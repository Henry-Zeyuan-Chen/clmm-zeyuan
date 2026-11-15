"""Descriptive frozen key geometry; not a mediation or task-performance test."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo))
from clmm.pretrain.adapter_registry import load_adapter_checkpoint
from clmm.hebbian.routing_decay import RoutingDecayAdapter
rows=[]
for rule in ('gradient','hebbian'):
 for seed in (90511,90515):
  parent,_=load_adapter_checkpoint(a.root/'parents'/f'{rule}-dense-{seed}.pt',torch.device('cpu'))
  for mapseed in (None,*range(90701,90706)):
   model=RoutingDecayAdapter.from_parent(parent,policy='content' if mapseed is None else 'random',inactive_decay='protected',random_seed=mapseed)
   raw=model.codebook.keys.detach();raw=raw/raw.norm(dim=-1,keepdim=True)
   with torch.no_grad():pre=torch.tanh(torch.einsum('kd,mdh->kmh',model.codebook.keys,model.key_encoder));pre=pre/pre.norm(dim=-1,keepdim=True)
   table=model.route_table;pair=[];learned=[]
   for i in range(64):
    for j in range(i+1,64):
     if table[i]!=table[j]:continue
     m=int(table[i]);pair.append(float(raw[i]@raw[j]));learned.append(float(pre[i,m]@pre[j,m]))
   rows.append(dict(parent=f'{rule}-dense-{seed}',policy='content' if mapseed is None else 'random',random_seed=mapseed,table_hash=model.routing_table_sha256,cardinalities=json.dumps(torch.bincount(table,minlength=4).tolist()),same_module_key_pairs=len(pair),raw_signed_cosine=float(np.mean(pair)),raw_squared_cosine=float(np.mean(np.array(pair)**2)),learned_pre_signed_cosine=float(np.mean(learned)),learned_pre_squared_cosine=float(np.mean(np.array(learned)**2))))
with (a.root/'analysis/router_geometry.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(dict(rows=len(rows),scope='Equal weighting of unordered within-module key pairs; excludes diagonal; frozen geometry, not causal mediation')))
