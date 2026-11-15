"""Bounded same-environment reload/inference check of fixed final checkpoints."""
import argparse
from contextlib import nullcontext
import csv
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.repo))
import numpy as np
import torch
from clmm.pretrain import load_checkpoint,reset_memory_at_segments
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True);torch.manual_seed(89102)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
a.out.mkdir(parents=True,exist_ok=False)
start=time.perf_counter();results=[]
def key(r):
    return r['cell'],r['family'],int(r['episode']),int(r['query_index']),r['condition']
for name in ('duplicate32-68101','paired32-68101','duplicate32-68105','paired32-68105','paired128-68101'):
    folder=a.root/name/'step-1600'
    model,_,_=load_checkpoint(folder/'checkpoint.pt',map_location='cuda');model.eval()
    with (folder/'evaluation/predictions.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    saved=np.load(folder/'evaluation/answer_logits.npy',allow_pickle=False)
    groups=[]
    for condition in ('online','reset_memory'):
        groups.append([i for i,r in enumerate(rows) if r['cell']=='acquire_k2' and r['family']=='seen' and r['condition']==condition][:16])
    wanted={key(rows[i]) for group in groups for i in group};prefixes={}
    with gzip.open(folder/'evaluation/evaluation_data.jsonl.gz','rt') as f:
        for line in f:
            item=json.loads(line)
            if item['type']=='sample' and key(item['row']) in wanted:
                prefixes[key(item['row'])]=item['prefix'].encode('ascii')
    for indices in groups:
        if time.perf_counter()-start>150:raise TimeoutError('Recheck150seconds')
        assert len(indices)==16
        condition=rows[indices[0]]['condition']
        x=torch.tensor([list(prefixes[key(rows[i])]) for i in indices],device='cuda')
        context=reset_memory_at_segments(model) if condition=='reset_memory' else nullcontext()
        with torch.no_grad(),context:logits=model(x)[:,-1].cpu().numpy()
        results.append(dict(run=name,condition=condition,queries=16,
            max_abs_logit_difference=float(np.max(np.abs(logits-saved[indices]))),
            prediction_mismatches=int(np.sum(logits.argmax(-1)!=saved[indices].argmax(-1)))))
record=dict(status='passed' if all(r['max_abs_logit_difference']<=1e-6 and r['prediction_mismatches']==0 for r in results) else 'mismatch',
            results=results,absolute_tolerance=1e-6,queries=160,elapsed_seconds=time.perf_counter()-start,
            scope='Final fixed checkpoints only; same-environment inference, no training rerun',
            environment=dict(python=sys.version,torch=str(torch.__version__),device=torch.cuda.get_device_name(),
                platform=platform.platform(),packages={d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()}))
(a.out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
shutil.copy2(__file__,a.out/Path(__file__).name)
print(json.dumps({k:v for k,v in record.items() if k!='environment'}),flush=True)
if record['status']!='passed':sys.exit(1)
