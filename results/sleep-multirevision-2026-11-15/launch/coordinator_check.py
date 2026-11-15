"""Coordinator recomputation of decisive numbers directly from saved rows."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]/'outputs/clmm-sleep-multirevision-2026-11-15'
packet=ROOT/'packet';data=json.loads((packet/'episodes.json').read_text());models=json.loads((packet/'models.json').read_text())
vectors={};gates={};score_count=0;write_count=0;resource_count=0
for model in models:
 folder=ROOT/'runs'/model['name'];meta=json.loads((folder/'metadata.json').read_text());assert meta['status']=='succeeded' and meta['additional_optimizer_steps']==0
 assert hashlib.sha256((packet/model['checkpoint']).read_bytes()).hexdigest()==model['checkpoint_sha256']
 with (folder/'evaluation/predictions.csv').open() as f:rows=list(csv.DictReader(f))
 score_count+=len(rows);groups={};own={}
 for row in rows:
  cell,arm,i,rank=row['cell'],row['arm'],int(row['episode']),int(row['rank']);q=data[cell][i]['queries'][rank]
  assert int(row['correct'])==int(int(row['prediction'])==q['target'])
  latest={}
  for k,v in zip(data[cell][i]['keys'],data[cell][i]['values']):latest[k]=v
  assert q['target']==latest[q['key']]
  if cell.startswith('pressure'):
   groups.setdefault((cell,arm,q['category'],i),[]).append(int(row['correct']))
   if q['category']!='recent':own.setdefault((cell,arm,i),[]).append(int(row['correct']))
 for category in ('stable','revised'):
  for arm in ('continuous','periodic'):
   vectors[(model['name'],arm,category)]=np.array([np.mean([np.mean(groups[(f'pressure_k16_p{phase}_d128',arm,category,i)]) for phase in (0,2)]) for i in range(128)])
 for phase in (0,2):
  for arm in ('online','repeat','continuous','periodic'):
   gates[(model['name'],phase,arm)]=float(np.mean([all(own[(f'pressure_k16_p{phase}_d0',arm,i)]) for i in range(128)]))
 with (folder/'evaluation/resources.csv').open() as f:resources=list(csv.DictReader(f))
 resource_count+=len(resources);write_count+=sum(int(r['actual_writes'])*int(r['batch']) for r in resources)
assert (score_count,write_count,resource_count)==(310272,8474112,3456)
ids=np.random.Generator(np.random.PCG64(91104)).integers(0,128,size=(5000,128));primary=[]
for cat in ('stable','revised'):
 arms={}
 for arm in ('continuous','periodic'):
  content=np.mean([vectors[(r['name'],arm,cat)] for r in models if r['policy']=='content'],axis=0)
  random=np.mean([vectors[(r['name'],arm,cat)] for r in models if r['policy']=='random'],axis=0)
  arms[arm]=.5*(content+random)
 difference=arms['periodic']-arms['continuous'];lower,upper=np.quantile(difference[ids].mean(axis=1),[.0125,.9875])
 primary.append(dict(category=cat,continuous=float(arms['continuous'].mean()),periodic=float(arms['periodic'].mean()),estimate=float(difference.mean()),lower=float(lower),upper=float(upper)))
analysis=json.loads((ROOT/'analysis/summary.json').read_text())
for mine in primary:
 other=next(r for r in analysis['primary'] if r['category']==mine['category'])
 for key in ('estimate','lower','upper'):assert np.isclose(mine[key],other[key],atol=1e-12)
result=dict(status='passed',scope='Direct row/target, per-history primary weighting/bootstrap and total resource recomputation; independent of analysis modules',primary=primary,score_vectors=score_count,causal_writes=write_count,resource_batches=resource_count,own_D0_all_pass=all(v>=.8 for v in gates.values()),own_D0=[dict(case=k[0],phase=k[1],arm=k[2],all6correct=v) for k,v in gates.items()])
(ROOT/'validation').mkdir(exist_ok=True);(ROOT/'validation/coordinator_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='own_D0'},indent=2))
