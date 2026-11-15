"""Direct raw-row recomputation of primary freshness effects and budgets."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]/'outputs/clmm-bounded-freshness-2026-11-15'
packet=ROOT/'packet';data=json.loads((packet/'episodes.json').read_text());models=json.loads((packet/'models.json').read_text())
vectors={};totals=dict(scores=0,writes=0,batches=0)
for model in models:
 folder=ROOT/'runs'/model['name'];meta=json.loads((folder/'metadata.json').read_text());assert meta['status']=='succeeded' and meta['additional_optimizer_steps']==0
 assert hashlib.sha256((packet/model['checkpoint']).read_bytes()).hexdigest()==model['checkpoint_sha256']
 with (folder/'evaluation/predictions.csv').open() as f:rows=list(csv.DictReader(f))
 totals['scores']+=len(rows);groups={}
 for row in rows:
  cell,arm,i,rank=row['cell'],row['arm'],int(row['episode']),int(row['rank']);q=data[cell][i]['queries'][rank]
  assert int(row['correct'])==int(int(row['prediction'])==q['target'])
  latest=dict(zip(data[cell][i]['keys'],data[cell][i]['values']));assert q['target']==latest[q['key']]
  if cell.startswith('pressure'):groups.setdefault((cell,arm,q['category'],i),[]).append(int(row['correct']))
 for category in ('stable','revised'):
  for freshness in ('literal','bounded'):
   suffix='' if freshness=='literal' else '_bounded'
   vectors[(model['name'],freshness,category)]=np.array([np.mean([np.mean(groups[(f'pressure_k16_p{p}_d128',schedule+suffix,category,i)]) for p in (0,2) for schedule in ('continuous','periodic')]) for i in range(128)])
 with (folder/'evaluation/resources.csv').open() as f:resources=list(csv.DictReader(f))
 totals['batches']+=len(resources);totals['writes']+=sum(int(r['actual_writes'])*int(r['batch']) for r in resources)
assert totals==dict(scores=433152,writes=13192704,batches=4608)
ids=np.random.Generator(np.random.PCG64(91204)).integers(0,128,size=(5000,128));primary=[]
for category in ('stable','revised'):
 arms={}
 for fresh in ('literal','bounded'):
  content=np.mean([vectors[(r['name'],fresh,category)] for r in models if r['policy']=='content'],axis=0)
  random=np.mean([vectors[(r['name'],fresh,category)] for r in models if r['policy']=='random'],axis=0)
  arms[fresh]=.5*(content+random)
 difference=arms['bounded']-arms['literal'];lower,upper=np.quantile(difference[ids].mean(axis=1),[.0125,.9875])
 primary.append(dict(category=category,literal=float(arms['literal'].mean()),bounded=float(arms['bounded'].mean()),estimate=float(difference.mean()),lower=float(lower),upper=float(upper)))
analysis=json.loads((ROOT/'analysis/summary.json').read_text())
for mine in primary:
 other=next(r for r in analysis['primary'] if r['category']==mine['category'])
 for key in ('estimate','lower','upper'):assert np.isclose(mine[key],other[key],atol=1e-12)
result=dict(status='passed',scope='Direct raw-row latest-target, primary weighting/bootstrap and resource recomputation; no analysis module imports',primary=primary,**totals)
(ROOT/'validation').mkdir(exist_ok=True);(ROOT/'validation/coordinator_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
