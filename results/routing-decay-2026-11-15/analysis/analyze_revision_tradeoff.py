"""Post-outcome exploratory paired revision/obsolete analysis; not co-primary."""
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
meta=json.loads((root/'main/metadata.json').read_text());values={};detail=[]
for case in meta['cases']:
 with (root/'main/runs'/case['name']/'evaluation/predictions.csv').open() as h:rows=[r for r in csv.DictReader(h) if r['cell']=='revise2' and r['arm']=='online']
 byep={e:[r for r in rows if int(r['episode'])==e] for e in range(128)}
 metrics=dict(both_correct=np.array([all(int(r['correct']) for r in byep[e]) for e in range(128)],dtype=float),
  revised_correct=np.array([int(next(r for r in byep[e] if r['category']=='revised')['correct']) for e in range(128)],dtype=float),
  obsolete=np.array([int(next(r for r in byep[e] if r['category']=='revised')['obsolete']) for e in range(128)],dtype=float))
 route='content' if case['policy']=='content' else f"random-{case['random_seed']}"
 values[(case['rule'],case['parent_seed'],route,case['inactive_decay'])]=metrics
 detail.append(dict(case=case['name'],rule=case['rule'],route=route,parent_seed=case['parent_seed'],inactive_decay=case['inactive_decay'],**{k:float(v.mean()) for k,v in metrics.items()}))
def write(name,rows):
 with (root/'analysis'/name).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
write('revision_tradeoff_per_parent.csv',detail)
draw=np.random.Generator(np.random.PCG64(90704)).integers(0,128,size=(5000,128));contrasts=[]
for rule in ('gradient','hebbian'):
 for route in ('content','mean5random'):
  maps=('content',) if route=='content' else tuple(f'random-{s}' for s in range(90701,90706))
  for metric in ('both_correct','revised_correct','obsolete'):
   arms={d:np.mean([values[(rule,s,m,d)][metric] for s in (90511,90515) for m in maps],axis=0) for d in ('protected','global')}
   delta=arms['protected']-arms['global'];lo,hi=np.quantile(delta[draw].mean(1),[.025,.975])
   contrasts.append(dict(rule=rule,route=route,metric=metric,protected=float(arms['protected'].mean()),global_decay=float(arms['global'].mean()),protected_minus_global=float(delta.mean()),lower95=float(lo),upper95=float(hi),scope='Exploratory post-outcome contrast; nominal95 fixed-parent/map paired-history interval; not co-primary'))
write('revision_tradeoff_contrasts.csv',contrasts)
(root/'analysis/revision_tradeoff_note.txt').write_text('Contrasts added after inspecting main outcomes. The original revise2 gate and its threshold remain unchanged. These secondary intervals describe the128 paired short-revision histories for fixed parents/maps and are not multiplicity-adjusted confirmatory tests. All maps/parents retained.\n')
print(json.dumps([r for r in contrasts if r['rule']=='hebbian' and r['metric']=='both_correct'],indent=2))
