"""Secondary nominal paired-history initial-to-final training contrasts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
from raw_eval_tools import read_csv,write_csv
from paired_analysis_tools import SEEDS,ROUTES

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
meta=json.loads((a.root/'metadata.json').read_text())
assert meta['status']=='succeeded' and meta['args']['mode']=='train' and len(meta['cases'])==24
values={}
raw_counts=defaultdict(int)
for case in meta['cases']:
    route='content' if case['policy']=='content' else f"random-{case['random_seed']}"
    saved_ids=None
    for stage,folder in [('initial','initial-evaluation'),('final','evaluation')]:
        rows=read_csv(a.root/'runs'/case['name']/folder/'predictions.csv')
        selected=[r for r in rows if r['arm']=='online' and
                  ((r['cell']=='revise2' and r['category']=='revised') or
                   (r['cell']=='pressure_k8_d64' and r['category']=='stable'))]
        ids=[tuple(r[k] for k in ('cell','episode','rank','key','target','old')) for r in selected]
        if saved_ids is None:
            saved_ids=ids
        else:
            assert ids==saved_ids
        for cell,category,label in [('revise2','revised','short_revision'),('pressure_k8_d64','stable','K8D64_stable')]:
            group=defaultdict(list)
            subset=[r for r in selected if r['cell']==cell and r['category']==category]
            for r in subset:
                correct=int(r['prediction'])==int(r['target'])
                assert int(r['correct'])==correct
                group[int(r['episode'])].append(correct)
            assert set(group)==set(range(128))
            assert all(len(v)==(1 if category=='revised' else 2) for v in group.values())
            values[(stage,case['revision_rule'],case['parent_seed'],route,label)]=np.asarray([np.mean(group[i]) for i in range(128)])
            raw_counts[stage]+=len(subset)
draws=np.random.Generator(np.random.PCG64(90804)).integers(0,128,size=(5000,128))
rows=[]
detail=[]
for rule in ('ema','local_delta'):
    for family,routes in [('content',('content',)),('mean5random',ROUTES[1:])]:
        for metric in ('short_revision','K8D64_stable'):
            initial=np.mean([values[('initial',rule,s,r,metric)] for s in SEEDS for r in routes],axis=0)
            final=np.mean([values[('final',rule,s,r,metric)] for s in SEEDS for r in routes],axis=0)
            delta=final-initial
            lo,hi=np.quantile(delta[draws].mean(1),[.025,.975])
            rows.append(dict(rule=rule,route_family=family,metric=metric,initial=float(initial.mean()),
                             final=float(final.mean()),final_minus_initial=float(delta.mean()),
                             lower95=float(lo),upper95=float(hi),histories=128,bootstrap_draws=5000,
                             bootstrap_seed=90804,scope='Secondary predeclared initial/final paired comparison; nominal95, fixed parents/maps, no primary change'))
    for s in SEEDS:
        for route in ROUTES:
            for metric in ('short_revision','K8D64_stable'):
                initial=values[('initial',rule,s,route,metric)]
                final=values[('final',rule,s,route,metric)]
                detail.append(dict(rule=rule,parent_seed=s,route=route,metric=metric,
                                   initial=float(initial.mean()),final=float(final.mean()),
                                   final_minus_initial=float((final-initial).mean())))
write_csv(a.out/'initial_to_final_contrasts.csv',rows)
write_csv(a.out/'initial_to_final_permap.csv',detail)
report=dict(status='passed',source_commit=meta['source_commit'],initial_final_query_ids_identical=True,
            checked_endpoint_rows=dict(raw_counts),contrasts=rows,
            scope='Secondary saved-output paired analysis only; no model execution or selection.')
(a.out/'initial_to_final_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
