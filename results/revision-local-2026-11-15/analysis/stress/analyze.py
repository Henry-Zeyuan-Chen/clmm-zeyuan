"""Independent exploratory stress raw audit and paired fixed-checkpoint analysis."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from raw_eval_tools import audit_eval,read_csv,sha,signatures_from_arrays,write_csv
from paired_analysis_tools import SEEDS,ROUTES

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
meta=json.loads((a.root/'metadata.json').read_text())
registry=json.loads((Path(meta['args']['trained'])/'metadata.json').read_text())
reference={c['name']:c for c in registry['cases']}
assert meta['offline_training'] is False and meta['expected_cases']==24
assert meta['source_commit']=='c9f5c153f786f1b82ac383d01c1d18c6c595946b'
assert sha(a.root/'driver_snapshot.py')==meta['driver_sha256']
assert sha(a.root/'design.json')==sha(a.out/'design.json')==meta['design_sha256']
source_reference=json.loads((a.out.parent/'final_source_audit.json').read_text())
source=Path(meta['args']['source'])
for name,digest in meta['source_sha256'].items():
    assert sha(source/name)==digest==source_reference['files'][name]
episodes=json.loads((a.root/'episodes.json').read_text())
assert set(episodes)=={'acquire2','acquire4','revise2',*(f'pressure_k8_r{r}_d{d}' for r in (1,4,16) for d in (0,64))}
assert all(len(eps)==128 for eps in episodes.values())
assert sha(a.root/'episodes.json')==meta['data']['episodes_sha256']
excluded=signatures_from_arrays(Path(meta['args']['trained'])/'old_training_exclusions.npz')
excluded|=signatures_from_arrays(Path(meta['args']['trained'])/'training_data.npz')
for label in ('prior','frozen','trained'):
    old=json.loads((Path(meta['args'][label])/'episodes.json').read_text())
    excluded|={ep['signature'] for eps in old.values() for ep in eps}
assert len(excluded)==meta['data']['excluded_signature_union']
assert not excluded.intersection(ep['signature'] for eps in episodes.values() for ep in eps)
spec=importlib.util.spec_from_file_location('_stress_streams_audit',source/'clmm/pretrain/kv_streams.py')
streams=importlib.util.module_from_spec(spec)
spec.loader.exec_module(streams)
short=streams.evaluation_data(91023,128,128,excluded,'gate')
assert all(episodes[k]==v for k,v in short.items())
joint=excluded|{ep['signature'] for eps in short.values() for ep in eps}
construction=json.loads((a.root/'construction.json').read_text())
assert sha(a.root/'episodes.json')==construction['episodes_sha256']
rng=np.random.default_rng(np.random.SeedSequence([91023,8,2,500]))
accepted=[]
attempts=0
while len(accepted)<128:
    attempts+=1
    candidate=int(rng.integers(2_000_000_000,3_000_000_000))
    derived={}
    for r in (1,4,16):
        for d in (0,64):
            base=streams.episode(candidate,8,d,2,True)
            keys=list(base['keys'][:8])
            values=list(base['values'][:8])
            for repeat in range(r-1):
                for target_index in (2,3):
                    keys.append(base['keys'][target_index])
                    values.append(base['values'][target_index])
            keys.extend(base['keys'][8:])
            values.extend(base['values'][8:])
            derived[(r,d)]={**base,'keys':keys,'values':values,'boundary':10+2*(r-1),
                            'old_reinforcement':r,'signature':streams.signature(keys,values)}
    signatures={ep['signature'] for ep in derived.values()}
    if len(signatures)!=6 or joint.intersection(signatures):
        continue
    i=len(accepted)
    assert all(episodes[f'pressure_k8_r{r}_d{d}'][i]==ep for (r,d),ep in derived.items())
    accepted.append(candidate)
    joint|=signatures
assert accepted==construction['accepted_base_seeds'] and attempts==construction['attempts']

cases={}
audits=[]
metrics=[]
incomplete=[]
gates=[]
draws=np.random.Generator(np.random.PCG64(91004)).integers(0,128,size=(5000,128))
blocks=np.random.Generator(np.random.PCG64(91004)).integers(0,8,size=(5000,8))
for case in meta['cases']:
    folder=a.root/'runs'/case['name']
    assert json.loads((folder/'case.json').read_text())==case
    original=reference[case['name']]
    for key in ('config','route_table','slow_sha256','checkpoint_sha256','revision_rule','parent_seed'):
        assert case[key]==original[key]
    assert sha(Path(meta['args']['trained'])/case['checkpoint_relative_to_trained'])==case['checkpoint_sha256']
    assert case['training_steps']==0 and case['offline_training'] is False
    if case['status']!='succeeded':
        incomplete.append(case['name'])
        continue
    result=audit_eval(folder/'evaluation',episodes,case['route_table'],case['revision_rule'])
    assert result['audit']['files']['scores.npy']==case['evaluation']['scores_sha256']
    assert case['evaluation']['pure_reads'] and case['evaluation']['slow_parameters_and_rng_unchanged']
    route='content' if case['policy']=='content' else f"random-{case['random_seed']}"
    ident=(case['revision_rule'],case['parent_seed'],route)
    assert ident not in cases
    cases[ident]=result
    audits.append(dict(case=case['name'],**result['audit']))
    for row in result['summaries']:
        metrics.append(dict(case=case['name'],rule=ident[0],parent_seed=ident[1],route=route,**row))
    ac=result['all_correct']
    shortgate={c:float(ac[(c,'online')].mean()) for c in ('acquire2','acquire4','revise2')}
    diff=ac[('acquire2','online')]-ac[('acquire2','rolled')]
    lo=float(np.quantile(diff.reshape(8,16).mean(1)[blocks].mean(1),.025))
    shortpass=shortgate['acquire2']>=.9 and shortgate['acquire4']>=.8 and shortgate['revise2']>=.9 and lo>0
    for r in (1,4,16):
        gates.append(dict(case=case['name'],rule=ident[0],parent_seed=ident[1],route=route,R=r,**shortgate,
                          rolled_lower95=lo,short_gate=bool(shortpass),
                          D0_four_target_all_correct=float(ac[(f'pressure_k8_r{r}_d0','online')].mean())))
expected={(rule,s,r) for rule in ('ema','local_delta') for s in SEEDS for r in ROUTES}
complete=set(cases)==expected and meta['status']=='succeeded'
write_csv(a.out/'per_case_metrics.csv',metrics)
write_csv(a.out/'short_and_stress_d0_gates.csv',gates)
contrasts=[]
aggregate=[]
if complete:
    def value(rule,routes,cell,category):
        return np.mean([cases[(rule,s,r)]['vectors'][(cell,'online',category)] for s in SEEDS for r in routes],axis=0)
    for family,routes in [('content',('content',)),('mean5random',ROUTES[1:])]:
        for r in (1,4,16):
            for d in (0,64):
                cell=f'pressure_k8_r{r}_d{d}'
                for category in ('stable','revised'):
                    arms={rule:value(rule,routes,cell,category) for rule in ('ema','local_delta')}
                    for rule,v in arms.items():
                        lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
                        aggregate.append(dict(rule=rule,route_family=family,R=r,D=d,cell=cell,category=category,
                                              accuracy=float(v.mean()),lower95=float(lo),upper95=float(hi)))
                    delta=arms['local_delta']-arms['ema']
                    lo,hi=np.quantile(delta[draws].mean(1),[.025,.975])
                    threshold=0 if category=='revised' else -.03
                    contrasts.append(dict(route_family=family,R=r,D=d,cell=cell,category=category,
                                          candidate=float(arms['local_delta'].mean()),baseline=float(arms['ema'].mean()),
                                          estimate=float(delta.mean()),lower95=float(lo),upper95=float(hi),
                                          prespecified_exploratory_focus=bool(r==16 and d==64),threshold=threshold,
                                          exploratory_inequality_pass=bool(lo>threshold),
                                          scope='Post-outcome exploratory; nominal95 paired-history interval, not original primary'))
    write_csv(a.out/'aggregate.csv',aggregate)
    write_csv(a.out/'contrasts.csv',contrasts)
report=dict(status='passed',full_phase_complete=complete,metadata_status=meta['status'],cases_audited=len(cases),
            incomplete_cases=incomplete,missing_cases=sorted(set(reference)-{x['case'] for x in audits}),
            predictions=sum(x['predictions'] for x in audits),causal_writes=sum(x['writes'] for x in audits),
            resource_batches=sum(x['resource_batches'] for x in audits),raw_audits=audits,
            source_commit=meta['source_commit'],driver_sha256=meta['driver_sha256'],
            exact_data_reconstruction=True,joint_RD_pairing_verified=True,zero_exclusion_overlap=True,
            source_immutability_verified=True,original_primary_unchanged=True,
            focus=[x for x in contrasts if x['prespecified_exploratory_focus']],
            bootstrap=dict(draws=5000,seed=91004,unit='paired whole history; fixed parents/maps',interval='nominal95'),
            scope='Exploratory post-outcome inference follow-up; no optimizer or rule selection; audit arithmetic imports no evaluator.')
(a.out/'raw_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='raw_audits'},indent=2))
