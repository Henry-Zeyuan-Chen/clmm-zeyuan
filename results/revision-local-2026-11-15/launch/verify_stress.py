"""Coordinator cross-check of stress construction, scores, contrasts and coverage."""
import argparse,csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=a.root;meta=json.loads((root/'stress/metadata.json').read_text());assert meta['status']=='succeeded' and len(meta['cases'])==24 and not meta['offline_training']
assert sha(root/'stress/driver_snapshot.py')==meta['driver_sha256']
assert all(sha(a.source/name)==digest for name,digest in meta['source_sha256'].items())
design=json.loads((root/'stress/design.json').read_text());assert design['recorded_utc']<=meta['started_utc'] and design['focus']['cell']=='pressure_k8_r16_d64'
data=json.loads((root/'stress/episodes.json').read_text());assert sha(root/'stress/episodes.json')==meta['data']['episodes_sha256']
construction=json.loads((root/'stress/construction.json').read_text());seeds=construction['accepted_base_seeds'];assert len(seeds)==128
spec=importlib.util.spec_from_file_location('_coordinator_stress_streams',a.source/'clmm/pretrain/kv_streams.py');streams=importlib.util.module_from_spec(spec);spec.loader.exec_module(streams)
for r in (1,4,16):
 for d in (0,64):
  cell=f'pressure_k8_r{r}_d{d}'
  for index,seed in enumerate(seeds):
   expected=streams.episode(seed,8,d,2,True);initial_keys=expected['keys'][:8];initial_values=expected['values'][:8]
   expected['keys']=initial_keys+[initial_keys[2],initial_keys[3]]*(r-1)+expected['keys'][8:]
   expected['values']=initial_values+[initial_values[2],initial_values[3]]*(r-1)+expected['values'][8:]
   expected.update(boundary=10+2*(r-1),old_reinforcement=r)
   expected['signature']=streams.signature(expected['keys'],expected['values']);ep=data[cell][index];assert ep==expected
   assert len(ep['keys'])==ep['boundary']+d
   for k in (2,3):
    key=initial_keys[k];old=initial_values[k]
    assert sum(x==key and v==old for x,v in zip(ep['keys'][:ep['boundary']-2],ep['values'][:ep['boundary']-2]))==r
    assert ep['keys'][ep['boundary']-4+k]==key and ep['values'][ep['boundary']-4+k]!=old
   latest=dict(zip(ep['keys'],ep['values']))
   assert all(latest[q['key']]==q['target'] for q in ep['queries'])
vectors={};required=[];count=0
for case in meta['cases']:
 assert case['status']=='succeeded' and case['training_steps']==0
 checkpoint=root/'trained'/case['checkpoint_relative_to_trained'];assert sha(checkpoint)==case['checkpoint_sha256']
 folder=root/'stress/runs'/case['name']
 for name in ('case.json','evaluation/episodes.json','evaluation/plans.json','evaluation/scores.npy','evaluation/predictions.csv','evaluation/summary.csv','evaluation/resources.csv'):
  path=folder/name;assert path.is_file() and path.stat().st_size>0;required.append(str(path.relative_to(root)))
 with (folder/'evaluation/predictions.csv').open() as h:rows=list(csv.DictReader(h))
 scores=np.load(folder/'evaluation/scores.npy');assert scores.shape==(6528,16) and len(rows)==6528
 pred=scores.argmax(-1);assert np.array_equal(pred,[int(r['prediction']) for r in rows]);assert np.array_equal(pred==[int(r['target']) for r in rows],[bool(int(r['correct'])) for r in rows])
 count+=len(rows)
 for category in ('stable','revised'):
  values=[]
  for e in range(128):
   selected=[int(r['correct']) for r in rows if r['cell']=='pressure_k8_r16_d64' and int(r['episode'])==e and r['category']==category]
   assert len(selected)==2;values.append(np.mean(selected))
  vectors.setdefault((case['policy'],case['revision_rule'],category),[]).append(values)
draw=np.random.Generator(np.random.PCG64(91004)).integers(0,128,size=(5000,128));focus=[]
with (root/'analysis/stress/contrasts.csv').open() as h:reported=list(csv.DictReader(h))
for policy in ('content','random'):
 for category in ('stable','revised'):
  candidate=np.mean(vectors[(policy,'local_delta',category)],axis=0);baseline=np.mean(vectors[(policy,'ema',category)],axis=0);delta=candidate-baseline;lo,hi=np.quantile(delta[draw].mean(1),[.025,.975])
  family='content' if policy=='content' else 'mean5random'
  ref=next(r for r in reported if r['route_family']==family and r['cell']=='pressure_k8_r16_d64' and r['category']==category)
  assert abs(float(ref['estimate'])-delta.mean())<1e-12 and abs(float(ref['lower95'])-lo)<1e-12 and abs(float(ref['upper95'])-hi)<1e-12
  focus.append(dict(family=family,category=category,candidate=float(candidate.mean()),baseline=float(baseline.mean()),delta=float(delta.mean()),lower95=float(lo),upper95=float(hi)))
assert count==156672 and len(required)==168
audit=json.loads((root/'analysis/stress/raw_audit.json').read_text());assert audit['status']=='passed'
proof=json.loads((root/'stress-cpu-recheck/verification.json').read_text());assert proof['status']=='passed' and proof['models']==24 and proof['queries']==1920
report=dict(status='passed',scope='Additionalexploratoryfollowup; originaltrainedprimaryremainsfalse',cases=24,scorevectors=count,expected_causal_writes=1078272,required_case_files=len(required),source_commit=meta['source_commit'],driver_sha256=meta['driver_sha256'],semantic_stream_reconstruction=True,old_reinforcement_counts_exact=True,query_targets_equal_latest_observation=True,all_argmax_flags_match=True,paired_focus_CIs_independently_recomputed=True,focus=focus,selected_CPUproof={k:proof[k] for k in ('models','queries','max_abs_error')})
(root/'validation/stress_coverage.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
