"""Audit saved CPU proof against original GPU batches; no model execution."""
import csv,hashlib,json,pathlib,numpy as np
OUT=pathlib.Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-key-binding-2026-11-15');P=OUT/'cpu-recheck'
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def prob(x):
 x=x.astype(np.float64);x=np.exp(x-x.max(-1,keepdims=True));return x/x.sum(-1,keepdims=True)
v=read(P/'verification.json');plan=read(OUT/'validation/cpu-proof-plan.json');actualplan=read(P/'plan.json');selected=read(P/'selected_rows.json');cpu=np.load(P/'cpu_logits.npy');gpu=np.load(P/'gpu_logits.npy')
assert v['status']=='passed' and v['models']==8 and v['queries']==len(selected)==2560 and cpu.shape==gpu.shape==(2560,256) and cpu.dtype==gpu.dtype==np.float32
assert np.isfinite(cpu).all() and np.isfinite(gpu).all() and plan['no_threshold_relaxation']
assert plan['absolute_logit_tolerance']==actualplan['max_logit_error']==v['plan']['max_logit_error']==.002
assert plan['absolute_probability_tolerance']==actualplan['max_probability_error']==v['plan']['max_probability_error']==.0002
assert plan['allowed_argmax_mismatches']==actualplan['prediction_mismatches_allowed']==v['plan']['prediction_mismatches_allowed']==0
script=OUT.parents[1]/'work/clmm-key-binding/cpu_recheck.py';assert sha(script)==plan['script_sha256']==v['script_sha256']
models={}
for path in (OUT/'gpu-main/result').rglob('metadata.json'):
 m=read(path)
 if m.get('mode') in ('initial','train'):models[m['checkpoint_sha256']]=path.parent
assert len(models)==8
cursor=0;groups_per_case=[]
for case in v['cases']:
 folder=models[case['checkpoint_sha256']];rows=csvrows(folder/'evaluation/predictions.csv');resources=csvrows(folder/'evaluation/resources.csv');raw=np.load(folder/'evaluation/answer_logits.npy');assert sha(folder/'checkpoint.pt')==case['checkpoint_sha256']
 seen=set()
 for check in case['checks']:
  ix=check['indices'];resource=resources[check['saved_batch']];assert json.loads(resource['row_indices'])==ix and len(ix)==16
  group,condition=check['group'],check['condition'];key=(group,condition);assert key not in seen;seen.add(key)
  first=next(r for r in resources if any(rows[i]['group']==group and rows[i]['condition']==condition for i in json.loads(r['row_indices'])));assert first==resource
  count=len(ix);assert np.array_equal(gpu[cursor:cursor+count],raw[ix])
  for j,i in enumerate(ix):assert selected[cursor+j]==dict(case=folder.name,**rows[i])
  le=float(np.abs(cpu[cursor:cursor+count]-gpu[cursor:cursor+count]).max());pe=float(np.abs(prob(cpu[cursor:cursor+count])-prob(gpu[cursor:cursor+count])).max());mm=int(np.sum(cpu[cursor:cursor+count].argmax(-1)!=gpu[cursor:cursor+count].argmax(-1)))
  assert le==check['max_logit_error'] and pe==check['max_probability_error'] and mm==check['prediction_mismatches']==0 and check['failed_rows']==[]
  assert le<=.002 and pe<=.0002;cursor+=count
 assert len(seen)==20 and case['queries']==320 and case['checkpoint_and_RNG_pure'];groups_per_case.append(len(seen))
assert cursor==2560
le=float(np.abs(cpu-gpu).max());pe=float(np.abs(prob(cpu)-prob(gpu)).max());mm=int(np.sum(cpu.argmax(-1)!=gpu.argmax(-1)))
assert le==v['max_logit_error'] and pe==v['max_probability_error'] and mm==v['prediction_mismatches']==0
result=dict(status='passed',kind='independent_saved_CPU_proof_artifact_audit',models_constructed=False,model_inference_by_reviewer=False,models=8,queries=2560,groups_per_model=groups_per_case,exact_original_GPU_batches=True,CPU_vectors_provenance_sha256=sha(P/'cpu_logits.npy'),GPU_vectors_provenance_sha256=sha(P/'gpu_logits.npy'),max_logit_error=le,max_probability_error=pe,prediction_mismatches=mm,original_tolerances_unchanged=True,prelaunch_script_hash_exact=True,runtime_weight_rng_purity_assertions_present=True,scope='Selected exact-batch CPU inference performed by coordinator, independently audited from saved vectors and source; not complete training reproduction')
(OUT/'analysis/cpu_proof_artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
