"""Selected actual inference/resume proof, separate from aggregate auditors."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,io,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--expected',type=Path);a=p.parse_args();sys.path.insert(0,str(a.source))
import numpy as np
import torch
from scripts.revision_local_experiment import load_checkpoint,slow_sha
from clmm.sleep import SleepSession
from scripts.kv_representation_control import equal_tree
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
started=time.perf_counter();packet=a.root/'packet';models=json.loads((packet/'models.json').read_text());data=json.loads((packet/'episodes.json').read_text());reports=[]
source_meta=json.loads((packet/'metadata.json').read_text())
for name,digest in source_meta['source_sha256'].items():assert sha(a.source/name)==digest
assert sha(packet/'episodes.json')==source_meta['data_sha256']
device=torch.device(a.device)
with torch.no_grad():
 for record in models:
  if time.perf_counter()-started>300:raise TimeoutError('Selected recheck exceeded 300 seconds')
  checkpoint=packet/record['checkpoint'];assert sha(checkpoint)==record['checkpoint_sha256']
  model,payload=load_checkpoint(checkpoint,device);model.eval().requires_grad_(False)
  before_slow=slow_sha(model);rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone() if a.device=='cuda' else None
  evaluation=(a.expected/record['name'] if a.expected else a.root/'runs'/record['name']/'evaluation')
  with (evaluation/'predictions.csv').open() as handle:rows=list(csv.DictReader(handle))
  scores=np.load(evaluation/'scores.npy',mmap_mode='r');index={(r['cell'],r['arm'],int(r['episode']),int(r['rank'])):i for i,r in enumerate(rows)}
  plans={(r['cell'],r['arm'],r['episode']):r['events'] for r in json.loads((evaluation/'plans.json').read_text()) if r['episode']<16}
  for cell,arms in [('gate_k16_r3',['online']),('pressure_k16_p2_d128',['online','continuous','periodic'])]:
   batch=data[cell][:16];queries=torch.tensor([[q['key'] for q in ep['queries']] for ep in batch],device=device)
   for arm in arms:
    session=SleepSession(model,16,arm,[ep['seed'] for ep in batch]);events=[[] for ep in batch];points=[]
    for i in range(len(batch[0]['keys'])):
     k=torch.tensor([ep['keys'][i] for ep in batch],device=device);v=torch.tensor([ep['values'][i] for ep in batch],device=device)
     emitted=session.prime(k,v) if i<batch[0]['boundary'] else session.observe(k,v)
     for b,group in enumerate(emitted):events[b].extend(group)
     if i in (batch[0]['boundary']-1,batch[0]['boundary'],21,22):
      before=session.read(queries).clone();raw=io.BytesIO();torch.save(session.export_state(),raw);raw.seek(0)
      session=SleepSession.from_state(model,torch.load(raw,map_location='cpu',weights_only=True))
      assert torch.equal(before,session.read(queries));points.append(i)
    assert all(not group for group in session.finish())
    assert all(group==plans[(cell,arm,b)] for b,group in enumerate(events))
    state=model.export_state(session.state);actual=session.read(queries).cpu().numpy();assert equal_tree(state,model.export_state(session.state))
    expected=np.stack([[scores[index[(cell,arm,b,q)]] for q in range(len(ep['queries']))] for b,ep in enumerate(batch)])
    stored=np.array([[int(rows[index[(cell,arm,b,q)]]['prediction']) for q in range(len(ep['queries']))] for b,ep in enumerate(batch)])
    error=float(np.abs(actual-expected).max());matches=np.array_equal(actual.argmax(-1),stored)
    assert error<=1e-5 and matches, dict(case=record['name'],cell=cell,arm=arm,max_abs_error=error,prediction_mismatches=int((actual.argmax(-1)!=stored).sum()))
    reports.append(dict(case=record['name'],cell=cell,arm=arm,queries=actual.shape[0]*actual.shape[1],max_abs_error=error,all_predictions_match=True,resume_indices=points,actual_writes=session.state.write_count,budget=session.budget(),checkpoint_sha256=sha(checkpoint)))
  assert before_slow==slow_sha(model) and torch.equal(rng,torch.random.get_rng_state())
  if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
  print(record['name'],flush=True)
result=dict(status='passed',scope='First 16 histories of neutral K16 gate and phase2 D128 with online/continuous/periodic across all12 fixed checkpoints; actual inference/resume, not full-grid numerical reproduction',models=12,queries=sum(r['queries'] for r in reports),max_abs_error=max(r['max_abs_error'] for r in reports),torch=str(torch.__version__),python=sys.version,numpy=np.__version__,device=a.device,device_name=torch.cuda.get_device_name() if a.device=='cuda' else 'CPU',source_sha256=sha(Path(__file__)),slurm_job=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.perf_counter()-started,cases=reports)
a.out.mkdir(parents=True,exist_ok=False);(a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
