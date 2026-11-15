"""Selected independent execution of saved reinforcement-stress inference."""
import argparse,csv,hashlib,io,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.source))
import numpy as np
import torch
from scripts.revision_local_experiment import load_checkpoint,slow_sha
from clmm.sleep import SleepSession
from scripts.kv_representation_control import equal_tree
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
started=time.perf_counter();meta=json.loads((a.root/'stress/metadata.json').read_text());assert meta['status']=='succeeded' and len(meta['cases'])==24
reports=[];cell='pressure_k8_r16_d64'
with torch.no_grad():
 for record in meta['cases']:
  if time.perf_counter()-started>180:raise TimeoutError('Stress selected proof exceeded180s')
  checkpoint=a.root/'trained'/record['checkpoint_relative_to_trained'];assert sha(checkpoint)==record['checkpoint_sha256']
  model,payload=load_checkpoint(checkpoint,torch.device('cpu'));model.eval().requires_grad_(False)
  before_slow=slow_sha(model);rng=torch.random.get_rng_state().clone()
  evaluation=a.root/'stress/runs'/record['name']/'evaluation'
  data=json.loads((evaluation/'episodes.json').read_text());batch=data[cell][:16]
  plans={(r['cell'],r['arm'],r['episode']):r['events'] for r in json.loads((evaluation/'plans.json').read_text())}
  with (evaluation/'predictions.csv').open() as h:rows=list(csv.DictReader(h))
  scores=np.load(evaluation/'scores.npy');index={(r['cell'],r['arm'],int(r['episode']),int(r['rank'])):i for i,r in enumerate(rows)}
  session=SleepSession(model,16,'online',[ep['seed'] for ep in batch]);events=[[] for ep in batch];points=[]
  queries=torch.tensor([[q['key'] for q in ep['queries']] for ep in batch])
  for i in range(len(batch[0]['keys'])):
   k=torch.tensor([ep['keys'][i] for ep in batch]);v=torch.tensor([ep['values'][i] for ep in batch])
   emitted=session.prime(k,v) if i<batch[0]['boundary'] else session.observe(k,v)
   for b,group in enumerate(emitted):events[b].extend(group)
   if i in (batch[0]['boundary']-1,batch[0]['boundary']):
    before=session.read(queries).clone();raw=io.BytesIO();torch.save(session.export_state(),raw);raw.seek(0)
    session=SleepSession.from_state(model,torch.load(raw,map_location='cpu',weights_only=True))
    assert torch.equal(before,session.read(queries));points.append(i)
  assert all(not group for group in session.finish())
  assert all(group==plans[(cell,'online',b)] for b,group in enumerate(events))
  state=model.export_state(session.state);actual=session.read(queries).numpy();assert equal_tree(state,model.export_state(session.state))
  expected=np.stack([[scores[index[(cell,'online',b,q)]] for q in range(len(ep['queries']))] for b,ep in enumerate(batch)])
  stored=np.array([[int(rows[index[(cell,'online',b,q)]]['prediction']) for q in range(len(ep['queries']))] for b,ep in enumerate(batch)])
  error=float(np.abs(actual-expected).max());assert error<=1e-5 and np.array_equal(actual.argmax(-1),stored)
  assert before_slow==slow_sha(model) and torch.equal(rng,torch.random.get_rng_state())
  reports.append(dict(case=record['name'],queries=actual.shape[0]*actual.shape[1],max_abs_error=error,all_predictions_match=True,resume_indices=points,actual_writes=session.state.write_count,checkpoint_sha256=sha(checkpoint)))
result=dict(status='passed',scope='First16historiesofR16D64 acrossall24existingtrainedcheckpoints; actualCPUinference/resume, not fullstress numerical reproduction',models=24,queries=sum(r['queries'] for r in reports),max_abs_error=max(r['max_abs_error'] for r in reports),torch=str(torch.__version__),source_sha256=sha(Path(__file__)),elapsed_seconds=time.perf_counter()-started,cases=reports)
(a.root/'stress-cpu-recheck').mkdir(exist_ok=False);(a.root/'stress-cpu-recheck/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
