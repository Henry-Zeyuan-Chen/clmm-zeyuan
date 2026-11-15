"""Selected score/actual-session resume proof across both complete phases."""
import argparse,csv,hashlib,io,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();sys.path.insert(0,str(a.repo))
import numpy as np
import torch
from scripts.revision_local_experiment import case_specs,load_checkpoint
from clmm.sleep import SleepSession
from scripts.kv_representation_control import equal_tree
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();device=torch.device(a.device);results=[];verified=[]
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def deadline():
  if time.perf_counter()-started>390:raise TimeoutError('Selected proof exceeded390s')
 with torch.no_grad():
  for phase,mode in (('frozen','frozen'),('trained','train')):
   meta=json.loads((a.root/phase/'metadata.json').read_text());assert meta['status']=='succeeded' and meta['scope']=='complete predeclared phase'
   expected={r['name'] for r in case_specs(mode)};assert {r['name'] for r in meta['cases']}==expected
   for case in meta['cases']:
    folder=a.root/phase/'runs'/case['name'];assert case['status']=='succeeded'
    for stage in (('evaluation',) if phase=='frozen' else ('initial-evaluation','evaluation')):
     deadline();name=phase+':'+case['name']+':'+stage;checkpoint=folder/('initial_checkpoint.pt' if stage=='initial-evaluation' else 'checkpoint.pt')
     model,payload=load_checkpoint(checkpoint,device);assert model.config.to_dict()==case['config'];model.eval().requires_grad_(False)
     slow={k:t.clone() for k,t in model.state_dict().items()};cpu_rng=torch.random.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
     evaluation=folder/stage;data=json.loads((evaluation/'episodes.json').read_text());plans={(r['cell'],r['arm'],r['episode']):r['events'] for r in json.loads((evaluation/'plans.json').read_text())}
     with (evaluation/'predictions.csv').open() as h:rows=list(csv.DictReader(h))
     scores=np.load(evaluation/'scores.npy');index={(r['cell'],r['arm'],int(r['episode']),int(r['rank'])):i for i,r in enumerate(rows)}
     for cell in ('acquire2','pressure_k8_d64'):
      deadline();batch=data[cell][:16];session=SleepSession(model,16,'online',[ep['seed'] for ep in batch]);emitted=[[] for ep in batch];points=[]
      for i in range(len(batch[0]['keys'])):
       k=torch.tensor([ep['keys'][i] for ep in batch],device=device);v=torch.tensor([ep['values'][i] for ep in batch],device=device)
       events=session.prime(k,v) if i<batch[0]['boundary'] else session.observe(k,v)
       for b,items in enumerate(events):emitted[b].extend(items)
       if i in (batch[0]['boundary']-1,batch[0]['boundary']):
        raw=io.BytesIO();torch.save(session.export_state(),raw);raw.seek(0);session=SleepSession.from_state(model,torch.load(raw,map_location='cpu',weights_only=True));points.append('prefill' if i==batch[0]['boundary']-1 else 'midblock')
      assert all(not items for items in session.finish());assert all(got==plans[(cell,'online',b)] for b,got in enumerate(emitted))
      q=torch.tensor([[x['key'] for x in ep['queries']] for ep in batch],device=device);before=model.export_state(session.state);actual=session.read(q).cpu().numpy();assert equal_tree(before,model.export_state(session.state))
      saved=np.stack([[scores[index[(cell,'online',b,q)]] for q in range(len(ep['queries']))] for b,ep in enumerate(batch)])
      stored=np.array([[int(rows[index[(cell,'online',b,q)]]['prediction']) for q in range(len(ep['queries']))] for b,ep in enumerate(batch)])
      error=float(np.max(np.abs(actual-saved)));assert error<=1e-5,(name,cell,error);assert np.array_equal(actual.argmax(-1),saved.argmax(-1)) and np.array_equal(actual.argmax(-1),stored),(name,cell,'prediction mismatch')
      results.append(dict(run=name,cell=cell,queries=actual.shape[0]*actual.shape[1],max_abs_error=error,all_argmax_match=True,stored_predictions_match=True,score_tolerance=1e-5,resume_points=points,checkpoint_sha256=sha(checkpoint)))
     assert all(torch.equal(t,model.state_dict()[k]) for k,t in slow.items());assert torch.equal(cpu_rng,torch.random.get_rng_state())
     if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
     verified.append(name)
 deadline();assert len(verified)==96
 report=dict(status='passed',scope='Selected actual inference/resume across48frozen+24initial+24final weightsets; not fullgridnumerical reproduction or retraining',device=str(device),torch=str(torch.__version__),source_sha256=sha(Path(__file__)),queries=sum(r['queries'] for r in results),max_abs_error=max(r['max_abs_error'] for r in results),verified_models=len(verified),expected_models=96,full_planned_model_coverage=True,groups=results,verified_weightsets=verified,elapsed_seconds=time.perf_counter()-started,pure_reads=True,slow_parameters_and_rng_unchanged=True)
 (a.out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('groups','verified_weightsets')},indent=2))
if __name__=='__main__':main()
