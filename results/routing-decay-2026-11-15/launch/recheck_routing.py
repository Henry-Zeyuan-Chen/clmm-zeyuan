"""Selected saved-score replay through the actual reference adapter/session API."""
import argparse
from datetime import datetime,timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--device',default='cuda')
args=p.parse_args()
sys.path.insert(0,str(args.repo))
import numpy as np
import torch
from clmm.pretrain.adapter_registry import load_adapter_checkpoint,fork_associative
from clmm.sleep import SleepSession
from scripts.kv_representation_control import equal_tree

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    start_utc=datetime.now(timezone.utc).isoformat()
    device=torch.device(args.device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    def deadline():
        if time.perf_counter()-started>240:raise TimeoutError('Selected recheck exceeded240seconds')
    results=[]
    from scripts.routing_decay_experiment import case_specs,load_checkpoint
    pipeline=json.loads((args.root/'main/metadata.json').read_text())
    assert pipeline['status']=='succeeded'
    records=pipeline['cases']
    expected={r['name'] for r in case_specs()}
    assert {r['name'] for r in records}==expected and all(r['status']=='succeeded' for r in records)
    with torch.no_grad():
        for item in records:
            name=item['name']
            if item['status']!='succeeded':raise AssertionError('Incomplete case')
            folder=args.root/'main/runs'/name
            meta=json.loads((folder/'case.json').read_text())
            checkpoint=folder/'checkpoint.pt'
            model,parent=load_checkpoint(checkpoint,device)
            assert model.config.to_dict()==meta['config']
            model.eval().requires_grad_(False)
            slow={k:t.clone() for k,t in model.state_dict().items()}
            cpu_rng=torch.random.get_rng_state().clone()
            gpu_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
            stage=folder/'evaluation'
            data=json.loads((stage/'episodes.json').read_text())
            plans={(r['cell'],r['arm'],r['episode']):r['events'] for r in json.loads((stage/'plans.json').read_text())}
            with (stage/'predictions.csv').open() as handle:rows=list(csv.DictReader(handle))
            score=np.load(stage/'scores.npy')
            index={(r['cell'],r['arm'],int(r['episode']),int(r['rank'])):j for j,r in enumerate(rows)}
            groups=[('acquire2','online'),('pressure_k8_d64','online')]
            for cell,arm in groups:
                deadline()
                batch=data[cell][:16]
                session=SleepSession(model,16,arm,[ep['seed'] for ep in batch])
                emitted=[[] for _ in batch]
                resume_points=[]
                for i in range(len(batch[0]['keys'])):
                    deadline()
                    key=torch.tensor([ep['keys'][i] for ep in batch],device=device)
                    value=torch.tensor([ep['values'][i] for ep in batch],device=device)
                    events=session.prime(key,value) if i<batch[0]['boundary'] else session.observe(key,value)
                    for b,group in enumerate(events):emitted[b].extend(group)
                    if i in (batch[0]['boundary']-1,batch[0]['boundary']):
                        raw=io.BytesIO()
                        torch.save(session.export_state(),raw)
                        raw.seek(0)
                        session=SleepSession.from_state(model,torch.load(raw,map_location='cpu',weights_only=True))
                        resume_points.append('prefill' if i==batch[0]['boundary']-1 else 'midblock')
                assert all(not group for group in session.finish())
                assert all(got==plans[(cell,arm,b)] for b,got in enumerate(emitted))
                query=torch.tensor([[q['key'] for q in ep['queries']] for ep in batch],device=device)
                before=model.export_state(session.state)
                actual=session.read(query).cpu().numpy()
                assert equal_tree(before,model.export_state(session.state))
                saved=np.stack([[score[index[(cell,arm,b,q)]] for q in range(len(ep['queries']))] for b,ep in enumerate(batch)])
                stored=np.array([[int(rows[index[(cell,arm,b,q)]]['prediction']) for q in range(len(ep['queries']))] for b,ep in enumerate(batch)])
                error=float(np.max(np.abs(actual-saved)))
                tolerance=1e-5  # Cross-device/CUDA2.11 vs CPU2.15 float32; all argmax must agree
                assert error<=tolerance and np.array_equal(actual.argmax(-1),saved.argmax(-1)),(name,cell,arm,error)
                assert np.array_equal(actual.argmax(-1),stored),(name,cell,arm,'stored prediction mismatch')
                results.append(dict(run=name,cell=cell,arm=arm,queries=actual.shape[0]*actual.shape[1],max_abs_error=error,
                    all_argmax_match=True,stored_predictions_match=True,score_tolerance=tolerance,resume_points=resume_points,
                    checkpoint_sha256=sha(checkpoint),live_events_exact=True))
            assert all(torch.equal(t,model.state_dict()[k]) for k,t in slow.items())
            assert torch.equal(cpu_rng,torch.random.get_rng_state())
            if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
    deadline()
    report=dict(status='passed',scope='Selected inference and actual combined-session resume, not retraining or complete-grid numerical reproduction',
        started_utc=start_utc,ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
        source_sha256=sha(Path(__file__)),device=str(device),torch=str(torch.__version__),
        queries=sum(r['queries'] for r in results),max_abs_error=max(r['max_abs_error'] for r in results),groups=results,
        verified_models=len(expected),expected_models=len(expected),full_planned_model_coverage=True,
        pure_reads=True,slow_parameters_and_rng_unchanged=True)
    (args.out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2))

if __name__=='__main__':main()
