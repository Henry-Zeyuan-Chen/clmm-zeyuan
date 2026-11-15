"""Selected saved-score reload with actual bounded streaming and mid-block resume."""
import argparse
import csv
from datetime import datetime,timezone
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
p.add_argument('--loader',choices=('strict','portable'),default='strict')
args=p.parse_args()
sys.path.insert(0,str(args.repo))
import numpy as np
import torch
if args.loader=='portable':
    from clmm.pretrain.kv_correlated_portable import load_portable_checkpoint as load_checkpoint
else:
    from scripts.kv_correlated_experiment import load_checkpoint
from scripts.kv_representation_control import equal_tree
from clmm.pretrain.kv_scheduler import StreamingReplay

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    args.out.mkdir(parents=True,exist_ok=False)
    started_utc=datetime.now(timezone.utc).isoformat()
    started=time.perf_counter()
    device=torch.device(args.device)
    def before_deadline():
        if time.perf_counter()-started>180:raise TimeoutError('Recheck exceeded180seconds')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rank=json.loads((args.root/'selection.json').read_text())['selected_rank']
    results=[]
    with torch.no_grad():
        for seed in (90511,90515):
            for stage,filename in (('trained','checkpoint.pt'),('initial','initial_checkpoint.pt')):
                run=args.root/f'comparison-{stage}-r{rank}-{seed}'
                checkpoint=args.root/f'r{rank}-{seed}'/filename
                model,provenance=load_checkpoint(checkpoint,device)
                model.eval().requires_grad_(False)
                slow={name:t.clone() for name,t in model.state_dict().items()}
                cpu_rng=torch.random.get_rng_state().clone()
                gpu_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
                folder=run/'evaluation'
                episodes=json.loads((folder/'episodes.json').read_text())
                saved_plans={(r['cell'],r['arm'],r['episode']):r['plan']['events'] for r in json.loads((folder/'plans.json').read_text())}
                with (folder/'predictions.csv').open() as handle:rows=list(csv.DictReader(handle))
                scores=np.load(folder/'scores.npy')
                lookup={(r['cell'],r['arm'],int(r['episode']),int(r['rank'])):i for i,r in enumerate(rows)}
                for cell in ('acquire2','pressure_k8_d0','pressure_k8_d64'):
                    for arm in (('online',) if cell=='acquire2' else ('online','repeat','continuous','periodic')):
                        before_deadline()
                        batch=episodes[cell][:16]
                        policies=[StreamingReplay(arm,ep['seed']) for ep in batch]
                        emitted=[[] for _ in batch]
                        state=model.initial_state(16)
                        boundary=batch[0]['boundary']
                        resume_points=[]
                        for index in range(len(batch[0]['keys'])):
                            before_deadline()
                            events=[]
                            for b,(ep,policy) in enumerate(zip(batch,policies)):
                                k,v=ep['keys'][index],ep['values'][index]
                                item=[policy.prime(k,v)] if index<boundary else policy.step(k,v)
                                events.append(item)
                                emitted[b].extend(item)
                            assert len({len(item) for item in events})==1
                            for offset in range(len(events[0])):
                                state=model.write(torch.tensor([item[offset]['key'] for item in events],device=args.device),
                                                  torch.tensor([item[offset]['value'] for item in events],device=args.device),state)
                            if index in (boundary-1,boundary):
                                raw=io.BytesIO()
                                torch.save(model.export_state(state),raw)
                                raw.seek(0)
                                state=model.import_state(torch.load(raw,map_location='cpu',weights_only=True))
                                policies=[StreamingReplay.from_state(json.loads(json.dumps(policy.export_state()))) for policy in policies]
                                resume_points.append('prefill' if index==boundary-1 else 'midblock')
                        for b,policy in enumerate(policies):
                            assert policy.finish()==[]  # all saved delays are full blocks
                            assert emitted[b]==saved_plans[(cell,arm,b)]
                        queries=torch.tensor([[q['key'] for q in ep['queries']] for ep in batch],device=args.device)
                        payload=model.export_state(state)
                        actual=model.read(queries,state).cpu().numpy()
                        assert equal_tree(payload,model.export_state(state))
                        reference=np.stack([[scores[lookup[(cell,arm,b,q)]] for q in range(len(ep['queries']))] for b,ep in enumerate(batch)])
                        error=float(np.max(np.abs(actual-reference)))
                        assert error<=1e-6 and np.array_equal(actual.argmax(-1),reference.argmax(-1)),(run.name,cell,arm,error)
                        results.append(dict(run=run.name,cell=cell,arm=arm,queries=actual.shape[0]*actual.shape[1],
                                            max_abs_error=error,all_argmax_match=True,events_exact=True,
                                            resume_points=resume_points,resumed_at_prefill_and_midblock=resume_points==['prefill','midblock'],checkpoint_sha256=sha(checkpoint)))
                        before_deadline()
                assert all(torch.equal(t,model.state_dict()[name]) for name,t in slow.items())
                assert torch.equal(cpu_rng,torch.random.get_rng_state())
                if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
    before_deadline()
    report=dict(status='passed',scope='Selected inference and real streaming resume, not retraining or full-grid numerical reproduction',
                started_utc=started_utc,ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
                source_sha256=sha(Path(__file__)),device=args.device,loader=args.loader,torch=str(torch.__version__),
                loader_source_sha256=sha(Path(sys.modules[load_checkpoint.__module__].__file__)),
                queries=sum(r['queries'] for r in results),max_abs_error=max(r['max_abs_error'] for r in results),
                groups=results,read_pure=True,slow_parameters_and_rng_unchanged=True)
    (args.out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2))

if __name__=='__main__':main()
