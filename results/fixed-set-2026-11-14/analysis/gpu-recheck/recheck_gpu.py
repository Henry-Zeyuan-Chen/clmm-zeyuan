"""Bounded same-environment final checkpoint reload and logit check."""
import argparse
from contextlib import nullcontext
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo))
import numpy as np
import torch
from clmm.pretrain import load_checkpoint,reset_memory_at_segments
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(90202)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
a.out.mkdir(parents=True,exist_ok=False)
start=time.perf_counter();result=[]
pipeline=json.loads((a.root/'pipeline.json').read_text());assert pipeline['status']=='succeeded'
for entry in pipeline['runs']:
    name=entry['name'];run=a.root/name;meta=json.loads((run/'metadata.json').read_text());assert meta['status']=='succeeded'
    folder=run/f"step-{max(c['step'] for c in meta['checkpoints']):04d}"
    model,_,_=load_checkpoint(folder/'checkpoint.pt',map_location='cuda');model.eval()
    with (folder/'evaluation/predictions.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    saved=np.load(folder/'evaluation/answer_logits.npy',allow_pickle=False)
    for condition in ('online','reset_memory'):
        if time.perf_counter()-start>180:raise TimeoutError('Reload check180seconds')
        indices=[i for i,r in enumerate(rows) if r['group']=='novel_values' and r['condition']==condition][:16]
        assert len(indices)==16
        x=torch.tensor([list(rows[i]['prefix'].encode('ascii')) for i in indices],device='cuda')
        context=reset_memory_at_segments(model) if condition=='reset_memory' else nullcontext()
        with torch.no_grad(),context:logits=model(x)[:,-1].cpu().numpy()
        result.append(dict(run=name,condition=condition,queries=len(indices),max_abs_logit_difference=float(np.max(np.abs(logits-saved[indices]))),prediction_mismatches=int(np.sum(logits.argmax(-1)!=saved[indices].argmax(-1)))))
record=dict(status='passed' if all(r['max_abs_logit_difference']<=1e-6 and r['prediction_mismatches']==0 for r in result) else 'mismatch',
    results=result,queries=sum(r['queries'] for r in result),absolute_tolerance=1e-6,elapsed_seconds=time.perf_counter()-start,
    scope='Final checkpoint inference reload at original batch16, no training reproduction',
    environment=dict(python=sys.version,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),platform=platform.platform(),packages={d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()}))
(a.out/'verification.json').write_text(json.dumps(record,indent=2)+'\n');shutil.copy2(__file__,a.out/Path(__file__).name)
print(json.dumps({k:v for k,v in record.items() if k!='environment'}),flush=True)
if record['status']!='passed':sys.exit(1)
