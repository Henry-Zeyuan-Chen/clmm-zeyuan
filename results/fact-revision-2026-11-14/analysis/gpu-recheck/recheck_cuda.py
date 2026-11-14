"""Repeat128 saved answers with the original GPU environment and batch geometry."""
import argparse
from contextlib import nullcontext
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.repo))
import numpy as np
import torch
from clmm.pretrain import load_checkpoint,reset_memory_at_segments
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(78102)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
started=time.perf_counter()
results=[]

def key(r):
    return r['cell'],r['family'],int(r['episode']),int(r['query_index']),r['condition']

for seed in (68101,68105):
    run=a.root/f'after-{seed}'
    model,_,_=load_checkpoint(a.root/f'adapt-{seed}/checkpoint.pt',map_location='cuda')
    model.eval()
    with (run/'predictions.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    saved=np.load(run/'answer_logits.npy',allow_pickle=False)
    selected=[]
    for cell,conditions in [('acquire_k2',('online','reset_memory')),
                            ('pressure_k8_d8',('continuous','periodic'))]:
        for condition in conditions:
            indices=[i for i,r in enumerate(rows) if r['cell']==cell and r['family']=='seen' and r['condition']==condition][:16]
            assert len(indices)==16
            selected.append(indices)
    wanted={key(rows[i]) for indices in selected for i in indices}
    prefixes={}
    with gzip.open(run/'evaluation_data.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            item=json.loads(line)
            if item['type']=='sample' and key(item['row']) in wanted:
                prefixes[key(item['row'])]=item['prefix'].encode('ascii')
    for indices in selected:
        if time.perf_counter()-started>180:
            raise TimeoutError('GPU recheck limit180seconds')
        condition=rows[indices[0]]['condition']
        x=torch.tensor([list(prefixes[key(rows[i])]) for i in indices],device='cuda')
        context=reset_memory_at_segments(model) if condition=='reset_memory' else nullcontext()
        with torch.no_grad(),context:
            logits=model(x)[:,-1].cpu().numpy()
        difference=np.max(np.abs(logits-saved[indices]))
        results.append(dict(seed=seed,cell=rows[indices[0]]['cell'],condition=condition,queries=16,
                            max_abs_logit_difference=float(difference),
                            prediction_mismatches=int(np.sum(logits.argmax(-1)!=saved[indices].argmax(-1)))))
        print(json.dumps(results[-1]),flush=True)
result=dict(status='passed' if all(r['max_abs_logit_difference']<=1e-6 and r['prediction_mismatches']==0 for r in results) else 'mismatch',
            queries=sum(r['queries'] for r in results),results=results,absolute_tolerance=1e-6,
            torch=str(torch.__version__),python=sys.version,device=torch.cuda.get_device_name(),
            batch_size=16,seed=78102,elapsed_seconds=time.perf_counter()-started,
            scope='128 answers rerun in original environment/batch geometry; no second training run.')
(a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
(a.out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps(result),flush=True)
