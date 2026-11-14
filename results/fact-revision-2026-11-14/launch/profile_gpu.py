"""Timing-only profile of the fixed checkpoint; not an efficacy experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
sys.path.insert(0, a.repo)
import torch
import torch.nn.functional as F
from clmm.pretrain import load_checkpoint
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
torch.manual_seed(78100)
model, config, record = load_checkpoint(a.checkpoint, map_location='cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
rows = []
for length in (63, 159, 351, 671):
    x = torch.randint(32, 127, (16, length), device='cuda')
    y = torch.randint(97, 113, (16,), device='cuda')
    elapsed = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x)[:, -1], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
    row = dict(length=length, batch=16, step_seconds=elapsed,
               peak_cuda_bytes=torch.cuda.max_memory_allocated())
    rows.append(row)
    print(json.dumps(row), flush=True)
result = dict(status='succeeded', torch=str(torch.__version__), python=sys.version,
              device=torch.cuda.get_device_name(), rows=rows,
              purpose='Random-input timing profile only; weights are discarded and no accuracy is used for budget selection.')
(out/'profile.json').write_text(json.dumps(result, indent=2)+'\n')
