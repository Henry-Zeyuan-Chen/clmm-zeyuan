"""Bounded cross-device numerical check on32 prespecified saved queries."""
import argparse
from contextlib import nullcontext
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(a.repo))
import numpy as np
import torch
from clmm.pretrain import load_checkpoint, reset_memory_at_segments
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
started = time.perf_counter()
rows = []
source_checks = {}

def key(row):
    return row['cell'], row['family'], int(row['episode']), int(row['query_index']), row['condition']

for seed in (68101, 68105):
    run = a.root / f'after-{seed}'
    model, _, _ = load_checkpoint(a.root / f'adapt-{seed}/checkpoint.pt')
    model.eval()
    for relative in ('clmm/pretrain/byte_lm.py', 'titans_pytorch/mac_transformer.py',
                     'titans_pytorch/neural_memory.py', 'titans_pytorch/memory_models.py'):
        local = (a.repo / relative).read_bytes().replace(b'\r\n', b'\n')
        saved = (run / 'sources' / relative).read_bytes().replace(b'\r\n', b'\n')
        assert local == saved, relative
        source_checks[relative] = hashlib.sha256(local).hexdigest()
    with (run/'predictions.csv').open(newline='') as handle:
        reference_rows = list(csv.DictReader(handle))
    reference_logits = np.load(run/'answer_logits.npy', allow_pickle=False)
    lookup = {key(row): (row, reference_logits[index]) for index, row in enumerate(reference_rows)}
    selected = []
    with gzip.open(run/'evaluation_data.jsonl.gz', 'rt', encoding='utf8') as handle:
        for line in handle:
            item = json.loads(line)
            if item['type'] != 'sample':
                continue
            r = item['row']
            gate = r['cell'] == 'acquire_k2' and r['family'] == 'seen' and r['episode'] < 2
            pressure = (r['cell'] == 'pressure_k8_d8' and r['category'] == 'stable'
                        and r['condition'] in ('continuous', 'periodic') and r['episode'] < 4)
            if gate or pressure:
                selected.append(item)
    assert len(selected) == 16
    for sample in selected:
        if time.perf_counter()-started > 180:
            raise TimeoutError('CPU recheck limit180seconds')
        r = sample['row']
        x = torch.tensor([list(sample['prefix'].encode('ascii'))])
        context = reset_memory_at_segments(model) if r['condition'] == 'reset_memory' else nullcontext()
        with torch.no_grad(), context:
            logits = model(x)[0, -1].numpy()
        reference, saved = lookup[key(r)]
        assert np.isfinite(logits).all()
        rows.append(dict(seed=seed, cell=r['cell'], family=r['family'], episode=r['episode'],
                         query_index=r['query_index'], condition=r['condition'],
                         max_abs_logit_difference=float(np.max(np.abs(logits-saved))),
                         reference_prediction=int(saved.argmax()), cpu_prediction=int(logits.argmax())))
with (a.out/'comparisons.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
result = dict(status='passed' if all(row['max_abs_logit_difference'] <= .001 and
                                    row['reference_prediction'] == row['cpu_prediction'] for row in rows) else 'mismatch',
              queries=len(rows), max_abs_logit_difference=max(row['max_abs_logit_difference'] for row in rows),
              prediction_mismatches=sum(row['reference_prediction'] != row['cpu_prediction'] for row in rows),
              prespecified_absolute_logit_tolerance=.001, source_lf_sha256=source_checks,
              torch=str(torch.__version__), python=sys.version, device='cpu', threads=2,
              elapsed_seconds=time.perf_counter()-started,
              scope='32selected saved queries only; cross-device inference, not a second training run.')
(a.out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
(a.out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps(result))
