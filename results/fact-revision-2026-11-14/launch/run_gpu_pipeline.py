"""Serial, bounded jobs in one existing Slurm GPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

root = Path('/insomnia001/depts/edu/users/zc2819/clmm/fact-revision-2026-11-14')
repo = root / 'repo'
python = '/insomnia001/depts/edu/users/zc2819/clmm/replay-2026-11-12/venv/bin/python'
script = repo / 'scripts/fact_revision_experiment.py'
design = root / 'design.json'
plan = []
for seed in (68101, 68105):
    plan.append((f'before-{seed}', 'evaluate', root / f'shared-checkpoints/lm-{seed}.pt', 'gate'))
for seed in (68101, 68105):
    plan.append((f'adapt-{seed}', 'train', root / f'shared-checkpoints/lm-{seed}.pt', 'gate'))
    plan.append((f'after-{seed}', 'evaluate', root / f'adapt-{seed}/checkpoint.pt', 'full'))
record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
              slurm_job_id=os.environ.get('SLURM_JOB_ID'), runs=[])

def save():
    (root / 'pipeline.json').write_text(json.dumps(record, indent=2)+'\n')

save()
for name, mode, checkpoint, suite in plan:
    command = [python, '-B', str(script), mode, '--checkpoint', str(checkpoint),
               '--out', str(root / name), '--design', str(design), '--device', 'cuda',
               '--steps', '1600', '--batch-size', '16', '--eval-batch-size', '16',
               '--train-seed', '78102', '--dev-seed', '78103', '--gate-episodes', '64',
               '--grid-episodes', '32', '--primary-episodes', '128', '--suite', suite,
               '--threads', '2', '--max-seconds', '1200' if mode == 'train' else '900']
    item = dict(name=name, command=command, status='running', started_utc=datetime.now(timezone.utc).isoformat())
    record['runs'].append(item)
    save()
    started = time.perf_counter()
    print(json.dumps(dict(starting=name)), flush=True)
    try:
        with (root / (name+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, timeout=1320 if mode == 'train' else 1020)
        item.update(status='succeeded' if result.returncode == 0 else 'failed', returncode=result.returncode)
    except BaseException as error:
        item.update(status='failed', exception=repr(error))
    item['elapsed_seconds'] = time.perf_counter()-started
    item['ended_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(item), flush=True)
    if item['status'] != 'succeeded':
        record['status'] = 'failed'
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        sys.exit(1)
record['status'] = 'succeeded'
record['ended_utc'] = datetime.now(timezone.utc).isoformat()
save()
