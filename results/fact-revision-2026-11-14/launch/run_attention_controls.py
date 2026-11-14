"""Run the predeclared, separate attention diagnostic after the main pipeline."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path('/insomnia001/depts/edu/users/zc2819/clmm/fact-revision-2026-11-14')
assert json.loads((root/'pipeline.json').read_text())['status'] == 'succeeded'
python = '/insomnia001/depts/edu/users/zc2819/clmm/replay-2026-11-12/venv/bin/python'
repo = root/'control-repo'
record = dict(status='running', runs=[], started_utc=datetime.now(timezone.utc).isoformat())

def save():
    (root/'attention-pipeline.json').write_text(json.dumps(record, indent=2)+'\n')

save()
for window in (32, 128):
    name = f'attention{window}'
    command = [python, '-B', str(repo/'scripts/fact_attention_control.py'),
               '--checkpoint', str(root/'shared-checkpoints/lm-68101.pt'),
               '--out', str(root/name), '--segment-len', str(window),
               '--steps', '1000', '--batch-size', '16', '--eval-batch-size', '16',
               '--train-seed', '78122', '--dev-seed', '78123', '--gate-episodes', '64',
               '--device', 'cuda', '--threads', '2', '--max-seconds', '600',
               '--design', str(root/'attention-control-design.json')]
    item = dict(name=name, command=command, status='running', started_utc=datetime.now(timezone.utc).isoformat())
    record['runs'].append(item)
    save()
    print(json.dumps(dict(starting=name)), flush=True)
    started = time.perf_counter()
    try:
        with (root/(name+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, timeout=660)
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
