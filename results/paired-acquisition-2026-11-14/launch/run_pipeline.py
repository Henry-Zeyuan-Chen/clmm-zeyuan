"""One bounded five-run diagnostic; no allocation renewal or checkpoint selection."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/insomnia001/depts/edu/users/zc2819/clmm/paired-acquisition-2026-11-14')
repo=root/'repo'
python='/insomnia001/depts/edu/users/zc2819/clmm/replay-2026-11-12/venv/bin/python'
parents=Path('/insomnia001/depts/edu/users/zc2819/clmm/fact-revision-2026-11-14/shared-checkpoints')
record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),runs=[])

def save():
    (root/'pipeline.json').write_text(json.dumps(record,indent=2)+'\n')

save()
specs=[(68101,'duplicate',32),(68101,'paired',32),(68105,'duplicate',32),
       (68105,'paired',32),(68101,'paired',128)]
for seed,arm,segment in specs:
    name=f'{arm}{segment}-{seed}'
    command=[python,'-B',str(repo/'scripts/fact_paired_acquisition.py'),
        '--checkpoint',str(parents/f'lm-{seed}.pt'),'--out',str(root/name),
        '--design',str(root/'design.json'),'--arm',arm,'--segment-len',str(segment),
        '--steps','1600','--batch-size','16','--eval-batch-size','16',
        '--train-seed','89102','--dev-seed','89103','--gate-episodes','64',
        '--strict-episodes','128','--lr','0.0003','--device','cuda',
        '--threads','2','--max-seconds','600']
    row=dict(name=name,command=command,status='running',started_utc=datetime.now(timezone.utc).isoformat())
    record['runs'].append(row);save()
    print(json.dumps(dict(starting=name)),flush=True)
    start=time.perf_counter()
    try:
        with (root/f'{name}.log').open('x') as f:
            result=subprocess.run(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=f,
                                  stderr=subprocess.STDOUT,timeout=660)
        meta=json.loads((root/name/'metadata.json').read_text())
        row.update(returncode=result.returncode,status='succeeded' if result.returncode==0 and meta['status']=='succeeded' else 'failed')
    except BaseException as error:
        row.update(status='failed',exception=repr(error))
    row.update(elapsed_seconds=time.perf_counter()-start,ended_utc=datetime.now(timezone.utc).isoformat())
    print(json.dumps(row),flush=True);save()
    if row['status']!='succeeded':
        record.update(status='failed',ended_utc=datetime.now(timezone.utc).isoformat());save();sys.exit(1)
record.update(status='succeeded',ended_utc=datetime.now(timezone.utc).isoformat());save()
