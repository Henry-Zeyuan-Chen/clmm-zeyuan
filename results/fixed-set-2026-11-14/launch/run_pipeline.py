"""One-hour maximum diagnostic, four fixed runs and one predeclared branch."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/insomnia001/depts/edu/users/zc2819/clmm/fixed-set-2026-11-14')
repo=root/'repo'
python='/insomnia001/depts/edu/users/zc2819/clmm/replay-2026-11-12/venv/bin/python'
parents=Path('/insomnia001/depts/edu/users/zc2819/clmm/fact-revision-2026-11-14/shared-checkpoints')
record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),runs=[])
started=time.perf_counter()
def save():(root/'pipeline.json').write_text(json.dumps(record,indent=2)+'\n')
def run(name,seed,segment,regime='fixed',contexts=32,keys=16,steps=2000):
    remaining=3300-(time.perf_counter()-started)
    if remaining<660:raise TimeoutError('Insufficient reserved pipeline budget for a further bounded run')
    command=[python,'-B',str(repo/'scripts/fixed_set_generalization.py'),
        '--checkpoint',str(parents/f'lm-{seed}.pt'),'--out',str(root/name),'--design',str(root/'design.json'),
        '--segment-len',str(segment),'--regime',regime,'--contexts',str(contexts),'--key-count',str(keys),
        '--steps',str(steps),'--batch-size','16','--bank-seed','90201','--train-seed','90202',
        '--dev-seed','90203','--eval-episodes','128','--eval-batch-size','16',
        '--lr','0.0003','--threads','2','--device','cuda','--max-seconds','600']
    item=dict(name=name,command=command,status='running',started_utc=datetime.now(timezone.utc).isoformat())
    record['runs'].append(item);save();print(json.dumps(dict(starting=name)),flush=True)
    start=time.perf_counter()
    with (root/f'{name}.log').open('x') as f:
        result=subprocess.run(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=660)
    meta=json.loads((root/name/'metadata.json').read_text())
    item.update(status='succeeded' if result.returncode==0 and meta['status']=='succeeded' else 'failed',
        returncode=result.returncode,elapsed_seconds=time.perf_counter()-start,ended_utc=datetime.now(timezone.utc).isoformat())
    save();print(json.dumps(item),flush=True)
    if item['status']!='succeeded':raise RuntimeError(f'Run failed: {name}')
    return meta

try:
    save()
    for seed in (68101,68105):
        for segment in (32,128):run(f'fixed{segment}-{seed}',seed,segment)
    primary=json.loads((root/'fixed128-68101/metadata.json').read_text())
    final=next(c for c in primary['checkpoints'] if c['step']==2000)
    def metric(group):return next(r['both_correct'] for r in final['summary'] if r['group']==group and r['condition']=='online')
    fit,transfer=metric('train'),metric('novel_values')
    branch=dict(selected_from='fixed128-68101/step-2000',fit_both_correct=fit,
        novel_values_both_correct=transfer,fit_threshold=.95,transfer_threshold=.8,
        recorded_utc=datetime.now(timezone.utc).isoformat())
    branch['choice']='tiny' if fit<.95 else ('fresh' if transfer<.8 else 'none')
    (root/'branch.json').write_text(json.dumps(branch,indent=2)+'\n')
    record['branch']=branch;save();print(json.dumps(branch),flush=True)
    if branch['choice']=='fresh':
        for segment in (32,128):run(f'fresh{segment}-68101',68101,segment,'fresh')
    elif branch['choice']=='tiny':run('tiny128-68101',68101,128,contexts=8,keys=4,steps=1600)
    record.update(status='succeeded',ended_utc=datetime.now(timezone.utc).isoformat());save()
except BaseException as error:
    record.update(status='failed',exception=repr(error),ended_utc=datetime.now(timezone.utc).isoformat())
    if record['runs'] and record['runs'][-1]['status']=='running':record['runs'][-1]['status']='interrupted'
    save();raise
