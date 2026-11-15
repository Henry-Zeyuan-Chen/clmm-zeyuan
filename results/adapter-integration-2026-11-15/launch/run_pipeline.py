"""Runtime-gated integration pilot; no performance-based configuration selection."""
import argparse
from datetime import datetime,timezone
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--python',required=True)
p.add_argument('--source-commit',required=True)
args=p.parse_args()
root=args.root
repo=root/'repo'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.source_commit
started=time.perf_counter()
record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),job_id=os.environ.get('SLURM_JOB_ID'),
            source_commit=args.source_commit,runs=[],skipped=[])
def save():(root/'pipeline.json').write_text(json.dumps(record,indent=2)+'\n')

def run(name,mode,*,seed=90511,rule='gradient',routing='dense',checkpoint=None,steps=1200,episodes=128,pressure=64,dev=90623,estimate=240):
    remaining=3150-(time.perf_counter()-started)
    if remaining<estimate+90:
        record['skipped'].append(dict(name=name,reason='Insufficient predicted remaining execution budget; not score-based'))
        save()
        return None
    command=[args.python,'-B',str(repo/'scripts/adapter_integration_experiment.py'),mode,'--out',str(root/name),
        '--shared-checkpoint',str(root/'shared-checkpoints/neural-90511.pt'),'--design',str(root/'design.json'),
        '--rule',rule,'--routing',routing,'--init-seed',str(seed),'--steps',str(steps),'--batch-size','32',
        '--eval-episodes',str(episodes),'--pressure-episodes',str(pressure),'--eval-batch-size','16',
        '--dev-seed',str(dev),'--device','cuda','--max-seconds','900']
    if checkpoint:command+=['--checkpoint',str(checkpoint)]
    item=dict(name=name,command=command,status='running',started_utc=datetime.now(timezone.utc).isoformat())
    record['runs'].append(item)
    save()
    print(json.dumps(dict(starting=name)),flush=True)
    tick=time.perf_counter()
    with (root/f'{name}.log').open('x') as handle:
        result=subprocess.run(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                              timeout=min(960,remaining))
    meta=json.loads((root/name/'metadata.json').read_text())
    item.update(status='succeeded' if result.returncode==0 and meta['status']=='succeeded' else 'failed',
        exit_code=result.returncode,elapsed_seconds=time.perf_counter()-tick,ended_utc=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps(dict(completed=name,status=item['status'])),flush=True)
    if item['status']!='succeeded':raise RuntimeError(f'Invocation failed: {name}')
    assert meta['source_commit']==args.source_commit
    return root/name

def projection(folder):
    with (folder/'training.csv').open() as handle:training=list(csv.DictReader(handle))
    steps=np.diff([float(r['elapsed_seconds']) for r in training])
    step_seconds=float(np.median(steps))
    def eval_estimate(stage,initial=False,fork=False,nnm=False):
        with (folder/stage/'resources.csv').open() as handle:resources=list(csv.DictReader(handle))
        total=0.
        for row in resources:
            short=not row['cell'].startswith('pressure')
            if (initial or fork) and not short and row['cell'] not in ('pressure_k8_d0','pressure_k8_d64'):continue
            if (initial or fork) and not short and row['arm']!='online':continue
            batches=8 if short else 4
            total+=float(row['elapsed_seconds'])*batches
        return total*(5 if nnm else 1)
    initial=eval_estimate('initial-evaluation',initial=True)
    final=eval_estimate('evaluation')
    fork=eval_estimate('evaluation',fork=True)
    nnm=eval_estimate('evaluation',nnm=True)
    pertrain=1200*step_seconds+initial+final+12
    estimate=1.25*(8*pertrain+8*(fork+8)+2*(nnm+8))
    return dict(step_seconds=step_seconds,per_train_seconds=pertrain,per_fork_seconds=fork+8,
                per_neural_anchor_seconds=nnm+8,projected_complete_seconds=estimate,
                rule='1.25padding; NNM evaluation5x associative batch cost; estimates contain no scores')

try:
    save()
    probe=run('throughput-gpu','train',steps=10,episodes=16,pressure=16,dev=90603,estimate=120)
    if probe is None:raise TimeoutError('No preflight execution budget')
    estimate=projection(probe)
    record['runtime_preflight']=estimate
    save()
    (root/'runtime-preflight.json').write_text(json.dumps(estimate,indent=2)+'\n')
    if estimate['projected_complete_seconds']>2400:
        record.update(status='completed_preflight_only',reason='Prospective runtime projection exceeds2400s; no main model outcomes consulted')
    else:
        for seed in (90511,90515):
            run(f'neural-anchor-{seed}','evaluate',seed=seed,checkpoint=root/f'shared-checkpoints/neural-{seed}.pt',
                estimate=estimate['per_neural_anchor_seconds'])
        complete=True
        for seed in (90511,90515):
            for rule,routing in (('gradient','dense'),('gradient','top1'),('hebbian','dense'),('hebbian','top1')):
                result=run(f'{rule}-{routing}-{seed}','train',seed=seed,rule=rule,routing=routing,
                    estimate=1.25*estimate['per_train_seconds'])
                if result is None:complete=False
        if complete:
            for seed in (90511,90515):
                for rule,routing in (('gradient','dense'),('gradient','top1'),('hebbian','dense'),('hebbian','top1')):
                    run(f'fork-{rule}-{routing}-{seed}','fork',seed=seed,rule=rule,routing=routing,
                        checkpoint=root/f'gradient-dense-{seed}/checkpoint.pt',estimate=1.25*estimate['per_fork_seconds'])
        record['status']='succeeded' if not record['skipped'] else 'completed_with_runtime_skips'
    record.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
    save()
except BaseException as error:
    record.update(status='failed',failure=repr(error),ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
    save()
    raise
