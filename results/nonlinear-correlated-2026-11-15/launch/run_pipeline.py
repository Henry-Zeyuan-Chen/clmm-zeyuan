"""Prospective finite calibration, confirmation, then matched-budget comparisons."""
import argparse
from datetime import datetime,timezone
import csv
import hashlib
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
record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),source_commit=args.source_commit,
            job_id=os.environ.get('SLURM_JOB_ID'),runs=[])
selection=dict(candidates=[],selected_rank=None,calibration_eligible=False,confirmed=False,comparison_scope=None)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def save():
    write(root/'pipeline.json',record)
    write(root/'selection.json',selection)

def run(name,rank,seed,mode='train',phase='calibration',dev=90503,pressure=32,checkpoint=None):
    remaining=3250-(time.perf_counter()-started)
    if remaining<120:raise TimeoutError('Pipeline budget nearly exhausted')
    command=[args.python,'-B',str(repo/'scripts/kv_correlated_experiment.py'),mode,'--out',str(root/name),
             '--design',str(root/'design.json'),'--phase',phase,'--rank',str(rank),'--codebook-seed','90501',
             '--init-seed',str(seed),'--train-seed','90502','--dev-seed',str(dev),'--steps','1200',
             '--batch-size','32','--eval-episodes','128','--pressure-episodes',str(pressure),
             '--eval-batch-size','16','--device','cuda','--threads','2','--max-seconds','900']
    if checkpoint is not None:command+=['--checkpoint',str(checkpoint)]
    item=dict(name=name,command=command,status='running',started_utc=datetime.now(timezone.utc).isoformat())
    record['runs'].append(item)
    save()
    print(json.dumps(dict(starting=name)),flush=True)
    t=time.perf_counter()
    with (root/f'{name}.log').open('x') as handle:
        result=subprocess.run(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,timeout=min(960,remaining))
    meta=json.loads((root/name/'metadata.json').read_text())
    item.update(status='succeeded' if result.returncode==0 and meta['status']=='succeeded' else 'failed',
                returncode=result.returncode,elapsed_seconds=time.perf_counter()-t,ended_utc=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps(dict(completed=name,status=item['status'])),flush=True)
    if item['status']!='succeeded':raise RuntimeError(f'Invocation failed: {name}')
    assert meta['source_commit']==args.source_commit
    return root/name

def gate(folder):
    with (folder/'evaluation/predictions.csv').open() as handle:rows=list(csv.DictReader(handle))
    def episode_scores(cell,arm='online',category=None,all_correct=False):
        group=[r for r in rows if r['cell']==cell and r['arm']==arm and (category is None or r['category']==category)]
        grouped={}
        for row in group:grouped.setdefault(int(row['episode']),[]).append(int(row['correct']))
        return np.array([float(all(grouped[i])) if all_correct else float(np.mean(grouped[i])) for i in sorted(grouped)])
    rates={cell:float(episode_scores(cell,all_correct=True).mean()) for cell in ('acquire2','acquire4','revise2')}
    delta=(episode_scores('acquire2',all_correct=True)-episode_scores('acquire2','rolled',all_correct=True)).reshape(8,16).mean(1)
    indices=np.random.default_rng(90504).integers(0,8,size=(5000,8))
    low,high=np.quantile(delta[indices].mean(1),[.025,.975])
    d0=episode_scores('pressure_k8_d0',all_correct=True)
    a0=episode_scores('pressure_k8_d0',category='stable').mean()
    a64=episode_scores('pressure_k8_d64',category='stable').mean()
    result=dict(**rates,normal_minus_rolled=float(delta.mean()),ci_low=float(low),ci_high=float(high),
                k8_d0_all_correct=float(d0.mean()),stable_d0=float(a0),stable_d64=float(a64),stable_drop=float(a0-a64))
    result['short_pass']=bool(rates['acquire2']>=.9 and rates['acquire4']>=.8 and rates['revise2']>=.9 and low>0)
    result['acquisition_pass']=bool(result['short_pass'] and d0.mean()>=.8)
    result['headroom_pass']=bool(.15<=a64<=.85 and a0-a64>=.1)
    result['eligible']=bool(result['acquisition_pass'] and result['headroom_pass'])
    return result

try:
    save()
    acquisition_candidates=[]
    for rank in (32,16,8):
        candidate=dict(rank=rank,runs=[],gates=[])
        selection['candidates'].append(candidate)
        save()
        for seed in (90511,90515):
            folder=run(f'r{rank}-{seed}',rank,seed)
            candidate['runs'].append(folder.name)
            candidate['gates'].append(dict(run=folder.name,**gate(folder)))
            save()
        if all(g['acquisition_pass'] for g in candidate['gates']):acquisition_candidates.append(rank)
        if all(g['eligible'] for g in candidate['gates']):
            selection.update(selected_rank=rank,calibration_eligible=True)
            break
    if selection['selected_rank'] is None:
        selection['selected_rank']=acquisition_candidates[-1] if acquisition_candidates else 8
        selection['fallback_reason']='No candidate passed all calibration gates; last both-seed short+D0 acquisition pass, else rank8, per declared diagnostic fallback'
    save()
    write(root/'selection-before-confirmation.json',selection)
    rank=selection['selected_rank']
    selection['confirmation']=[]
    for seed in (90511,90515):
        folder=run(f'confirm-r{rank}-{seed}',rank,seed,'evaluate','confirmation',90513,128,root/f'r{rank}-{seed}/checkpoint.pt')
        selection['confirmation'].append(dict(run=folder.name,**gate(folder)))
        save()
    selection['confirmed']=bool(all(g['acquisition_pass'] for g in selection['confirmation']))
    selection['fresh_headroom_both']=bool(all(g['headroom_pass'] for g in selection['confirmation']))
    selection['comparison_scope']='qualified calibration comparison; per-K actual D0 gates still apply' if selection['calibration_eligible'] and selection['confirmed'] else 'ineligible diagnostic comparison; no retention-efficacy claim'
    save()
    write(root/'comparison-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),selection=selection,
           selection_sha256=sha(root/'selection.json'),design_sha256=sha(root/'design.json'),no_replay_outcomes_inspected=True))
    for seed in (90511,90515):
        for stage,filename in (('trained','checkpoint.pt'),('initial','initial_checkpoint.pt')):
            run(f'comparison-{stage}-r{rank}-{seed}',rank,seed,'evaluate','comparison',90523,128,root/f'r{rank}-{seed}'/filename)
    record.update(status='succeeded',ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
    save()
except BaseException as error:
    record.update(status='failed',failure=repr(error),ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
    save()
    raise
