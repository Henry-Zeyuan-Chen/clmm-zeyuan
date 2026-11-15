"""Bounded selection/confirmation pipeline for shared typed memory controls."""
from datetime import datetime,timezone
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

root=Path('/insomnia001/depts/edu/users/zc2819/clmm/kv-control-2026-11-14')
repo=root/'repo';python='/insomnia001/depts/edu/users/zc2819/clmm/replay-2026-11-12/venv/bin/python'
record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),job_id=os.environ.get('SLURM_JOB_ID'),runs=[])
selection=dict(candidates=[],selected=None,confirmed=False)
started=time.perf_counter()
def save():
    (root/'pipeline.json').write_text(json.dumps(record,indent=2)+'\n')
    (root/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
def run(name,mode='train',method='neural',eta=.5,seed=90301,checkpoint=None,dev=90303,suite='gate'):
    if time.perf_counter()-started>3100:raise TimeoutError('Pipeline execution budget nearly exhausted')
    command=[python,'-B',str(repo/'scripts/kv_representation_control.py'),mode,'--out',str(root/name),
        '--design',str(root/'design.json'),'--method',method,'--step-size',str(eta),'--init-seed',str(seed),
        '--train-seed','90302','--dev-seed',str(dev),'--steps','1200','--batch-size','32',
        '--eval-episodes','128','--pressure-episodes','32','--eval-batch-size','16',
        '--suite',suite,'--device','cuda','--threads','2','--max-seconds','900']
    if checkpoint:command+=['--checkpoint',str(checkpoint)]
    item=dict(name=name,command=command,status='running',started_utc=datetime.now(timezone.utc).isoformat());record['runs'].append(item);save()
    print(json.dumps(dict(starting=name)),flush=True);start=time.perf_counter()
    with (root/f'{name}.log').open('x') as f:
        result=subprocess.run(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=min(960,max(1,3250-(time.perf_counter()-started))))
    meta=json.loads((root/name/'metadata.json').read_text())
    item.update(status='succeeded' if result.returncode==0 and meta['status']=='succeeded' else 'failed',returncode=result.returncode,
        elapsed_seconds=time.perf_counter()-start,ended_utc=datetime.now(timezone.utc).isoformat());save();print(json.dumps(item),flush=True)
    if item['status']!='succeeded':raise RuntimeError(f'Invocation failed: {name}')
    return root/name
def gate(folder):
    with (folder/'evaluation/summary.csv').open() as f:summary=list(csv.DictReader(f))
    def rate(cell):return float(next(r['all_correct'] for r in summary if r['cell']==cell and r['arm']=='online' and r['category']=='all'))
    with (folder/'evaluation/predictions.csv').open() as f:rows=list(csv.DictReader(f))
    episode={}
    for arm in ('online','rolled'):
        chosen=[r for r in rows if r['cell']=='acquire2' and r['arm']==arm]
        episode[arm]={i:float(all(int(r['correct']) for r in chosen if int(r['episode'])==i)) for i in range(128)}
    delta=np.array([episode['online'][i]-episode['rolled'][i] for i in range(128)]).reshape(8,16).mean(1)
    ix=np.random.default_rng(90304).integers(0,8,size=(5000,8));low,high=np.quantile(delta[ix].mean(1),[.025,.975])
    result=dict(acquire2=rate('acquire2'),acquire4=rate('acquire4'),revise2=rate('revise2'),
        normal_minus_rolled=float(delta.mean()),ci_low=float(low),ci_high=float(high),bootstrap_unit='8 cyclic-roll blocks of16 episodes')
    result['passed']=result['acquire2']>=.9 and result['acquire4']>=.8 and result['revise2']>=.9 and low>0
    return result
try:
    save();run('delta-full','evaluate','delta',dev=90323,suite='full')
    for tag,eta in (('s05',.5),('s10',1.)):
        candidate=dict(tag=tag,step_size=eta,runs=[],gates=[])
        selection['candidates'].append(candidate);save()
        for seed in (90301,90305):
            folder=run(f'{tag}-{seed}',eta=eta,seed=seed)
            candidate['runs'].append(folder.name);candidate['gates'].append(gate(folder));save()
        if all(g['passed'] for g in candidate['gates']):selection['selected']=tag;save();break
    if selection['selected']:
        chosen=next(c for c in selection['candidates'] if c['tag']==selection['selected'])
        selection['confirmation']=[];save()
        for seed,name in zip((90301,90305),chosen['runs']):
            folder=run(f'confirm-{name}','evaluate',eta=chosen['step_size'],seed=seed,checkpoint=root/name/'checkpoint.pt',dev=90313)
            selection['confirmation'].append(dict(run=folder.name,gate=gate(folder)));save()
        selection['confirmed']=all(c['gate']['passed'] for c in selection['confirmation']);save()
        if selection['confirmed']:
            for seed,name in zip((90301,90305),chosen['runs']):
                run(f'pressure-{name}','evaluate',eta=chosen['step_size'],seed=seed,checkpoint=root/name/'checkpoint.pt',dev=90323,suite='full')
    record.update(status='succeeded',ended_utc=datetime.now(timezone.utc).isoformat());save()
except BaseException as error:
    record.update(status='failed',failure=repr(error),ended_utc=datetime.now(timezone.utc).isoformat());save();raise
