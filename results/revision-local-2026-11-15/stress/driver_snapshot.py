"""Post-outcome exploratory old-value reinforcement stress; frozen CPU inference."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
START=time.perf_counter()
DEADLINE=START+600
sys.dont_write_bytecode=True

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def check_deadline():
    if time.perf_counter()>=DEADLINE:
        raise TimeoutError('Exploratory stress global600-second deadline reached')

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--trained',type=Path,required=True)
p.add_argument('--frozen',type=Path,required=True)
p.add_argument('--prior',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--analysis',type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.source))
import numpy as np
import torch
from clmm.pretrain.kv_streams import training_data,evaluation_data,episode,signature
from scripts.adapter_integration_experiment import evaluate
from scripts.revision_local_experiment import load_checkpoint
from scripts.routing_decay_experiment import slow_sha

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
COMMIT='c9f5c153f786f1b82ac383d01c1d18c6c595946b'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip()==COMMIT
assert subprocess.check_output(['git','status','--porcelain'],cwd=a.source,text=True)==''
a.out.mkdir(parents=True,exist_ok=False)
a.analysis.mkdir(parents=True,exist_ok=False)
sources={str(path.relative_to(a.source)).replace('\\','/'):sha(path)
         for path in a.source.rglob('*.py') if '.git' not in path.parts}
driver=Path(__file__).resolve()
driver_sha=sha(driver)
shutil.copy2(driver,a.out/'driver_snapshot.py')
design=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
    role='Exploratory follow-up motivated after seeing trained short-revision ceiling; not a new primary',
    original_joint_primary='Failed revision superiority at ceiling; unchanged and not rescued by this follow-up',
    source_commit=COMMIT,driver_sha256=driver_sha,
    models='All24 fixed trained checkpoints; EMA/local_delta x original initializers90511/90515 x content/all5maps90701..90705; no weights selected or changed',
    data=dict(seed=91023,histories=128,batch=16,short='Original acquire2/acquire4/revise2; online/reset/rolled',
        pressure='K8, two true conflicting revisions; R1/4/16 x D0/64; online only',
        reinforcement='Between8initialfacts and2existingrevisions, repeat oldbindings(initialkey2,initialkey3) R-1cycles; preserve originaldelay andquerylabels; boundary10+2*(R-1)',
        pairing='Same base seed/key/value/distractor histories across all six R/D cells; rejectcandidate jointly if any cell signature excluded or duplicated',
        base_rng='PCG64 SeedSequence([91023,8,2,500]); candidate seeds uniform[2000000000,3000000000)',
        exclusions='Exact old90502 andnew90802 training plus all actual prior90723/frozen90823/trained90923 histories; also freshshort signatures'),
    focus=dict(cell='pressure_k8_r16_d64',revision='local_delta minus EMA revised-query accuracy',
        stable='local_delta minus EMA stable-query accuracy, exploratory noninferiority margin.03',
        families=['content','mean5random'],bootstrap='128wholehistories jointly acrossfixedparents/maps;5000PCG64seed91004draws;nominal95percentile;exploratory,notconfirmatory'),
    report='AllR/Dcurves, all24cases, D0allfourtargetaccuracy andshortgates; no dropping failures or configuration selection',
    resources=dict(device='CPU only',torch_threads=2,global_deadline_seconds=600,optimizer_steps=0,
        persistent_trace_bytes=65536,route_table_bytes=512,extra_buffers=None,
        note='Allmethodsreceiveidenticaloldrepeats/observations; extra projectionarithmeticmeansnotidenticalFLOPs'))
save(a.analysis/'design.json',design)
shutil.copy2(a.analysis/'design.json',a.out/'design.json')
meta=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),source_commit=COMMIT,
    driver_sha256=driver_sha,driver_snapshot='driver_snapshot.py',source_sha256=sources,
    command=[sys.executable,*sys.argv],args={k:str(v) for k,v in vars(a).items()},
    environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device='cpu',
        torch_threads=2,deterministic=True,dtype='float32'),offline_training=False,expected_cases=24,cases=[],
    design_sha256=sha(a.out/'design.json'))
save(a.out/'metadata.json',meta)
try:
    check_deadline()
    excluded=set()
    exclusion_counts={}
    for seed in (90502,90802):
        _,_,seen=training_data(1200,32,seed)
        excluded|=seen
        exclusion_counts[f'training_{seed}']=len(seen)
    for label,path in [('prior90723',a.prior),('frozen90823',a.frozen),('trained90923',a.trained)]:
        data=json.loads((path/'episodes.json').read_text())
        ids={ep['signature'] for eps in data.values() for ep in eps}
        exclusion_counts[label]=len(ids)
        excluded|=ids
    short=evaluation_data(91023,128,128,excluded,'gate')
    short_ids={ep['signature'] for eps in short.values() for ep in eps}
    joint_excluded=excluded|short_ids
    data=dict(short)
    for r in (1,4,16):
        for d in (0,64):
            data[f'pressure_k8_r{r}_d{d}']=[]
    generator=np.random.default_rng(np.random.SeedSequence([91023,8,2,500]))
    attempts=0
    accepted=[]
    while len(accepted)<128:
        check_deadline()
        attempts+=1
        if attempts>10000:
            raise RuntimeError('Joint stress history rejection limit')
        seed=int(generator.integers(2_000_000_000,3_000_000_000))
        cells={}
        for r in (1,4,16):
            for d in (0,64):
                ep=episode(seed,8,d,2,True)
                extra_keys=[ep['keys'][2],ep['keys'][3]]*(r-1)
                extra_values=[ep['values'][2],ep['values'][3]]*(r-1)
                ep['keys']=ep['keys'][:8]+extra_keys+ep['keys'][8:]
                ep['values']=ep['values'][:8]+extra_values+ep['values'][8:]
                ep['boundary']=10+2*(r-1)
                ep['old_reinforcement']=r
                ep['signature']=signature(ep['keys'],ep['values'])
                cells[f'pressure_k8_r{r}_d{d}']=ep
        ids={ep['signature'] for ep in cells.values()}
        if len(ids)!=6 or joint_excluded.intersection(ids):
            continue
        for cell,ep in cells.items():
            data[cell].append(ep)
        joint_excluded|=ids
        accepted.append(seed)
    save(a.out/'episodes.json',data)
    save(a.out/'construction.json',dict(exclusion_counts=exclusion_counts,
        excluded_signature_union=len(excluded),attempts=attempts,accepted_base_seeds=accepted,
        episodes_sha256=sha(a.out/'episodes.json')))
    meta['data']=dict(cells=list(data),histories_per_cell=128,episodes_sha256=sha(a.out/'episodes.json'),
        exclusion_counts=exclusion_counts,excluded_signature_union=len(excluded),candidate_attempts=attempts)
    registry=json.loads((a.trained/'metadata.json').read_text())
    assert registry['status']=='succeeded' and len(registry['cases'])==24
    (a.out/'runs').mkdir()
    save(a.out/'metadata.json',meta)
    eval_args=argparse.Namespace(eval_batch_size=16)
    for original in registry['cases']:
        check_deadline()
        tick=time.perf_counter()
        checkpoint=a.trained/'runs'/original['name']/'checkpoint.pt'
        assert sha(checkpoint)==original['checkpoint_sha256']
        model,payload=load_checkpoint(checkpoint,torch.device('cpu'))
        model.eval().requires_grad_(False)
        assert payload['training']['step']==1200
        assert slow_sha(model)==original['slow_sha256']
        folder=a.out/'runs'/original['name']
        folder.mkdir()
        record={k:original[k] for k in ('name','parent_seed','policy','random_seed','revision_rule','route_table','route_cardinalities','config','slow_sha256')}
        record.update(status='running',source_commit=COMMIT,checkpoint_sha256=sha(checkpoint),
            checkpoint_relative_to_trained=str(checkpoint.relative_to(a.trained)).replace('\\','/'),
            offline_training=False,training_steps=0)
        meta['cases'].append(record)
        save(folder/'case.json',record)
        save(a.out/'metadata.json',meta)
        record['evaluation']=evaluate(model,eval_args,folder/'evaluation',excluded,DEADLINE,
            data_override=data,pressure_arms_override=('online',))
        assert slow_sha(model)==record['slow_sha256']
        record.update(status='succeeded',elapsed_seconds=time.perf_counter()-tick)
        save(folder/'case.json',record)
        save(a.out/'metadata.json',meta)
        print(json.dumps(dict(completed=record['name'],elapsed_seconds=record['elapsed_seconds'])),flush=True)
        del model
    assert sha(driver)==driver_sha and sha(a.out/'driver_snapshot.py')==driver_sha
    assert all(sha(a.source/name)==digest for name,digest in sources.items())
    check_deadline()
    meta['status']='succeeded'
except BaseException as error:
    meta.update(status='failed',failure=repr(error),traceback=traceback.format_exc())
    raise
finally:
    meta.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-START)
    save(a.out/'metadata.json',meta)
