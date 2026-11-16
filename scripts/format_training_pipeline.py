"""Single-owner one/two-GPU pipeline; reserved output follows terminal lock."""
import argparse,json,os,signal,subprocess,sys,time,traceback,hashlib
from pathlib import Path
from datetime import datetime,timezone
P=argparse.ArgumentParser();P.add_argument('--base',required=True,type=Path);P.add_argument('--worker',choices=['engineering','main','confirmation']);P.add_argument('--parent',type=int);P.add_argument('--gpu',type=int);P.add_argument('--hard-stop',type=float)
A=P.parse_args();B=A.base.resolve();SRC=B/'source';PACK=B/'packet';RUN=B/'runs';SCRIPT=Path(__file__).resolve()
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def left():return A.hard_stop-time.time()
def slurm_time_seconds(value):
    """Parse Slurm's actual remaining time, refusing unknown/unlimited values."""
    pieces=value.strip().split('-')
    if len(pieces)>2:raise ValueError('Invalid Slurm time')
    days=int(pieces[0]) if len(pieces)==2 else 0
    clock=pieces[-1].split(':')
    if len(clock) not in (2,3) or any(not p.isdigit() for p in clock):raise ValueError('Unknown Slurm time')
    numbers=list(map(int,clock))
    if numbers[-1]>=60 or numbers[-2]>=60:raise ValueError('Invalid Slurm clock')
    return days*86400+sum(n*factor for n,factor in zip(reversed(numbers),(1,60,3600)))
def invoke(mode,out,**kw):
    out.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'scripts/format_training_experiment.py',mode,'--packet',str(PACK),'--out',str(out),'--device','cuda:0','--threads','2','--max-seconds',str(max(1,int(left()-30)))]
    for k,v in kw.items():cmd+=['--'+k.replace('_','-'),str(v)]
    with (out.parent/(out.name+'.log')).open('w') as log:subprocess.run(cmd,cwd=SRC,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,left()-10))
    print(json.dumps(dict(stage=A.worker,out=str(out),status='succeeded',utc=utc())),flush=True)
def worker():
    arms=read(PACK/'protocol.json')['arms']
    if A.worker=='engineering':
        arm=arms[A.gpu];out=RUN/'engineering'/arm
        invoke('train',out,parent=68101,encoding='digits',arm=arm,dataset='engineering')
        invoke('evaluate',RUN/'engineering-evaluation'/arm,checkpoint=out/'model.pt',encoding='digits',case='engineering-'+arm,dataset='engineering')
    elif A.worker=='main':
        order=[('digits',arms[0]),('digits',arms[1]),('atomic',arms[1]),('atomic',arms[0])]
        if A.parent==68105:order=order[::-1]
        for encoding,arm in order:invoke('train',RUN/'main'/f'{A.parent}-{encoding}-{arm}',parent=A.parent,encoding=encoding,arm=arm,dataset='main')
    else:
        for encoding in ['digits','atomic']:
            init=next(c for c in read(PACK/'initializers.json') if c['parent']==A.parent and c['encoding']==encoding)
            invoke('evaluate',RUN/'confirmation'/f'{A.parent}-{encoding}-initial',checkpoint=PACK/init['checkpoint'],encoding=encoding,case=f'{A.parent}-{encoding}-initial',dataset='confirmation',decision_lock=B/'decision_lock.json')
            for arm in arms:
                name=f'{A.parent}-{encoding}-{arm}'
                invoke('evaluate',RUN/'confirmation'/name,checkpoint=RUN/'main'/name/'model.pt',encoding=encoding,case=name,dataset='confirmation',decision_lock=B/'decision_lock.json')
def parallel(stage,devices):
    tasks=list(enumerate([68101,68105]))
    for offset in range(0,len(tasks),len(devices)):
        processes=[];logs=[]
        try:
            for gpu,parent in tasks[offset:offset+len(devices)]:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[gpu%len(devices)]);log=(B/f'{stage}-{parent}.log').open('w');logs.append(log)
                cmd=[sys.executable,str(SCRIPT),'--base',str(B),'--worker',stage,'--gpu',str(gpu),'--parent',str(parent),'--hard-stop',str(A.hard_stop)]
                processes.append(subprocess.Popen(cmd,cwd=SRC,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(p.poll() is None for p in processes):
                if left()<15:raise TimeoutError('Pipeline resource deadline')
                if any(p.poll() not in (None,0) for p in processes):raise RuntimeError(f'{stage} worker failed; preserve logs')
                time.sleep(5)
            assert all(p.returncode==0 for p in processes)
        finally:
            for p in processes:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for p in processes:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            for f in logs:f.close()
def pipeline():
    assert os.environ.get('SLURM_JOB_ID'),'Substantial work requires Slurm'
    protocol=read(PACK/'protocol.json');arms=protocol['arms'];vectors=protocol['confirmation']['total_vectors']
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assert len(devices) in (1,2) and all(devices)
    assert len(devices)==protocol['resources']['gpus']
    actual_left=slurm_time_seconds(subprocess.check_output(['squeue','-h','-j',os.environ['SLURM_JOB_ID'],'-o','%L'],text=True))
    assert 180<actual_left<=protocol['resources'].get('max_wall_seconds',14400),'Allocation must fit the declared maximum'
    A.hard_stop=time.time()+min(13500,actual_left-180)
    state=dict(status='running',started_utc=utc(),slurm_job_id=os.environ['SLURM_JOB_ID'],actual_slurm_seconds=actual_left,hard_stop_epoch=A.hard_stop,devices=devices,source_commit=read(PACK/'metadata.json')['source_commit'],packet_metadata_sha256=sha(PACK/'metadata.json'))
    audit=read(B/'analysis/packet_audit.json');assert audit['status']=='passed' and audit['packet_metadata_sha256']==sha(PACK/'metadata.json')
    write(B/'pipeline_state.json',state)
    try:
        state['stage']='engineering';write(B/'pipeline_state.json',state);parallel('engineering',devices)
        rates=[read(RUN/'engineering'/a/'metadata.json')['warm_step_seconds'] for a in arms]
        evals=[read(RUN/'engineering-evaluation'/a/'metadata.json') for a in arms]
        rate=max(e['forward_seconds']/e['queries'] for e in evals)
        initial_delta=None
        if arms==['constant','warmup']:
            import numpy as np
            a0=np.load(RUN/'engineering'/arms[0]/'first_logits.npy');a1=np.load(RUN/'engineering'/arms[1]/'first_logits.npy')
            initial_delta=float(np.max(np.abs(a0-a1)));assert initial_delta<=1e-5,'Paired initial functions differ'
        projection=1.3*(8*6000*max(rates)/len(devices)+vectors*rate/len(devices)+240)
        gate=dict(created_utc=utc(),before_main=True,gpus=len(devices),confirmation_vectors=vectors,initial_pair_max_logit_delta=initial_delta,worst_warm_step_seconds=max(rates),worst_eval_seconds_per_query=rate,projected_seconds=projection,remaining_seconds=left(),proceed=projection<left(),padding=1.3)
        write(B/'resource_decision.json',gate)
        if not gate['proceed']:raise RuntimeError('Full eight-run design exceeds allocation; no main training started')
        state['stage']='main_training';write(B/'pipeline_state.json',state);parallel('main',devices)
        terminals=[]
        for init in read(PACK/'initializers.json'):
            first=[]
            for arm in arms:
                path=RUN/'main'/f"{init['id']}-{arm}";m=read(path/'metadata.json')
                assert m['status']=='succeeded' and m['completed_steps']==6000 and m['supervised_answers']==96000 and m['training_input_bytes']==9120000
                assert m['registered_parameters']==839248 and m['initializer']==init and m['exact_reload'] and m['optimizer_reset']
                assert sha(path/'model.pt')==m['checkpoint_sha256'] and sha(path/'optimizer.pt')==m['optimizer_sha256']
                if arms==['constant','warmup']:
                    assert sha(path/'first_logits.npy')==m['first_logits_sha256']
                    first.append(np.load(path/'first_logits.npy'))
                terminals.append(dict(parent=init['parent'],encoding=init['encoding'],arm=arm,status=m['status'],completed_steps=6000,checkpoint_sha256=m['checkpoint_sha256'],state_sha256=m['state_sha256'],metadata_sha256=sha(path/'metadata.json'),ended_utc=m['ended_utc'],path=str(path.relative_to(B))))
            if first:assert float(np.max(np.abs(first[0]-first[1])))<=1e-5
        lock=dict(status='all_eight_terminal_locked',created_utc=utc(),packet_metadata_sha256=sha(PACK/'metadata.json'),terminals=terminals,initializers=read(PACK/'initializers.json'),no_adaptive_training=True)
        assert not (B/'decision_lock.json').exists();write(B/'decision_lock.json',lock)
        state['stage']='confirmation';write(B/'pipeline_state.json',state);parallel('confirmation',devices)
        cases=list((RUN/'confirmation').glob('*/metadata.json'));assert len(cases)==12 and sum(read(p)['queries'] for p in cases)==vectors
        state.update(status='succeeded',stage='complete',ended_utc=utc(),remaining_seconds=left(),terminal_runs=8,confirmation_capabilities=12,confirmation_vectors=vectors)
    except BaseException:
        state.update(status='failed',ended_utc=utc(),traceback=traceback.format_exc());raise
    finally:write(B/'pipeline_state.json',state)
if __name__=='__main__':
    if A.worker:worker()
    else:pipeline()
