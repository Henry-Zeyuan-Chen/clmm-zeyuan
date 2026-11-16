"""Single-owner, bounded two-GPU pipeline; no reserved output before terminal lock."""
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
def invoke(mode,out,**kw):
    out.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'scripts/format_training_experiment.py',mode,'--packet',str(PACK),'--out',str(out),'--device','cuda:0','--threads','2','--max-seconds',str(max(1,int(left()-30)))]
    for k,v in kw.items():cmd+=['--'+k.replace('_','-'),str(v)]
    with (out.parent/(out.name+'.log')).open('w') as log:subprocess.run(cmd,cwd=SRC,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,left()-10))
    print(json.dumps(dict(stage=A.worker,out=str(out),status='succeeded',utc=utc())),flush=True)
def worker():
    if A.worker=='engineering':
        arm=('familiar','mixed')[A.gpu];out=RUN/'engineering'/arm
        invoke('train',out,parent=68101,encoding='digits',arm=arm,dataset='engineering')
        invoke('evaluate',RUN/'engineering-evaluation'/arm,checkpoint=out/'model.pt',encoding='digits',case='engineering-'+arm,dataset='engineering')
    elif A.worker=='main':
        order=[('digits','familiar'),('digits','mixed'),('atomic','mixed'),('atomic','familiar')]
        if A.parent==68105:order=order[::-1]
        for encoding,arm in order:invoke('train',RUN/'main'/f'{A.parent}-{encoding}-{arm}',parent=A.parent,encoding=encoding,arm=arm,dataset='main')
    else:
        for encoding in ['digits','atomic']:
            init=next(c for c in read(PACK/'initializers.json') if c['parent']==A.parent and c['encoding']==encoding)
            invoke('evaluate',RUN/'confirmation'/f'{A.parent}-{encoding}-initial',checkpoint=PACK/init['checkpoint'],encoding=encoding,case=f'{A.parent}-{encoding}-initial',dataset='confirmation',decision_lock=B/'decision_lock.json')
            for arm in ['familiar','mixed']:
                name=f'{A.parent}-{encoding}-{arm}'
                invoke('evaluate',RUN/'confirmation'/name,checkpoint=RUN/'main'/name/'model.pt',encoding=encoding,case=name,dataset='confirmation',decision_lock=B/'decision_lock.json')
def parallel(stage,devices):
    processes=[];logs=[]
    try:
        for gpu,parent in enumerate([68101,68105]):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[gpu]);log=(B/f'{stage}-{parent}.log').open('w');logs.append(log)
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
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assert len(devices)==2 and all(devices)
    A.hard_stop=time.time()+13500
    state=dict(status='running',started_utc=utc(),slurm_job_id=os.environ['SLURM_JOB_ID'],devices=devices,source_commit=read(PACK/'metadata.json')['source_commit'],packet_metadata_sha256=sha(PACK/'metadata.json'))
    audit=read(B/'analysis/packet_audit.json');assert audit['status']=='passed' and audit['packet_metadata_sha256']==sha(PACK/'metadata.json')
    write(B/'pipeline_state.json',state)
    try:
        state['stage']='engineering';write(B/'pipeline_state.json',state);parallel('engineering',devices)
        rates=[read(RUN/'engineering'/a/'metadata.json')['warm_step_seconds'] for a in ['familiar','mixed']]
        evals=[read(RUN/'engineering-evaluation'/a/'metadata.json') for a in ['familiar','mixed']]
        rate=max(e['forward_seconds']/e['queries'] for e in evals)
        projection=1.3*(4*6000*max(rates)+350208*rate/2+240)
        gate=dict(created_utc=utc(),before_main=True,worst_warm_step_seconds=max(rates),worst_eval_seconds_per_query=rate,projected_seconds=projection,remaining_seconds=left(),proceed=projection<left(),padding=1.3)
        write(B/'resource_decision.json',gate)
        if not gate['proceed']:raise RuntimeError('Full eight-run design exceeds allocation; no main training started')
        state['stage']='main_training';write(B/'pipeline_state.json',state);parallel('main',devices)
        terminals=[]
        for init in read(PACK/'initializers.json'):
            for arm in ['familiar','mixed']:
                path=RUN/'main'/f"{init['id']}-{arm}";m=read(path/'metadata.json')
                assert m['status']=='succeeded' and m['completed_steps']==6000 and m['supervised_answers']==96000 and m['training_input_bytes']==9120000
                assert m['registered_parameters']==839248 and m['initializer']==init and m['exact_reload'] and m['optimizer_reset']
                assert sha(path/'model.pt')==m['checkpoint_sha256'] and sha(path/'optimizer.pt')==m['optimizer_sha256']
                terminals.append(dict(parent=init['parent'],encoding=init['encoding'],arm=arm,status=m['status'],completed_steps=6000,checkpoint_sha256=m['checkpoint_sha256'],state_sha256=m['state_sha256'],metadata_sha256=sha(path/'metadata.json'),ended_utc=m['ended_utc'],path=str(path.relative_to(B))))
        lock=dict(status='all_eight_terminal_locked',created_utc=utc(),packet_metadata_sha256=sha(PACK/'metadata.json'),terminals=terminals,initializers=read(PACK/'initializers.json'),no_adaptive_training=True)
        assert not (B/'decision_lock.json').exists();write(B/'decision_lock.json',lock)
        state['stage']='confirmation';write(B/'pipeline_state.json',state);parallel('confirmation',devices)
        cases=list((RUN/'confirmation').glob('*/metadata.json'));assert len(cases)==12 and sum(read(p)['queries'] for p in cases)==350208
        state.update(status='succeeded',stage='complete',ended_utc=utc(),remaining_seconds=left(),terminal_runs=8,confirmation_capabilities=12,confirmation_vectors=350208)
    except BaseException:
        state.update(status='failed',ended_utc=utc(),traceback=traceback.format_exc());raise
    finally:write(B/'pipeline_state.json',state)
if __name__=='__main__':
    if A.worker:worker()
    else:pipeline()
