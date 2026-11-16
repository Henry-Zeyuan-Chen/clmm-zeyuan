"""One GPU, fixed four-case diagnostic, runtime-only admission and hard stop."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE/'result';SOURCE=BASE/'source'
SCRIPT=SOURCE/'scripts/key_binding_experiment.py'
START=time.monotonic();LIMIT=2100
META=dict(status='running',source_commit='5e8c61829ff85ecea3c9314d4ccd8dfef3268b68',started_epoch=time.time(),slurm_job=os.environ.get('SLURM_JOB_ID'),workers=1,preflight=[],initial=[],cases=[])
ROOT.mkdir(exist_ok=True)
def save():
    target=ROOT/'pipeline.json';temp=ROOT/'pipeline.json.tmp'
    temp.write_text(json.dumps(META,indent=2),encoding='utf8');temp.replace(target)
def run(mode,packet,out,parent,encoding):
    left=LIMIT-(time.monotonic()-START)
    if left<=0:raise TimeoutError('Hard2100s pipeline cap')
    out.parent.mkdir(parents=True,exist_ok=True);log=ROOT/'logs'/f'{out.parent.name}-{out.name}.log';log.parent.mkdir(exist_ok=True)
    cmd=[sys.executable,'-u','-B',str(SCRIPT),mode,'--packet',str(packet),'--out',str(out),'--parent',str(parent),'--encoding',encoding,'--device','cuda:0','--threads','2','--eval-batch','16','--max-seconds',str(int(min(left,1800)))]
    t=time.monotonic()
    with log.open('w',encoding='utf8') as f:
        process=subprocess.Popen(cmd,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONHASHSEED':'0'})
        try:code=process.wait(timeout=left)
        except BaseException:process.kill();process.wait();raise
    if code:raise RuntimeError(f'{out.name} exit{code}; original log preserved')
    meta=json.loads((out/'metadata.json').read_text());assert meta['status']=='succeeded' and meta['reload_max_error']==0
    return dict(name=out.name,mode=mode,parent=parent,encoding=encoding,elapsed_seconds=time.monotonic()-t,
                training_seconds=meta['training_elapsed_seconds'],evaluation_seconds=meta['evaluation']['elapsed_seconds'],
                command=cmd,log=str(log),checkpoint_sha256=meta['checkpoint_sha256'])
try:
    save()
    for encoding in ('digits','atomic'):
        META['preflight'].append(run('preflight',BASE/'engineering/packet',ROOT/'engineering'/encoding,68101,encoding));save()
    train=max(r['training_seconds'] for r in META['preflight']);evaluation=max(r['evaluation_seconds'] for r in META['preflight'])
    projection=1.25*(4*100*train+8*16*evaluation)
    META.update(projected_main_seconds=projection,runtime_gate=projection<=1800);save()
    print(json.dumps(dict(projected_main_seconds=projection,proceed=projection<=1800)),flush=True)
    if projection<=1800:
        META['main_started_epoch']=time.time();save()
        for parent in (68101,68105):
            for encoding in ('digits','atomic'):
                record=run('initial',BASE/'packet',ROOT/'initial'/f'{parent}-{encoding}',parent,encoding)
                META['initial'].append(record);save();print(json.dumps(dict(initial=record['name'])),flush=True)
        for parent in (68101,68105):
            for encoding in ('digits','atomic'):
                record=run('train',BASE/'packet',ROOT/'runs'/f'{parent}-{encoding}',parent,encoding)
                META['cases'].append(record);save();print(json.dumps(dict(completed=record['name'],seconds=record['elapsed_seconds'])),flush=True)
        META['main_elapsed_seconds']=time.time()-META['main_started_epoch'];META['status']='succeeded'
    else:META['status']='preflight_only'
except BaseException as error:
    META.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    META.update(ended_epoch=time.time(),elapsed_seconds=time.monotonic()-START);save()
