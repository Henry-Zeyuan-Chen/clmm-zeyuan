"""Execution-only amendment: two parent workers, unchanged frozen experiment."""
import concurrent.futures,hashlib,json,os,subprocess,sys,threading,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE/'result';SOURCE=BASE/'source'
SCRIPT=SOURCE/'scripts/key_binding_experiment.py'
START=time.monotonic();LIMIT=2400;LOCK=threading.Lock()
AMEND=BASE/'execution-amendment.json'
META=dict(status='running',source_commit='5e8c61829ff85ecea3c9314d4ccd8dfef3268b68',started_epoch=time.time(),slurm_job=os.environ.get('SLURM_JOB_ID'),workers=2,
          execution_amendment_sha256=hashlib.sha256(AMEND.read_bytes()).hexdigest(),preflight=[],initial=[],cases=[])
ROOT.mkdir(exist_ok=True)
def save():
    with LOCK:
        target=ROOT/'pipeline.json';temp=ROOT/'pipeline.json.tmp'
        temp.write_text(json.dumps(META,indent=2),encoding='utf8');temp.replace(target)
def run(mode,packet,out,parent,encoding,slot):
    left=LIMIT-(time.monotonic()-START)
    if left<=0:raise TimeoutError('Hard2400s amended pipeline cap')
    out.parent.mkdir(parents=True,exist_ok=True);log=ROOT/'logs'/f'{out.parent.name}-{out.name}.log';log.parent.mkdir(exist_ok=True)
    cmd=[sys.executable,'-u','-B',str(SCRIPT),mode,'--packet',str(packet),'--out',str(out),'--parent',str(parent),'--encoding',encoding,'--device',f'cuda:{slot}','--threads','2','--eval-batch','16','--max-seconds',str(int(min(left,1800)))]
    t=time.monotonic()
    with log.open('w',encoding='utf8') as f:
        process=subprocess.Popen(cmd,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONHASHSEED':'0'})
        try:code=process.wait(timeout=left)
        except BaseException:process.kill();process.wait();raise
    if code:raise RuntimeError(f'{out.name} exit{code}; original log preserved')
    meta=json.loads((out/'metadata.json').read_text());assert meta['status']=='succeeded' and meta['reload_max_error']==0
    return dict(name=out.name,mode=mode,parent=parent,encoding=encoding,slot=slot,elapsed_seconds=time.monotonic()-t,
                training_seconds=meta['training_elapsed_seconds'],evaluation_seconds=meta['evaluation']['elapsed_seconds'],
                command=cmd,log=str(log),checkpoint_sha256=meta['checkpoint_sha256'])
def worker(mode,parent,slot):
    for encoding in ('digits','atomic'):
        record=run(mode,BASE/'packet',ROOT/('initial' if mode=='initial' else 'runs')/f'{parent}-{encoding}',parent,encoding,slot)
        META['initial' if mode=='initial' else 'cases'].append(record);save()
        print(json.dumps(dict(mode=mode,completed=record['name'],seconds=record['elapsed_seconds'])),flush=True)
try:
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(run,'preflight',BASE/'engineering/packet',ROOT/'engineering'/encoding,68101,encoding,slot) for slot,encoding in enumerate(('digits','atomic'))]
        META['preflight']=[f.result() for f in fs]
    train=max(r['training_seconds'] for r in META['preflight']);evaluation=max(r['evaluation_seconds'] for r in META['preflight'])
    projection=1.25*(4*100*train+8*16*evaluation)/2
    META.update(projected_main_seconds=projection,runtime_gate=projection<=2300);save()
    print(json.dumps(dict(projected_main_seconds=projection,proceed=projection<=2300)),flush=True)
    if projection<=2300:
        META['main_started_epoch']=time.time();save()
        for mode in ('initial','train'):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(worker,mode,parent,slot) for slot,parent in enumerate((68101,68105))]
                for f in fs:f.result()
        META['main_elapsed_seconds']=time.time()-META['main_started_epoch'];META['status']='succeeded'
    else:META['status']='preflight_only'
except BaseException as error:
    META.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    META.update(ended_epoch=time.time(),elapsed_seconds=time.monotonic()-START);save()
