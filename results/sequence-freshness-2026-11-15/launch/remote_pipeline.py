"""One-allocation, two-worker sequence experiment with runtime-only admission."""
import concurrent.futures,json,os,subprocess,sys,time,traceback,threading
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source';ROOT=BASE/'result'
SCRIPT=SOURCE/'scripts/sequence_freshness_experiment.py'
START=time.monotonic();LIMIT=5000
META=dict(status='running',started_epoch=time.time(),source_commit='95da59531ccf5d5f3fe2729553e6d980da598352',slurm_job=os.environ.get('SLURM_JOB_ID'),workers=2,steps=2000,preflight=[],initial=[],cases=[])
ROOT.mkdir(exist_ok=True)
LOCK=threading.Lock()
def save():
    with LOCK:
        target=ROOT/'pipeline.json';temporary=ROOT/'pipeline.json.tmp'
        temporary.write_text(json.dumps(META,indent=2),encoding='utf8');temporary.replace(target)
def run(mode,packet,out,parent,segment,arm,slot):
    left=LIMIT-(time.monotonic()-START)
    if left<=0:raise TimeoutError('Hard5000s pipeline deadline')
    out.parent.mkdir(parents=True,exist_ok=True);log=ROOT/'logs'/f'{out.parent.name}-{out.name}.log';log.parent.mkdir(exist_ok=True)
    cmd=[sys.executable,'-u','-B',str(SCRIPT),mode,'--packet',str(packet),'--out',str(out),'--parent',str(parent),'--segment',str(segment),'--arm',arm,'--device',f'cuda:{slot}','--threads','2','--eval-batch','16','--max-seconds',str(int(min(left,4000)))]
    begin=time.monotonic()
    with log.open('w',encoding='utf8') as f:
        process=subprocess.Popen(cmd,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONHASHSEED':'0'})
        try:code=process.wait(timeout=left)
        except BaseException:process.kill();process.wait();raise
    if code:raise RuntimeError(f'Case failed{code}: {log}')
    meta=json.loads((out/'metadata.json').read_text());assert meta['status']=='succeeded' and meta['reload_max_error']==0
    return dict(name=out.name,mode=mode,parent=parent,segment=segment,arm=arm,slot=slot,elapsed_seconds=time.monotonic()-begin,training_seconds=meta['training_elapsed_seconds'],evaluation_seconds=meta['evaluation']['elapsed_seconds'],command=cmd,log=str(log),checkpoint_sha256=meta['checkpoint_sha256'])
def worker(slot,segment):
    values=[]
    for parent in (68101,68105):
        record=run('initial',BASE/'packet',ROOT/'initial'/f'{parent}-w{segment}',parent,segment,'periodic',slot)
        META['initial'].append(record);save();print(json.dumps(dict(initial=record['name'])),flush=True)
    for parent in (68101,68105):
        for arm in ('periodic','periodic_bounded'):
            record=run('train',BASE/'packet',ROOT/'runs'/f'{parent}-w{segment}-{arm}',parent,segment,arm,slot)
            META['cases'].append(record);save();print(json.dumps(dict(completed=record['name'],seconds=record['elapsed_seconds'],count=len(META['cases']))),flush=True)
    return values
try:
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(run,'preflight',BASE/'engineering/packet',ROOT/'engineering'/f'w{segment}',68101,segment,'periodic',slot) for slot,segment in enumerate((32,128))]
        META['preflight']=[f.result() for f in fs]
    train=max(r['training_seconds'] for r in META['preflight']);evaluation=max(r['evaluation_seconds'] for r in META['preflight'])
    projection=1.25*(8*100*train+12*8*evaluation)/2
    META.update(projected_main_seconds=projection,runtime_gate=projection<=4200);save()
    print(json.dumps(dict(projected_main_seconds=projection,proceed=projection<=4200)),flush=True)
    if projection<=4200:
        META['main_started_epoch']=time.time();save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(worker,slot,segment) for slot,segment in enumerate((32,128))]
            for f in fs:f.result()
        META['main_elapsed_seconds']=time.time()-META['main_started_epoch'];META['status']='succeeded'
    else:META['status']='preflight_only'
except BaseException as error:
    META.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    META.update(ended_epoch=time.time(),elapsed_seconds=time.monotonic()-START);save()
