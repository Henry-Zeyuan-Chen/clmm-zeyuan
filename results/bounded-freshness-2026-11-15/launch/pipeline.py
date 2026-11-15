import concurrent.futures, json, os, subprocess, sys, time, traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'outputs/clmm-bounded-freshness-2026-11-15'
SOURCE=Path(__file__).parent/'source-checkout'
SCRIPT=SOURCE/'scripts/freshness_experiment.py'
OLD=BASE/'outputs/clmm-revision-local-2026-11-15/trained'
META={'status':'running','started_epoch':time.time(),'commands':[]}
START=time.monotonic();LIMIT=4200
def save():
    (OUT/'pipeline.json').write_text(json.dumps(META,indent=2),encoding='utf8')
def run(args,log):
    left=LIMIT-(time.monotonic()-START)
    if left<=0: raise TimeoutError('Pipeline hard deadline')
    cmd=[sys.executable,'-X','utf8','-B',str(SCRIPT),*map(str,args)]
    log.parent.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    with log.open('w',encoding='utf8') as f:
        p=subprocess.Popen(cmd,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        try: code=p.wait(timeout=left)
        except BaseException:
            p.kill();p.wait();raise
    result=dict(command=cmd,log=str(log),elapsed=time.monotonic()-begin,returncode=code)
    if code: raise RuntimeError(result)
    return result
def prep(packet,seed,n):
    return run(['prepare','--out',packet,'--trained',OLD,'--history-root',BASE/'outputs','--design',OUT/'design.json','--dev-seed',seed,'--episodes',n],packet.parent/'prepare.log')
def case(packet,out,name):
    return run(['evaluate','--out',out,'--packet',packet,'--model',name,'--max-seconds',1500],out.parent.parent/'logs'/f'{out.parent.name}-{name}.log')
try:
    engineering=OUT/'engineering';packet=engineering/'packet'
    META['commands'].append(prep(packet,91203,16));save()
    names=[r['name'] for r in json.loads((packet/'models.json').read_text())]
    reps=[next(n for n in names if '90511' in n and 'content' in n),next(n for n in names if '90515' in n and '90701' in n)]
    META['representatives']=reps;META['preflight_sequential']=[];save()
    for name in reps:
        META['preflight_sequential'].append(case(packet,engineering/'sequential'/name,name));save()
    begin=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(case,packet,engineering/'concurrent'/name,name) for name in reps]
        META['preflight_concurrent']=[f.result() for f in futures]
    wall=time.monotonic()-begin
    projected=wall*8*6*1.25
    META.update(concurrent_wall_seconds=wall,projected_main_seconds=projected,preflight_pass=projected<=3600);save()
    print(json.dumps({'preflight_wall':wall,'projected_main':projected,'proceed':projected<=3600}),flush=True)
    if projected<=3600:
        packet=OUT/'packet';META['commands'].append(prep(packet,91223,128));save()
        META['main_started_epoch']=time.time();META['main']=[];save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(case,packet,OUT/'runs'/name,name):name for name in names}
            for f in concurrent.futures.as_completed(futures):
                record=f.result();record['name']=futures[f];META['main'].append(record);save()
                print(json.dumps({'finished':record['name'],'seconds':record['elapsed'],'count':len(META['main'])}),flush=True)
        META['main_elapsed_seconds']=time.time()-META['main_started_epoch']
    META['status']='succeeded' if projected<=3600 else 'preflight_only'
except BaseException as error:
    META.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
    META.update(ended_epoch=time.time(),elapsed_seconds=time.monotonic()-START);save()
