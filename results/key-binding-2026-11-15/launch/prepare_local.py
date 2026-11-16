"""Sequential frozen-source CPU smoke and two declared data packets."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]/'outputs/clmm-key-binding-2026-11-15'
SOURCE=WORK/'source-frozen'
PY=WORK.parent/'clmm-test-env/Scripts/python.exe'
OLD=ROOT.parent/'clmm-sequence-freshness-2026-11-15/analysis/future_exclusion_signatures.json'
PARENTS=ROOT.parent/'clmm-shared-pretrain-2026-11-14'
SCRIPT=SOURCE/'scripts/key_binding_experiment.py'
ROOT.mkdir(exist_ok=True)
META=dict(status='running',commands=[],source_commit='5e8c61829ff85ecea3c9314d4ccd8dfef3268b68',started_epoch=time.time())
def save():(ROOT/'preparation.json').write_text(json.dumps(META,indent=2),encoding='utf8')
def run(name,args,limit=240):
    t=time.monotonic();cmd=[str(PY),'-X','utf8','-u','-B',str(SCRIPT),*map(str,args)]
    with (WORK/f'{name}.log').open('w',encoding='utf8') as f:
        result=subprocess.run(cmd,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
    META['commands'].append(dict(name=name,command=cmd,exit_code=result.returncode,elapsed_seconds=time.monotonic()-t));save()
    if result.returncode:raise RuntimeError(f'{name} failed; original log preserved')
def prepare(name,steps,contexts,n,train,dev,extra=()):
    out=ROOT/name
    args=['prepare','--out',out,'--parents',PARENTS,'--exclusions',OLD,'--steps',steps,'--contexts',contexts,'--episodes',n,
          '--train-seed',train,'--dev-seed',dev,'--design',ROOT/'design.json']
    for p in extra:args+=['--exclude-packet',ROOT/p]
    run('prepare-'+name.replace('/','-'),args)
try:
    assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='99db2ebb75ccf49faad2fc3cd3a4bff9bb8ff42cc5e842063a695feba8d02a7b'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==META['source_commit']
    prepare('engineering-cpu/packet',8,2,2,91406,91416)
    run('cpu-smoke',['train','--packet',ROOT/'engineering-cpu/packet','--out',ROOT/'engineering-cpu/run',
        '--parent',68101,'--encoding','atomic','--device','cpu','--max-seconds',120])
    prepare('engineering/packet',20,8,16,91403,91413,('engineering-cpu/packet',))
    prepare('packet',2000,8,256,91402,91423,('engineering-cpu/packet','engineering/packet'))
    META['status']='succeeded'
except BaseException as error:
    META.update(status='failed',error=repr(error));raise
finally:
    META['ended_epoch']=time.time();save();print(json.dumps(META,indent=2))
