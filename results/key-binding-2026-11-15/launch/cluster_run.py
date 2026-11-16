"""Single bounded allocation, exact upload/download hashes, own-job cleanup."""
import hashlib,json,subprocess,tarfile,time,traceback
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-key-binding-2026-11-15'
REMOTE='/insomnia001/depts/edu/users/zc2819/clmm/key-binding-2026-11-15'
SSH=r'C:\Windows\System32\OpenSSH\ssh.exe';SCP=r'C:\Windows\System32\OpenSSH\scp.exe';CONFIG=r'C:\Users\henry\.ssh\config'
PY='/insomnia001/depts/edu/users/zc2819/clmm/replay-2026-11-12/venv/bin/python'
SOURCE='5e8c61829ff85ecea3c9314d4ccd8dfef3268b68'
OUT=ROOT/'cluster';OUT.mkdir(exist_ok=False)
META=dict(status='running',job_id=None,commands=[],started_epoch=time.time(),pending_limit_seconds=600,allocation_limit_seconds=2700,no_renewal=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save():(OUT/'metadata.json').write_text(json.dumps(META,indent=2),encoding='utf8')
def run(args,timeout=120):
    t=time.monotonic();r=subprocess.run(args,text=True,encoding='utf8',errors='replace',capture_output=True,timeout=timeout)
    META['commands'].append(dict(args=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,elapsed_seconds=time.monotonic()-t));save()
    if r.returncode:raise RuntimeError(r.stderr or r.stdout)
    return r.stdout.strip()
def ssh(cmd,timeout=120):return run([SSH,'-F',CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=8','insomnia-login',cmd],timeout)
def scp(src,dest,recursive=False,timeout=240):return run([SCP,'-F',CONFIG,'-o','BatchMode=yes',*(['-r'] if recursive else []),src,dest],timeout)
try:
    assert json.loads((ROOT/'preparation.json').read_text())['status']=='succeeded'
    for folder in (ROOT/'packet',ROOT/'engineering/packet'):
        assert json.loads((folder/'metadata.json').read_text())['source_commit']==SOURCE
    archive=WORK/'input-upload.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(ROOT/'packet',arcname='packet');tar.add(ROOT/'engineering/packet',arcname='engineering/packet')
        tar.add(WORK/'protocol.bundle',arcname='source.bundle');tar.add(WORK/'remote_pipeline.py',arcname='remote_pipeline.py')
        tar.add(WORK/'environment_probe.py',arcname='environment_probe.py')
    META.update(upload_sha256=sha(archive),upload_bytes=archive.stat().st_size);save()
    ssh(f'mkdir -p {REMOTE}; hostname; insomnia status; insomnia resources; squeue -u zc2819 -h -o "%i %T %j %M %R"')
    scp(str(archive),f'insomnia-login:{REMOTE}/input-upload.tar.gz')
    assert ssh(f'sha256sum {REMOTE}/input-upload.tar.gz').split()[0]==META['upload_sha256']
    ssh(f'cd {REMOTE} && tar -xzf input-upload.tar.gz && git -c core.autocrlf=false clone -q --no-checkout source.bundle source && git -c core.autocrlf=false -C source checkout -q --detach {SOURCE}',180)
    job=ssh(f'sbatch --parsable -A edu -p short --qos=free_short -N1 -n1 -c4 --mem=16G --time=00:45:00 --gres=gpu:A6000:1 --exclude=ins081 --job-name=clmm-binding --output={REMOTE}/allocation-%j.log --wrap="sleep 2700"').split(';')[0]
    assert job.isdigit();META['job_id']=job;save();waiting=time.monotonic()
    while True:
        state=ssh(f'squeue -j {job} -h -o "%u %T %N"')
        if state.startswith('zc2819 RUNNING '):break
        if not state or time.monotonic()-waiting>600:raise TimeoutError(f'Allocation not running: {state}')
        time.sleep(10)
    META.update(verified_running=state,running_epoch=time.time());save();ssh(f'scontrol show job -o {job}')
    probe=f'{PY} -u -B {REMOTE}/environment_probe.py 1'
    ssh(f'timeout --kill-after=10s 90s srun --input=/dev/null --jobid={job} --overlap -N1 -n1 -c4 {probe} > {REMOTE}/import-probe.log 2>&1',115)
    META['phase']='pipeline';save()
    ssh(f'timeout --kill-after=20s 2150s srun --input=/dev/null --jobid={job} --overlap -N1 -n1 -c4 {PY} -u -B {REMOTE}/remote_pipeline.py > {REMOTE}/pipeline.log 2>&1',2200)
    META['phase']='retrieve';save()
    ssh(f'timeout --kill-after=10s 120s srun --input=/dev/null --jobid={job} --overlap -N1 -n1 -c4 tar -czf {REMOTE}/result.tar.gz -C {REMOTE} result',145)
    digest=ssh(f'sha256sum {REMOTE}/result.tar.gz').split()[0];target=WORK/'result-download.tar.gz'
    scp(f'insomnia-login:{REMOTE}/result.tar.gz',str(target));assert sha(target)==digest
    META['download_sha256']=digest
    with tarfile.open(target) as tar:tar.extractall(ROOT/'gpu-output',filter='data')
    for name in ('import-probe.log','pipeline.log'):scp(f'insomnia-login:{REMOTE}/{name}',str(OUT/name))
    result=json.loads((ROOT/'gpu-output/result/pipeline.json').read_text())
    META.update(pipeline_status=result['status'],status=result['status'])
except BaseException as error:
    META.update(status='failed',failure=repr(error),traceback=traceback.format_exc())
    if META['job_id']:
        for name in ('import-probe.log','pipeline.log'):
            try:scp(f'insomnia-login:{REMOTE}/{name}',str(OUT/name))
            except BaseException:pass
        try:
            (ROOT/'gpu-output').mkdir(exist_ok=True);scp(f'insomnia-login:{REMOTE}/result',str(ROOT/'gpu-output'),True)
        except BaseException as recovery:META['partial_retrieval_error']=repr(recovery)
    raise
finally:
    if META['job_id']:
        try:
            ssh(f'scancel {META["job_id"]}');META['cancel_requested']=True;time.sleep(2)
            final=ssh(f'sacct -j {META["job_id"]} --format=JobID,User,State,Elapsed,AllocTRES,ExitCode -P; squeue -u zc2819 -h -o "%i %T %j %M %R"')
            (OUT/'scheduler-final.txt').write_text(final+'\n',encoding='utf8')
        except BaseException as error:META['closeout_error']=repr(error)
    META['ended_epoch']=time.time();save();print(json.dumps({k:v for k,v in META.items() if k!='commands'},indent=2))
