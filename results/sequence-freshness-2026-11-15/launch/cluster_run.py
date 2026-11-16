"""Upload frozen inputs and run one bounded two-GPU Insomnia allocation."""
import hashlib,json,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-sequence-freshness-2026-11-15'
REMOTE='/insomnia001/depts/edu/users/zc2819/clmm/sequence-freshness-2026-11-15'
SSH=r'C:\Windows\System32\OpenSSH\ssh.exe';SCP=r'C:\Windows\System32\OpenSSH\scp.exe';CONFIG=r'C:\Users\henry\.ssh\config'
PY='/insomnia001/depts/edu/users/zc2819/clmm/replay-2026-11-12/venv/bin/python'
SOURCE='95da59531ccf5d5f3fe2729553e6d980da598352'
OUT=ROOT/'cluster';OUT.mkdir(exist_ok=False)
META=dict(status='running',job_id=None,commands=[],started_epoch=time.time(),pending_limit_seconds=1200,allocation_limit_seconds=5400,no_renewal=True)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save():(OUT/'metadata.json').write_text(json.dumps(META,indent=2),encoding='utf8')
def run(args,timeout=120):
 begin=time.monotonic();result=subprocess.run(args,text=True,encoding='utf8',errors='replace',capture_output=True,timeout=timeout)
 META['commands'].append(dict(args=args,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,elapsed_seconds=time.monotonic()-begin));save()
 if result.returncode:raise RuntimeError(result.stderr or result.stdout or str(args))
 return result.stdout.strip()
def ssh(command,timeout=120):return run([SSH,'-F',CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=8','insomnia-login',command],timeout)
try:
 for folder in (ROOT/'packet-final',ROOT/'engineering/packet-final'):
  assert json.loads((folder/'metadata.json').read_text())['source_commit']==SOURCE
 archive=WORK/'input-upload.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  tar.add(ROOT/'packet-final',arcname='packet')
  tar.add(ROOT/'engineering/packet-final',arcname='engineering/packet')
  tar.add(WORK/'protocol-executed.bundle',arcname='source.bundle')
  tar.add(WORK/'remote_pipeline.py',arcname='remote_pipeline.py')
 META['upload_sha256']=sha(archive);META['upload_bytes']=archive.stat().st_size;save()
 ssh(f'mkdir -p {REMOTE}; hostname; squeue -u zc2819 -h -o "%i %T %j %M %R"')
 run([SCP,'-F',CONFIG,str(archive),f'insomnia-login:{REMOTE}/input-upload.tar.gz'],timeout=240)
 assert ssh(f'sha256sum {REMOTE}/input-upload.tar.gz').split()[0]==META['upload_sha256']
 ssh(f'cd {REMOTE} && tar -xzf input-upload.tar.gz && git -c core.autocrlf=false clone -q --no-checkout source.bundle source && git -c core.autocrlf=false -C source checkout -q --detach {SOURCE}',timeout=180)
 job=ssh(f'sbatch --parsable -A edu -p short --qos=free_short -N1 -n1 -c8 --mem=16G --time=01:30:00 --gres=gpu:A6000:2 --exclude=ins081 --job-name=clmm-sequence --output={REMOTE}/allocation-%j.log --wrap="sleep 5400"').split(';')[0]
 assert job.isdigit();META['job_id']=job;save();waiting=time.monotonic()
 while True:
  state=ssh(f'squeue -j {job} -h -o "%u %T %N"')
  if state.startswith('zc2819 RUNNING '):break
  if not state or time.monotonic()-waiting>1200:raise TimeoutError(f'Allocation not running: {state}')
  time.sleep(10)
 META['verified_running']=state;META['running_epoch']=time.time();save()
 ssh(f'scontrol show job -o {job}')
 # A bounded import-only probe prevents silently holding resources on a stalled filesystem.
 probe=f"{PY} -u -B -c 'import time; t=time.time(); print(\"PY_START\",flush=True); import torch; print(\"TORCH_IMPORTED\",torch.__version__,time.time()-t,flush=True); print(torch.cuda.device_count(),flush=True); assert torch.cuda.device_count()==2'"
 ssh(f'timeout --kill-after=10s 90s srun --input=/dev/null --jobid={job} --overlap -N1 -n1 -c8 {probe} > {REMOTE}/import-probe.log 2>&1',timeout=115)
 META['phase']='pipeline';save()
 ssh(f'timeout --kill-after=20s 5050s srun --input=/dev/null --jobid={job} --overlap -N1 -n1 -c8 {PY} -u -B {REMOTE}/remote_pipeline.py > {REMOTE}/pipeline.log 2>&1',timeout=5100)
 META['phase']='retrieve';save()
 ssh(f'timeout --kill-after=10s 180s srun --input=/dev/null --jobid={job} --overlap -N1 -n1 -c4 tar -czf {REMOTE}/result.tar.gz -C {REMOTE} result',timeout=205)
 digest=ssh(f'sha256sum {REMOTE}/result.tar.gz').split()[0]
 target=WORK/'result-download.tar.gz'
 run([SCP,'-F',CONFIG,f'insomnia-login:{REMOTE}/result.tar.gz',str(target)],timeout=240);assert sha(target)==digest
 META['download_sha256']=digest
 with tarfile.open(target) as tar:tar.extractall(ROOT/'gpu-output',filter='data')
 for name in ('import-probe.log','pipeline.log'):
  run([SCP,'-F',CONFIG,f'insomnia-login:{REMOTE}/{name}',str(OUT/name)])
 result=json.loads((ROOT/'gpu-output/result/pipeline.json').read_text())
 META['pipeline_status']=result['status'];META['status']='succeeded' if result['status']=='succeeded' else result['status']
except BaseException as error:
 META.update(status='failed',failure=repr(error),traceback=traceback.format_exc())
 if META['job_id']:
  for name in ('import-probe.log','pipeline.log'):
   try:run([SCP,'-F',CONFIG,f'insomnia-login:{REMOTE}/{name}',str(OUT/name)])
   except BaseException:pass
  try:
   (ROOT/'gpu-output').mkdir(exist_ok=True)
   run([SCP,'-F',CONFIG,'-r',f'insomnia-login:{REMOTE}/result',str(ROOT/'gpu-output')],timeout=240)
  except BaseException as retrieval_error:META['partial_retrieval_error']=repr(retrieval_error)
 raise
finally:
 if META['job_id']:
  try:
   ssh(f'scancel {META["job_id"]}');META['cancel_requested']=True;time.sleep(2)
   final=ssh(f'sacct -j {META["job_id"]} --format=JobID,User,State,Elapsed,AllocTRES,ExitCode -P; squeue -u zc2819 -h -o "%i %T %j %M %R"')
   (OUT/'scheduler-final.txt').write_text(final+'\n',encoding='utf8')
  except BaseException as error:META['closeout_error']=repr(error)
 META['ended_epoch']=time.time();save();print(json.dumps({k:v for k,v in META.items() if k!='commands'},indent=2))
