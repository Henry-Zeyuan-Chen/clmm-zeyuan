"""One bounded Insomnia allocation for selected checkpoint portability proof."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-sleep-multirevision-2026-11-15'
REMOTE='/insomnia001/depts/edu/users/zc2819/clmm/sleep-multirevision-2026-11-15'
SSH=r'C:\Windows\System32\OpenSSH\ssh.exe';SCP=r'C:\Windows\System32\OpenSSH\scp.exe';CONFIG=r'C:\Users\henry\.ssh\config'
PY='/insomnia001/depts/edu/users/zc2819/clmm/replay-2026-11-12/venv/bin/python'
OUT=ROOT/'cluster';OUT.mkdir(exist_ok=False)
meta=dict(status='running',allocation_limit_seconds=900,pending_limit_seconds=120,selected_inference_limit_seconds=420,commands=[],started_epoch=time.time(),job_id=None)
def save():(OUT/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
def command(args,timeout=90):
 begin=time.monotonic();r=subprocess.run(args,text=True,encoding='utf8',errors='replace',capture_output=True,timeout=timeout)
 meta['commands'].append(dict(args=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,elapsed_seconds=time.monotonic()-begin));save()
 if r.returncode:raise RuntimeError(r.stderr or r.stdout)
 return r.stdout.strip()
def ssh(cmd,timeout=90):return command([SSH,'-F',CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=8','insomnia-login',cmd],timeout)
try:
 command([sys.executable,'-X','utf8',str(WORK/'portable_packet.py')],timeout=180)
 archive=WORK/'portable-proof.tar.gz';digest=hashlib.sha256(archive.read_bytes()).hexdigest();meta['upload_sha256']=digest;save()
 ssh(f'mkdir -p {REMOTE}; hostname; squeue -u zc2819 -h -o "%i %T %j %M %R"')
 command([SCP,'-F',CONFIG,str(archive),f'insomnia-login:{REMOTE}/portable-proof.tar.gz'],timeout=180)
 remote_sha=ssh(f'sha256sum {REMOTE}/portable-proof.tar.gz').split()[0];assert remote_sha==digest
 ssh(f'cd {REMOTE} && tar -xzf portable-proof.tar.gz && git -c core.autocrlf=false clone -q --no-checkout protocol.bundle source && git -c core.autocrlf=false -C source checkout -q --detach 06c0a35b543c25437659765128b81718e5bfa1c5',timeout=180)
 job=ssh(f'sbatch --parsable -A edu -p short --qos=free_short -N1 -n1 -c4 --mem=16G --time=00:15:00 --job-name=clmm-sleep-proof --output={REMOTE}/allocation-%j.log --gres=gpu:A6000:1 --wrap="sleep 900"').split(';')[0]
 assert job.isdigit();meta['job_id']=job;save();begin=time.monotonic()
 while True:
  status=ssh(f'squeue -j {job} -h -o "%u %T %N"')
  if status.startswith('zc2819 RUNNING '):break
  if not status or time.monotonic()-begin>120:raise TimeoutError(f'Allocation not running: {status}')
  time.sleep(5)
 meta['verified_running_status']=status;save()
 ssh(f'scontrol show job -o {job}')
 ssh(f'timeout 420s srun --input=/dev/null --jobid={job} --overlap -N1 -n1 -c4 {PY} -B {REMOTE}/recheck.py --root {REMOTE}/root --source {REMOTE}/source --device cuda --expected {REMOTE}/root/portable-proof/expected --out {REMOTE}/gpu-recheck > {REMOTE}/gpu-recheck.log 2>&1',timeout=450)
 local=ROOT/'gpu-recheck';local.mkdir(exist_ok=False)
 for name in ('verification.json',):command([SCP,'-F',CONFIG,f'insomnia-login:{REMOTE}/gpu-recheck/{name}',str(local/name)])
 command([SCP,'-F',CONFIG,f'insomnia-login:{REMOTE}/gpu-recheck.log',str(local/'execution.log')])
 result=json.loads((local/'verification.json').read_text());assert result['status']=='passed' and result['queries']==5184 and result['models']==12
 meta['status']='succeeded';meta['verification_max_abs_error']=result['max_abs_error']
except BaseException as error:
 meta.update(status='failed',error=repr(error),traceback=traceback.format_exc())
 if meta['job_id']:
  try:ssh(f'tail -n 60 {REMOTE}/gpu-recheck.log')
  except BaseException:pass
 raise
finally:
 if meta['job_id']:
  try:
   ssh(f'scancel {meta["job_id"]}')
   time.sleep(2)
   final=ssh(f'sacct -j {meta["job_id"]} --format=JobID,User,State,Elapsed,AllocTRES,ExitCode -P; squeue -u zc2819 -h -o "%i %T %j %M %R"')
   (OUT/'scheduler-final.txt').write_text(final+'\n',encoding='utf8');meta['cancel_requested']=True
  except BaseException as error:meta['closeout_error']=repr(error)
 meta['ended_epoch']=time.time();save()
 print(json.dumps({k:v for k,v in meta.items() if k!='commands'},indent=2))
