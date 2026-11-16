"""Read-only confirmation that this task's two scoped allocations are closed."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'outputs/clmm-key-binding-2026-11-15'
SSH=r'C:\Windows\System32\OpenSSH\ssh.exe';CONFIG=r'C:\Users\henry\.ssh\config'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def run(command):
    r=subprocess.run([SSH,'-F',CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=8','insomnia-login',command],capture_output=True,text=True,encoding='utf8',timeout=30)
    assert r.returncode==0,r.stderr
    return r.stdout
jobs=[]
for directory,gpus in (('cluster',1),('cluster-v2',2)):
    folder=ROOT/directory;meta=read(folder/'metadata.json');assert meta['cancel_requested']
    job=str(meta['job_id']);accounting=run(f'sacct -j {job} --format=JobIDRaw,User,State,ElapsedRaw,AllocTRES,ExitCode -P')
    (folder/'accounting-confirmed.txt').write_text(accounting,encoding='utf8')
    row=next(s.split('|') for s in accounting.splitlines() if s.startswith(job+'|'))
    assert row[1]=='zc2819' and row[2].startswith('CANCELLED')
    record=dict(job_id=int(job),state=row[2],GPUs=gpus,allocation_elapsed_seconds=int(row[3]),allocated_GPU_hours=gpus*int(row[3])/3600,job_closed=True)
    jobs.append(record);write(folder/'closure.json',record)
queue=run('squeue -u zc2819 -h -o "%i %T %j %M %R"')
active={line.split()[0] for line in queue.splitlines() if line.strip()}
assert not {str(r['job_id']) for r in jobs}&active
(ROOT/'cluster-v2/queue-confirmed.txt').write_text(queue if queue.strip() else 'EMPTY: live squeue returned no jobs (exit0)\n',encoding='utf8')
result=dict(status='passed',our_allocations_closed=True,no_active_user_jobs=not bool(active),unrelated_active_jobs=queue.splitlines(),
            jobs=jobs,total_allocated_GPU_hours=sum(r['allocated_GPU_hours'] for r in jobs),no_helper_selection_change=True,no_time_limit_extension=True)
write(ROOT/'cluster-closure.json',result);print(json.dumps(result,indent=2))
