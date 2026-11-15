"""One bounded routing/decay factorial; runtime-only continuation rule."""
import argparse,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--python',required=True);p.add_argument('--source-commit',required=True)
a=p.parse_args();root=a.root;repo=root/'repo';started=time.perf_counter()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==a.source_commit
record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),source_commit=a.source_commit,job_id=os.environ.get('SLURM_JOB_ID'),commands=[])
def save():(root/'pipeline-cpu.json').write_text(json.dumps(record,indent=2)+'\n')
def call(command,log,timeout):
 entry=dict(command=command,log=log,status='running');record['commands'].append(entry);save()
 tick=time.perf_counter()
 with (root/log).open('x') as h:r=subprocess.run(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,timeout=timeout)
 entry.update(exit_code=r.returncode,status='succeeded' if r.returncode==0 else 'failed',elapsed_seconds=time.perf_counter()-tick);save()
 if r.returncode:raise RuntimeError(log+' failed')
def command(name,small=False):
 return [a.python,'-B',str(repo/'scripts/routing_decay_experiment.py'),'--out',str(root/name),'--parents',str(root/'parents'),'--design',str(root/'design.json'),'--source-commit',a.source_commit,'--device','cpu','--max-seconds','180' if small else '2850','--dev-seed','90703' if small else '90723','--eval-episodes','16' if small else '128','--pressure-episodes','16' if small else '128','--case-limit','1' if small else '48']
try:
 save();call(command('engineering-cpu',True),'engineering-cpu.log',240)
 m=json.loads((root/'engineering-cpu/metadata.json').read_text());percase=m['cases'][0]['elapsed_seconds']
 projection=48*8*percase*1.3+90
 record['runtime_preflight']=dict(engineering_case_seconds=percase,projected_main_seconds=projection,rule='48cases x8historybatches x1.3padding +90; no score consulted')
 save()
 if projection>2550:record.update(status='preflight_only',reason='Runtime bound exceeded; no performance selection')
 else:
  call(command('main'),'main.log',2920)
  call([a.python,'-B',str(root/'recheck_routing.py'),'--root',str(root),'--repo',str(repo),'--out',str(root/'gpu-recheck'),'--device','cuda'],'gpu-recheck.log',300)
  record['status']='succeeded'
except BaseException as e:record.update(status='failed',failure=repr(e));raise
finally:
 record.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started);save()
