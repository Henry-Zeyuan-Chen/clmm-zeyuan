"""Bounded runtime-only continuation to a fixed24-case matched training grid."""
import argparse,csv,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--python',required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
r=a.root;repo=r/'repo';started=time.perf_counter()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==a.source_commit
record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),job_id=os.environ.get('SLURM_JOB_ID'),source_commit=a.source_commit,commands=[])
def save():(r/'training-pipeline.json').write_text(json.dumps(record,indent=2)+'\n')
def call(cmd,log,timeout):
 item=dict(command=cmd,log=log,status='running');record['commands'].append(item);save();t=time.perf_counter()
 with (r/log).open('x') as h:result=subprocess.run(cmd,cwd=repo,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,timeout=timeout)
 item.update(status='succeeded' if result.returncode==0 else 'failed',exit_code=result.returncode,elapsed_seconds=time.perf_counter()-t);save()
 if result.returncode:raise RuntimeError(log+' failed')
def command(name,small=False):
 return [a.python,'-B',str(repo/'scripts/revision_local_experiment.py'),'train','--out',str(r/name),'--parents',str(r/'parents'),'--design',str(r/'design.json'),'--source-commit',a.source_commit,'--train-device','cuda','--eval-device','cpu','--dev-seed','90803' if small else '90923','--eval-episodes','16' if small else '128','--pressure-episodes','16' if small else '128','--steps','10' if small else '1200','--max-seconds','240' if small else '2800']+(['--case-limit','2'] if small else [])
try:
 save();call(command('engineering-gpu',True),'engineering-gpu.log',300)
 m=json.loads((r/'engineering-gpu/metadata.json').read_text());cases=[]
 for case in m['cases']:
  with (r/'engineering-gpu/runs'/case['name']/'training.csv').open() as h:trace=list(csv.DictReader(h))
  step=float(np.median(np.diff([float(x['elapsed_seconds']) for x in trace])))
  estimate=1200*step+8*(case['initial_evaluation_seconds']+case['evaluation_seconds'])+3
  cases.append(dict(rule=case['revision_rule'],step_seconds=step,projected_case_seconds=estimate))
 projection=1.25*12*sum(x['projected_case_seconds'] for x in cases)
 record['runtime_preflight']=dict(cases=cases,projected_full24_seconds=projection,continuation_bound=2700,rule='12pairedconfigurations xsum2rulecosts x1.25padding; no efficacy consulted');save()
 if projection>2700:record.update(status='preflight_only',reason='Full24-case projection exceeds declaredruntimebound; no scoreselection')
 else:
  call(command('trained'),'trained.log',2900)
  call([a.python,'-B',str(r/'recheck_revision.py'),'--root',str(r),'--repo',str(repo),'--out',str(r/'gpu-recheck'),'--device','cuda'],'gpu-recheck.log',420)
  record['status']='succeeded'
except BaseException as e:record.update(status='failed',failure=repr(e));raise
finally:record.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started);save()
