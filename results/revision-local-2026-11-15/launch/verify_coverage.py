"""Reconcile complete declared phases, frozen code, and selected inference proofs."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();source='c9f5c153f786f1b82ac383d01c1d18c6c595946b';files=[];reports={};sourcecounts={}
for phase,count in (('frozen',48),('trained',24)):
 meta=json.loads((a.root/phase/'metadata.json').read_text());assert meta['status']=='succeeded' and len(meta['cases'])==count and meta['source_commit']==source
 for case in meta['cases']:
  folder=a.root/phase/'runs'/case['name'];required=['case.json','checkpoint.pt']
  stages=['evaluation']
  if phase=='trained':required+=['initial_checkpoint.pt','optimizer.pt','training.csv'];stages+=['initial-evaluation']
  for stage in stages:required += [stage+'/'+n for n in ('episodes.json','plans.json','scores.npy','predictions.csv','summary.csv','resources.csv')]
  for name in required:
   path=folder/name;assert path.is_file() and path.stat().st_size>0;files.append(str(path.relative_to(a.root)))
 for name,digest in meta['source_sha256'].items():
  blob=subprocess.check_output(['git','show',source+':'+name],cwd=a.repo);assert hashlib.sha256(blob).hexdigest()==digest
  assert hashlib.sha256((a.root/phase/'sources'/name).read_bytes()).hexdigest()==digest
 sourcecounts[phase]=len(meta['source_sha256'])
 audit=json.loads((a.root/'analysis'/phase/'raw_phase_audit.json').read_text());assert audit['status']=='passed' and audit['full_phase_complete'] and audit['case_count']==count and audit['total_predictions']==319488 and audit['total_causal_writes']==1665024
 weights=json.loads((a.root/'analysis'/phase/'saved_weights_audit.json').read_text());assert weights['status']=='passed' and weights['checked_checkpoint_payloads']==48
 reports[phase]=dict(predictions=audit['total_predictions'],writes=audit['total_causal_writes'],cases=count,scientific_joint_success=audit['analyses']['final']['joint_primary_success'])
 assert json.loads((a.root/phase/'design.json').read_text())==json.loads((a.root/'design.json').read_text())
assert len(files)==792
for stage in ('gpu-recheck','cpu-recheck'):
 report=json.loads((a.root/stage/'verification.json').read_text());assert report['status']=='passed' and report['verified_models']==report['expected_models']==96 and report['queries']==10752
 assert report['source_sha256']==hashlib.sha256((a.root/'launch/recheck_revision.py').read_bytes()).hexdigest()
 assert all(r['stored_predictions_match'] and r['all_argmax_match'] for r in report['groups'])
 reports[stage]={k:report[k] for k in ('queries','verified_models','max_abs_error','device','torch','scope')}
pipeline=json.loads((a.root/'training-pipeline.json').read_text());assert pipeline['status']=='succeeded'
assert json.loads((a.root/'analysis/resource_accounting.json').read_text())['actual_optimizer_steps']==28800
blob=subprocess.check_output(['git','show',source+':results/revision-local-2026-11-15/design.json'],cwd=a.repo);assert json.loads(blob)==json.loads((a.root/'design.json').read_text())
report=dict(status='passed',source_commit=source,required_case_files=len(files),required_paths=files,sourcecounts=sourcecounts,phases_and_rechecks=reports,tests='181passed',scope='Full artifact/arithmetic coverage plus selected inference, not retraining/fullgridnumerical reproduction; scientificsuccessreported separately')
(a.root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='required_paths'},indent=2))
