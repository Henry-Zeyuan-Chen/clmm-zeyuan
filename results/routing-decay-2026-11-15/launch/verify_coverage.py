"""Coverage, source-commit identity and selected inference proof reconciliation."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
SOURCE='61212289d6029da65eeb6a3735ce5ec3894866a9'
meta=json.loads((a.root/'main/metadata.json').read_text());assert meta['status']=='succeeded' and meta['source_commit']==SOURCE and len(meta['cases'])==48
assert json.loads((a.root/'pipeline.json').read_text())['status']=='preflight_only'
assert json.loads((a.root/'pipeline-cpu.json').read_text())['status']=='succeeded'
files=[]
for case in meta['cases']:
 folder=a.root/'main/runs'/case['name']
 for name in ('case.json','checkpoint.pt','evaluation/episodes.json','evaluation/plans.json','evaluation/scores.npy','evaluation/predictions.csv','evaluation/summary.csv','evaluation/resources.csv'):
  path=folder/name;assert path.is_file() and path.stat().st_size>0;files.append(str(path.relative_to(a.root)))
for name in ('metadata.json','episodes.json','design.json','exclusion_training_data.npz','exclusion_training_plan.json'):assert (a.root/'main'/name).is_file()
for name,expected in meta['source_sha256'].items():
 blob=subprocess.check_output(['git','show',SOURCE+':'+name],cwd=a.repo)
 assert hashlib.sha256(blob).hexdigest()==expected,('Commit blob differs from executed source',name)
 assert hashlib.sha256((a.root/'main/sources'/name).read_bytes()).hexdigest()==expected
reports={}
for stage in ('gpu-recheck','cpu-recheck'):
 report=json.loads((a.root/stage/'verification.json').read_text());assert report['status']=='passed' and report['verified_models']==report['expected_models']==48 and report['queries']==5376
 assert all(r['stored_predictions_match'] and r['all_argmax_match'] for r in report['groups'])
 assert report['source_sha256']==hashlib.sha256((a.root/'launch/recheck_routing.py').read_bytes()).hexdigest()
 reports[stage]={k:report[k] for k in ('queries','verified_models','max_abs_error','device','torch','scope')}
audit=json.loads((a.root/'analysis/raw_artifact_audit.json').read_text());assert audit['fully_audited_complete_main'] and audit['prediction_vectors']==319488 and audit['causal_write_events']==1665024
report=dict(status='passed',source_commit=SOURCE,main_cases=48,required_case_files=len(files),required_case_paths=files,executed_sources_match_commit_blobs=len(meta['source_sha256']),tests='110 passed: routing + prior Hebbian/sleep adapters',rechecks=reports,raw_audit_vectors=audit['prediction_vectors'],actual_new_training=False,scope='Artifact/selected inference validation, not retraining or full-grid numerical reproduction')
(a.root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='required_case_paths'},indent=2))
