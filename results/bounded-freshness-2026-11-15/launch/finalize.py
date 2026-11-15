"""Bind completed evidence, checks and the preserved user edit into the log."""
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-bounded-freshness-2026-11-15';REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pipeline=read(ROOT/'pipeline.json');analysis=read(ROOT/'analysis/summary.json');check=read(ROOT/'validation/coordinator_check.json');cpu=read(ROOT/'cpu-recheck/verification.json');prior=read(WORK/'preexisting_worktree.json')
assert pipeline['status']=='succeeded' and len(pipeline['main'])==12 and pipeline['elapsed_seconds']<4200
assert analysis['status']=='passed_with_documented_launch_coverage_omission'
assert check['status']==cpu['status']=='passed'
assert cpu['queries']==9216 and cpu['max_abs_error']==0 and cpu['models']==12
assert sha(REPO/prior['path'])==prior['sha256']
supplement=read(ROOT/'analysis/supplemental_exclusion_audit.json')
assert supplement['omitted_prior_unique_histories']==1 and supplement['main_overlap_with_extended_prior_union']==0
assert read(ROOT/'validation/supplemental_crosscheck.json')['status']=='passed'
with (ROOT/'analysis/gates.csv').open() as f:gates=list(csv.DictReader(f))
gate_summary={}
for arm in sorted({r['arm'] for r in gates}):
 rows=[r for r in gates if r['arm']==arm];values=[float(r['own_D0_six_target_all_correct']) for r in rows]
 gate_summary[arm]=dict(passed=sum(r['own_full_retention_gate']=='True' for r in rows),total=len(rows),minimum_all_six_correct=min(values),maximum_all_six_correct=max(values))
result=dict(status='passed',scope='Fixed-checkpoint component freshness comparison with CPU execution, independent full raw/source/data audit and selected actual CPU resume/reproduction',source_commit='9ab3901860548d2d87a9afc61e8ce58f950a03ce',baseline_tag='35a549f137548d48aecc26e33c6f8a34a6efc97c',models=12,tests_passed=417,test_log_sha256=sha(WORK/'tests.log'),audited_totals=analysis['audited_totals'],primary=check['primary'],common_gates_all_pass=analysis['all_common_checkpoint_gates_pass'],own_D0_gates=gate_summary,cpu_selected_proof=dict(models=12,queries=9216,max_abs_error=0,script_sha256=cpu['source_sha256']),new_GPU_allocations=0,GPU_numerical_proof='not run; main small-matrix evaluation and verification use CPU',analysis_inventory_sha256=sha(ROOT/'analysis/final_artifact_inventory.json'),figures_reviewed=len(read(ROOT/'analysis/figures/manifest.json')['figures']),main_elapsed_seconds=pipeline['main_elapsed_seconds'],pipeline_elapsed_seconds=pipeline['elapsed_seconds'],new_optimizer_steps=0,all_our_inference_stopped=True,preexisting_user_edit_preserved=prior,model_size_decision='Retain81921parametercomponent now; qualify held-out acquisition in existing839248parameter sequence model before increasing capacity')
result['status']=analysis['status']
result['postlaunch_exclusion_coverage']=dict(launch_signatures=89093,extended_signatures=89094,older_unique_history_omitted_by_filename_collector=1,observed_overlap=0,independent_and_producer_regeneration_unchanged=True,launch_packets_modified=False)
(ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
event=dict(time_utc=datetime.now(timezone.utc).isoformat(),event='bounded_freshness_complete',**result,output_root=ROOT.as_posix(),all_conditions_retained=True,full_sequence_qualified=False)
with (REPO/'docs/EXPERIMENT_LOG.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(event,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('primary','own_D0_gates')},indent=2))
