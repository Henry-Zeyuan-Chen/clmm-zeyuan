import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-sleep-multirevision-2026-11-15';REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pipeline=read(ROOT/'pipeline.json');analysis=read(ROOT/'analysis/summary.json');check=read(ROOT/'validation/coordinator_check.json');cpu=read(ROOT/'cpu-recheck/verification.json');cluster=read(ROOT/'cluster/metadata.json')
assert pipeline['status']=='succeeded' and len(pipeline['main'])==12 and pipeline['elapsed_seconds']<3300
assert analysis['status']==check['status']==cpu['status']=='passed'
assert cpu['queries']==5184 and cpu['max_abs_error']==0 and cpu['models']==12
assert cluster['status']=='failed' and cluster['cancel_requested'] and cluster['job_id']=='13921270'
scheduler=(ROOT/'cluster/scheduler-confirmed.txt').read_text(encoding='utf8')
assert '13921270|zc2819|CANCELLED' in scheduler
assert not any(line.startswith('13921270 ') for line in scheduler.splitlines())
closeout=dict(status='closed',job_id='13921270',allocation_state='CANCELLED',allocation_elapsed_seconds=456,allocated_GPU_hours=456/3600,selected_gpu_proof='unverified: timeout with kernel I/O wait observed, no scores',no_active_user_jobs=True,helper_selection_changed=False,no_renewal=True,scheduler_sha256=sha(ROOT/'cluster/scheduler-confirmed.txt'))
(ROOT/'cluster/closeout.json').write_text(json.dumps(closeout,indent=2)+'\n',encoding='utf8')
result=dict(status='passed',scope='Completed CPU comparison, full saved-output/data/source audit and selected actual CPU reproduction; GPU numerical proof is unverified',source_commit='06c0a35b543c25437659765128b81718e5bfa1c5',baseline_tag='35a549f137548d48aecc26e33c6f8a34a6efc97c',models=12,tests_passed=359,test_log_sha256=sha(WORK/'tests.log'),audited_totals=analysis['audited_totals'],primary=check['primary'],common_gates_all_pass=analysis['all_common_checkpoint_gates_pass'],all_CP_own_D0_gates_pass=analysis['all_continuous_periodic_retention_gates_pass'],cpu_selected_proof=dict(models=12,queries=5184,max_abs_error=0,script_sha256=cpu['source_sha256']),gpu=closeout,analysis_inventory_sha256=sha(ROOT/'analysis/final_artifact_inventory.json'),figures_reviewed=5,main_elapsed_seconds=pipeline['main_elapsed_seconds'],pipeline_elapsed_seconds=pipeline['elapsed_seconds'],new_optimizer_steps=0,all_inference_stopped=True)
(ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
event=dict(time_utc=datetime.now(timezone.utc).isoformat(),event='sleep_multirevision_complete',**result,output_root=ROOT.as_posix(),all_conditions_retained=True,full_sequence_qualified=False,team_choice='Literal versus bounded version-aware refresh; any applied-content change needs a fresh split and matched write/storage controls')
with (REPO/'docs/EXPERIMENT_LOG.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(event,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('primary','gpu')},indent=2))
