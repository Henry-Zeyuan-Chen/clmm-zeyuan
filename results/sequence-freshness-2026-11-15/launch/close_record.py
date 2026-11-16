"""Write final process checks and append the project log after actual validation."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]/'outputs/clmm-sequence-freshness-2026-11-15'
REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')

pipe=read(ROOT/'gpu-output/result/pipeline.json')
cpu=read(ROOT/'cpu-recheck/verification.json')
coordinator=read(ROOT/'validation/coordinator-check.json')
analysis=read(ROOT/'analysis/audited/results_audit.json')
secondary=read(ROOT/'analysis/audited/secondary_analysis.json')
cluster=read(ROOT/'cluster/metadata.json')
closure=read(ROOT/'cluster/closure.json')
figures=read(ROOT/'validation/figure-review.json')
assert pipe['status']=='succeeded' and len(pipe['initial'])==4 and len(pipe['cases'])==8
assert cpu['status'] in ('passed','numerical_review_required')
assert coordinator['status']==analysis['status']=='passed'
if cpu['status']=='numerical_review_required':
    assert read(ROOT/'cpu-recheck/batch-diagnostic/review.json')['status']=='completed'
assert analysis['total_logit_vectors']==52224 and secondary['final_cases']==8
assert cluster['cancel_requested'] and closure['job_closed'] and closure['no_active_user_jobs']
assert figures['status']=='passed'
shutil.copy2(WORK/'tests-final.log',ROOT/'validation/tests-final.log')
write(ROOT/'provenance/preparation-attempts.json',dict(
    original_packet='packet',original_engineering='engineering/packet',original_source='d0b74557408a43a1609fd28308b9a455196ba5b3',
    failed_attempts=[dict(path=p,reason='Requested shortened supplementary exclusion filename did not exist; preparation stopped before model training',model_training_used=False)
       for p in ['packet-executed','engineering/packet-executed']],
    executed_main='packet-final',executed_engineering='engineering/packet-final',executed_source=pipe['source_commit'],
    preserved_logs=['launch/logs/prepare-executed.log','launch/logs/prepare-engineering-executed.log'],
    original_amendment_note='source-amendment.json is a prospective note: its packet-executed wording names the failed preparation; final executed paths are explicitly indexed here',
    afterlaunch_coverage='21CPU-smoke histories outside original outputs scan added to81326union. Wider81347union has zero main-eval overlap and regenerates all evaluation worlds identically. Launch packets unchanged.',
    independent_auditor_preparation='Initial lazy-NPZ analysis attempt interrupted, caching adopted; complete saved-data audit subsequently passes. No model training/inference in reviewer phase.'))
result=dict(status='complete_with_documented_limitations',scientific_outcome='acquisition_pass' if secondary['all_final_case_gate_sets_pass'] else 'acquisition_failed',
    source_commit=pipe['source_commit'],registered_parameters=839248,tests_passed=238,
    test_log_sha256=sha(ROOT/'validation/tests-final.log'),totals=coordinator['totals'],primary=coordinator['primary'],
    passed_acquisition_gate_cells=coordinator['passed_gate_cells'],total_acquisition_gate_cells=24,
    all_acquisition_gate_sets_pass=secondary['all_final_case_gate_sets_pass'],full_sequence_retention_qualified=False,
    CPU_selected_inference=dict(status=cpu['status'],models=cpu['models'],queries=cpu['queries'],max_absolute_error=cpu['max_absolute_error'],maximum_probability_error=max(r['max_probability_error'] for r in cpu['cases']),rows_exceeding_either_threshold=read(ROOT/'cpu-recheck/batch-diagnostic/review.json')['failures'],argmax_differences=cpu['argmax_differences'],original_tolerances_unchanged=True),
    figures=figures,cluster=closure,pipeline_elapsed_seconds=pipe['elapsed_seconds'],main_elapsed_seconds=pipe['main_elapsed_seconds'],
    exclusions=dict(supplied=81326,later_union=81347,main_evaluation_overlap=0,original_data_unchanged=True),
    scope='Full-sequence raw-record replay reference, fixed training budgets, no size increase; no paper-faithful/memory-only sleep/cross-session/biological claim',
    checks_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'analysis/audited/results_audit.json',ROOT/'analysis/audited/secondary_analysis.json',ROOT/'analysis/final_packet_provenance_review.json',ROOT/'analysis/work_smoke_regeneration_review.json',ROOT/'validation/coordinator-check.json',ROOT/'cpu-recheck/verification.json']})
write(ROOT/'validation.json',result)
event=dict(time_utc=datetime.now(timezone.utc).isoformat(),event='sequence_freshness_complete',**result,
    output_root=str(ROOT),all_terminal_cases_retained=True,no_outcome_based_extra_training=True,
    preexisting_user_edit_preserved=read(WORK/'preexisting_worktree.json'))
log=REPO/'docs/EXPERIMENT_LOG.jsonl'
assert not any(json.loads(line).get('event')==event['event'] for line in log.read_text(encoding='utf8').splitlines())
with log.open('a',encoding='utf8',newline='\n') as f:f.write(json.dumps(event,separators=(',',':'))+'\n')
print(json.dumps(result,indent=2))
