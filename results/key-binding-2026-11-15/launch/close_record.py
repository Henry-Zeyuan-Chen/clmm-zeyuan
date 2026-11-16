"""Bind actual checked results, usable-model scopes, and final project log."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-key-binding-2026-11-15'
REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
pipe=read(ROOT/'gpu-main/result/pipeline.json');assert pipe['status']=='succeeded'
coordinator=read(ROOT/'validation/coordinator-check.json');assert coordinator['status']=='passed'
audit=read(ROOT/'analysis/main_output_audit.json');tensors=read(ROOT/'analysis/main_tensor_audit.json')
assert audit['status']==tensors['status']=='passed' and audit['vectors']==81920
assert tensors['terminal_checkpoints']==tensors['initial_checkpoints']==4
for ours in coordinator['primary']:
    other=next(r for r in audit['co_primary'] if 'acquire_'+r['endpoint']==ours['group'])
    assert abs(ours['estimate']-other['atomic_minus_digits'])<1e-12
    assert abs(ours['lower']-other['ci97_5'][0])<1e-12 and abs(ours['upper']-other['ci97_5'][1])<1e-12
assert audit['admission_cells_pass']==coordinator['passed_gate_cells']
cpu=read(ROOT/'cpu-recheck/verification.json');assert cpu['status'] in ('passed','numerical_review_required') and cpu['queries']==2560
plan=read(ROOT/'validation/cpu-proof-plan.json');assert plan['script_sha256']==cpu['script_sha256']
closure=read(ROOT/'cluster-closure.json');assert closure['our_allocations_closed']
figures=read(ROOT/'validation/figure-review.json');assert figures['status']=='passed'
amendment=read(ROOT/'execution-amendment.json')
assert sha(ROOT/'packet/metadata.json')==amendment['main_packet_metadata_sha256']
assert sha(ROOT/'execution-amendment.json')==pipe['execution_amendment_sha256']
shutil.copy2(WORK/'tests.log',ROOT/'validation/tests.log')
models=[]
for folder in sorted((ROOT/'gpu-main/result/runs').iterdir()):
    meta=read(folder/'metadata.json')
    gates=[r for r in coordinator['gates'] if (r['parent'],r['encoding'])==(meta['parent_seed'],meta['encoding'])]
    assert len(gates)==6
    config=dict(meta['parent']['config']);config['segment_len']=128
    models.append(dict(name=folder.name,path=str((folder/'checkpoint.pt').relative_to(ROOT)),encoding=meta['encoding'],
        checkpoint_sha256=sha(folder/'checkpoint.pt'),state_sha256=meta['final_state_sha256'],source_commit=meta['source_commit'],
        config=config,codec_path='packet/key_codec.json',codec_sha256=sha(ROOT/'packet/key_codec.json'),
        registered_parameters=839248,completed_updates=2000,selection='Fixed terminal, no checkpoint selection',
        acquisition_gates=gates,all_six_acquisition_gates_pass=all(g['passed'] for g in gates),retention_qualified=False,
        scope='Controlled key encoding only; no native32, delayed, cross-session, replay or biological qualification'))
write(ROOT/'models.json',dict(models=models,static_codec_serialized_bytes=(ROOT/'packet/key_codec.json').stat().st_size,codec_is_external_preprocessing=True))
result=dict(status='complete' if cpu['status']=='passed' else 'complete_with_documented_limitations',source_commit=pipe['source_commit'],
    registered_parameters=839248,tests_passed=109,test_log_sha256=sha(ROOT/'validation/tests.log'),totals=coordinator['totals'],
    primary=coordinator['primary'],primary_matches_independent_audit=True,acquisition_gate_cells_pass=coordinator['passed_gate_cells'],
    acquisition_gate_cells_total=24,all_six_gate_models_pass=sum(m['all_six_acquisition_gates_pass'] for m in models),
    full_sequence_retention_qualified=False,CPU_selected=dict(status=cpu['status'],queries=cpu['queries'],models=cpu['models'],max_logit_error=cpu['max_logit_error'],
        max_probability_error=cpu['max_probability_error'],prediction_mismatches=cpu['prediction_mismatches'],tolerances_unchanged=True),
    cluster=closure,pipeline_elapsed_seconds=pipe['elapsed_seconds'],main_elapsed_seconds=pipe['main_elapsed_seconds'],
    figures=figures,model_registry_sha256=sha(ROOT/'models.json'),execution_amendment_sha256=sha(ROOT/'execution-amendment.json'),
    scope='Fixed-size encoding bundle; independent same-budget models. No tokenizer mechanism, retention/sleep/Hebbian efficacy claim')
write(ROOT/'validation.json',result)
event=dict(time_utc=datetime.now(timezone.utc).isoformat(),event='key_binding_complete',**result,all_terminal_conditions_retained=True,no_outcome_based_extra_training=True,
           output_root=str(ROOT),preserved_preexisting_user_edit=read(WORK/'preexisting_worktree.json'))
log=REPO/'docs/EXPERIMENT_LOG.jsonl'
assert not any(json.loads(s).get('event')==event['event'] for s in log.read_text(encoding='utf8').splitlines())
with log.open('a',encoding='utf8',newline='\n') as f:f.write(json.dumps(event,separators=(',',':'))+'\n')
print(json.dumps(result,indent=2))
