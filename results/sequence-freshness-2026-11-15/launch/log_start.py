import json
from datetime import datetime,timezone
from pathlib import Path
log=Path(r'C:\Users\henry\Documents\Codex\clmm-replay\docs\EXPERIMENT_LOG.jsonl')
event=dict(time_utc=datetime.now(timezone.utc).isoformat(),event='sequence_freshness_started',
    branch='sequence-bounded-freshness',source_commit='95da59531ccf5d5f3fe2729553e6d980da598352',
    registered_parameters=839248,terminal_models=8,initial_models=4,steps_per_terminal=2000,
    answers_per_terminal=32000,input_bytes_per_terminal=3646208,parents=[68101,68105],segments=[32,128],
    training_arms=['periodic','periodic_bounded'],tests_passed=238,
    cluster_job=13922356,cluster_node='ins080',allocated_GPUs=2,allocation_minutes=90,
    pipeline_limit_seconds=5000,padded_projection_seconds=2821.6674887540285,runtime_gate_passed=True,
    no_renewal=True,helper_selection_changed=False,training_seed=91302,evaluation_seed=91323,
    evaluation_worlds=128,revision_worlds=64,source_files=48,
    exclusions=dict(original=42416,supplied=81326,later_work_smoke_union=81347,evaluation_overlap=0,
       original_worlds_and_arrays_unchanged=True),
    admission='Each terminal model >=80%both-correct train-seen/dev-seen/dev-heldout; memory dependence separate',
    scope='Full raw-sequence replay reference; no size increase or paper-faithful MAC/sleep claim',
    output_root='C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-sequence-freshness-2026-11-15')
assert not any(json.loads(s).get('event')==event['event'] for s in log.read_text(encoding='utf8').splitlines())
with log.open('a',encoding='utf8',newline='\n') as f:f.write(json.dumps(event,separators=(',',':'))+'\n')
