"""Check the separately located, earlier CPU smoke packet against frozen main data."""
import json
from pathlib import Path
from audit_exclusions import structured_histories
from audit_packet import read_json,sha
ROOT=Path(__file__).resolve().parents[1]
smoke=ROOT.parent.parent/'work/clmm-sequence-freshness/smoke-packet'
if not smoke.exists():
    smoke=Path('C:/Users/henry/Documents/Codex/2026-10-22/i/work/clmm-sequence-freshness/smoke-packet')
packet=ROOT/'packet-final';declared=set(read_json(packet/'excluded_signatures.json'))
train=structured_histories(read_json(smoke/'training_plan.json.gz'))
evaluation=structured_histories(read_json(smoke/'episodes.json'))
main=structured_histories(read_json(packet/'episodes.json'))
combined=declared|train|evaluation
assert not main&combined
result=dict(status='passed',scope='earlier CPU-smoke semantic histories outside outputs root; no model execution',
    frozen_exclusions=len(declared),cpu_smoke_training_histories=len(train),cpu_smoke_evaluation_histories=len(evaluation),
    additional_histories=len(combined-declared),expanded_exclusions=len(combined),overlap_with_main_evaluation=len(main&combined),
    main_evaluation_histories=len(main),inputs={str(smoke/name):sha(smoke/name) for name in ['training_plan.json.gz','episodes.json']})
(ROOT/'analysis/work_smoke_exclusion_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
(ROOT/'analysis/work_smoke_expanded_signatures.json').write_text(json.dumps(sorted(combined),indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
