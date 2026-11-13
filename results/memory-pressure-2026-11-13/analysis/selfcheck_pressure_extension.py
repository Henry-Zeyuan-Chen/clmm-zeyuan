"""Check amendment admission and original-two gating without model inference."""
import copy
import json
from pathlib import Path

import pressure_analysis as a

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
design_path=ROOT/'design.json'
design=json.loads(design_path.read_text())
before=copy.deepcopy(design)
amendment=a.load_training_amendment(ROOT/'training-amendment-2.json',design_path,design)
assert design==before
assert {row[0] for row in amendment['revised_training']['seeds']}=={48101,48103,48105}

original=json.loads((HERE/'original-two'/'primary_decision.json').read_text())
rows=[]
for record in original['checkpoints']:
    rows.extend([record['primary'],record['recent'],*record['online_anchors']])
assert a.continuation(rows,'grid')['both_checkpoint_signal'] is False
passing_first=[row for row in rows if row['init_seed']==48101]
third=copy.deepcopy(passing_first)
for row in third:
    row.update(run='supplementary48105',init_seed=48105,train_seed=48106)
assert a.continuation(passing_first+third,'grid')['both_checkpoint_signal'] is False
assert a.continuation(rows+third,'grid')['both_checkpoint_signal'] is False
assert a.continuation(rows+third,'grid')['supplementary_checkpoint_init_seeds']==[48105]
assert a.continuation(rows+third,'grid')['original_checkpoint_set_complete'] is True
positive_second=copy.deepcopy(passing_first)
for row in positive_second:
    row.update(run='mockOriginal48103',init_seed=48103,train_seed=48104)
assert a.continuation(passing_first+positive_second,'grid')['both_checkpoint_signal'] is True
assert a.continuation(passing_first+positive_second+third,'grid')['both_checkpoint_signal'] is True
assert a.continuation(passing_first+passing_first,'grid')['both_checkpoint_signal'] is False

for folder in ('grid-48101','grid-48103','original-two','grid-48101-longgap','grid-48103-longgap'):
    saved=HERE/folder
    manifest=json.loads((saved/'source_manifest.json').read_text())
    for name,digest in manifest.items():
        assert a.sha256(saved/name)==digest
result=dict(status='passed',base_design_unchanged=True,amendment_parent_validated=True,
    original_two_fail_retained=True,passing_third_cannot_replace_original=True,
    duplicate_seed_cannot_trigger=True,archived_sources_unchanged=True,
    analysis_script_sha256=a.sha256(HERE/'pressure_analysis.py'),
    amendment_sha256=a.sha256(ROOT/'training-amendment-2.json'))
out=HERE/'pressure_extension_selfcheck.json'
assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
