"""Deterministic estimator and protocol checks; no training or model inference."""
import copy
import json
from pathlib import Path

import numpy as np
import analyze_shared_lm as a

here=Path(__file__).resolve().parent
design=json.loads((here.parent/'design.json').read_text())
a.validate_design(design)
corrupt=copy.deepcopy(design)
corrupt['training']['initialization_seeds']=[68101,68107]
try:
    a.validate_design(corrupt)
except ValueError:
    pass
else:
    raise AssertionError('Unplanned seed accepted')

# The denominator must follow resampled whole windows, not an average of ratios.
num=np.array([1.,3.]); den=np.array([1,9]); index=np.array([[0,1],[0,0],[1,1]])
np.testing.assert_allclose(a.ratio_bootstrap(num,den,index),[.4,1.,1/3],rtol=0,atol=1e-15)
np.testing.assert_array_equal(a.ratio_bootstrap(num,den,index)-a.ratio_bootstrap(num,den,index),np.zeros(3))

# Explicit alignment yields a constant0.3-nat paired change despite window heterogeneity.
groups={}
for stage,conditions,lengths in [('initial',['normal'],[256]),('final',a.CONDITIONS,[256,1024])]:
    for length in lengths:
        for condition in conditions:
            cell={}
            for window in range(4):
                rate=2.+window*.2
                if stage=='initial': rate+=.3
                if condition=='reset_memory_at_segments': rate+=.05
                if condition=='zero_memory_output': rate+=.1
                late=max(0,length-256)
                cell[window]=dict(nll=rate*length,late_nll=rate*late,bytes=length,late_bytes=late)
            groups[stage,condition,length]=cell
meta=dict(configuration=dict(eval_windows=4,init_seed=68101),checkpoint_sha256='synthetic')
_,contrasts,decision,_=a.analyze(Path('synthetic'),meta,groups,False)
gain=next(row for row in contrasts if row['comparison']=='initial_minus_final')
np.testing.assert_allclose([gain['difference_nats_per_byte'],gain['ci_low_nats'],gain['ci_high_nats']],.3,atol=1e-14,rtol=0)
assert decision['baseline_accepted'] and decision['baseline_with_useful_context_accepted']
_,_,smoke_decision,_=a.analyze(Path('synthetic'),meta,groups,True)
assert not smoke_decision['pilot_decision_applicable'] and not smoke_decision['baseline_accepted']

result=dict(status='passed',unplanned_seed_rejected=True,ratio_of_sums_checked=True,
    paired_identity_and_constant_effect_checked=True,smoke_cannot_certify_pilot=True,
    source_sha256=a.sha256(here/'analyze_shared_lm.py'))
out=here/'selfcheck.json'
assert not out.exists()
a.save_json(out,result)
print(json.dumps(result,indent=2))
