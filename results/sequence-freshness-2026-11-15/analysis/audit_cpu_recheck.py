"""Audit coordinator CPU/GPU comparison artifacts; never executes a model."""
import collections,csv,json
from pathlib import Path
import numpy as np
from audit_packet import read_json,sha
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'cpu-recheck';plan=read_json(source/'plan.json');reported=read_json(source/'verification.json')
rows=read_json(source/'selected_rows.json');cpu=np.load(source/'cpu_logits.npy');gpu=np.load(source/'gpu_logits.npy')
assert cpu.shape==gpu.shape==(1200,256) and np.isfinite(cpu).all() and np.isfinite(gpu).all()
full={}
for directory in ['initial','runs']:
    for case in (ROOT/'gpu-output/result'/directory).iterdir():
        if case.is_dir():full[case.name]=np.load(case/'evaluation/answer_logits.npy')
for index,row in enumerate(rows):
    assert row['case'] in full and int(row['episode']) in (0,1)
    assert np.array_equal(gpu[index],full[row['case']][int(row['score_index'])])
    assert int(row['prediction'])==int(gpu[index].argmax())
assert collections.Counter(row['case'] for row in rows)=={name:100 for name in full}
def probability(logits):
    logits=logits.astype(np.float64);value=np.exp(logits-logits.max(axis=1,keepdims=True));return value/value.sum(axis=1,keepdims=True)
logit_error=np.abs(cpu.astype(np.float64)-gpu.astype(np.float64)).max(axis=1)
prob_error=np.abs(probability(cpu)-probability(gpu)).max(axis=1)
logit_fail=logit_error>plan['absolute_logit_tolerance'];prob_fail=prob_error>plan['maximum_probability_error'];failed=logit_fail|prob_fail
prediction_differences=int((cpu.argmax(axis=1)!=gpu.argmax(axis=1)).sum())
assert reported['status']=='numerical_review_required' and reported['queries']==len(rows) and reported['argmax_differences']==prediction_differences==0
assert np.isclose(logit_error.max(),reported['max_absolute_error'],rtol=0,atol=1e-12)
failures=[]
for i in np.flatnonzero(failed):
    row=rows[i]
    assert row['group'].startswith('revision') and row['condition']=='normal' and int(row['input_bytes'])==351
    failures.append(dict(case=row['case'],score_index=int(row['score_index']),group=row['group'],arm=row['arm'],
        absolute_logit_error=float(logit_error[i]),maximum_probability_error=float(prob_error[i]),
        logit_threshold_failed=bool(logit_fail[i]),probability_threshold_failed=bool(prob_fail[i])))
acquisition=[i for i,row in enumerate(rows) if row['group'].startswith('acquire')]
assert not failed[acquisition].any()
diagnostic=read_json(source/'batch-diagnostic/review.json')
assert {(r['case'],r['score_index']) for r in diagnostic['rows']}=={(r['case'],r['score_index']) for r in failures}
assert diagnostic['failures']==int(failed.sum()) and not diagnostic['exact_batch_all_within_original_tolerance']
assert all(row['exact_batch_original_subset_CPU_error']==0 for row in diagnostic['rows'])
assert diagnostic['all_predicted_bytes_equal'] and diagnostic['checkpoint_and_RNG_pure']
result=dict(status='audited_documented_failure',scope='saved numerical-comparison evidence only; no reviewer inference',
    models=len(full),selected_vectors=len(rows),absolute_logit_tolerance=plan['absolute_logit_tolerance'],maximum_probability_tolerance=plan['maximum_probability_error'],
    maximum_absolute_logit_error=float(logit_error.max()),maximum_probability_error=float(prob_error.max()),argmax_differences=prediction_differences,
    absolute_logit_failures=int(logit_fail.sum()),probability_failures=int(prob_fail.sum()),combined_failures=int(failed.sum()),
    affected_models=len({r['case'] for r in failures}),all_acquisition_rows_within_tolerance=True,
    exact_GPU_cohort_CPU_equals_prior_subset_CPU=True,
    maximum_CPU_single_versus_batch_error=max(row['exact_batch_single_CPU_error'] for row in diagnostic['rows']),
    interpretation='Original numerical acceptance remains unmet; selected-subset batching does not explain the observed CPU/GPU difference. Hardware and Torch version differ; cause not isolated.',
    failures=failures,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [source/'plan.json',source/'verification.json',source/'selected_rows.json',source/'cpu_logits.npy',source/'gpu_logits.npy',source/'batch-diagnostic/review.json']})
(ROOT/'analysis/cpu_recheck_artifact_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k not in ['failures','input_sha256']}))
