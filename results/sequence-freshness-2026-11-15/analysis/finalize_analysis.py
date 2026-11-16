"""Write final independent interpretation, figure evidence map and analysis inventory."""
from __future__ import annotations
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from audit_results import load_csv,write_csv
from audit_packet import read_json,sha
ROOT=Path(__file__).resolve().parent
summary=load_csv(ROOT/'audited/case_summary.csv');counter=load_csv(ROOT/'audited/counterfactual_metrics.csv')
final=[r for r in summary if r['mode']=='train' and r['arm']=='online' and r['condition']=='normal']
cases=sorted({(int(r['segment']),int(r['parent']),r['training_arm'],r['case']) for r in final})
lines=['| Parent | Segment | Training policy | Fresh train /128 | Dev seen /128 | Dev heldout /128 | Joint four /128 |',
       '|---|---:|---|---:|---:|---:|---:|']
figure_source=[]
for segment,parent,policy,case in cases:
    counts={g:int(next(r for r in final if r['case']==case and r['group']==g)['both_correct_count']) for g in ['acquire_train_seen','acquire_dev_seen','acquire_dev_heldout']}
    cf=int(next(r for r in counter if r['case']==case)['correct_count'])
    lines.append(f"| {parent} | {segment} | {'bounded' if policy.endswith('_bounded') else 'literal'} | {counts['acquire_train_seen']} | {counts['acquire_dev_seen']} | {counts['acquire_dev_heldout']} | {cf} |")
    for group in ['acquire_train_seen','acquire_train_swapped','acquire_dev_seen','acquire_dev_heldout']:
        r=next(r for r in final if r['case']==case and r['group']==group)
        figure_source.append(dict(case=case,parent=parent,segment=segment,training_policy=policy,group=group,correct_count=int(r['both_correct_count']),episodes=int(r['episodes']),accuracy=float(r['both_correct'])))
    figure_source.append(dict(case=case,parent=parent,segment=segment,training_policy=policy,group='joint_four_answer_correct',correct_count=cf,episodes=128,accuracy=cf/128))
write_csv(ROOT/'figures/acquisition_figure_source.csv',figure_source)
table='\n'.join(lines)
interpretation=f'''# Independent result interpretation

The bounded raw-record integration is operational, but none of the eight unchanged-size sequence models acquires a reliable reusable two-key rule under this fixed recipe. All 24 required acquisition gates fail. No sequence checkpoint is admitted to retention or sleep efficacy comparisons.

## Acquisition and counterfactual results

Each number below counts complete distinct-value episodes; each episode has two independently answered queries with identical wording apart from the requested key. The admission threshold is at least 103/128 both-correct (80%). Joint four requires both the original and value-swapped pair to be entirely correct.

{table}

On fresh training-key/seen-wording episodes, every generated answer lies in that context's two current values, but each model returns the identical byte for both requested keys in 124–128 of 128 episodes. Per-question accuracy is 49.61–50.39%, versus 3.91–6.64% for query-only controls. These saved predictions show access to context values without reliable key selection; they do not identify an internal storage mechanism or prove an always-first/last rule. The first-value rate varies by model. All eight joint-four counts are 0/128.

The two declared bounded-training minus literal-training co-primary both-correct effects, averaged across the four fixed parent/geometry cases, are:

- Fresh train keys / seen wording: 0.0000 percentage points, adjusted 97.5% paired interval [-0.9766, +0.8813].
- Dev keys / jointly heldout fact and query wording: 0.0000 points, adjusted 97.5% interval [-0.9766, +0.9766].

Intervals use 5,000 whole-episode PCG64 draws, seed 91304, after averaging fixed-case differences within each episode. They describe history uncertainty conditional on these checkpoints; they do not estimate training-procedure variability. A zero effect at an acquisition floor, or degenerate secondary intervals from all-zero observations, is not population equivalence. These replay-ineligible K2 evaluation prefixes are identical across replay policies: the primary contrasts measure effects induced during adaptation, not direct test-time freshness repair.

Both native32 and independently trained native128 models fail. Native128 acquisition fits inside direct attention, and all 6,144 acquisition normal/reset pairs of 256-logit vectors in the six native128 initial/final cases are exactly equal as expected. The wider models' failure prevents isolating a neural-memory-only bottleneck. Nonzero memory gradients at every one of 16,000 main optimizer steps verify gradient plumbing; they do not establish useful key binding.

## Revision diagnostics

Every terminal checkpoint has 0/64 both-correct revision pairs under all six normal arms on both seen and heldout wording. Bounded replay can increase partial per-question revised accuracy; the figure deliberately shows this partial metric while stating that every complete pair fails. The 240 nominal secondary contrast rows retain all training policies, arms, template groups and obsolete-answer rates. These are stale-conflict/revision diagnostics of an unqualified baseline, not successful retention, sleep efficacy, or full-model replication of the earlier typed-component result.

The full transformer consumes the raw replay text, including attention, embeddings, feedforward computation and native neural memory. The fixed preprocessing parser recognizes the four supplied fact grammars, while the learned model receives unchanged original wording. Template transfer is conditional on this provided parser. The controller retains at most twelve raw records/384 payload bytes at rest and may use thirteen/416 bytes in a transition lookup, apart from metadata and complete model input/activation costs. Queries and oracle targets never enter replay selection. Full prefixes are recomputed independently per answer; persistent deployed sessions are not tested.

## Integrity, selection and resources

Independent saved-artifact checks pass all 52,224 main 256-logit vectors, exact row/prefix/target binding, full-vocabulary predictions, legal-answer mass, NLL, paired summaries, resource-row coverage, source/parent identities and checkpoint/optimizer tensors. Prediction bytes agree exactly with independent raw-score recomputation; the largest continuous-score rounding difference is 1.11e-6. All 48 executable source snapshots match Git blobs at `95da59531ccf5d5f3fe2729553e6d980da598352`.

All four initial states exactly equal their original parents. The eight terminal states retain 839,248 registered elements and original tensor shapes. AdamW state covers 830,960 active elements, with the same eight unused original-memory-model/frequency tensors across all trained models. All terminal optimizer states are at step 2000. The eight recovery-only midpoint checkpoints and optimizers are finite at step 1000; no midpoint is selected or evaluated as an efficacy endpoint.

Main totals are 16,000 optimizer updates, 256,000 supervised answer forks, 29,169,664 training input bytes, and 16,000 nonzero memory-gradient updates. K1 deliberately duplicates its one query to retain two forks per context. Per-case curriculum is 256 K1, 1,564 K2 acquisition and 180 revision updates. Evaluation totals are 52,224 answers, 10,269,696 input bytes and 3,264 resource batches. Separately audited engineering runs contribute 1,088 score vectors and two twenty-step runs; they are excluded from the main effect estimate.

The runtime-only preflight projection is 2,821.6675 seconds against the 4,200-second gate. Main execution takes 1,542.8670 seconds; the complete remote pipeline takes 1,583.0394 seconds, below the 5,000-second pipeline cap. Scheduler release/accounting is owned and recorded separately by the coordinator; this reviewer performed no GPU, SSH or model inference work.

## Numerical reproduction limitation

The coordinator's prespecified CPU check covers 1,200 selected scores from all twelve main models. All 1,200 predicted bytes match saved GPU predictions, and checkpoint/RNG purity checks pass. The original numerical acceptance nevertheless fails: maximum absolute logit error is 0.0187702179 against 0.002, and maximum probability error is 0.0007565087 against 0.0002. Ten rows in two terminal models fail at least one threshold (six logit failures, six probability failures, with two overlapping). All are 351-byte normal revision probes; every selected acquisition row remains within tolerance.

The post-failure diagnostic uses the exact original GPU cohorts. CPU scores equal the prior subset-CPU scores exactly on the ten flagged rows, so selected-subset batching does not explain the discrepancy. Single-row versus batch CPU perturbation is at most 0.00043869. Hardware and Torch version differ (GPU2.11 versus CPU2.15); the cause is not isolated. No tolerance was relaxed. This is a documented numerical reproduction limitation, distinct from the passing saved-GPU integrity audit. The reviewer audited these arrays and comparison artifacts without executing a model.

## Exclusion history and future use

The original 42,416-signature collector omitted JSONL/raw-prefix histories. A pre-execution amendment used an independently audited 81,326-signature union; original and final model inputs, semantic worlds, samples, plans and parents remain identical. A separately located CPU smoke adds 21 signatures. Actual frozen-producer generation under the wider 81,347 union reproduces every fixed evaluation episode exactly, with zero overlap. Original packets, failed preparation attempts and all failures are preserved.

`future_exclusion_signatures.json` contains 85,123 signatures for subsequent studies: the later 81,347 base plus all now-consulted real and actually presented raw-record histories from main and engineering training/evaluation. It is explicitly future-only and never rewrites a launch packet or its union.

The supported next decision is to repair and qualify reusable key-conditioned acquisition before scaling or making retention claims. The result is specific to this representation, objective, curriculum and 2,000-step budget. It neither proves that larger models are necessary nor that this architecture cannot learn the task.
'''
(ROOT/'result_interpretation.md').write_text(interpretation,encoding='utf8')
claims=[
 dict(claim_id='integration',claim='Causal exact-byte bounded replay plumbing at unchanged model size',status='measured_engineering',evidence='wrapper_review.json;final_packet_review.json;final_saved_execution_review.json;final_run_provenance.json',limit='Full-transformer reference, known grammar parser, no deployed persistent-session test'),
 dict(claim_id='acquisition',claim='All eight final models qualify on fresh train/dev/heldout acquisition',status='contradicted_under_fixed_recipe',evidence='audited/acquisition_gates.csv;audited/case_summary.csv',limit='0/24 gates; each BOTH-correct0–2/128; no architecture impossibility claim'),
 dict(claim_id='primary_benefit',claim='Bounded training improves declared fresh-assignment/template both-correct endpoints',status='unsupported_floor_result',evidence='audited/primary_estimates.csv',limit='Both point effects0pp; conditional97.5% intervals crosszero; no population equivalence'),
 dict(claim_id='counterfactual',claim='Reliable key-specific retrieval under paired assignment swaps',status='contradicted_under_fixed_recipe',evidence='audited/counterfactual_metrics.csv;audited/query_behavior.csv',limit='0/128joint4 for all8; changing predictions alone insufficient'),
 dict(claim_id='revision',claim='Complete revised two-key acquisition under six normal replay arms',status='contradicted_under_fixed_recipe',evidence='audited/case_summary.csv;audited/revision_contrasts.csv',limit='0/64both-correct every finalcase/arm/family; partial accuracy remains diagnostic'),
 dict(claim_id='saved_integrity',claim='Recorded GPU outputs and budgets reconstruct from raw artifacts',status='measured_audited',evidence='audited/results_audit.json;final_run_provenance.json;resources_and_midpoints_review.json',limit='Saved-artifact audit, not reviewer model inference or retraining'),
 dict(claim_id='cpu_numerics',claim='Selected CPU inference meets original full numerical tolerances',status='failed_documented',evidence='cpu_recheck_artifact_review.json;../cpu-recheck/verification.json;../cpu-recheck/batch-diagnostic/review.json',limit='10combined failures;0/1200argmaxdifferences; cause notisolated; no tolerance relaxation'),
 dict(claim_id='scale',claim='Larger capacity or sleep efficacy is established by this experiment',status='unsupported',evidence='audited/acquisition_gates.csv;result_interpretation.md',limit='No larger model trained; prerequisite acquisition fails')]
write_csv(ROOT/'claim_evidence_map.csv',claims)
figures=['acquisition_counts','primary_paired_effects','revision_diagnostics']
figure_review=dict(status='passed',scope='Current PNGs visually inspected; PDF/SVG exported from same Matplotlib figures',
    observations={'acquisition_counts':'All8cases and exact denominators legible; joint4 separately identified; full-scale colorbar preserves acquisition floor.',
                  'primary_paired_effects':'Both co-primary fixed-case intervals and zero effects legible; wrapped axis caption and97.5% bootstrap scope fully visible.',
                  'revision_diagnostics':'Per-question metric explicit; all0/64pair failures stated; both trained geometries/template families/all6arms retained; nominal fixed-parent intervals labeled.'},
    figures={f'{name}.{ext}':sha(ROOT/'figures'/f'{name}.{ext}') for name in figures for ext in ['png','pdf','svg']},
    source_tables={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'figures/acquisition_figure_source.csv',ROOT/'audited/primary_estimates.csv',ROOT/'figures/revision_figure_source.csv']})
(ROOT/'figure_review.json').write_text(json.dumps(figure_review,indent=2)+'\n',encoding='utf8')
readme='''# Independent sequence replay evidence review

All eight 839,248-parameter terminal models fail reusable two-key acquisition: 0/24 required gates, 0/128 joint-four swap success per model, and 0/64 complete revision-pair success under every normal arm. Both co-primary bounded-minus-literal effects are zero with intervals crossing zero. Integration and saved-artifact integrity pass; no retention/sleep efficacy admission follows.

Read [result_interpretation.md](result_interpretation.md) for the complete negative conclusion and limitations. The coordinator's CPU check preserves all 1,200 sampled predicted bytes but fails its original numerical tolerance in ten rows; this limitation is independently verified in [cpu_recheck_artifact_review.json](cpu_recheck_artifact_review.json).

## Final artifacts

- `audited/results_audit.json`: all 52,224 main score vectors, exact raw-prefix/label binding, independent metrics, paired fixed-case primary estimates and budgets.
- `audited/acquisition_gates.csv`, `counterfactual_metrics.csv`, `query_behavior.csv`, `memory_dependence.csv`, `revision_contrasts.csv`: all per-case and secondary development diagnostics.
- `final_run_provenance.json`: all twelve main checkpoints, terminal optimizers, parent identity, tensor shape/count/finite checks and 48 exact source Git blobs.
- `resources_and_midpoints_review.json`: eight recovery-only midpoint checks, 16,000 main updates/256,000 forks/29,169,664 training bytes, evaluation/resource coverage and runtime gate.
- `engineering_audited/` and `engineering_run_provenance.json`: separately scoped twenty-step engineering evidence, never pooled into main effects.
- `figures/`: three stable PNG/PDF/SVG numerical figures; source tables are preserved. `figure_review.json` records visual QA and hashes.
- `claim_evidence_map.csv`: exact supported and unsupported claims.
- `future_exclusion_signatures.json` and `future_exclusion_manifest.json`: future-only85,123-history union, including actually presented contexts; launch packets remain immutable.
- `final_analysis_inventory.json` and `.sha256`: final artifact inventory and evidence counts.

## Source, data and exclusion provenance

Executed source is `95da59531ccf5d5f3fe2729553e6d980da598352`. `packet-final` and `engineering/packet-final` preserve exact source snapshots. `source_amendment_review.json` proves original/main-final training tensors, semantic worlds, plans, exact samples and parents unchanged after the performance/exclusion amendment. `final_packet_review.json`, `final_saved_execution_review.json`, `final_packet_provenance_review.json`, `final_engineering_packet_review.json` and `final_exclusion_review.json` preserve prelaunch audits.

The original structural collector's JSONL/raw-prefix omission is documented. The pre-execution81,326 union is immutable. Work-directory CPU smoke adds21 signatures; `work_smoke_regeneration_review.json` proves actual frozen-producer regeneration under81,347 produces exactly identical fixed evaluation episodes. The original CPU smoke packet/run are archived at `../engineering-cpu`, with66 file hashes. The failed `packet-executed` preparations remain preserved; their missing-filename error is not a model outcome.

`initial_protocol_review.md`, `wrapper_review.json` and helper/selfcheck records preserve the prospective review. Earlier `smoke_*` artifacts concern the separate eight-step CPU engineering run. They are not main efficacy evidence.

## Reproduction

Use Python with NumPy, Matplotlib for plotting, and PyTorch only for CPU tensor inspection. Main saved-data/statistics audits import no model code. The bridge review imports policy modules only; synthetic selfchecks use producer data functions to create fixtures.

```text
python audit_packet.py --training-worlds ../packet-final/training_plan.json.gz --training-arrays ../packet-final/training.npz --evaluation-worlds ../packet-final/episodes.json --exclusions ../packet-final/excluded_signatures.json
python audit_saved_plans.py --packet ../packet-final
python audit_results.py --packet ../packet-final --case <four-initial-and-eight-terminal-case-folders> --out audited --require-complete
python audit_provenance.py --packet ../packet-final --repo <frozen-LF-source-checkout> --case <same-case-folders> --out final_run_provenance.json
python secondary_analysis.py --audited audited
python analyze_behavior.py
python plot_results.py --audited audited --out figures
python audit_resources.py
python audit_cpu_recheck.py
python build_future_exclusions.py
python finalize_analysis.py
```

Whole-episode paired bootstraps use seed91304 and5,000 draws. Co-primary intervals are97.5%; secondary intervals are nominal95%. These condition on fixed checkpoints and cannot measure general training variability. The reviewer performed no model construction, training, inference, GPU or SSH work during this review. CPU checkpoint/optimizer tensors were inspected without building a model. Scheduler closure and the actual CPU inference were performed separately by the coordinator.
'''
(ROOT/'README.md').write_text(readme,encoding='utf8')
exclude={'final_analysis_inventory.json','final_analysis_inventory.sha256'}
files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in exclude]
records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
inventory=dict(status='complete_with_documented_limitations',created_utc=datetime.now(timezone.utc).isoformat(),scope='Independent analysis only',
    counts=dict(analysis_files=len(records),analysis_bytes=sum(r['bytes'] for r in records),main_checkpoints=12,terminal_optimizers=8,recovery_midpoints=8,
                main_score_vectors=52224,engineering_score_vectors=1088,main_optimizer_updates=16000,main_answer_forks=256000,main_training_input_bytes=29169664,
                memory_nonzero_gradient_steps=16000,main_evaluation_resource_batches=3264,acquisition_gates=24,acquisition_gates_passed=0,joint_four_success_per_model=0,
                revision_both_correct_per_case_arm_family=0,source_git_blobs=48,figures=3,figure_exports=9,cpu_selected_vectors=1200,cpu_combined_numerical_failures=10,cpu_argmax_differences=0,
                future_exclusion_histories=85123),
    omissions='Operational __pycache__ files and this self-manifest/hash are omitted from file list.',
    limitations=['No admitted acquisition or retention baseline','Original coordinator CPU numerical tolerance unmet in10selected rows','Fixed-model history intervals only','No reviewer model execution; coordinator owns scheduler accounting'],files=records)
path=ROOT/'final_analysis_inventory.json';path.write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf8')
(ROOT/'final_analysis_inventory.sha256').write_text(sha(path)+'  final_analysis_inventory.json\n',encoding='ascii')
print(json.dumps(dict(status=inventory['status'],files=len(records),bytes=inventory['counts']['analysis_bytes'],inventory_sha256=sha(path))))

