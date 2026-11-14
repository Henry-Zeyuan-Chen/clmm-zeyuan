# Descriptive attention-control behavior

This is a posthoc analysis of independently certified saved outputs. No training or inference was rerun. Both controls used the same original checkpoint, 16,000 K2-only training examples, and the same 64 development episodes per short cell.

For two-fact acquisition, attention32 scored 52.34% and attention128 scored 50.78%. The paired attention128-minus32 difference was -1.56 percentage points (95% episode-bootstrap interval -5.47 to +2.34). Both remain below the descriptive 80% target.

| Control | Query first fact | Query second fact | Legal-answer mass |
|---|---:|---:|---:|
| attention32 | 32/64 (50.00%) | 35/64 (54.69%) | 99.88% |
| attention128 | 31/64 (48.44%) | 34/64 (53.12%) | 99.75% |

Both fixed-first and fixed-last value heuristics achieve 51.56% over these balanced K2 queries; coincident values account for their slight advantage over 50%. Thus about 50% accuracy alone does not establish correct key–value binding.

| Control | Predict first value | Predict last value | Neither |
|---|---:|---:|---:|
| attention32 | 59/124 (47.58%) | 65/124 (52.42%) | 0/124 |
| attention128 | 59/124 (47.58%) | 65/124 (52.42%) | 0/124 |

The value-choice table excludes episodes whose two fact values coincide. These are descriptive alignments, not identification of an internal selection rule.

For attention32, among 29 episodes with different fact values and identical query wording, changing only the requested key changed the top prediction in 0 episodes; both queries were answered correctly in 0.

Across all 62 different-value episodes, predictions changed in 5; this wider set also permits query wording to change, so it does not isolate a requested-key change.

For attention128, among 29 episodes with different fact values and identical query wording, changing only the requested key changed the top prediction in 0 episodes; both queries were answered correctly in 0.

Across all 62 different-value episodes, predictions changed in 1; this wider set also permits query wording to change, so it does not isolate a requested-key change.

The zero-event subsets have degenerate empirical-bootstrap intervals. Their observed counts do not establish a zero event probability beyond this development sample.

| Control | Revision: current value | Revision: obsolete value | Neither value | One-fact accuracy |
|---|---:|---:|---:|---:|
| attention32 | 48.44% | 33/64 (51.56%) | 0/64 | 0.00% |
| attention128 | 48.44% | 33/64 (51.56%) | 0/64 | 100.00% |

The controls trained only on two distinct facts with no revisions. One-fact inputs also have a different length (63 rather than 95 bytes), while revisions reuse one key. Their results measure transfer beyond that training recipe. The observed behavior does not establish reliable arbitrary-key acquisition or revision handling, and does not rescue the main replay experiment.

All estimates and subsets are saved in CSV, including heldout templates, reset conditions, counterfactual assignments, query-template strata, and episode responses. The intervals condition on this fixed parent and development sample.
