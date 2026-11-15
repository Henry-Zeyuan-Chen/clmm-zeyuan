# Audited fixed-set fitting and assignment transfer

Saved-artifact audit: **verified**. Source `5bccffd9084b543ca086721be2bd46e6a2ea27a6`. This recomputes persisted evidence without reproducing training or model inference.

All four fixed runs retain their original parent, data, optimizer budget and final step-2000 endpoint. Both queries are scored against all 256 byte logits. Earlier stages remain descriptive.

| Fixed run | Exact train both correct | Novel assignments both correct | Novel 95% interval | Heldout keys both correct | Heldout keys + wording both correct | Query-only accuracy |
|---|---:|---:|---:|---:|---:|---:|
| fixed32-68101 | 32/32 (100.00%) | 0/128 (0.00%) | [0.00, 0.00] | 0/128 (0.00%) | 1/128 (0.78%) | 25.00% |
| fixed128-68101 | 32/32 (100.00%) | 0/128 (0.00%) | [0.00, 0.00] | 1/128 (0.78%) | 1/128 (0.78%) | 25.00% |
| fixed32-68105 | 32/32 (100.00%) | 0/128 (0.00%) | [0.00, 0.00] | 2/128 (1.56%) | 1/128 (0.78%) | 25.00% |
| fixed128-68105 | 32/32 (100.00%) | 2/128 (1.56%) | [0.00, 3.91] | 2/128 (1.56%) | 0/128 (0.00%) | 23.44% |

The frozen primary trigger is `fixed128-68101/step-2000`: 32/32 exact-bank pairs and 0/128 novel-assignment pairs. It selects the **fresh** branch using the declared 31/32 fitting and 103/128 transfer counts.

The novel assignments use the same ordered key pairs while excluding both original letters. This is a deliberate value shift. The widened window exposes all 95 input bytes to attention and retains the neural-memory path. Joint heldout wording does not isolate fact wording from query wording.

## Exploratory fresh-training branch

Fresh training restarts both geometries from the original primary parent and changes the assignment distribution. Its reference bank is an evaluation population, not its exact fixed training set. The amended probe contains 128 distinct contexts absent from both complete fresh training histories and uses identical examples for all four endpoints.

| Training-disjoint final endpoint | Both correct | Cluster 95% interval | First-key accuracy | Second-key accuracy | Posthoc original-bank answer rate |
|---|---:|---:|---:|---:|---:|
| fixed32-68101 | 0/128 (0.00%) | [0.00, 0.00] | 0.78% | 5.47% | 86.33% |
| fixed128-68101 | 0/128 (0.00%) | [0.00, 0.00] | 22.66% | 18.75% | 28.12% |
| fresh32-68101 | 1/128 (0.78%) | [0.00, 2.34] | 42.19% | 56.25% | 0.00% |
| fresh128-68101 | 3/128 (2.34%) | [0.00, 5.47] | 50.78% | 49.22% | 0.00% |

Paired fresh-minus-fixed effects on these training-disjoint examples:
- fresh32-68101 minus fixed32-68101: +0.78 percentage points [0.00, 2.34].
- fresh128-68101 minus fixed128-68101: +2.34 percentage points [0.00, 5.47].

The original fresh-regime `novel_values` population is retained descriptively because exact training-context overlap can occur:
- fresh32-68101: 46/128 contexts seen in training; 82 unseen contexts remain.
- fresh128-68101: 46/128 contexts seen in training; 82 unseen contexts remain.

The original-bank answer rates are posthoc descriptions of incorrect old labels, since both old letters are absent from each new context. They do not establish an internal retrieval or memorization mechanism.

## Evidence limits

Bootstrap intervals use 5,000 draws with seed 90204. Novel values resample complete base-pair clusters and retain their assignment replicates; other populations use complete two-question episodes. Parent checkpoints are reported separately. Train-bank intervals are descriptive and zero-event bootstrap intervals do not imply zero population error.

A successful exact-bank fit demonstrates finite-set trainability under this recipe. Failed transfer establishes a limitation of the tested training procedure, not architecture impossibility. Exact-set fitting or same-key transfer alone does not qualify the project retention baseline; revision, causal-state/reset and session requirements remain unresolved.

The audit checks 192,000 supervised training answers, 18,240,000 training input bytes and 45,056 saved evaluation logit vectors across 6 runs. All original initial states, declared checkpoints/optimizer counters, source hashes, matching budgets, and required 128-window exact reset no-ops pass.
