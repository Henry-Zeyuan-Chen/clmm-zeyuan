# Fixed-set fitting and assignment transfer

This experiment supports [Zeyuan’s shared training/evaluation role](ZEYUAN_STATUS.md). It follows the failed two-key acquisition tests and asks whether the complete model can fit a finite set of associations, then use the requested key under novel assignments. It is a prerequisite diagnostic for the sleep and selective Hebbian comparisons, not a new project contribution claim.

## Fixed protocol

Source `5bccffd9084b543ca086721be2bd46e6a2ea27a6` adds the data factory, runner and tests. A fixed bank contains 32 two-fact contexts over 16 training keys, with each key appearing four times and assigned at least two different values. Ordered key pairs are unique; every answer letter a–p appears four times, and the two values within a context differ. The existing byte serialization and canonical seen templates are used. Both questions are independently supervised without exposing the other's answer.

For each original language parent (68101 and 68105), compare attention segment 32 with 128 while preserving every initial learned tensor, parameter count, training array and optimizer budget. Both models retain neural memory. The wider window allows ordinary attention to see all 95 input bytes; it is not a separately trained memory-free baseline.

Training cycles through shuffled epochs of the same 32 contexts, eight contexts/16 answers per update, for exactly 2,000 updates. AdamW learning rate is 0.0003, weight decay 0.01, clipping 1. Bank/training/evaluation seeds are 90201/90202/90203. Save and evaluate fixed checkpoints at 0/200/1000/2000; the final step is primary. All data, raw logits, plans, tensor initialization and optimizer states are retained.

Evaluate both questions, ordinary and reset-memory conditions, on:

- **Exact fitting bank:** the 32 supervised contexts.
- **Novel values on familiar pairs:** four assignments per original ordered pair, excluding both original letters. This deliberate value shift tests transfer beyond each pair's supervised labels, rather than generic IID generalization.
- **Heldout keys:** 128 fresh dev-key associations with canonical wording.
- **Heldout-key templates:** identical keys/values/seeds with heldout fact and query wording, measuring template transfer conditional on unseen keys.
- **Query-only bank control:** the original query without fact context. This shorter-input control checks whether successful predictions are possible from the query alone.

Report full-256 per-query and both-correct accuracy, query rank, prediction changes, NLL and legal answer mass. For novel assignments, uncertainty resamples 32 base pairs with all four variants retained; heldout-key intervals use complete episodes. Use 5,000 draws, seed 90204, separately per fixed checkpoint. No reserved test keys are evaluated.

## Declared continuation

Only parent 68101 with window 128 at final step 2000 selects the follow-up. Fitting requires at least 31/32 both-correct; adequate novel-assignment transfer requires at least 103/128.

- If fitting succeeds but novel-assignment transfer fails, run two additional 2,000-step models from the original primary parent, windows 32/128, with fresh random assignments every update over the same 16 training keys. This exploratory comparison changes the training distribution, not the architecture or update rule. Its `reference_bank` evaluation is not presented as its literal training set.
- If fitting fails, run one smaller 8-context/4-key fitting check, with each key still assigned several values, for 1,600 steps at window 128. Failure remains budget-specific; success is a fitting sanity check.
- Otherwise, record the demonstrated transfer and remaining project requirements without another training search.

All work is limited to one one-hour GPU allocation, no renewal and at most two follow-up runs. Final model selection is fixed-step throughout. No checkpoint is admitted as a retention baseline on fitting alone: revisions, unseen-key/wording scope and persistent-state behavior remain explicit gates.

## Results and handoff

All four fixed-bank runs and the two declared fresh-assignment follow-ups completed. Every run received exactly 2,000 updates, 32,000 supervised answers and 3,040,000 input bytes. All 24 checkpoints, including step zero, remain available. The independent audit verifies all source/parent identities, initialization tensors, optimizer steps, training arrays, paired evaluations and reported scores.

| Fixed-bank model | Exact fitting bank, both correct | Novel values, both correct | Heldout keys, both correct |
|---|---:|---:|---:|
| Window 32, parent 68101 | 32/32 | 0/128 | 0/128 |
| Window 128, parent 68101 | 32/32 | 0/128 | 1/128 |
| Window 32, parent 68105 | 32/32 | 0/128 | 2/128 |
| Window 128, parent 68105 | 32/32 | 2/128 | 2/128 |

The finite fitting problem is solvable under this budget in both geometries and both fixed parents. It does not establish a reusable rule for new assignments. On the primary 32-window model, 219/256 novel-value questions (85.55%) are answered with that pair/key's old supervised target, even though both old letters are excluded from the new context. The corresponding old-answer rates are 33.20%, 57.81% and 27.34% for primary128, sensitivity32 and sensitivity128. This post-hoc behavior is consistent with reliance on old assignments; it does not identify an internal storage mechanism. Query-only performance is far below full-bank fitting, and all original per-query/rank/ablation outcomes remain in the tables.

### Result-directed fresh-assignment training

The declared selector, primary parent/window128 at step2000, fits 32/32 but transfers to 0/128 novel assignments, so it selects the **fresh** branch. Both follow-ups restart from the original primary language checkpoint and receive the same 2,000-step budget with new random values and key pairs over the 16-key vocabulary every update. They are a data-distribution diagnostic, not a sleep/Hebbian method comparison.

Independent review identified a finite-universe overlap risk before the branch ran. A prospective local amendment adds a seed-90205 probe that rejects every exact fact context present in either fresh run's training history. All four primary-parent endpoints receive the identical accepted set of four variants per ordered pair. Selection is independent of model predictions. The original probe is preserved: 46/128 of its contexts occur in fresh training, so it cannot be presented as wholly unseen assignments for that regime. The extra probe has zero such overlap; the independent auditor reconstructs every acceptance/rejection and all raw scores.

| Primary-parent model | Disjoint novel assignments, both correct | Per-query accuracy | Key-switch prediction changes |
|---|---:|---:|---:|
| Fixed-bank training, window32 | 0/128 | 3.13% | 127/128 |
| Fixed-bank training, window128 | 0/128 | 20.70% | 95/128 |
| Fresh-assignment training, window32 | 1/128, 0.78% | 49.22% | 4/128 |
| Fresh-assignment training, window128 | 3/128, 2.34% | 50.00% | 6/128 |

Fresh-minus-fixed both-correct differences are +0.78 percentage points [95% interval 0.00, 2.34] at window32 and +2.34 [0.00, 5.47] at window128. These clustered intervals describe the fixed primary-parent checkpoints and sampled base pairs. Neither model reaches the 80% assignment-transfer target. Both fresh models get 1/128 both-correct on unseen dev keys and 0/128 under jointly heldout fact/query wording. The 128-window reset is an exact no-op, as expected. The follow-up reduces old-bank-answer behavior, but roughly 50% per-question accuracy and rare key-switch changes do not establish correct two-key retrieval.

**Diagnosis:** the model can fit the finite bank; reliable assignment/key transfer fails under the tested training recipes. This does not prove that the architecture cannot learn the task with a different representation, objective, capacity or training budget. Because the direct-attention models fail transfer too, the result cannot isolate a neural-memory-only defect. A prediction that changes with the query is insufficient: the fixed-bank model changes its answer frequently while still returning obsolete assignments.

### Validation and resources

The 98-test scoped suite passes: 32 data tests, 6 runner tests and 60 existing fact-stream tests. The final CPU smoke completes four reduced-budget checkpoint/evaluation stages and tensor/RNG preservation guards. The independent saved-artifact auditor passes 30 corruption/pairing/selection selfchecks and verifies 192,000 training answers, 18,240,000 input bytes and 45,056 raw 256-logit vectors. The separate disjoint-probe selector passes five CPU synthetic checks. The artifact audit does not execute the model or repeat training.

A same-environment GPU reload check exactly reproduces all saved logits for 192 selected final-checkpoint answers across the six models and ordinary/reset conditions. Figures are generated from audited tables and exported as PNG/PDF. Allocation 13912639 on an RTX A6000 at ins089 was released deliberately after work and retrieval; accounting records 37 minutes 22 seconds (about 0.62 GPU-hours). No allocation renewal or desktop-target change occurred.

### Handoff and next Zeyuan work

`checkpoint_admission.json` records each model's measured scope and unresolved requirements. These checkpoints are fitting/transfer controls; none is admitted to a project retention comparison. The original language parents remain available. The full artifact root is `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-fixed-set-2026-11-14`; the adjacent verified ZIP contains source history, all checkpoints/optimizers, raw prefixes/logits, data and analysis.

This diagnostic has answered its question, so another pressure sweep or indefinite fixed-bank training is not the next useful use of compute. Zeyuan’s next baseline work should target the key/value representation and reusable association-learning objective, with this fitting/novel-assignment suite retained as a regression check. A controlled shared key/value encoding is a useful next positive control for the two memory update mechanisms; it must be labeled separately from the byte-language model. A stronger sequence encoder or training recipe then needs its own bounded comparison before the same retention claims are made.

In parallel with that baseline choice, the common method interface still needs explicit persistent state, write/read/reset semantics, frozen outer weights and revision validation. Member A and Member B can continue their independent implementations against the existing controlled memory testbed and shared data contract, while the full-sequence acquisition gap is exposed rather than hidden. See [Zeyuan status](ZEYUAN_STATUS.md) for how this supports the proposal's sleep-timing and selective-Hebbian aims.
