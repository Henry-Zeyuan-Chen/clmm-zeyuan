# CLMM working handoff

## Resume here: nonlinear correlated-key experiment

Completed local implementation, bounded calibration, fresh confirmation, matched-budget comparisons and audited analysis on November15,2026. Branch `nonlinear-correlated-control`. **One operational closeout remains unverified:** SSH to Insomnia began timing out after all main artifacts and the strict GPU proof were downloaded. Check job13914782 and retrieve the optional portable-GPU result when access returns. Do not rerun training or reopen configuration selection.

Read [NONLINEAR_CORRELATED.md](NONLINEAR_CORRELATED.md), [MEMBER_C_STATUS.md](MEMBER_C_STATUS.md) and `results/nonlinear-correlated-2026-11-15`. Zeyuan is Member C, responsible for shared baseline/pretraining/evaluation. A owns sleep, B owns the Hebbian rule/routing. The full byte model remains unqualified for novel two-key transfer; the new result is a typed memory-component baseline.

## Current scientific result

- The first declared rank32 configuration qualifies both original seeds90511/90515; ranks16/8 were not run. Both trained models pass short acquisition/revision and fresh confirmation128/128 in every cell. Actual final K8D0 is128/128 both, K16 is125/128 and124/128. K32 fails75/128,77/128 and remains diagnostic.
- At K8D64, trained stable accuracy averages39.06% online,40.43% repeat-current,35.35% continuous and34.77% periodic. Initial/untrained online accuracy averages29.10%. Training improves delayed accuracy by9.96pp[5.08,14.84] despite initial short accuracy already being perfect; late online coordinateMSE worsens, so accuracy and MSE must remain separate.
- Prespecified periodic-minus-continuous primary:−0.59pp,95%paired interval[−2.15,+0.98]. No established periodic advantage. Two fixed seeds/one codebook,128 shared histories; bootstrap resamples same history IDs across both seeds, not256 independent seed×history rows. No equivalence or broad sleep/Hebbian efficacy claim.
- Continuous/periodic are below the lower-budget online anchor in the planned secondary K8D64 comparison. This frozen small-FIFO timing baseline is useful evidence, not a reason to tune the same histories until a positive result appears. All256 primary stable queries are finally evicted;102 had earlier replay.

## Source and reusable artifacts

- Main frozen source: `b5f27ff4738ded4832538573d454d2461786d38b`.
- Post-run portable loader: `4b119efd9df2f3bc68a223fca4b2da951026f2d8`. Use `clmm.pretrain.kv_correlated_portable.load_portable_checkpoint`. It preserves saved literal codebook bits and every model tensor; the original strict constructor rejects cross-environment normalization bits. This did not alter any model run, selection or score. It also separates CPU-aliased frozen/template and initial-memory storage before loading.
- Output root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-nonlinear-correlated-2026-11-15`.
- Work scripts: sibling `work/clmm-nonlinear-correlated`.
- Final component weights: `r32-90511/checkpoint.pt`, `r32-90515/checkpoint.pt`; their `initial_checkpoint.pt` files are exact untrained anchors. Checkpoint registry records hashes/config/actual D0 abilities. All64 IDs/atomicbindings are familiar; fresh complete histories are the scope.
- Complete recovery ZIP: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/CLMM_Nonlinear_Correlated_2026-11-15.zip`. Adjacent verification JSON records actual ZIP checks and final handoff commit. The archive contains full data/plans/scores/weights/source snapshots and a complete branch bundle; the compact Git index omits raw traces and weights. Read MANIFEST.sha256 after extracting.
- Design fixes ranks32/16/8,1200updatesB32, acquisition/D0/headroom gates and diagnostic fallback before main outcomes. `selection-before-confirmation.json` and `comparison-freeze.json` precede replay. No fallback was needed. Actual training totals357888labels/559936writes from two runs, not planned arrays saved again in evaluation folders.

## Validation actually observed

352 focused tests plus21 portable-adapter tests. Independent current-state SGD/nested-codebook/seed alignment and persistent-state checks. Main artifact audit143616scores/1536000causal writes/2232serialization batches. Strict same-environment GPU:2432selectedanswers and all coordinates exactly reproduced. Portable Windows CPU:2432argmaxes match,max raw-score error3.58e−7. Both use actual bounded streaming/resume; these are selected inference checks, not retraining or full-grid numerical reproduction. Figures are generated from saved CSVs and visually checked.

## Cluster closeout: verify rather than assume

Allocation13914782,ins083,A6000,4CPUs,16G,one-hour limit, observedRUNNING during execution. Scheduler showed start2026-11-15T01:03:18 and scheduled end02:03:18. Main pipeline completed946.50seconds; strict GPU proof was retrieved. A further portable GPU check launched in separate `verification-repo` with180s internal/240s external limit, but SSH channel loss then repeated connection timeouts prevented result retrieval and final scheduler/cancel confirmation. The one-hour limit was not renewed; actual final state/elapsed must not be invented from a scheduled end. No desktop helper selection or unrelated job was changed.

Remote root: `/insomnia001/depts/edu/users/zc2819/clmm/nonlinear-correlated-2026-11-15`. When access returns, inspect `sacct -j13914782`/`squeue -j13914782`; if still running, verify ownership/task completion before cancelling it. Retrieve `gpu-portable-recheck/verification.json` and its log if present, then append the observed closure to the log. No background monitoring is active after this session. A user-facing VPN reconnect question was issued; elapsed time is not a connection response.

## Next project work

Freeze rank32,K8/K16 as the shared qualified component regime. Member A can integrate its actual sleep update and compare it against this continuous timing anchor. Member B can implement the Hebbian update alone, then selective routing. Member C keeps common encoding, state/load API, data histories, acquisition gates and resource accounting. A later buffer-coverage or replay-content ablation should be specified on fresh histories before changing the FIFO; do not conflate it with timing alone. Full sequence-model encoding/pretraining remains separate.

Earlier evidence: [KV_CONTROL.md](KV_CONTROL.md) records the orthogonal ceiling and single linear calibration; [FIXED_SET_GENERALIZATION.md](FIXED_SET_GENERALIZATION.md) records byte fitting/transfer failure; [SHARED_PRETRAIN.md](SHARED_PRETRAIN.md) records both original byte parents. Keep those failures/limits visible. No Google Doc edits, teammate messages, push, PR or merge occurred. Preserve baseline-v0.6.0 at35a549f137548d48aecc26e33c6f8a34a6efc97c. Append consequential work to EXPERIMENT_LOG.jsonl.
