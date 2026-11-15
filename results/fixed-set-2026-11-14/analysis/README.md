# Independent saved-artifact audit

`audit_fixed_set.py` runs on CPU and loads tensor dictionaries with `weights_only=True`. It never constructs or forwards a model and never imports the experiment runner. It imports the frozen fact-stream and association-data modules only after checking every source SHA-256 and, when `--repo` is supplied, its exact Git object. It independently recomputes full-256 and forced-choice predictions, NLL, legal answer mass, semantic pairing, summaries and bootstrap analysis.

The checks cover the declared source revision and design, all original parent tensors and step-zero checkpoints, optimizer counters and finite states at every stage, complete training arrays and labels, epoch coverage, matched 32/128 bytes and targets, key-split isolation, evaluation populations and prefixes, independent question labels, all raw 256-logit rows and reported summaries, and exact 128-window reset no-ops. The branch is reconstructed only from the frozen `fixed128-68101/step-2000` endpoint and compared with `branch.json`. Failed or missing prescribed runs prevent verification; smoke runs are never certified.

Run synthetic corruption checks first:

```powershell
python .\test_audit_selfchecks.py
```

After all main and required conditional run directories have been downloaded:

```powershell
python .\audit_fixed_set.py `
  --root 'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-fixed-set-2026-11-14' `
  --parents 'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-shared-pretrain-2026-11-14' `
  --expected-commit 5bccffd9084b543ca086721be2bd46e6a2ea27a6 `
  --repo 'C:\Users\henry\Documents\Codex\clmm-replay' `
  --out 'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-fixed-set-2026-11-14\analysis\final'
```

The output must be a new directory under this experiment's `analysis`. `verification.json` is written with `status: verified` only after all checks pass. The audit verifies persisted evidence; it does not reproduce training or model inference. Runtime RNG/tensor invariants are assertions in the frozen runner rather than a separate inference reproduction here.

`metrics.csv` retains every run, stage, population and inference condition. `paired_differences.csv` contains 128-minus-32, normal-minus-reset and any exploratory fresh-minus-fixed both-correct differences. The latter maps the fresh `reference_bank` group to the same fixed arm's `train` examples. `decisions.csv` leaves the fixed-fit gate blank for fresh runs because their reference bank is not the fixed training set. `budgets.csv` preserves answers, input bytes and elapsed time. `training_context_overlap.csv` quantifies exact model-visible context overlap with each run's training history and reports final accuracy among unseen contexts; episode seeds are metadata and do not make identical byte contexts new.

For `train_query_only` rows in the overlap table, the count refers to the episode's underlying semantic fact context. Those facts are absent from the actual 31-byte model input. These rows do not establish training/probe leakage and should not be used as evidence of such leakage.

If the fresh branch is selected, verification additionally requires `disjoint-transfer` and the prospectively recorded `transfer-amendment.json`. The auditor does not import the external probe. It independently reconstructs the first four accepted seed-90205 candidates per base pair after excluding both original letters, duplicates and the union of both fresh training-context sets. It checks the selected source indices and rejection counters, frozen external-script SHA-256, all source/input/output hashes, every one of the 2,048 raw logit rows, summary copies and exact 128-window reset no-ops. `disjoint_metrics.csv`, `disjoint_differences.csv` and `disjoint_selection.json` retain this training-disjoint comparison. These are development results rather than an untouched final test.

`posthoc_old_assignment_accuracy` reports the fraction of novel-value queries answered with that base pair's original bank target; `posthoc_both_old_assignment` reports the corresponding two-answer rate. Both original letters are excluded from each new context, so these are wrong old answers. These requested posthoc descriptions do not change the primary gates and do not establish an internal mechanism. The original overlapping fresh-regime population remains descriptive; the disjoint amendment supplies the fresh branch's assignment-transfer evidence.

Novel-value intervals resample the 32 fixed ordered key-pair clusters while retaining all four assignment replicates. The tiny control instead has eight base pairs with sixteen replicates each. Other intervals and differences resample complete two-question episodes. There are 5,000 draws with seed 90204, and no pooling across parent checkpoints or checkpoint selection. Intervals describe these fixed checkpoints and sampled development units. Train-bank intervals are descriptive empirical resampling, not a population generalization estimate. A zero-event bootstrap interval does not prove zero population error or equivalence.

The novel-values population excludes both original letters for its ordered pair and is a deliberate shift, not IID generalization. Heldout wording changes facts and queries jointly on heldout dev keys, so it does not isolate their individual effects. The 128-window model retains neural memory while exposing the complete 95-byte input to attention. Query-only examples are shorter and are a diagnostic, not a causal proof of binding. Passing an exact-set or same-key check alone does not qualify the retention baseline.

After verification, `python .\plot_audited_metrics.py --report .\final --out .\figures` writes PNG/PDF figures exclusively from audited tables. Figure generation refuses an absent verification record or an existing destination.

All 30 synthetic selfchecks pass, including nine checks for the added disjoint-selection path; the execution output is saved in `selfchecks.log`.
