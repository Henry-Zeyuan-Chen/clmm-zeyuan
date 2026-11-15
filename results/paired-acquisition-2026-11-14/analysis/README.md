# Independent saved-artifact audit

`audit_paired_acquisition.py` implements the frozen prospective analysis in the parent `design.json`. It runs on CPU, loads checkpoint dictionaries with `weights_only=True`, and never constructs or forwards a model. It does not import `fact_paired_acquisition.py` or the common experiment runner. It loads the saved v1 fact-stream module after verifying its SHA-256 and optionally its exact Git object; this supplies the existing Episode validator, serialization, and label-independent `render_query` API.

Run the synthetic corruption checks independently of all experiment artifacts:

```powershell
python .\test_audit_selfchecks.py
```

After all five GPU run directories have been downloaded, run from this directory:

```powershell
python .\audit_paired_acquisition.py `
  --root 'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-paired-acquisition-2026-11-14' `
  --parents 'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-shared-pretrain-2026-11-14' `
  --expected-commit d82030974596207a37044ba3951fbbbe242e2d00 `
  --repo 'C:\Users\henry\Documents\Codex\clmm-replay' `
  --out 'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-paired-acquisition-2026-11-14\analysis\audited-results'
```

The output directory must be new and under `analysis`. Missing artifacts, incomplete run status, changed sources, unexpected configuration/budgets, duration over 600 seconds, mismatched arrays or labels, duplicate identities, and broken semantic pairing fail the audit. It expects exactly the five declared main runs, three stages (200/1200/1600), 1600 training steps, batches of 16 answers from eight contexts, and the declared dev populations. The earlier `smoke-cpu` directory is never certified by this command.

Checks include:

- Every frozen source file and every source-to-Git object when `--repo` is supplied; matching source manifests across all runs and source provenance in every checkpoint.
- Parent SHA-256 from the design, original-parent metadata, initial tensors, stage configurations, optimizer settings and counters, finite tensors, exact final-checkpoint flag, and saved checkpoint hashes.
- Every training byte/target reconstructed from saved validated Episodes, declared query indices, complete real context and `render_query`; candidate sampling, rejection conditioning, curriculum, context matching, and per-step/total byte and answer counts.
- Exact K1 input/target arrays and step-200 learned tensors across paired/duplicate arms separately for each parent.
- Every gate/strict raw logit vector and prediction; full-256 and forced-choice scoring; all original gate and strict summaries; no replay in acquisition probes; dev-only keys and exact fixed evaluation populations across runs/stages.
- Strict distinct-value pairs that share context and differ only in requested key; semantic identity checks before any paired comparison; bit-exact 256-logit reset no-op in the auxiliary 128-byte run.

Output tables retain all runs, stages, template families, and inference conditions. `primary_differences.csv` contains strict seen both-correct paired-minus-duplicate differences separately for each parent; only step 1600 is primary. `k2_ablation_differences.csv` contains natural K2 ordinary-minus-reset accuracy intervals. Both use 5000 complete-episode bootstrap draws, seed 89104, percentile 95% intervals. `gate_decisions.csv` and `decisions.json` apply the exact four final paired32 acquisition requirements and the separate useful-pairing-lead rule. `audit.json` has status `passed`; the packaging alias `verification.json` has status `verified` and explicitly limits verification to saved artifacts. Both are written only after every required main run passes. No seed pooling, checkpoint selection, pressure inference, or automatic replay continuation occurs.

This verifies the integrity and scoring of saved evidence. It does not reproduce training or checkpoint inference. Passing a saved-artifact audit does not imply that the model passes acquisition.

After the audit, `plot_audited_metrics.py --report .\final --out .\figures` writes a four-panel PNG/PDF using only audited tables. It refuses an absent verification record or an existing output directory. The plots retain both fixed parents, stage trajectories, the separate 128-byte auxiliary, natural K2 ordinary/reset values, and the primary paired episode intervals.
