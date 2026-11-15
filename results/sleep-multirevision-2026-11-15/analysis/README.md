# Independent artifact analysis

This directory belongs to the independent design/artifact reviewer. Analysis imports NumPy, Matplotlib, and the Python standard library; it imports no CLMM model, data-generator, scheduler, or evaluator implementation. The reviewer did not run model inference, training, or accelerator work. The coordinator owns separately reported numerical model rechecks.

## Reproduce

Use Python with NumPy and Matplotlib. The prepared-packet audit also needs Git access to commit `06c0a35b543c25437659765128b81718e5bfa1c5` in the local repository. Preserve the full raw output root and original prior-output files, or relocate the paths in packet provenance and provide an equivalent local Git bundle/checkout.

```powershell
$analysis = 'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-sleep-multirevision-2026-11-15\analysis'
$prior = 'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-revision-local-2026-11-15'
$repo = 'C:\Users\henry\Documents\Codex\clmm-replay'
python "$analysis\audit_pretrained_inventory.py" --trained "$prior\trained"
python "$analysis\audit_prepared_packet.py" --packet "$analysis\..\packet" --prior-root $prior --repo $repo
python "$analysis\run_analysis.py" --final
python "$analysis\stream_telemetry.py" --evaluation "$analysis\..\runs\90511-content-local_delta\evaluation"
python "$analysis\plot_results.py"
```

`run_analysis.py` without `--final` audits every newly completed case and caches hash-identified arithmetic outputs in `cases`. It writes final statistical tables only when all twelve cases have passed the complete raw audit. The optional `--model NAME` restricts a progressive audit to a named case; it does not select cases for the final analysis. Source artifacts and inputs remain preserved; a changed cached input is rejected.

## What the checks establish

`pretrained_inventory.json` authenticates all prescribed terminal checkpoint files against the prior training records, with no checkpoint unpickling. `packet_audit.json` checks all twelve copied checkpoints, all preserved source files against the frozen Git blobs, exact reconstruction of the complete declared exclusion union, array equality to the old actual training data, all literal stream geometry and metadata, and independent private-PCG64 regeneration of every new stream. The two initializers and all five random maps are retained.

Every complete case audit verifies score finiteness, all saved argmaxes and correctness/MSE labels, complete query coverage, obsolete-any versus immediate-penultimate labels, all literal source/key/value/timing events, identical continuous/periodic extra sequences, stale-at-application observer labels, FIFO eligibility, final direct write information, reset/rolled source indices, module occupancy/projection/zero-decay counts, 64-KiB trace state, 8+4 literal-record limits, write opportunities, and full resource-batch coverage. A successful saved-artifact arithmetic check is not model inference reproduction.

The expected full grid contains 310,272 score vectors, 8,474,112 causal evaluation writes and 3,456 resource batches. Reset and rolled controls have their own full arrival plans followed by the declared state manipulation; these writes are included. No optimizer steps are added.

## Tables and uncertainty

`summary.json` holds the two co-primary contrasts and audit totals. `gates.csv` retains common checkpoint admission and every arm/phase's own D0 six-target all-correct gate. `aggregate.csv` contains the full arm/family/phase/delay/query-group curves. `contrasts.csv` contains the matched timing and repeat/online controls. `per_condition.csv` preserves all parent/map results. `diagnostics.csv` contains independently derived error and applied-write exposure summaries, while `replay_telemetry.csv` contains model-independent event counts.

The primary estimand is periodic minus continuous accuracy at final tail D128, separately for four stable targets and two revised targets. It averages the two phases within each base history, the two fixed learned initializers, and a balanced routing family weight of one-half content plus one-half the mean of all five fixed random maps. A simple mean over all twelve checkpoints would use a different estimand.

The 5,000 percentile-bootstrap draws use private PCG64 seed 91104, resampling the same 128 whole-history IDs jointly across arms, phases, delays, parents and maps. Two two-sided 97.5% intervals give Bonferroni coverage for the two co-primary endpoints. Other displayed intervals are nominal 95% secondary intervals. The online-minus-rolled acquisition-gate interval resamples eight groups of sixteen histories because the rolled-state controls couple histories within each batch. These intervals describe history variability conditional on the fixed models/maps, not training-seed or routing-map population uncertainty. No joint superiority criterion is invented after inspecting outcomes.

All conditions enter the primary contrast. An arm-specific D0 failure remains visible and labels its delay curve as a revision/interference diagnostic rather than admitted retention. D0 means no final tail after the three revision windows and two intervening distractor gaps; it still has 56 post-prefill observations. The neutral admission gate instead puts all three genuine changes per revised key in a 22-observation common online-only prefill.

Stable early targets at positions 0/1 cannot be directly replayed after FIFO8 receives the sixteen-fact prefill. Stable eligible targets at positions 12/13 can be replayed while resident. Both groups enter the prespecified stable endpoint equally by target count, and are separately reported. “Eligible” does not imply actually sampled. Six current query values are distinct after each individual genuine revision, and each revised key has four unique versions. Obsolete errors are classified per queried key.

The timing contrast holds all learned tensors fixed within each checkpoint. Content/random checkpoints were separately trained previously, so their family difference also includes those training-induced tensor differences. Stale restoration of lawful old snapshot records is an intended consequence of this timing intervention. It does not establish biological sleep, general lifelong learning superiority, or a map-only routing effect.

## Numerical figures

`plot_results.py` draws all figures from the saved independent source tables and exports PNG/PDF/SVG. `figures/manifest.json` binds input/output hashes and records visual review. Curve error bars are nominal 95%; the two co-primary contrast bars are 97.5%. `revision_error_and_stale_replay` separates immediate-penultimate, earlier-obsolete, and other revised-target errors, and shows the model-independent stale-extra counts. None of the figures uses generated imagery or fabricated numerical placeholders.

The prospective review and later evidence interpretation are separate artifacts: `initial_protocol_review.md` was written before outcomes; the final interpretation is written only after the full grid and its independent arithmetic pass.
