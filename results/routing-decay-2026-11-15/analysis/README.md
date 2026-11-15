# Independent analysis and validation

The prospective review is `design_review.md`. It was written before this run's outcomes. The source is frozen at `61212289d6029da65eeb6a3735ce5ec3894866a9`; base repository parent is `65609b3648a8e36b3ad0bee02202f84522a9b550`.

The completed readout is `result_interpretation.md`. All48 cases are retained. Content routing shows no advantage over the five fixed random maps; inactive protection strongly raises late stable accuracy, but the Hebbian protected family fails its own short revision gate. The four inspected figures include this stability/revision tradeoff and visible admission status.

## Completed before outcome analysis

- `parent_input_audit.json`: all four dense parents use identical saved training arrays, equal to deterministic regeneration of 1,200 batches of 32 at seed90502. There are 38,393 distinct full training histories among 38,400 samples. All64 keys,16 values and1,024 bindings are familiar in each parent's training data. Literal codebooks/prototypes match across parents. New90723 histories have zero full-history training overlap and share intended D0/16/64 prefixes within K. The content module sizes are [14,20,9,21]; all five fixed random tables preserve these sizes exactly.
- `adapter_independent_audit.json`: CPU equation/state checks cover all48 parent/policy/decay combinations. Parent tensors remain bit-identical. Active protected/global results are exact; the independent modulewise formula differs by at most1.49e-8 per coordinate. Content/protected exactly reproduces old top-one traces and scores for12 writes per parent. Input-state/slow-weight/RNG purity, zero64KiB fast state, private fixed maps, policy-bound imports and serialization are checked. This is not a GPU or training reproduction.

## Reproduce without training

From any checkout containing the frozen source, run the first two scripts with a CPU Python environment containing torch and NumPy. Replace only the filesystem paths if the archive has been relocated.

```powershell
$auditPython = 'C:/Users/henry/Documents/Codex/2026-10-22/i/work/clmm-test-env/Scripts/python.exe'
$plotPython = 'C:/Users/henry/AppData/Local/Programs/Python/Python313/python.exe'
$experimentRoot = 'C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-routing-decay-2026-11-15'
$previousRoot = 'C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-adapter-integration-2026-11-15'
$repoPath = 'C:/Users/henry/Documents/Codex/clmm-replay'
& $auditPython "$experimentRoot/analysis/audit_parent_inputs.py" --repo $repoPath --parents $previousRoot --out "$experimentRoot/analysis/parent_input_audit.json"
& $auditPython "$experimentRoot/analysis/audit_adapter_independent.py" --repo $repoPath --parents $previousRoot --out "$experimentRoot/analysis/adapter_independent_audit.json"
& $auditPython "$experimentRoot/analysis/audit_saved_checkpoints.py" --root "$experimentRoot/main" --parents "$experimentRoot/parents" --out "$experimentRoot/analysis/saved_checkpoint_audit.json"
& $plotPython "$experimentRoot/analysis/analyze_independent.py" --root "$experimentRoot/main" --out "$experimentRoot/analysis" --saved-training "$previousRoot/gradient-dense-90511/training_data.npz"
& $plotPython "$experimentRoot/analysis/plot_independent.py" --analysis "$experimentRoot/analysis"
```

The analysis imports no evaluator implementation. It checks the frozen source/scope, case and checkpoint hashes, literal score/prediction/MSE agreement, all histories/queries and control state sources, saved summaries, every causal online plan, actual write budgets, per-history module association and decay exposure,64KiB fast state, resumed-read exactness, table cardinalities and private permutations. `saved_checkpoint_audit.json` independently compares every saved parent tensor and routing table without constructing an adapter.

Rechecking the original saved training arrays requires the previous adapter-integration artifact directory (or its recovery archive). The main numerical analysis can run entirely from this experiment root by omitting the optional `--saved-training` comparison; it still reconstructs and checks all saved exclusion signatures. Flattened parent checkpoints are retained under this root's `parents/`.

## Statistics and interpretation

All reported maps and parents are fixed. The128 shared history IDs are the resampling units. Stable-query accuracy first averages the two stable queries within each history; conditions then average the two fixed parents and, where applicable, the five fixed random maps. Main bootstrap draws resample the same history IDs jointly across all conditions5,000 times at private seed90704. The two Hebbian co-primary contrasts use97.5% percentile intervals (1.25th/98.75th percentiles) for Bonferroni nominal familywise95% coverage. Other intervals are unadjusted95% development intervals. Neither interval estimates training-seed or routing-map population uncertainty.

Short admission uses episode-all-correct acquire2/revise2>=.90 and acquire4>=.80. The acquire2 normal-minus-rolled all-correct contrast must have a positive lower95% bound, bootstrapping the8 independent16-history rolled cycles. Reset remains diagnostic. Each K requires D0 all-four-original-target accuracy>=.80 in both fixed parents for that rule/table/decay family. Failures remain in every table and figure; related effects are acquisition/interference diagnostics rather than admitted retention improvements.

One retained active association/read module per observation and64KiB fast state are matched. The inherited vectorized code computes candidate changes and decoded reads for all four modules before masking. It therefore supplies no sparse-arithmetic or FLOP-saving evidence. Global inactive decay intentionally applies four EMA decays per observation versus one in the protected condition. Per-stream module occupancy and post-target same/other-module exposure can differ despite matching global key cardinalities.

This is a frozen-transfer mechanism diagnostic conditional on four dense-trained parents and one synthetic correlated codebook. Routing jointly changes reads and writes. Effects do not establish biological plausibility, learned or natural-language semantic routing, a general Hebbian advantage, a fraction of the historical dense/top-one benefit explained, or complete Titans integration. Runtime-only changes documented at the experiment root do not alter maps, histories, checkpoints or estimands.
