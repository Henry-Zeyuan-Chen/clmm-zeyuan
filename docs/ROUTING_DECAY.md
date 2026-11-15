# Content allocation and inactive-trace decay

Completed November 15, 2026 on local branch `routing-decay-controls`. Frozen executable source: `61212289d6029da65eeb6a3735ce5ec3894866a9`; dense parent training source: `a41bb097af3e7eed356d13e24e4a0d5bc6677b14`. This extends the [reference adapter pilot](ADAPTER_INTEGRATION.md), rather than completing Titans/MAC integration.

Inactive-trace protection changes late stable-query scores substantially. The fixed cosine router has no demonstrated advantage over cardinality-matched random allocation. Protection also exposes a Hebbian revision failure, so the Hebbian co-primary comparisons remain acquisition/interference diagnostics under the preset admission rule. The admitted gradient comparison supports the protection effect independently.

## Controlled comparison

Four existing dense-trained parents (gradient/Hebbian, initializers90511/90515) retain every learned tensor bit, including encoders, decoder, W0, alpha and eta. Every condition starts with zero H. There is no new optimization or checkpoint selection.

Within each parent, six fixed key-to-module tables cross two decay policies, giving48 conditions:

| Factor | Conditions |
|---|---|
| Content allocation | Cosine argmax against literal keys0,16,32,48, materialized once on CPU |
| Random allocation | Five preselected PCG64 permutations90701–90705 of content labels over64 keys |
| Protected inactive traces | Active EMA/update unchanged; other three traces unchanged |
| Global decay | Identical active update; other three traces multiply by1−eta on every observation |

All tables preserve exact module sizes `[14,20,9,21]`, use the same assignment for writing/reading, and never inspect values or evaluation labels. The actual512-byte table and SHA are saved in each config/checkpoint. Online writes make no random routing decisions.

Conditions receive identical online-only streams, one retained association update and one effective read module per observation, and64KiB persistent fast traces per stream. The implementation computes candidates for all four modules before masking. This matches retained updates, not sparse arithmetic or FLOPs. Global decay intentionally applies decay to4 rather than1 trace; storage and learned weights stay identical.

Fresh development seed90723 supplies128 histories per short cell and per K8/K16 × D0/16/64 pressure cell. Whole histories are training-disjoint; all64 keys,16 values and1,024 bindings are familiar. Delay cells share prefixes within K. These are new compositions in a fixed correlated codebook, not open-vocabulary/language generalization. Pressure arms are online only to isolate routing/decay from replay timing.

## Results and admission

K8D64 stable-query accuracy averages the two fixed parents; random additionally averages all five fixed maps. All maps and failures are retained.

| Parent family | Content/protected | Random/protected | Content/global | Random/global |
|---|---:|---:|---:|---:|
| Gradient |41.02%|48.32%|5.47%|5.55%|
| Hebbian |33.40%|38.63%|4.69%|5.35%|

The Hebbian co-primary contrasts use paired-history bootstrap97.5% intervals, nominally adjusted for two comparisons:

| Contrast | Difference, percentage points |97.5% interval|Admission|
|---|---:|---|---|
| Content−mean5random, protected |−5.23|[−11.52,+1.33]|Diagnostic; own revision gate failed|
| Protected−global, content |+28.71|[+23.24,+34.57]|Diagnostic; own revision gate failed|

Secondary gradient contrasts pass admission: content−random is−7.30points (nominal95% interval[−13.48,−1.05]); content protection is+35.55points[+30.47,+40.82]. This supports inactive protection and favors the fixed random allocation on these histories. It does not rescue the failed Hebbian admission or establish an absolute Hebbian-versus-gradient causal effect: those parents were trained separately.

The bootstrap resamples128 history IDs jointly across the fixed parents/maps,5,000draws, seed90704. Intervals describe history variability conditional on these checkpoints/codebook/tables, not population uncertainty across training seeds/maps. This factorial does not measure a percentage of the earlier dense→top-one gain.

Preset short gates use episode-all-correct: acquire2/revise2≥.90, acquire4≥.80, and acquire2 normal-minus-rolled lower95%>0 (eight16-history cyclic blocks). Reset is a diagnostic. Per-K admission additionally requires D0 all four original targets correct≥.80 in both initializers; the recent distractor is excluded.

All48 conditions pass pressure K8D0 alone (minimum91.41%), but four protected Hebbian90515 conditions fail short revision: content89.06%, random90703 88.28%,90704 86.72%,90705 88.28%. Thresholds remain unchanged. Hebbian table/decay families admitted in both parents are8/12 atK8 and4/12 atK16; gradient is12/12 at both. Five global Hebbian parent conditions additionally fail K16D0. Full gates and per-map results are retained in the analysis CSVs.

## Revision tradeoff: exploratory analysis

After seeing gate failures we added a paired short-revision contrast, explicitly outside the co-primary tests. Hebbian content/protected scores92.19% both-correct versus99.22% global: −7.03points, nominal95% interval[−10.55,−3.91]. Random means are91.33% versus99.22%. All Hebbian short-revision errors return obsolete old values; the other stable key is correct.

The component shows a stability/revision tradeoff. Protecting inactive traces helps late stable queries, while retaining old contributions can impede replacing a binding. This is a useful project failure, not a reason to relax admission. Saved raw/learned key geometry and same/other-module exposure counts are descriptive; they do not establish mediation of the routing difference. Read and write allocation change jointly, and equal global table sizes do not match every history's collisions.

## Implementation and checks

- `clmm/hebbian/routing_decay.py`: versioned config, saved fixed router, inactive-decay switch, strict checkpoint tensors and policy-bound fast-state schema2. The prior reference adapter/config remain unchanged.
- `scripts/routing_decay_experiment.py`: slow-only dense-parent forks, dedicated `clmm.routing_decay.v1` checkpoints, no training and complete48-condition evaluation.
- Shared evaluator additions: optional data/pressure-arm overrides, per-history association/decay counts, and target-specific following same/other-module writes after the last real target observation. Exposure is computed after state updates; future query keys/labels never enter a write.

110 relevant CPU tests pass. Independent formula/state checks cover all48 constructions; actual saved-payload audits verify all nine original parent tensors bit for bit. Raw audit reconstructs319,488scorevectors,1,665,024causal writes,5,760resource/serialization batches and7,296serialization boundaries. Actual histories, exclusion arrays, predictions, summaries, plans, budgets and exposure are checked. All384required case files exist; all29executed source files match frozen Git blobs.

Selected actual SleepSession reload/resume checks cover all48 conditions on both devices,5,376queries each. Every saved prediction agrees. Maximum coordinate errors are9.54e−7 on GPU(Torch2.11) and3.58e−7 on Windows CPU(Torch2.15), within the predeclared cross-device1e−5 tolerance. These are inference/artifact checks, not retraining or full-grid numerical reproduction.

## Compute and recovery

GPU engineering projected2,874s, exceeding the conservative2,550s continuation limit; the original pipeline stopped before main outcomes. A prospective runtime-only amendment tested CPU for these small matrices. CPU projected862s and ran the identical48-case/128-history grid in534.46s. Both engineering runs and the original stopped pipeline are preserved. Main Torch2.11.0+cu128 runs on CPU; GPU handled the preflight and selected rechecks.

Job13918985(ins087,A6000,4CPU/16G) was explicitly cancelled after13m25s,0.2236allocatedGPUhours, including CPU execution and transfer. Task steps completed exit0; the sleeping allocation wrapper was cancelled intentionally. No renewal, helper-selection change, unrelated cancellation, active task job or background monitoring remains. Transfer tar exit0 and remote/local SHA agree.

Full root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-routing-decay-2026-11-15`.
Compact Git index: [results/routing-decay-2026-11-15](../results/routing-decay-2026-11-15).
Complete recovery ZIP: sibling `outputs/CLMM_Routing_Decay_2026-11-15.zip`, including parent/case weights, raw outputs, exclusions, sources, scripts, branch bundle and unchanged baseline tag. Adjacent verification JSON identifies every-file SHA/CRC and handoff commit. Fast fingerprints do not identify slow learned values: retain the registry's exact checkpoint SHA.

To reproduce, clone the package bundle, check out6121228, set up the recorded environment, and run `scripts/routing_decay_experiment.py` with a new output directory, packaged `parents/`, `design.json`, frozen source commit,90723 and default128counts. Relocate absolute launch paths. Follow `analysis/README.md` for analysis; full raw artifacts are required, rather than the compact Git index.

## Next shared-baseline work and team choices

Member C should qualify A/B's intended methods on acquisition **and revisions**, then compare continuous/periodic replay with identical routing, decay and replay budgets. Keep fixed random routing as a control. Do not select a favorable map or tune eta on the consulted90723 histories.

Agree whether inactive decay is global/protected, what forgetting clock the intended method uses, and how its local rule replaces conflicting values without a revision/query-label oracle. Implement those choices behind the functional write/read/session API. A new declared-budget training comparison needs initializers and fresh histories before claiming equally trained routing efficacy. No such training was launched here.

Full Titans/MAC, the paper-faithful roadmap and the byte-model fresh-assignment transfer failure remain separate. No Google Doc editing, teammate messages or GitHub publication occurred.
