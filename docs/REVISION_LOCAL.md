# Key-local revision and the shared component baseline

Completed November 15, 2026 on `hebbian-revision-local`. Executable source: `c9f5c153f786f1b82ac383d01c1d18c6c595946b`; parent handoff `a98c849`. This extends [ROUTING_DECAY.md](ROUTING_DECAY.md), with reference components rather than full Titans/MAC or final teammate methods.

The new rule repairs the existing dense-trained Hebbian checkpoints while increasing late stable recall. After equal training from identical original initializers, both rules score 100% on short revisions. Local delta scores **95.51%/97.07% late stable accuracy** (content/random), compared with EMA's **24.02%/28.52%**. The trained joint primary criterion fails because revision superiority ties at the ceiling; both stable constraints pass. Qualification and superiority remain separate.

## Mechanism and fixed design

For the routed module, let `x=tanh(key_encoder(key))`, `y=tanh(value_encoder(observed_value))`, `s=max(sum(x²),1e-8)`, and `P_x H=(x/s) outer (xᵀH)`. The update is:

`H_new = clamp(d * (H - e * P_x H) + eta * x outer y, -1, 1)`.

All inactive modules stay unchanged. Four frozen rules cross active bulk decay and local erasure:

| Rule | d | e |
|---|---:|---:|
| Original EMA |1−eta|0|
| Additive |1|0|
| Erasure plus EMA |1−eta|1|
| Local delta |1|1|

All have the same first write at H=0. Erasure uses current fast memory and the incoming key, without a revision flag, query label, history lookup or new buffer. Without clipping or an active epsilon floor, local delta overwrites raw `xᵀH` with `eta * ||x||² * y`, preserving orthogonal trace directions. This is not an exact decoded-output guarantee: reads also use W0, elementwise alpha and a learned decoder.

Targeted residual/projection updates are established in [Yang et al., NeurIPS 2024, sections 2.2 and 3.3](https://papers.neurips.cc/paper_files/paper/2024/file/d13a3eae72366e61dfdc7eea82eeb685-Paper-Conference.pdf). This is a delta-rule-inspired reference control with matched first-write amplitude, not biological novelty or a reproduction of that architecture.

The 48 frozen conditions use two existing dense-trained parents, all six maps and four rules. Every original tensor remains identical; additional optimization is zero. The 24 matched training runs compare EMA/local delta from the two ORIGINAL UNTRAINED initializers, with exact paired initial tensor values. All use 1,200 AdamW steps, batch 32, lr .001, weight decay 0, clipping 1, and shared training seed 90802. Initial and terminal models are saved/evaluated; no best checkpoint or map selection occurs. Stage 2 was fixed independently of stage 1 quality.

Content uses fixed cosine prototypes at raw keys 0,16,32,48. Five fixed random maps 90701–90705 permute assignments while preserving module sizes `[14,20,9,21]`. Each table is consistent for reading/writing. Each phase has 128 histories per short cell and K8/K16 × D0/16/64 pressure cell; pressure is online only. Frozen development 90823 excludes both training histories and prior 90723; trained development 90923 additionally excludes frozen histories. Atomic keys/values/bindings are covered by training; exact whole histories are disjoint. This is fixed-codebook composition, not language generalization.

Each observation retains one module update, with 64KiB persistent traces, a 512-byte table and 81,921 parameters. Projection adds arithmetic. Writes, storage, data and optimizer opportunities match; FLOPs do not. All four module candidates are computed before masking.

## Frozen diagnostic

Percentages average two fixed dense-trained parents; random also averages all five maps. Stable means K8D64 stable queries.

| Rule | Revised/content | Revised/random | Stable/content | Stable/random |
|---|---:|---:|---:|---:|
| EMA |92.97|93.75|34.96|41.25|
| Additive |53.52|52.34|84.96|88.01|
| Erasure plus EMA |100.00|100.00|35.35|43.16|
| Local delta |100.00|100.00|90.04|92.42|

Removing active bulk decay explains most stable gains, but additive loses revision ability. Erasure restores replacement of stale values. The compound candidate achieves both here; its stable gain cannot be credited solely to erasure. Additive fails revision/D0 admission and remains a mechanism diagnostic. All 12 candidate parent/map conditions pass short and K8/K16 gates. EMA's 98 short revision errors all return obsolete values.

Candidate−EMA revised gains: +7.03 points content (97.5% interval [3.52,11.33]) and +6.25 random [4.22,8.44]. Stable gains: +55.08 [48.83,61.72] and +51.17 [47.66,54.77]. All four diagnostic inequalities pass. This does not replace the separately trained primary analysis.

## Matched training and initial controls

| Rule/stage | Revised/content | Revised/random | Stable/content | Stable/random |
|---|---:|---:|---:|---:|
| EMA initial |100.00|100.00|22.66|30.31|
| EMA trained |100.00|100.00|24.02|28.52|
| Local delta initial |100.00|100.00|88.87|91.64|
| Local delta trained |100.00|100.00|95.51|97.07|

Original untrained initializers already solve short revision. The earlier repair concerns particular dense-trained parents, not an inherent failure of every Hebbian initializer. Both matched methods remain at 100%, so trained revision superiority is not established. The declared four-inequality joint primary result is **false**.

Trained stable differences are +71.48 points (97.5% interval [65.23,77.73]) for content and +68.55 [65.66,71.41] for random. Both revision differences are 0 [0,0]. All 24 initial and all 24 terminal conditions pass original short and K8/K16D0 gates. Candidate terminal D0 four-target all-correct is 100% at both K values; EMA minima are 99.22% K8 and 96.09% K16.

Most candidate advantage exists before training. Secondary paired initial→final stable gains are +6.64 points content (nominal 95% interval [3.71,9.96]) and +5.43 random [3.87,6.99]. EMA changes +1.37 [−3.52,6.05] and −1.80 [−4.06,.39], without an established late gain. Training supplies a smaller additional benefit for local delta; it does not explain the entire architectural gap.

Primary constraints use two-sided 97.5% percentile intervals: lower .0125 for four one-sided Bonferroni comparisons. Revision requires lower >0; stable noninferiority requires lower >−.03, a prospectively fixed 3-point engineering tolerance. Bootstrap: 5,000 PCG64 draws, seed 90804, jointly resampling 128 history IDs across fixed parents/maps for each endpoint. These are conditional history intervals, not population seed/map uncertainty.

Short gates remain acquire2/revise2 both-correct ≥.90, acquire4 ≥.80, and normal-minus-rolled acquire2 blocked lower95 >0 (eight cycles of16). Candidate additionally needs per-K D0 all four original targets correct ≥.80 in both initializers/every relevant map. Comparator revision failure was prospectively exempted only for repair comparisons; other comparator gates remain required. Reset is diagnostic; no failures are dropped or thresholds relaxed.

## Checks, resources and recovery

`clmm/hebbian/revision.py` supplies the rules and strict schema-3 state. `scripts/revision_local_experiment.py` supplies both phases and dedicated `clmm.revision.local.v1` checkpoints. Use its `load_checkpoint`, with generic SleepSession. Fast fingerprints do not bind learned values: retain exact slow checkpoint SHA from the registry.

181 relevant tests pass. Independent formula/source checks cover all 48 constructions and 155 tracked source files. Full raw audits check 638,976 scorevectors, 3,330,048 causal evaluation writes and 11,520 resource batches across 96 evaluation weightsets. Frozen/initial tensors and all 24 optimizer payloads are audited. Actual main training: 28,800 steps, 6,707,712 observations and 4,313,088 answers. Engineering runs are separate. All 792 required case files and executed-source hashes are reconciled.

Selected inference/resume checks cover all 96 weightsets, 10,752 queries on each device, with all saved predictions matching. Maximum coordinate errors are 9.54e−7 GPU and 3.58e−7 WindowsCPU, below the declared 1e−5 cross-device tolerance. These are inference/artifact checks, not training or full-grid numerical reproduction. Figures derive from audited CSVs with vector exports and visual inspection.

Pending CPU job 13919399 was cancelled without running after a multi-hour queue estimate. Frozen evaluation ran locally from an isolated LF checkout of the same source in 536.02s. GPU preflight projected 1,189.77s below the 2,700s limit; matched training/initial+final evaluation took 1,022.75s and the complete pipeline 1,150.50s. Training uses Torch2.11 CUDA/CPU evaluation; frozen/Windows proof use Torch2.15 CPU. The prospective execution amendment changes hardware only.

Insomnia13919414 (ins080,A6000,4CPU/16G) was explicitly cancelled after 23m01s: 0.3836 allocated GPUhours including CPU evaluation, checks and transfer. All task steps completed exit0; sleeping-wrapper cancellation was intentional. No renewal, helper-selection change, active task job or background monitoring remains. Both transfer hashes agree; tar exits0.

Full root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-revision-local-2026-11-15`.
Compact index: [results/revision-local-2026-11-15](../results/revision-local-2026-11-15).
Recovery ZIP: sibling `outputs/CLMM_Revision_Local_2026-11-15.zip`, including four original parents, 48 frozen weights, 24 initial/final pairs/optimizers, all raw outputs, sources, engineering records, analyses and branch bundle. Adjacent verification identifies every-file SHA/CRC and final handoff commit. Preserve baseline-v0.6.0 unchanged.

Clone the bundle, check out c9f5c153, use the recorded environment and relocate launch paths. Run frozen with90823 and trained with90923/train90802, default1200steps/128counts. Phase analysis READMEs contain reproduction commands; full raw artifacts are required. Registry entries bind exact checkpoints and admission.

## Exploratory reinforcement stress after the ceiling result

The trained short-revision ceiling motivated one additional CPU-only follow-up, declared before its inference on fresh91023 histories. It reuses all24 terminal checkpoints without optimization. Between the original8facts and2genuine revisions, each old binding receives R=1,4,16 total presentations. One subsequent observation changes each binding; D=0/64 distractors follow. All six cells use the same base histories, and every method gets identical writes. All training/prior/frozen/main histories are excluded. No revision flag enters the model.

| R16D64 endpoint | Local delta/content | EMA/content | Local delta/random | EMA/random |
|---|---:|---:|---:|---:|
| Revised queries |99.61%|0.39%|98.87%|2.03%|
| Stable queries |96.68%|18.36%|97.58%|22.46%|

At R4/R16 D0, local delta replaces values correctly100% while EMA revised accuracy is0% in every condition. All3072 R16D0 EMA revised predictions return obsolete old values. This is immediate replacement failure, not solely later forgetting. The EMA all-four-target D0 gate fails there, so interpret the follow-up as exploratory revision/interference performance, rather than an admitted EMA retention comparison.

Focus R16D64 revised gains are+99.22points content (nominal95% interval[98.05,100]) and+96.84random[95.70,97.81]. Stable gains are+78.32[73.63,82.81] and+75.12[72.58,77.66]. Bootstrap5000 paired-history draws, seed91004; fixed parents/maps. These exploratory intervals do not replace the original primary or establish final-test generalization. All reinforcement/delay cells and failures remain visible.

All24 cases finish in351.51s CPU under the600s limit. Raw audit checks156672scorevectors/1078272writes/2880resource batches, exact repeated-old streams, latest-value targets, joint pairing/exclusions, budgets and checkpoint/source links. Coordinator independently reconstructs the data, argmax flags, focus means/intervals and168case files. Selected CPU inference/resume reproduces1920 R16D64queries across all24 reused models exactly. No new parameters, optimizer steps, buffers or GPU allocation.

Driver `work/clmm-revision-local/revision_stress.py` is preserved as `stress/driver_snapshot.py`, SHA `f7162bbf3d86be1da4dd663e52c2ca97d1434c1087e380ffc1876c35556a8ab3`, alongside the frozen c9 core. See `analysis/stress/README.md`, design, interpretation and figures. Main plus follow-up total795648audited vectors and4408320evaluation writes. The original trained joint primary remains **not met**.

## Next Member C milestone and team choices

Use qualified local-delta weights and EMA comparators as shared component baselines for A/B, retaining both initializers and random controls. Next test intended adapters on multiple genuine revisions and admitted K16 pressure with identical checkpoints, routing, replay budgets and explicit forgetting clocks. Simple short revisions are at a ceiling; the reinforced-old-value fixture provides a useful harder diagnostic. Consulted90823/90923/91023 are development sets, not untouched final tests.

The team should decide whether local delta is a strong control or part of B's intended mechanism. Known delta-style erasure is not biological novelty. A/B's final modules, full Titans/MAC, the paper-faithful roadmap and byte-model fresh-assignment transfer remain separate. No Google Doc edits, teammate messages or GitHub publication occurred.
