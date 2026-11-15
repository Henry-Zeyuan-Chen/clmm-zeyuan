# CLMM working handoff

## Resume here

**Active revision-local follow-up:** branch `hebbian-revision-local`; design under `outputs/clmm-revision-local-2026-11-15/design.json`.48 frozen2x2diagnosticconditions and24matched-training EMA/local-delta conditions from untrained paired initializers. No main outcomes yet. Both cosine/all5randommaps, same state/write/optimizerbudgets; nextjob/statusmustbeverifiedfromactualfiles.

**Completed November15,2026: content/random routing × inactive-trace decay.** Local branch `routing-decay-controls`, executable source `61212289d6029da65eeb6a3735ce5ec3894866a9`, parent handoff65609b3. Start with [ROUTING_DECAY.md](ROUTING_DECAY.md), [MEMBER_C_STATUS.md](MEMBER_C_STATUS.md), and `results/routing-decay-2026-11-15`. No active task job, pending connection check or background monitoring remains.

Zeyuan is Member C: shared pretraining, weights and evaluation. A owns sleep, B owns Hebbian/routing. Latest fetched team main1e8de19 contained only empty sleep/Hebbian folders; these are reference component implementations, not teammate-final methods. Full Titans/MAC and byte-model fresh-assignment transfer remain unresolved.

## What changed

`clmm/hebbian/routing_decay.py` adds fixed cosine/random tables, protected/global inactive decay, versioned configs and strict state/checkpoint validation. Existing adapter behavior is preserved. New runner `scripts/routing_decay_experiment.py` freezes dense parents and saves dedicated `clmm.routing_decay.v1` weights. Shared evaluator has optional overrides and post-update exposure logging.110 relevant tests pass.

48 conditions =4dense-trained parents ×(content+5fixed maps90701–90705) ×2decays. Parents: gradient/Hebbian, seeds90511/90515, frozen training sourcea41bb09. All learned/codebook/prototype bits remain identical within a parent, and every session starts zero H. No new training, optimizer or map selection. One retained update/effective read per observation,64KiB persistent trace,512-byte table. Arithmetic still computes four candidates; global decay intentionally multiplies4traces instead of1.

Main development90723:128histories in each short/acquisition/revision andK8/K16×D0/16/64cell. Online pressure only. Whole histories are training-disjoint; atomic codebook/bindings are familiar. Original bootstrap/admission preserved, failures retained.

## Evidence and interpretation

K8D64 stable scores, contentprotected/randomprotected/contentglobal/randomglobal:
- Gradient:41.02/48.32/5.47/5.55percent. All12table×decay families pass both-parent admission atK8/K16. Secondary contentprotection+35.55pp95[30.47,40.82]; content−random−7.30pp[−13.48,−1.05].
- Hebbian:33.40/38.63/4.69/5.35percent. Co-primary content−random−5.23pp97.5[−11.52,+1.33]; contentprotection+28.71pp[23.24,34.57]. **Both are diagnostics, not admitted retention claims.** Four protected90515 shortrevision conditions fail .90; content=.890625. Both-parent families admitted8/12atK8,4/12atK16. Five globalHbparentconditions additionallyfailK16D0.

Post-outcome exploratory revision contrast: Hbcontent92.19%protected versus99.22%global, difference−7.03pp95[−10.55,−3.91]; allshortrevision errors return obsolete values. This exposes a stability/revision tradeoff. Cosine routing has no demonstrated advantage. Do not call this biology, pure semantic routing, equally retrained efficacy, population-map inference, or a percentage of the historical dense→top1gain. Routes jointly alter read/write allocation. Intervals condition on fixedparents/maps/codebook.

Independent raw audit:319,488vectors,1,665,024causalwrites,5,760resourcebatches,7,296serializationboundaries; actual data/summary/plans/budgets/exposure exact. All48 saved checkpoints retain all9parent tensors bitforbit.384casefiles and29executed-source Gitblobs checked. GPU/WindowsCPU rechecks each cover5,376queries/all48models; allpredictionsmatch, maxerrors9.54e−7/3.58e−7. Four measured figures rendered and inspected. These checks do not retrain or numerically reproduce every grid output.

## Artifacts and recovery

Full output root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-routing-decay-2026-11-15`.
Work scripts: sibling `work/clmm-routing-decay`.
Recovery ZIP: sibling `outputs/CLMM_Routing_Decay_2026-11-15.zip`; adjacent verificationJSON binds every-fileSHA/CRC and handoffcommit. Contains fullbranchbundle/baseline tag,4parents,48caseweights, bothengineeringruns, rawscores/plans, sources and analysis. Compact Git index omits weights/fullrawdata.

Use dedicated `scripts.routing_decay_experiment.load_checkpoint`, not the old associative registry loader, for new48casefiles. `RoutingDecayAdapter.from_parent` requires a dense base parent; supplies identicalslowbits/newconfig/zeroH. Exactslowcheckpoint SHA must accompany everyfaststate: config fingerprints bindtable/policy/decay/codebook, not learnedweights. `checkpoint_admission.json` records compatibility and gates. Saved byte weights, nonlinear neural weights and associative weights remain distinct architectures.

Main `pipeline-cpu.json` succeeded. Original `pipeline.json` is intentionally `preflight_only`: GPU projection2874s exceeded2550bound, before main. Saved runtime-only amendment moved unchanged grid to CPU (projection862s; main534.46s), GPUusedforengineering/recheck. Preserve bothpipeline records. Launch scripts need path relocation; frozen source6121228 and environment versions are authoritative. Analysis README has reproduction commands requiring fullrawroot.

## Cluster closed

Job13918985(ins087,A6000,4CPU/16G) explicitlyCANCELLED after13m25s,0.2236allocatedGPUh includingCPUmain/transfer. Tasksteps COMPLETEDexit0; wrappercancelintentional. Tar exits0, remote/localSHAagree, transferextracted651members. No activeuserjobs; desktophelperselection unchanged. Earlier13918002cancelled and13914782TIMEOUT remain resolved in prior handoffs.

## Next concrete work / team choices

Member C should maintain the common API/checkpoint/causal-evaluation contract, qualify A/B's intended modules on acquisition and genuine revision, and then rerun matched continuous/periodic replay with identical routing/decay. Keep random allocation as a mandatory control. Do not select a favorable map or tune eta on90723.

Team decisions: inactive global/protected forgettingclock; intended route (fixed/learned/topk); and error-correcting revision behavior of B's local rule without revision/query-label oracles. A new equal-training-budget comparison needs declared observations/labels/optimizeropportunities, exact initializers and fresh histories. Do not solve admission by dropping seeds or relaxing thresholds. No new training study started here. Fullsequence bridge and roadmap remain separate milestones.

Prior [ADAPTER_INTEGRATION.md](ADAPTER_INTEGRATION.md), [NONLINEAR_CORRELATED.md](NONLINEAR_CORRELATED.md), [KV_CONTROL.md](KV_CONTROL.md), and [FIXED_SET_GENERALIZATION.md](FIXED_SET_GENERALIZATION.md) retain earlier evidence/limits. No GoogleDoc edits, teammate messages, pushes/PRs/merges. Preserve baseline-v0.6.0 at35a549f137548d48aecc26e33c6f8a34a6efc97c. Inspect real files/Git/scheduler on resume and append EXPERIMENT_LOG.jsonl.
