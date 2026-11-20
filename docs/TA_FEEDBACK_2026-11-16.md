# TA feedback: persistence, benchmarks and the vanilla comparator

Recorded November 16 from the email pasted by Zeyuan. This is a planning
update, not a new experiment or an interim-report draft. The TA supports
investigating memory update rules, but asks for an explicit account of how
updates survive across segments, recognizable evaluation foundations, and a
same-size vanilla Titans comparator. The email's offer to meet is not an
instruction to contact or schedule anyone.

## Decision and current evidence

Prioritize a streaming state contract and a benchmark-backed retention pilot
over another generic training-recipe sweep. The proposed 500-step head
alignment study remains unlaunched and is now secondary to this milestone.
It may be used as a bounded acquisition repair if the pilot cannot learn its
short-context task, rather than becoming the project's main research question.

Recent 0.84M experiments establish useful acquisition/format diagnostics.
Their 95-byte inputs fit inside attention128, so those results do not establish
cross-segment retention or an advantage of a biological/sleep update. The
latest constant/warm-up comparison has no fully admitted terminal (0/8).
Earlier typed key/value studies have explicit carried states and stronger
component controls, but do not replace the intended raw-sequence comparison.
These completed results and their archives remain unchanged.

## The state that must persist

Separate fixed learned parameters from episode state. At test time, the slow
model parameters stay fixed while fast memory evolves as the stream arrives.
The data-independent learned persistent tokens are a different component;
their presence alone does not answer the TA's question about storing new facts.
The Titans paper's sections3.1 and4.1 describe online associative updates and
using updated memory in later segments. The inspected version is arXiv v1;
this note is not a fresh full-paper reproduction audit.
[Titans v1](https://arxiv.org/html/2501.00663v1).

A useful implementation-neutral contract is

```
output_s, state_(s+1) = process(segment_s, state_s)
```

The next segment receives the returned state, rather than the initializer.
This statement specifies the data flow; the actual write rule, read timing,
decay and consolidation schedule still need their exact method definitions.

The inspected vendored `NeuralMemState` contains `seq_index`, `weights`,
`cache_store_segment`, `cache_store_mask`, `states` and `updates`.
`NeuralMemory.forward(state=...)` carries the fast-weight/update state;
`MemoryAsContextTransformer.forward(cache=..., return_cache=True)` includes
neural-memory state in its returned cache. `mem_state_detach` detaches tensor
graphs while preserving tensor values. These APIs exist; their availability
does not prove semantic retention by our trained model or final adapters.

For the existing typed Hebbian reference, reads use
`W_effective = W0 + alpha * trace`; `write` returns a new state carrying the
updated trace. `detach_state`, export and import preserve that trace's values.
That is a concrete component persistence mechanism. It is not evidence that
Member B's intended full-sequence plastic layer has already been integrated.

The shared sequence contract must specify:

- Which fast weights/traces, momentum, partial-chunk buffers and position
  counters are carried, and which boundary initializes a new episode.
- Whether and how the computation graph is detached during training, without
  confusing detachment with zeroing memory.
- The causal read/write order and update source. Evaluation may use arriving
  input-derived associations, not future QA answers or hidden answer labels.
- Sleep-buffer state, replay placement, storage limits and consolidation writes.
  Reset at independent episode boundaries; do not reset at every segment.
- How state serialization binds to the exact slow checkpoint and configuration.

## First milestone: prove the state path, then measure retention

Zeyuan owns the common harness and evidence; A/B own their intended update
implementations. No GPU allocation is started by this planning update.

1. Audit vanilla streaming and introduce the same interface for adapters.
   Check split versus uninterrupted execution at supported chunk boundaries,
   save/resume equivalence, and a new episode's clean reset. Preserve partial
   chunks, positions and attention policy; a convenient API split is not
   automatically equivalent to a full forward pass.
2. Use a tiny learnable fact task as the acquisition positive control. Train
   explicitly for the chosen short-attention streaming setting. Do not relabel
   current attention128 weights as a trained native32 baseline.
3. Present facts once, then enough distractors that the evidence leaves the
   actual raw attention cache and, for a memory-only diagnostic, replay storage.
   Query without feeding the old prompt back. Track delay in both segments
   and actual bytes/tokens under the selected tokenizer.
4. Compare carried memory with a matched reset-memory intervention and a
   retrieval-disabled intervention. Keep attention/position/input handling
   identical; resetting the whole cache changes more than neural memory.
   A changed future logit establishes a functioning state path, not correct
   semantic recall by itself. Require acquisition and delayed-answer evidence.
5. Evaluate stable facts and changed facts separately, including stale-answer
   errors. A reset gap alone is not a general memory-quality result. Save raw
   predictions/state provenance and a fixed analysis plan before main runs.

Use two clearly labeled settings: full-system performance with bounded caches
and buffers, and a mechanistic fast-memory-only read after controlled cache/
buffer removal. For sleep, consolidation occurs before that removal. The
second setting is an ablation, not an unmodified official benchmark protocol.

## Same-size comparison

The main comparator must retain the unmodified Titans-style gradient-surprise
memory update in the same small architecture. Match the tokenizer, attention
window, memory width/depth/capacity, initial slow weights, training examples,
adaptation budgets and evaluation streams. Train each update rule under its
own dynamics with matched training opportunity; swapping a new rule only at
test time into vanilla-trained weights is a separate diagnostic.

Start with vanilla, intended Hebbian/plastic update and intended sleep update.
For sleep, also keep continuous replay with identical buffer capacity, sampled
records and extra writes; this separates replay timing from extra exposure.
Online-only is a useful lower-update anchor, not an automatically matched
compute control. Report learned parameter counts, fast-state/replay bytes,
write counts, wall time and measured compute differences for every method.
Do not add unused parameters merely to manufacture equal parameter counts.

Call the existing comparator an unmodified vendored/lucidrains-based Titans
reference until the paper-deviation checklist is verified. Same-size code
comparison does not establish paper-faithful MAC reproduction. In particular,
record the memory write inputs and read/write timing; the roadmap and
ambiguity register remain relevant. The original baseline tag is preserved.

## Benchmark foundations checked

These are source-backed candidates, not downloaded datasets or executed scores.
Pin exact revisions, splits, licenses and generation settings before use.

| Source | Verified coverage | Proposed role and limitation |
| --- | --- | --- |
| [RULER](https://github.com/NVIDIA/RULER) | Configurable retrieval with multiple keys/values/queries and distractor choices; variable tracking is also provided. | First engineering foundation for controlled retrieval and delay. Shorter lengths or byte-format rewrites must be labeled as adaptations; keep the official protocol separate. |
| [BABILong](https://github.com/booydar/babilong) | bAbI facts embedded among irrelevant PG19 text; tasks require tracking/reasoning over distributed facts. | Strong project-facing target; Titans v1 section5.4 also evaluates BABILong. Start with short-context acquisition and qa1, then increase distractor length; do not assume a0.84M byte model can already solve the official tasks. |
| [LongMemEval, original ICLR2025 benchmark](https://github.com/xiaowu0162/LongMemEval) | Explicit knowledge-update and temporal/multi-session memory categories with timestamped histories. | Relevant foundation for revisions. The chat-assistant setting is a larger capability jump; reserve as an extension unless a suitable language baseline exists. This refers to original LongMemEval, not its separate V2 benchmark. |

Our controlled revision/distractor suite remains useful for causal analysis.
If we add revisions to a benchmark that lacks that task, preserve and score
its original examples separately, define latest-truth semantics explicitly,
and call the new suite benchmark-derived rather than claiming an official
leaderboard score. Hidden supporting-fact labels stay in evaluation metadata.

## Concrete priority order and team choices

1. Zeyuan will validate streaming state, implement carry and reset checks,
   and qualify a short learnable baseline using the vanilla update.
2. Zeyuan will pin a RULER-derived retrieval pilot and a short BABILong
   acquisition subset and record protocol changes before training.
3. The team will integrate the intended sleep and plastic modules and run
   matched vanilla comparisons once the acquisition task is learnable.
4. Extend delays, interference and genuine revisions; use the result to decide
   whether a broader benchmark or a larger pretrained model is justified.

Team choices to settle are the precise plastic update/decay equation and
state boundary, sleep's write target and replay access, the primary benchmark
and whether its official or adapted protocol is the claim, and any genuine
parameter/storage differences. Do not resolve these by silently changing
current failed admission gates. Standard behavioral defaults and a vanilla
engineering harness can progress before all method choices are settled.

The next report should state the explicit persistent state/update equation,
baseline and evaluation protocol, then distinguish measured progress from
planned work. It should not present acquisition-only plots as retention
evidence. This note does not draft or submit that report, edit the team's
Google Doc, send a TA reply, or schedule a meeting.
