# Member C project update · November 17, 2026

The next shared milestone is a streaming evaluator and a same-size Titans
baseline that can recall facts after they leave the attention window. That
gives the sleep and Hebbian tracks a common comparison for their memory
updates and directly addresses the TA feedback.

| Completed work | Result and scope |
| --- | --- |
| Replay freshness on the typed key/value reference | At a delay of 128 arrivals, revised-fact accuracy increased from 83.46% with literal replay to 93.93% with bounded freshness at equal write budgets. Equal-write repeat-current scored 94.52%. This motivates a stale-replay control in the sleep study. |
| Record selection and value decoding on the 839,248-parameter byte model | All four decoders passed the acquisition gate when given the correct record index. Ordinary key-to-value decoding passed in zero of four runs. The explicit index is a positive control for value decoding. |
| Paired optimizer experiment on the same model | Across four initializer pairs at 6,000 updates per run, 500-step warm-up raised familiar-wording development recall from 66.31% to 71.58% and reduced new-wording recall from 24.51% to 17.87%. Zero of eight terminal models passed all acquisition gates. Constant LR remains the reference recipe. |

The latest sequence evaluation measures joint recall of both answers across
original and reversed records on 256 fresh development worlds per case.
Both records were visible inside 128-byte attention, so the scope of those
scores is acquisition and wording transfer. The typed reference runs have
explicit carried state and support component controls.

Zeyuan will prepare a common state interface and test carry, reset and
save/resume. The first retention test will present facts once, move them
outside attention, and compare carried fast weights with a matched reset.
The main comparator will use the vendored Titans gradient-surprise update
with matched model size and training opportunity. We will record fast-state
bytes, replay storage and writes, including a matched continuous replay
control for the sleep schedule.

The retrieval pilot will use configurable multi-key retrieval and distractors
from RULER as a foundation. A short BABILong task is the next benchmark
target once the acquisition control works. Adapted tasks will carry their
own protocol and scores. [RULER](https://github.com/NVIDIA/RULER) ·
[BABILong](https://github.com/booydar/babilong)

Team main at `fbb267e` includes Hebbian memory, padding and the causal
segment mask. The full MAC forward pass and switching are in progress.
`HebbianMemory` reads from the incoming trace and returns the next trace
after writing. Our component check reproduced the next read and trace
exactly after save/resume, and resetting the trace changed the read.

The current Hebbian design makes the persistent context slots plastic.
The gradient-based long-term memory stays in place. Use vanilla fixed slots
and a reset within the plastic system to measure the role of trace updates.
The Titans gradient reference and the carried-trace test are available
as separate review patches.
