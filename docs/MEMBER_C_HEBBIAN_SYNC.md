# Hebbian memory and MAC host sync

Team main at `fbb267e` contains the Hebbian memory and routing modules, config
validation, segment padding, and a causal mask that prepends context columns.
It provides the building blocks for the MAC host. The host level forward pass,
choice of write inputs, and switching interface are still in progress.

The Hebbian component reads from the incoming state before writing the current
segment. It returns a position counter and traces, and the next call uses them
to form the effective weights `base + alpha * traces`. Detaching cuts gradient
history and preserves both values. The component uses physical segment width
for its cursor, including padding. Delay and write counts in the evaluator
should separately count valid content. Byte zero is valid under the existing
256-byte codec, so the byte runner should carry an explicit validity mask
or use a reserved padding ID with documented vocabulary accounting.

The new component test uses two fixed segments and a mask on the first write.
The second read and post-write traces match exactly after saving and restoring
the state. A trace reset at the same cursor changes the second read, and the
episode initializer stays zero. This uses fixed slow weights on CPU and
establishes that carried traces can affect the next read. Semantic recall and
end-to-end host behavior still require their own evaluation.

The team design makes the persistent context slots plastic and keeps the
vendored adaptive long-term memory. For the Hebbian comparison, keep that
long-term path and the attention and training conditions fixed. Compare
vanilla fixed persistent slots with the plastic slots at a matched model
budget. Within the Hebbian host, compare carried traces with a matched reset
or disabled writes. Report realized learned parameter counts and carried bytes
in each case. A claim about replacing the gradient update would require a
separate experiment that actually changes that update.

The first host check should run the same past stream and next query across
carry, resume, and reset modes. Verify position and mask equality at the
boundary, future-token causality, and exact same-environment resume. Follow
with an acquisition check and out-of-window delayed questions, where the
target facts are absent from raw attention. The MAC host owner can keep
developing the switching interface while Member C prepares those evaluations.

Validation ran 48 current Hebbian, host, streaming, and Titans reference
checks, plus 61 local adapter checks. One adapter check skipped because its
optional checkpoint path was absent in this worktree.
