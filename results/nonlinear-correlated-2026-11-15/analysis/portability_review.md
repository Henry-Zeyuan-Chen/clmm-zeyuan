# Post-run checkpoint compatibility review

The frozen implementation and results remain at `b5f27ff4738ded4832538573d454d2461786d38b`. A cross-environment loader issue was discovered during artifact review, not a model-training failure: the same CPU Gaussian construction normalized under the run's PyTorch 2.11 environment and local PyTorch 2.15 differs by at most 2.98e−8 per coordinate. This changes the canonical tensor hash, so the frozen constructor's exact regenerated-hash check rejects the saved configuration locally.

The saved rank-32 codebook hash is `701cbfb55a6be11487bb02b21a2392374a361bd2b4e003225e4b72b0acaceb0a`. These saved literal tensor bits are the representation used for every reported model result. Replacing them with locally regenerated vectors would change the representation and is not an acceptable reload fix.

Reviewed the separate post-run `clmm/pretrain/kv_correlated_portable.py` and its tests. Source SHA256: `788d89b01139a689bb2a401f925016018a9bd5c570afa89886637602b3ac9837`. No blocker was identified:

- Loading preserves saved key/value tensors exactly and validates their canonical hash, dtype, shape, finiteness, zero padding, unit norm and seeded Gaussian construction. The 2e−7 absolute / 1e−6 relative tolerance applies only to numerical-construction validation. It does not renormalize or change a saved element.
- The method name, configuration JSON/fingerprint, architecture and inherited write/read mechanism match the frozen state contract. Saved fast payloads retain their exact codebook identity.
- CPU initialization can alias frozen model-template matrices and initial-memory matrices. The compatibility constructor separates their storage without changing values or RNG consumption, so strict loading preserves both distinct saved copies.
- Source tests cover slight normalization differences, literal-bit/state handoff, frozen/trainable copy separation, and rejection of material geometry/hash/value/dtype corruption.

I inspected coordinator-run verification artifacts, rather than rerunning the model. `gpu-recheck/verification.json` reports 2,432 selected queries and genuine streaming resumes under the frozen strict loader with maximum score error 0. `cpu-portable-recheck/verification.json` reports the same query count under PyTorch 2.15 CPU and this portable loader with maximum error 3.5762786865234375e−7. Both report pure reads and unchanged slow parameters/RNG. The CPU verification's loader hash matches the reviewed source.

This is a compatibility/inference check, not retraining, a new method, a hyperparameter change, or full-grid numerical reproduction. Preserve the frozen source snapshots, original failed reload record, portable helper/tests and both verification artifacts separately. The original result interpretation and model selection are unchanged.

An additional portable-loader GPU check was attempted after these completed checks, but SSH disconnected/timeouts prevented observing its result or scheduler closure. Its status is **unknown**. This review does not claim it passed or that the allocation was released. It is separate from the eight successful main invocations and the observed strict-GPU/portable-CPU verification records above.
