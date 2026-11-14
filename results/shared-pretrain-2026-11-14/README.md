# Shared sequence-model pilot

See [research notes](../../docs/SHARED_PRETRAIN.md) for scope and results. `analysis/` contains independently checked metrics and paired-window intervals, with separate execution, baseline-adequacy and memory-context decisions. The two seeds share their training and development arrays; they are not pooled.

The CPU execution amendment was written before either full pilot. `cluster-observations.json` separately records the interrupted/unverified GPU diagnostic and cleanup evidence. The prior failed two-seed replay screen remains unchanged.

Raw byte-window/offset arrays, checkpoints, optimizer states, source snapshots, failures, plots and the full Git bundle are in `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-shared-pretrain-2026-11-14` and the adjacent `CLMM_Shared_Pretrain_2026-11-14.zip`. Large weights are intentionally outside Git. Source for both pilots is `8cc20ac`; the later handoff commit only records evidence.
