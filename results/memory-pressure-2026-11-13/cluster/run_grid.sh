#!/bin/bash
set -euo pipefail
base=/insomnia001/depts/edu/users/zc2819/clmm/pressure-2026-11-13
python=/insomnia001/depts/edu/users/zc2819/clmm/replay-2026-11-12/venv/bin/python
seed=${1:?Specify training initialization seed}
case "$seed" in 48101|48103|48105) ;; *) exit 2 ;; esac
cd "$base/repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$python" scripts/memory_pressure_probe.py \
  --checkpoints "$base/train-curriculum-$seed/checkpoint.pt" \
  --out "$base/grid-$seed" --device cuda --facts 16 64 256 \
  --delays 0 64 256 1024 --episodes 256 --batch-size 64 --seed 58111 \
  --aux-decay-scale 0 --buffer-size 32 --block-size 8 --design "$base/design.json"
