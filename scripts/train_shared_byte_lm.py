"""Bounded training of the shared lucidrains byte-level sequence baseline."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
import torch.nn.functional as F
from clmm.pretrain import (ByteLMConfig, build_model, fixed_windows, load_checkpoint,
                           load_enwik8, sample_windows, save_checkpoint,
                           zero_memory_output, reset_memory_at_segments)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def capture_sources(out):
    paths = [*sorted((REPO / "clmm/pretrain").glob("*.py")),
             *sorted((REPO / "titans_pytorch").glob("*.py")),
             REPO / "clmm/__init__.py", Path(__file__), REPO / "pyproject.toml",
             REPO / "requirements-lock.txt"]
    hashes = {}
    for source in paths:
        name = source.relative_to(REPO).as_posix()
        destination = out / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[name] = sha256(destination)
    return hashes


@torch.no_grad()
def evaluate(model, windows, offsets, *, length, stage, condition, batch_size, device, deadline=math.inf):
    model.eval()
    if condition == "normal":
        context = nullcontext()
    elif condition == "zero_memory_output":
        context = zero_memory_output(model)
    elif condition == "reset_memory_at_segments":
        context = reset_memory_at_segments(model)
    else:
        raise ValueError(condition)
    rows = []
    with context:
        for start in range(0, len(windows), batch_size):
            if time.perf_counter() >= deadline:
                raise TimeoutError("Evaluation wall-clock limit reached")
            data = windows[start:start + batch_size, :length + 1].to(device)
            logits = model(data[:, :-1])
            if logits.shape[:2] != data[:, :-1].shape or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid heldout logits")
            losses = F.cross_entropy(logits.transpose(1, 2), data[:, 1:], reduction="none").double()
            if not torch.isfinite(losses).all():
                raise RuntimeError("Nonfinite heldout next-byte loss")
            for index, per_byte in enumerate(losses.cpu()):
                tail = per_byte[256:]
                rows.append(dict(stage=stage, condition=condition, length=length,
                                 window=start + index, offset=int(offsets[start + index]),
                                 bytes=length, nll_sum=float(per_byte.sum()),
                                 late_bytes=len(tail), late_nll_sum=float(tail.sum())))
    return rows


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["stage"], row["condition"], row["length"])
        group = groups.setdefault(key, dict(stage=key[0], condition=key[1], length=key[2],
                                           windows=0, bytes=0, nll_sum=0., late_bytes=0,
                                           late_nll_sum=0.))
        group["windows"] += 1
        for field in ("bytes", "nll_sum", "late_bytes", "late_nll_sum"):
            group[field] += row[field]
    for group in groups.values():
        group["nll_per_byte"] = group["nll_sum"] / group["bytes"]
        group["bits_per_byte"] = group["nll_per_byte"] / math.log(2)
        group["late_nll_per_byte"] = group["late_nll_sum"] / group["late_bytes"] if group["late_bytes"] else None
    return list(groups.values())


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if min(args.steps, args.batch_size, args.eval_windows, args.eval_batch_size, args.seq_len) <= 0:
        raise ValueError("Counts and lengths must be positive")
    if args.seq_len != 256:
        raise ValueError("This pilot fixes the training length at 256 bytes")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Use an empty run directory: {out}")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    config = ByteLMConfig()
    command = [sys.executable, *sys.argv]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    source_hashes = capture_sources(out)
    if args.design:
        shutil.copy2(args.design, out / "design.json")
    meta = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                command=command, repo_commit=commit, source_sha256=source_hashes,
                configuration={**vars(args), "out": str(out), "data": str(args.data.resolve()),
                               "design": str(args.design.resolve()) if args.design else None},
                model_config=asdict(config),
                design_sha256=sha256(args.design) if args.design else None,
                environment=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                                 cuda=torch.version.cuda, device=str(device),
                                 device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                                 numpy=np.__version__, dtype="float32", deterministic_algorithms=True,
                                 tf32=False, cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"]),
                selection="Fixed final state; no dev-selected checkpoint. All failures retained.",
                interpretation="Shared integration pilot. Ablations are inference interventions, not separately trained architecture comparisons.",
                state_protocol="Reset between windows; causal within-window updates. No cross-session state is claimed.")
    meta["environment"]["packages"] = {name: importlib.metadata.version(name) for name in
        ("einops", "einx", "tensordict", "x-transformers", "assoc-scan", "hyper-connections")}
    save_json(out / "metadata.json", meta)
    started = time.perf_counter()
    deadline = started + args.max_seconds
    model = None
    completed = 0
    evaluation_rows = []
    try:
        corpus = load_enwik8(args.data)
        long_windows, dev_offsets = fixed_windows(corpus, "dev", 1024, args.eval_windows, args.dev_seed)
        np.savez_compressed(out / "dev_windows.npz", tokens=long_windows.numpy().astype(np.uint8),
                            offsets=np.asarray(dev_offsets, dtype=np.int64))
        meta["data"] = dict(path=str(args.data.resolve()), compressed_sha256=sha256(args.data),
                            raw_sha256=corpus.sha256, bytes=len(corpus.data), split_bounds=corpus.split_bounds,
                            dev_windows_sha256=sha256(out / "dev_windows.npz"), test_evaluated=False)
        torch.manual_seed(args.init_seed)
        model = build_model(config).to(device)
        meta["parameters"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        generator = torch.Generator().manual_seed(args.train_seed)
        evaluation_rows.extend(evaluate(model, long_windows, dev_offsets, length=256, stage="initial",
                                       condition="normal", batch_size=args.eval_batch_size, device=device, deadline=deadline))
        training_offsets = []
        training_start = time.perf_counter()
        memory_gradient_observed = False
        with (out / "training.csv").open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "predicted_bytes", "loss", "gradient_norm", "elapsed_seconds"])
            writer.writeheader()
            for step in range(1, args.steps + 1):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Per-run wall-clock limit reached")
                data, offsets = sample_windows(corpus, "train", args.seq_len, args.batch_size, generator)
                training_offsets.append(np.asarray(offsets, dtype=np.int64))
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = model(data.to(device), return_loss=True)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                memory_gradient_observed |= any("memory_model" in name and p.grad is not None and
                    bool(p.grad.abs().sum() > 0) for name, p in model.named_parameters())
                optimizer.step()
                synchronize(device)
                completed = step
                row = dict(step=step, predicted_bytes=step * args.batch_size * args.seq_len,
                           loss=float(loss.detach()), gradient_norm=float(grad_norm),
                           elapsed_seconds=time.perf_counter() - training_start)
                writer.writerow(row)
                handle.flush()
                if step == 1 or step % 50 == 0 or step == args.steps:
                    print(json.dumps(row), flush=True)
        np.save(out / "training_offsets.npy", np.stack(training_offsets))
        meta["training_seconds"] = time.perf_counter() - training_start
        meta["memory_gradient_observed"] = memory_gradient_observed
        if not memory_gradient_observed:
            raise RuntimeError("No nonzero gradient observed for neural-memory model parameters")
        checkpoint_info = dict(repo_commit=commit, source_sha256=source_hashes, data=meta["data"])
        info = dict(init_seed=args.init_seed, train_seed=args.train_seed, dev_seed=args.dev_seed,
                    steps=completed, predicted_bytes=completed * args.batch_size * args.seq_len,
                    optimizer="AdamW", lr=args.lr, weight_decay=args.weight_decay,
                    selection=meta["selection"])
        meta["checkpoint_sha256"] = save_checkpoint(out / "checkpoint.pt", model, config,
                                                     source_info=checkpoint_info, training_info=info)
        torch.save(optimizer.state_dict(), out / "optimizer.pt")
        model.eval()
        if time.perf_counter() >= deadline:
            raise TimeoutError("Wall-clock limit reached before reload checks")
        with torch.no_grad():
            sample = long_windows[:2, :256].to(device)
            expected = model(sample)
            restored, restored_config, payload = load_checkpoint(out / "checkpoint.pt", map_location=device,
                                                                 expected_config=config)
            restored.eval()
            actual = restored(sample)
            meta["checkpoint_reload_max_abs_difference"] = float((actual - expected).abs().max())
            if not torch.equal(actual, expected):
                raise RuntimeError("Checkpoint reload logits are not exact")
            # A token's own output may change; earlier outputs must not.
            changed = sample.clone()
            changed[:, 127:] = (changed[:, 127:] + 17) % 256
            delta = float((model(changed)[:, :127] - expected[:, :127]).abs().max())
            meta["trained_suffix_causality_max_abs_difference"] = delta
            if delta > 1e-6:
                raise RuntimeError("Trained model violates prefix causality")
            del restored, actual, expected, payload
        frozen = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for length in (256, 1024):
            for condition in ("normal", "zero_memory_output", "reset_memory_at_segments"):
                evaluation_rows.extend(evaluate(model, long_windows, dev_offsets, length=length, stage="final",
                                               condition=condition, batch_size=args.eval_batch_size, device=device, deadline=deadline))
                if not all(torch.equal(frozen[k], v) for k, v in model.state_dict().items()):
                    raise RuntimeError("Evaluation changed outer parameters")
        write_rows(out / "evaluation.csv", evaluation_rows)
        save_json(out / "summary.json", aggregate(evaluation_rows))
        meta["outer_parameters_unchanged_during_evaluation"] = True
        meta["summary"] = aggregate(evaluation_rows)
        if not all(sha256(REPO / name) == digest and sha256(out / "sources" / name) == digest
                   for name, digest in source_hashes.items()):
            raise RuntimeError("Source changed during run")
        if time.perf_counter() >= deadline:
            raise TimeoutError("Wall-clock limit reached before completion")
        meta["status"] = "succeeded"
    except Exception as exc:
        meta.update(status="failed", exception=str(exc), traceback=traceback.format_exc())
        if evaluation_rows:
            write_rows(out / "evaluation.partial.csv", evaluation_rows)
        if model is not None:
            torch.save(model.state_dict(), out / "failed_or_partial_state.pt")
        if "training_offsets" in locals() and training_offsets:
            np.save(out / "training_offsets.partial.npy", np.stack(training_offsets))
        raise
    finally:
        synchronize(device)
        meta.update(completed_steps=completed, elapsed_seconds=time.perf_counter() - started,
                    ended_utc=datetime.now(timezone.utc).isoformat())
        if device.type == "cuda":
            meta.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        save_json(out / "metadata.json", meta)
    print(json.dumps(dict(status=meta["status"], out=str(out), steps=completed,
                         summary=meta["summary"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO / "data/enwik8.gz")
    parser.add_argument("--design", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--eval-windows", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--init-seed", type=int, default=68101)
    parser.add_argument("--train-seed", type=int, default=68102)
    parser.add_argument("--dev-seed", type=int, default=68103)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seconds", type=float, default=1200.)
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())
