"""Conditional K2 task-learnability diagnostic with 32- or 128-byte attention.

Both arms start from the primary shared byte checkpoint and train on identical
K2/D0/no-revision inputs. Only the attention segment geometry changes between
arms. This supplemental diagnostic is not a replay baseline or a replacement
for failed acquisition seeds; it does not apply the v1 memory acquisition gate.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import fact_revision_experiment as common
import numpy as np
import torch
import torch.nn.functional as F

from clmm.pretrain import build_model, load_checkpoint, save_checkpoint
from clmm.pretrain.fact_streams import RECORD_BYTES, QUERY_PREFIX_BYTES, encode_example, make_episode


INPUT_LENGTH = 2 * RECORD_BYTES + QUERY_PREFIX_BYTES
PRIMARY_INIT_SEED = 68101


def make_training_inputs(steps, batch_size, train_seed, *, deadline=None):
    """Materialize geometry-independent paired inputs and every sampling choice."""
    if any(type(value) is not int or value < 1 for value in (steps, batch_size)):
        raise ValueError("steps and batch_size must be positive integers")
    if type(train_seed) is not int or train_seed < 0:
        raise ValueError("train_seed must be a nonnegative integer")
    rng = np.random.default_rng(train_seed)
    inputs = np.empty((steps, batch_size, INPUT_LENGTH), dtype=np.uint8)
    targets = np.empty((steps, batch_size), dtype=np.uint8)
    episode_seeds = np.empty((steps, batch_size), dtype=np.int64)
    query_indices = np.empty((steps, batch_size), dtype=np.uint8)
    for step in range(steps):
        if deadline is not None:
            common.before_deadline(deadline)
        seeds = rng.integers(1, 1_000_000_000, size=batch_size)
        indices = rng.integers(0, 2, size=batch_size)
        for index, (seed, query_index) in enumerate(zip(seeds, indices)):
            episode = make_episode(int(seed), 2, 0, "train", revision_fraction=0., include_recent=False)
            prefix, target, _ = encode_example(episode, episode.queries[int(query_index)], "online")
            if len(prefix) != INPUT_LENGTH:
                raise AssertionError("K2 online examples must be exactly 95 input bytes")
            inputs[step, index] = np.frombuffer(prefix, dtype=np.uint8)
            targets[step, index] = target
        episode_seeds[step] = seeds
        query_indices[step] = indices
    return dict(input_bytes=inputs, targets=targets, episode_seeds=episode_seeds, query_indices=query_indices)


def training_data_sha256(data):
    """Hash array names, types, shapes and bytes independently of NPZ packaging."""
    digest = hashlib.sha256()
    for name in sorted(data):
        array = np.ascontiguousarray(data[name])
        descriptor = json.dumps([name, array.dtype.str, list(array.shape)], separators=(",", ":"))
        digest.update(descriptor.encode("ascii") + b"\n")
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_control_model(checkpoint, segment_len, device="cpu"):
    """Strict-copy the primary parent's tensors into the requested geometry."""
    if segment_len not in (32, 128) or type(segment_len) is not int:
        raise ValueError("segment_len must be 32 or 128")
    parent, parent_config, payload = load_checkpoint(checkpoint, map_location="cpu")
    info = payload["training_info"]
    if (info.get("init_seed") != PRIMARY_INIT_SEED or info.get("task") is not None
            or info.get("parent_checkpoint_sha256") is not None):
        raise ValueError("Use the original primary shared checkpoint with init_seed 68101, not a task-adapted checkpoint")
    if (parent_config.segment_len, parent_config.memory_segment_len, parent_config.memory_batch_size) != (32, 8, 32):
        raise ValueError("Parent must have shared 32/8/32 attention/memory geometry")
    config = replace(parent_config, segment_len=segment_len)
    # Building a new module must not consume the training RNG differently by arm.
    with torch.random.fork_rng(devices=[]):
        model = build_model(config)
    model.load_state_dict(parent.state_dict(), strict=True)
    parent_state, control_state = parent.state_dict(), model.state_dict()
    identical = (parent_state.keys() == control_state.keys()
                 and all(torch.equal(value, control_state[key]) for key, value in parent_state.items()))
    parent_count = sum(parameter.numel() for parameter in parent.parameters())
    control_count = sum(parameter.numel() for parameter in model.parameters())
    if not identical or parent_count != control_count:
        raise AssertionError("Control initialization must exactly preserve parent tensors and parameter count")
    changes = [name for name, value in asdict(parent_config).items() if value != asdict(config)[name]]
    if changes != ([] if segment_len == 32 else ["segment_len"]):
        raise AssertionError("Only the attention segment_len may change")
    metadata = dict(parent_config=asdict(parent_config), control_config=asdict(config),
                    changed_config_fields=changes, initial_state_tensors_identical=True,
                    parent_parameter_count=parent_count, parameter_count=control_count,
                    primary_parent_init_seed=info["init_seed"],
                    attention_segment_len=segment_len,
                    record_bytes=RECORD_BYTES, training_input_length=INPUT_LENGTH,
                    local_attention_contains_entire_training_example=INPUT_LENGTH <= segment_len,
                    memory_reset_expected_noop_on_training_length=INPUT_LENGTH <= segment_len)
    return model.to(device), config, metadata


def _save_training_inputs(data, out):
    np.savez_compressed(out / "training_inputs.npz", **data)
    plan = [dict(step=step + 1, facts=2, delay=0, revision_fraction=0., split="train",
                 condition="online", input_length=INPUT_LENGTH,
                 episode_seeds=data["episode_seeds"][step].tolist(),
                 query_indices=data["query_indices"][step].tolist())
            for step in range(len(data["input_bytes"]))]
    common.json_file(out / "training_plan.json", plan)


def train_control(model, config, args, out, device, deadline, training_data, *, source_info=None):
    """Train exactly the fixed step count; persist planned inputs and failures."""
    inputs, targets = training_data["input_bytes"], training_data["targets"]
    if (inputs.shape != (args.steps, args.batch_size, INPUT_LENGTH)
            or targets.shape != (args.steps, args.batch_size)
            or inputs.dtype != np.uint8 or targets.dtype != np.uint8):
        raise ValueError("Training arrays do not match fixed K2 batch geometry")
    if not np.all((targets >= ord("a")) & (targets <= ord("p"))):
        raise ValueError("Training targets must be legal answer bytes")
    _save_training_inputs(training_data, out)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    completed, succeeded, rows = 0, False, []
    started = time.perf_counter()
    parent_hash = common.sha256(args.checkpoint)
    try:
        model.train()
        with (out / "training.csv").open("w", newline="", encoding="utf8") as handle:
            writer = None
            for step in range(args.steps):
                common.before_deadline(deadline)
                x = torch.tensor(inputs[step].astype(np.int64), device=device)
                y = torch.tensor(targets[step].astype(np.int64), device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)[:, -1]
                if logits.shape != (args.batch_size, 256) or not torch.isfinite(logits).all():
                    raise RuntimeError("Training requires finite full-vocabulary answer logits")
                loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                common.sync(device)
                completed = step + 1
                row = dict(step=completed, facts=2, delay=0, revision_fraction=0., batch=args.batch_size,
                           input_length=INPUT_LENGTH, attention_segment_len=config.segment_len,
                           input_bytes=int(x.numel()), supervised_answers=args.batch_size,
                           loss=float(loss.detach()), accuracy=float((logits.argmax(-1) == y).float().mean()),
                           gradient_norm=float(norm), elapsed_seconds=time.perf_counter() - started)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                if completed == 1 or completed % 50 == 0 or completed == args.steps:
                    print(json.dumps(row), flush=True)
        common.before_deadline(deadline)
        training_info = dict(task="fact-attention-control", primary_parent_init_seed=PRIMARY_INIT_SEED,
                             steps=completed, parent_checkpoint_sha256=parent_hash,
                             train_seed=args.train_seed, fixed_final_checkpoint=True,
                             optimizer="AdamW", lr=args.lr, weight_decay=.01, gradient_clip=1.,
                             facts=2, delay=0, revision_fraction=0., condition="online",
                             input_length=INPUT_LENGTH, answer_vocabulary=256,
                             attention_segment_len=config.segment_len,
                             record_bytes=RECORD_BYTES, training_data_sha256=training_data_sha256(training_data))
        digest = save_checkpoint(out / "checkpoint.pt", model, config,
                                 source_info=source_info, training_info=training_info)
        succeeded = True
        return dict(completed_steps=completed, output_checkpoint_sha256=digest,
                    training_input_sha256=common.sha256(out / "training_inputs.npz"),
                    training_data_sha256=training_data_sha256(training_data),
                    training_plan_sha256=common.sha256(out / "training_plan.json"),
                    training_seconds=time.perf_counter() - started,
                    input_bytes=sum(row["input_bytes"] for row in rows),
                    supervised_answers=completed * args.batch_size)
    finally:
        torch.save(optimizer.state_dict(), out / "optimizer.pt")
        common.json_file(out / "training_progress.json", dict(completed_steps=completed,
                         planned_steps=args.steps, fixed_final_checkpoint_saved=succeeded))
        if not succeeded:
            torch.save(model.state_dict(), out / "partial_state.pt")


def label_reset_diagnostic(datasets, segment_len):
    """Label reset interpretation while preserving the common evaluation data."""
    return [dict(cell=dataset["spec"]["cell"], family=dataset["spec"]["family"],
                 attention_segment_len=segment_len, input_lengths=sorted({len(sample["prefix"]) for sample in dataset["samples"]}),
                 reset_expected_noop=all(len(sample["prefix"]) <= segment_len for sample in dataset["samples"]),
                 interpretation=("Within one attention segment: memory reset is expected to be a no-op."
                                 if all(len(sample["prefix"]) <= segment_len for sample in dataset["samples"])
                                 else "Crosses attention segments: memory reset removes cross-segment memory state."))
            for dataset in datasets]


def validate_args(args):
    for name in ("steps", "batch_size", "eval_batch_size", "gate_episodes", "grid_episodes",
                 "primary_episodes", "max_seconds", "threads"):
        value = getattr(args, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if args.segment_len not in (32, 128) or type(args.segment_len) is not int:
        raise ValueError("segment_len must be 32 or 128")
    if args.suite != "gate" or args.mode != "attention_control":
        raise ValueError("This supplemental diagnostic only supports the gate evaluation suite")
    if args.block_size != 4 or args.buffer_size != 8:
        raise ValueError("Shared evaluation uses block_size=4 and buffer_size=8")
    if not math.isfinite(args.lr) or args.lr != .0005:
        raise ValueError("The fixed control learning rate is 0.0005")
    for name in ("train_seed", "dev_seed"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")


def run(args):
    validate_args(args)
    if args.out.exists():
        raise FileExistsError("--out must be a NEW directory")
    started = time.perf_counter()
    deadline = started + args.max_seconds
    out, device, model = args.out.resolve(), torch.device(args.device), None
    meta = dict(mode=args.mode, status="initializing", parent_checkpoint_path=str(args.checkpoint.resolve()),
                attention_segment_len=args.segment_len, v1_memory_gate_applied=False,
                started_utc=datetime.now(timezone.utc).isoformat(),
                args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    try:
        # Exactly one initialization in this process; evaluation does not re-run
        # thread setup (set_num_interop_threads cannot be called twice).
        out, device, meta = common.initialize(args)
        driver_name = Path(__file__).relative_to(REPO).as_posix()
        driver_snapshot = out / "sources" / driver_name
        driver_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), driver_snapshot)
        meta["source_sha256"][driver_name] = common.sha256(driver_snapshot)
        parent_hash = common.sha256(args.checkpoint)
        meta.update(parent_checkpoint_path=str(args.checkpoint.resolve()), parent_checkpoint_sha256=parent_hash,
                    source_commit=meta["repo_commit"], driver_source_snapshot=driver_name,
                    driver_source_sha256=meta["source_sha256"][driver_name],
                    source_snapshot_is_reconstructable=True, output_checkpoint_sha256=None,
                    conditional_supplemental_diagnostic=True, replaces_primary_result=False,
                    v1_memory_gate_applied=False,
                    interpretation="K2 task learnability and local attention access diagnostic; not a fair replay baseline.")
        common.json_file(out / "metadata.json", meta)
        common.before_deadline(deadline)
        model, config, initialization = build_control_model(args.checkpoint, args.segment_len, device)
        meta.update(initialization)
        torch.save({name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                   out / "initial_state.pt")
        meta["initial_state_sha256"] = common.sha256(out / "initial_state.pt")
        common.json_file(out / "metadata.json", meta)
        training_data = make_training_inputs(args.steps, args.batch_size, args.train_seed, deadline=deadline)
        source_info = dict(repo_commit=meta["source_commit"], source_sha256=meta["source_sha256"],
                           supplemental_diagnostic="fact_attention_control")
        meta.update(train_control(model, config, args, out, device, deadline, training_data, source_info=source_info))
        common.before_deadline(deadline)
        datasets = common.build_evaluation(args, out)
        reset_labels = label_reset_diagnostic(datasets, config.segment_len)
        common.json_file(out / "reset_diagnostic.json", reset_labels)
        meta["reset_diagnostic"] = reset_labels
        meta.update(common.evaluate(model, datasets, args, out, device, deadline))
        if common.sha256(args.checkpoint) != parent_hash:
            raise AssertionError("Parent checkpoint changed during the control")
        if common.sha256(out / "checkpoint.pt") != meta["output_checkpoint_sha256"]:
            raise AssertionError("Fixed final checkpoint changed during evaluation")
        for name, digest in meta["source_sha256"].items():
            if common.sha256(REPO / name) != digest or common.sha256(out / "sources" / name) != digest:
                raise AssertionError(f"Source changed during run: {name}")
        common.before_deadline(deadline)
        meta["status"] = "succeeded"
    except BaseException as error:
        meta.update(status="failed", failure_type=type(error).__name__, failure=str(error), traceback=traceback.format_exc())
        if out.exists() and model is not None:
            torch.save(model.state_dict(), out / "partial_state.pt")
        if (out / "training_progress.json").exists():
            meta["completed_steps"] = json.loads((out / "training_progress.json").read_text())["completed_steps"]
        raise
    finally:
        if out.exists():
            meta["elapsed_seconds"] = time.perf_counter() - started
            meta["ended_utc"] = datetime.now(timezone.utc).isoformat()
            if device.type == "cuda" and torch.cuda.is_available():
                meta["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            common.json_file(out / "metadata.json", meta)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.set_defaults(mode="attention_control", suite="gate", grid_episodes=32, primary_episodes=128,
                        block_size=4, buffer_size=8)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--design", type=Path)
    result.add_argument("--segment-len", type=int, choices=(32, 128), required=True)
    result.add_argument("--steps", type=int, default=1000)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--eval-batch-size", type=int, default=16)
    result.add_argument("--lr", type=float, default=.0005)
    result.add_argument("--train-seed", type=int, default=78122)
    result.add_argument("--dev-seed", type=int, default=78123)
    result.add_argument("--gate-episodes", type=int, default=64)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--max-seconds", type=int, default=600)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
