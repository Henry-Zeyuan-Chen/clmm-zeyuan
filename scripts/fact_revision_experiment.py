"""Bounded task adaptation and paired replay evaluation of shared byte checkpoints.

This is Member C's evaluation reference: copied records traverse the complete
unchanged transformer. It does not implement a sleep-specific memory module.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import io
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
from clmm.pretrain import load_checkpoint, save_checkpoint, reset_memory_at_segments
from clmm.pretrain.fact_streams import (
    VALUES, KEY_POOL_HASHES, make_episode, encode_example, schedule_episode,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_file(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def before_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("Experiment wall-clock limit reached")


def capture_sources(out):
    sources = [*sorted((REPO / "clmm/pretrain").glob("*.py")),
               *sorted((REPO / "titans_pytorch").glob("*.py")),
               REPO / "clmm/__init__.py", Path(__file__), REPO / "pyproject.toml",
               REPO / "requirements-lock.txt"]
    hashes = {}
    for source in sources:
        name = source.relative_to(REPO).as_posix()
        destination = out / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[name] = sha256(destination)
    return hashes


def initialize(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.train_seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    meta = dict(status="running", mode=args.mode, started_utc=datetime.now(timezone.utc).isoformat(),
                command=[sys.executable, *sys.argv],
                repo_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                source_sha256=capture_sources(out), checkpoint_path=str(args.checkpoint.resolve()),
                checkpoint_sha256=sha256(args.checkpoint),
                args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                key_pool_sha256=KEY_POOL_HASHES,
                environment=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                                 platform=platform.platform(), device=str(device),
                                 device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                                 dtype="float32", deterministic_algorithms=True, tf32=False),
                interpretation="Synthetic byte association; raw-record replay reference, not a sleep module or natural-language benchmark.")
    if args.design:
        shutil.copy2(args.design, out / "design.json")
        meta["design_sha256"] = sha256(args.design)
    json_file(out / "metadata.json", meta)
    return out, device, meta


def training_spec(index, steps, rng):
    """Fixed curriculum by progress fraction, independent of observed losses."""
    fraction = index / steps
    if fraction < .2:
        return 1, 0, 0.0
    if fraction < .4:
        facts = int(rng.choice([1, 2]))
        return facts, int(rng.choice([0, 2])), float(rng.choice([0., .5]))
    facts = int(rng.choice([1, 2, 4, 8], p=[.15, .25, .35, .25]))
    return facts, int(rng.choice([0, 4, 8])), .5


def train(model, config, args, out, device, deadline):
    rng = np.random.default_rng(args.train_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rows, plan, input_chunks, targets, offsets = [], [], [], [], [0]
    model.train()
    start = time.perf_counter()
    trace = out / "training.csv"
    for step in range(args.steps):
        before_deadline(deadline)
        facts, delay, revision_fraction = training_spec(step, args.steps, rng)
        seeds = rng.integers(1, 1_000_000_000, size=args.batch_size)
        episodes = [make_episode(int(seed), facts, delay, "train", revision_fraction=revision_fraction)
                    for seed in seeds]
        query_indices = [int(rng.integers(len(ep.queries))) for ep in episodes]
        examples = [encode_example(ep, ep.queries[index], "online")
                    for ep, index in zip(episodes, query_indices)]
        x_np = np.stack([np.frombuffer(item[0], dtype=np.uint8) for item in examples])
        y_np = np.array([item[1] for item in examples], dtype=np.uint8)
        x = torch.tensor(x_np.astype(np.int64), device=device)
        y = torch.tensor(y_np.astype(np.int64), device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)[:, -1]
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        sync(device)
        row = dict(step=step + 1, facts=facts, delay=delay, revision_fraction=revision_fraction,
                   batch=args.batch_size, input_bytes=int(x.numel()), supervised_answers=args.batch_size,
                   loss=float(loss.detach()), accuracy=float((logits.argmax(-1) == y).float().mean()),
                   gradient_norm=float(norm), elapsed_seconds=time.perf_counter() - start)
        rows.append(row)
        with trace.open("a", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if step == 0:
                writer.writeheader()
            writer.writerow(row)
        plan.append(dict(step=step + 1, facts=facts, delay=delay, revision_fraction=revision_fraction,
                         episode_seeds=[int(seed) for seed in seeds], query_indices=query_indices,
                         input_length=x.shape[1]))
        input_chunks.append(x_np.reshape(-1))
        offsets.append(offsets[-1] + x_np.size)
        targets.append(y_np)
        if (step + 1) % 50 == 0 or step == 0 or step + 1 == args.steps:
            print(json.dumps(row), flush=True)
    np.savez_compressed(out / "training_inputs.npz", input_bytes=np.concatenate(input_chunks),
                        step_offsets=np.array(offsets, dtype=np.int64), targets=np.stack(targets))
    json_file(out / "training_plan.json", plan)
    torch.save(optimizer.state_dict(), out / "optimizer.pt")
    digest = save_checkpoint(out / "checkpoint.pt", model, config,
                             training_info=dict(task="fact-revision", steps=args.steps,
                                                parent_checkpoint_sha256=sha256(args.checkpoint),
                                                train_seed=args.train_seed, fixed_final_checkpoint=True))
    return dict(completed_steps=len(rows), output_checkpoint_sha256=digest,
                training_input_sha256=sha256(out / "training_inputs.npz"),
                training_plan_sha256=sha256(out / "training_plan.json"),
                input_bytes=sum(row["input_bytes"] for row in rows),
                supervised_answers=sum(row["supervised_answers"] for row in rows))


def counterfactual_episode(episode):
    """Bijective value relabeling; query words/keys stay fixed, labels must move."""
    mapping = dict(zip(VALUES, VALUES[1:] + VALUES[:1]))
    events = tuple(replace(event, value=mapping[event.value]) for event in episode.events)
    queries = tuple(replace(query, target=mapping[query.target],
                            old_value=mapping[query.old_value] if query.old_value else None)
                    for query in episode.queries)
    return replace(episode, events=events, queries=queries)


def cells(args):
    result = []
    for family in ("seen", "heldout"):
        for name, facts, rev in (("acquire_k1", 1, 0.), ("acquire_k2", 2, 0.),
                                 ("revise_k1", 1, 1.), ("counterfactual_k2", 2, 0.)):
            result.append(dict(cell=name, facts=facts, delay=0, revision_fraction=rev,
                               family=family, episodes=args.gate_episodes,
                               conditions=["online", "reset_memory"], all_queries=True))
    if args.suite == "full":
        for facts in (4, 8, 16):
            for delay in (0, 8, 32):
                primary = facts == 8 and delay == 8
                result.append(dict(cell=f"pressure_k{facts}_d{delay}", facts=facts, delay=delay,
                                   revision_fraction=.5, family="seen",
                                   episodes=args.primary_episodes if primary else args.grid_episodes,
                                   conditions=["online", "repeat_current", "continuous", "periodic"],
                                   all_queries=False))
    return result


def query_selection(episode, all_queries):
    if all_queries:
        return list(range(len(episode.queries)))
    # One key per category per episode, selected from seed without inspecting values or predictions.
    result = []
    rng = np.random.default_rng(episode.seed + 97)
    for category in ("stable", "revised", "recent"):
        indices = [i for i, query in enumerate(episode.queries) if query.category == category]
        if indices:
            result.append(int(rng.choice(indices)))
    return result


def build_evaluation(args, out):
    datasets = []
    for spec in cells(args):
        # Share semantic episodes across delays/templates and acquisition counterfactuals.
        group = 1 if spec["cell"] == "revise_k1" else (2 if spec["cell"].startswith("pressure") else 0)
        seed_rng = np.random.default_rng(args.dev_seed + spec["facts"] * 101 + group * 10007)
        episode_seeds = seed_rng.integers(2_000_000_000, 3_000_000_000, size=spec["episodes"])
        dataset = dict(spec=spec, episodes=[], samples=[])
        for index, seed in enumerate(episode_seeds):
            episode = make_episode(int(seed), spec["facts"], spec["delay"], "dev", spec["family"],
                                   spec["revision_fraction"])
            if spec["cell"] == "counterfactual_k2":
                episode = counterfactual_episode(episode)
            dataset["episodes"].append(episode.to_dict())
            queries = query_selection(episode, spec["all_queries"])
            continuous = schedule_episode(episode, "continuous", args.block_size, args.buffer_size)
            periodic = schedule_episode(episode, "periodic", args.block_size, args.buffer_size)
            repeat = schedule_episode(episode, "repeat_current", args.block_size, args.buffer_size)
            assert continuous.metadata["selected_replay_event_ids"] == periodic.metadata["selected_replay_event_ids"]
            assert continuous.metadata["extra_record_count"] == periodic.metadata["extra_record_count"] == repeat.metadata["extra_record_count"]
            for condition in spec["conditions"]:
                for query_index in queries:
                    query = episode.queries[query_index]
                    prefix, target, record = encode_example(
                        episode, query, "online" if condition == "reset_memory" else condition,
                        args.block_size, args.buffer_size)
                    plan = record["execution_plan"]
                    query_events = [event for event in plan if event["key"] == query.key]
                    same_target_replays = [event for event in query_events if event["kind"] == "replay"]
                    row = dict(cell=spec["cell"], family=spec["family"], facts=spec["facts"], delay=spec["delay"],
                               episode=index, episode_seed=int(seed), query_index=query_index,
                               key=query.key, category=query.category, target=target,
                               old_target=ord(query.old_value) if query.old_value else -1,
                               condition=condition, input_length=len(prefix), prefix_sha256=record["prefix_sha256"],
                               online_records=record["online_record_count"], extra_records=record["extra_record_count"],
                               replay_records=record["replay_record_count"],
                               stale_replays=record["stale_replay_count"],
                               target_replays=len(same_target_replays),
                               target_stale_replays=sum(event["stale"] for event in same_target_replays),
                               target_resident=record["target_in_final_buffer"],
                               online_age=record["online_events_since_latest_target"],
                               last_occurrence_age=record["records_since_last_target_key_record"],
                               last_observed_value=ord(episode.events[-1].value),
                               last_presented_value=ord(plan[-1]["value"]),
                               last_key_presented_value=ord(query_events[-1]["value"]))
                    dataset["samples"].append(dict(prefix=prefix, row=row, audit=record))
        datasets.append(dataset)
    # Compress complete semantic events, schedules and prefixes. No future source is passed to model.
    import gzip
    with (out / "evaluation_data.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf8", newline="\n") as handle:
                for dataset in datasets:
                    handle.write(json.dumps(dict(type="episodes", spec=dataset["spec"], episodes=dataset["episodes"])) + "\n")
                    for sample in dataset["samples"]:
                        handle.write(json.dumps(dict(type="sample", row=sample["row"],
                                                     prefix=sample["prefix"].decode("ascii"), audit=sample["audit"])) + "\n")
    with gzip.open(out / "evaluation_data.jsonl.gz", "rb") as handle:
        canonical_hash = hashlib.sha256(handle.read()).hexdigest()
    (out / "evaluation_data_canonical.sha256").write_text(canonical_hash + "\n", encoding="ascii")
    return datasets


def score_logits(logits, targets):
    if logits.ndim != 2 or logits.shape[-1] != 256 or not torch.isfinite(logits).all():
        raise RuntimeError("Invalid answer logits")
    losses = F.cross_entropy(logits, targets, reduction="none")
    log_probs = F.log_softmax(logits, dim=-1)
    legal = logits[:, ord(VALUES[0]):ord(VALUES[-1]) + 1]
    legal_mass = log_probs[:, ord(VALUES[0]):ord(VALUES[-1]) + 1].exp().sum(-1)
    legal_losses = F.cross_entropy(legal, targets - ord(VALUES[0]), reduction="none")
    if not torch.isfinite(losses).all() or not torch.isfinite(legal_losses).all():
        raise RuntimeError("Nonfinite answer loss")
    return dict(prediction=logits.argmax(-1), forced_prediction=legal.argmax(-1) + ord(VALUES[0]),
                nll=losses, forced_nll=legal_losses, legal_probability_mass=legal_mass)


@torch.no_grad()
def evaluate(model, datasets, args, out, device, deadline):
    model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    rows, runtimes, saved_logits = [], [], []
    for dataset in datasets:
        grouped = defaultdict(list)
        for sample in dataset["samples"]:
            grouped[(sample["row"]["condition"], len(sample["prefix"]))].append(sample)
        for (condition, length), samples in grouped.items():
            sync(device)
            started = time.perf_counter()
            context = reset_memory_at_segments(model) if condition == "reset_memory" else nullcontext()
            with context:
                for start in range(0, len(samples), args.eval_batch_size):
                    before_deadline(deadline)
                    batch = samples[start:start + args.eval_batch_size]
                    x = torch.tensor(np.stack([np.frombuffer(sample["prefix"], dtype=np.uint8)
                                                for sample in batch]).astype(np.int64), device=device)
                    targets = torch.tensor([sample["row"]["target"] for sample in batch], device=device)
                    logits = model(x)[:, -1]
                    values = {key: value.cpu().tolist() for key, value in score_logits(logits, targets).items()}
                    saved_logits.append(logits.cpu().numpy())
                    for i, sample in enumerate(batch):
                        row = dict(sample["row"])
                        row.update({key: value[i] for key, value in values.items()})
                        row.update(correct=int(row["prediction"] == row["target"]),
                                   forced_correct=int(row["forced_prediction"] == row["target"]),
                                   obsolete=int(row["old_target"] >= 0 and row["prediction"] == row["old_target"]))
                        rows.append(row)
            sync(device)
            timing = dict(cell=dataset["spec"]["cell"], family=dataset["spec"]["family"],
                          condition=condition, samples=len(samples), input_length=length,
                          elapsed_seconds=time.perf_counter() - started)
            runtimes.append(timing)
            print(json.dumps(timing), flush=True)
            write_csv(out / "predictions.csv", rows)
            np.save(out / "answer_logits.npy", np.concatenate(saved_logits))
    unchanged = all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    if not unchanged:
        raise AssertionError("Outer checkpoint state changed during evaluation")
    write_csv(out / "runtime.csv", runtimes)
    summary = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cell"], row["family"], row["category"], row["condition"])].append(row)
    for (cell, family, category, condition), group in groups.items():
        summary.append(dict(cell=cell, family=family, category=category, condition=condition,
                            queries=len(group), episodes=len({row["episode"] for row in group}),
                            accuracy=float(np.mean([row["correct"] for row in group])),
                            forced_accuracy=float(np.mean([row["forced_correct"] for row in group])),
                            obsolete_rate=float(np.mean([row["obsolete"] for row in group])),
                            nll=float(np.mean([row["nll"] for row in group])),
                            legal_probability_mass=float(np.mean([row["legal_probability_mass"] for row in group]))))
    write_csv(out / "summary.csv", summary)
    return dict(evaluation_queries=len(rows), checkpoint_unchanged=unchanged,
                evaluation_data_sha256=sha256(out / "evaluation_data.jsonl.gz"),
                evaluation_data_canonical_sha256=(out / "evaluation_data_canonical.sha256").read_text().strip(),
                answer_logits_sha256=sha256(out / "answer_logits.npy"),
                predictions_sha256=sha256(out / "predictions.csv"))


def run(args):
    if min(args.steps, args.batch_size, args.eval_batch_size, args.gate_episodes,
           args.grid_episodes, args.primary_episodes, args.max_seconds, args.threads) <= 0:
        raise ValueError("Counts and limits must be positive")
    out, device, meta = initialize(args)
    started = time.perf_counter()
    deadline = started + args.max_seconds
    model = None
    try:
        model, config, record = load_checkpoint(args.checkpoint, map_location=device)
        if config.segment_len != 32 or config.memory_segment_len != 8 or config.memory_batch_size != 32:
            raise ValueError("Fact protocol requires the fixed shared checkpoint geometry")
        meta["parameter_count"] = sum(p.numel() for p in model.parameters())
        if args.mode == "train":
            meta.update(train(model, config, args, out, device, deadline))
            before_deadline(deadline)
            reloaded, _, _ = load_checkpoint(out / "checkpoint.pt", map_location=device)
            probe = make_episode(2_000_000_001, 2, 0, "dev", revision_fraction=0.)
            prefix, _, _ = encode_example(probe, probe.queries[0], "online")
            x = torch.tensor([list(prefix)], device=device)
            with torch.no_grad():
                model.eval()
                reloaded.eval()
                delta = float((model(x) - reloaded(x)).abs().max())
            if delta != 0:
                raise AssertionError("Checkpoint reload logits differ")
            meta["reload_max_abs_delta"] = delta
        else:
            datasets = build_evaluation(args, out)
            meta.update(evaluate(model, datasets, args, out, device, deadline))
        for name, digest in meta["source_sha256"].items():
            if sha256(REPO / name) != digest or sha256(out / "sources" / name) != digest:
                raise AssertionError(f"Source changed during run: {name}")
        before_deadline(deadline)
        meta["status"] = "succeeded"
    except BaseException as error:
        meta.update(status="failed", failure_type=type(error).__name__, failure=str(error),
                    traceback=traceback.format_exc())
        if model is not None:
            torch.save(model.state_dict(), out / "partial_state.pt")
        raise
    finally:
        meta["elapsed_seconds"] = time.perf_counter() - started
        meta["ended_utc"] = datetime.now(timezone.utc).isoformat()
        if device.type == "cuda":
            meta["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        json_file(out / "metadata.json", meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.0005)
    parser.add_argument("--train-seed", type=int, default=78102)
    parser.add_argument("--dev-seed", type=int, default=78103)
    parser.add_argument("--gate-episodes", type=int, default=64)
    parser.add_argument("--grid-episodes", type=int, default=32)
    parser.add_argument("--primary-episodes", type=int, default=128)
    parser.add_argument("--suite", choices=("gate", "full"), default="gate")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())
