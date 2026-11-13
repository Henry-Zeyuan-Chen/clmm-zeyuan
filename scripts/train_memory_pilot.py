"""Train a bounded nonlinear Titans memory pilot on latest-value recall.

Example:
    python scripts/train_memory_pilot.py --out results/deep-pilot --device cuda
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from research_memory import NKEYS, NVALS, REPOSITORY, Reader


def make_episodes(generator, batch_size, writes):
    """Sample repeated keys, then a uniform distinct observed key's latest value."""
    keys = torch.randint(NKEYS, (batch_size, writes), generator=generator)
    values = torch.randint(NVALS, (batch_size, writes), generator=generator)
    observed = (keys[:, :, None] == torch.arange(NKEYS)).any(dim=1)
    query_scores = torch.rand(batch_size, NKEYS, generator=generator)
    query = query_scores.masked_fill(~observed, -1).argmax(dim=1)
    matches = keys == query[:, None]
    positions = torch.arange(writes).expand(batch_size, writes)
    latest = positions.masked_fill(~matches, -1).max(dim=1).values
    target = values.gather(1, latest[:, None]).squeeze(1)
    revised = (matches & (values != target[:, None])).any(dim=1)
    return {
        "keys": keys, "values": values, "query": query, "target": target,
        "delay": writes - 1 - latest, "revised": revised,
    }


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def capture_sources(output):
    paths = [Path(__file__), Path(__file__).with_name("research_memory.py")]
    paths.extend(sorted((REPOSITORY / "titans_pytorch").rglob("*.py")))
    paths.extend(REPOSITORY / name for name in ("pyproject.toml", "requirements-lock.txt"))
    hashes = {}
    for source in paths:
        relative = source.relative_to(REPOSITORY)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return hashes


@torch.no_grad()
def evaluate(model, args, device, deadline):
    model.eval()
    generator = torch.Generator().manual_seed(args.eval_seed)
    totals = {name: 0 for name in ("episodes", "matched", "rolled", "initial")}
    by_delay = {}
    fields = ["episode", "query", "target", "delay", "revised", "matched",
              "rolled", "initial", "keys", "values"]
    with (args.out / "heldout.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while totals["episodes"] < args.eval_episodes and time.perf_counter() < deadline:
            count = min(args.batch_size, args.eval_episodes - totals["episodes"])
            episode = make_episodes(generator, count, args.episode_length)
            inputs = {name: tensor.to(device) for name, tensor in episode.items()}
            fast = model.write(inputs["keys"], inputs["values"])
            snapshot = {name: weights.clone() for name, weights in fast.items()}
            states = {
                "matched": fast,
                "rolled": {name: weights.roll(1, dims=0) for name, weights in fast.items()},
                "initial": model.memory.init_weights(count),
            }
            predictions = {
                name: model.read(inputs["query"], state).argmax(dim=-1).cpu()
                for name, state in states.items()
            }
            if not all(torch.equal(weights, snapshot[name]) for name, weights in fast.items()):
                raise RuntimeError("A held-out read changed the fast memory")
            for index in range(count):
                delay = int(episode["delay"][index])
                target = int(episode["target"][index])
                row = {name: int(prediction[index]) for name, prediction in predictions.items()}
                delay_counts = by_delay.setdefault(delay, {"episodes": 0, "matched": 0})
                delay_counts["episodes"] += 1
                delay_counts["matched"] += row["matched"] == target
                for name in predictions:
                    totals[name] += row[name] == target
                row.update(
                    episode=totals["episodes"] + index,
                    query=int(episode["query"][index]), target=target, delay=delay,
                    revised=bool(episode["revised"][index]),
                    keys=json.dumps(episode["keys"][index].tolist()),
                    values=json.dumps(episode["values"][index].tolist()),
                )
                writer.writerow(row)
            totals["episodes"] += count
            handle.flush()
    return {"counts": totals, "by_delay": by_delay,
            "complete": totals["episodes"] == args.eval_episodes}


def run(args):
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {args.out}")
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.init_seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)

    config = {"dim": args.dim, "depth": args.depth,
              "expansion_factor": args.expansion_factor, "max_inner_lr": args.max_inner_lr}
    model = Reader(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.train_seed)
    version = subprocess.run(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=False)
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "repository_commit": version.stdout.strip(),
        "configuration": {**vars(args), "out": str(args.out)}, "model_config": model.config,
        "sources_sha256": capture_sources(args.out),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "device": str(device), "dtype": "float32",
                        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()},
        "selection": "Final training state only; held-out outcomes do not select a checkpoint.",
        "protocol": "Independent uniform keys and values with replacement. Uniform distinct observed query key. Latest matching value is the target. Each episode resets fast state.",
        "inner_update": {"loss": "mean squared error over features", "heads": 1,
                         "write_chunk": 1, "gradient_refresh": 1, "momentum": False,
                         "max_inner_lr": args.max_inner_lr,
                         "step_and_decay": "learned", "memory_residual_wrapper": False},
        "registered_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fast_state_parameters_per_episode": sum(parameter.numel() for parameter in model.memory.memory_model_parameters),
    }
    metadata["environment"]["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("einops", "einx", "tensordict", "x-transformers", "assoc-scan")
    }
    save_json(args.out / "metadata.json", metadata)
    synchronize(device)
    start = time.perf_counter()
    deadline = start + args.max_seconds
    completed_steps = 0
    active_parameters = set()
    try:
        model.train()
        with (args.out / "training.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "episodes_seen", "loss", "accuracy", "gradient_norm",
                             "mean_query_delay", "revised_query_fraction", "elapsed_seconds"])
            for step in range(1, args.steps + 1):
                if time.perf_counter() >= deadline:
                    break
                episode = make_episodes(generator, args.batch_size, args.episode_length)
                inputs = {name: tensor.to(device) for name, tensor in episode.items()}
                optimizer.zero_grad(set_to_none=True)
                fast = model.write(inputs["keys"], inputs["values"])
                logits = model.read(inputs["query"], fast)
                loss = F.cross_entropy(logits, inputs["target"])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip, error_if_nonfinite=True)
                active_parameters.update(name for name, parameter in model.named_parameters() if parameter.grad is not None)
                optimizer.step()
                synchronize(device)
                completed_steps = step
                accuracy = float((logits.argmax(-1) == inputs["target"]).float().mean())
                writer.writerow([step, step * args.batch_size, float(loss.detach()), accuracy,
                                 float(gradient_norm), float(episode["delay"].float().mean()),
                                 float(episode["revised"].float().mean()), time.perf_counter() - start])
                handle.flush()
                if step == 1 or step % args.log_every == 0 or step == args.steps:
                    print(f"step {step}/{args.steps}, loss {float(loss.detach()):.4f}, accuracy {accuracy:.3f}, elapsed {time.perf_counter() - start:.1f}s", flush=True)

        checkpoint = {
            "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "config": model.config, "steps": completed_steps,
            "init_seed": args.init_seed, "train_seed": args.train_seed,
        }
        torch.save(checkpoint, args.out / "checkpoint.pt")
        metadata["training_seconds"] = time.perf_counter() - start
        metadata["checkpoint_sha256"] = hashlib.sha256((args.out / "checkpoint.pt").read_bytes()).hexdigest()
        metadata["evaluation"] = evaluate(model, args, device, deadline)
        complete = completed_steps == args.steps and metadata["evaluation"]["complete"]
        metadata["status"] = "succeeded" if complete else "budget_exhausted"
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["steps_completed"] = completed_steps
        metadata["elapsed_seconds"] = time.perf_counter() - start
        metadata["outer_gradient_parameters"] = sorted(active_parameters)
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if device.type == "cuda":
            metadata["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        save_json(args.out / "metadata.json", metadata)
    print(json.dumps({"status": metadata["status"], "steps": completed_steps,
                      "evaluation": metadata["evaluation"]["counts"], "out": str(args.out)}), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--expansion-factor", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episode-length", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--max-inner-lr", type=float, default=1.0,
                        help="Upper bound of the learned sigmoid-scaled inner write step")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--init-seed", type=int, default=41101)
    parser.add_argument("--train-seed", type=int, default=41102)
    parser.add_argument("--eval-seed", type=int, default=51141)
    parser.add_argument("--eval-episodes", type=int, default=1024)
    parser.add_argument("--max-seconds", type=float, default=600.0,
                        help="Combined training/evaluation budget, checked between batches; excludes setup and source capture")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    positive = (args.dim, args.depth, args.expansion_factor, args.steps, args.batch_size,
                args.episode_length, args.lr, args.clip, args.eval_episodes,
                args.max_seconds, args.threads, args.log_every)
    if min(positive) <= 0 or args.weight_decay < 0:
        parser.error("Dimensions, budgets, learning rate and clip must be positive; weight decay must be nonnegative")
    if not math.isfinite(args.max_inner_lr) or args.max_inner_lr <= 0:
        parser.error("--max-inner-lr must be finite and positive")
    if args.batch_size < 2 or args.eval_episodes % args.batch_size == 1:
        parser.error("Held-out batches need at least two episodes for the rolled-memory control")
    return args


if __name__ == "__main__":
    run(parse_args())
