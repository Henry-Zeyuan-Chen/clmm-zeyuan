"""Audit saved symbolic-memory training artifacts without running inference.

Example:
    python training_audit.py --run ../train-48101 --out training_audit_48101.json
    python training_audit.py --run ../train-recovery-48101 --design-amendment ../training-amendment-1.json
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def load_design(design_path, amendment_path=None):
    """Apply only an explicitly supplied amendment, retaining both input hashes."""
    design = json.loads(design_path.read_text(encoding="utf-8"))
    provenance = {"design_path": str(design_path.resolve()), "design_sha256": sha256(design_path)}
    if amendment_path is None:
        return design, provenance
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    require(amendment["parent_design"] == design_path.name, "Amendment names a different parent design")
    aliases = {"heldout_episode_length": "episode_length", "maximum_seconds_each": "max_seconds_per_run"}
    allowed = {"max_inner_lr", "curriculum", "steps", "batch_size", "nkeys", "nvals", "dim", "depth", "seeds"}
    annotations = {"unchanged", "selection"}
    for name, value in amendment["revised_training"].items():
        require(name in allowed or name in aliases or name in annotations, f"Unknown amended field: {name}")
        if name not in annotations:
            design["training"][aliases.get(name, name)] = value
    provenance.update(design_amendment_path=str(amendment_path.resolve()),
                      design_amendment_sha256=sha256(amendment_path),
                      design_amendment=amendment,
                      effective_training_design=design["training"])
    return design, provenance


def training_schedule(planned):
    """Independently reconstruct the planned schedule, without importing trainer code."""
    specification = planned.get("curriculum", "")
    if not specification.strip():
        stages = [{"episode_length": planned["episode_length"], "steps": planned["steps"]}]
    else:
        stages = []
        for stage in specification.split(","):
            length, count = [int(value) for value in stage.split(":")]
            stages.append({"episode_length": length, "steps": count})
    require(all(stage["episode_length"] > 0 and stage["steps"] > 0 for stage in stages), "Nonpositive curriculum stage")
    require(sum(stage["steps"] for stage in stages) == planned["steps"], "Curriculum total differs from planned steps")
    return stages


def audit(run, design):
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["configuration"]
    planned = design["training"]
    require(metadata["status"] == "succeeded", f"Run status is {metadata['status']}")
    for name in ("nkeys", "nvals", "dim", "depth", "expansion_factor", "max_inner_lr",
                 "steps", "batch_size", "episode_length", "lr", "weight_decay"):
        require(config[name] == planned[name], f"Configuration differs from design: {name}")
    require(config.get("curriculum", "") == planned.get("curriculum", ""), "Curriculum differs from explicit design/amendment")
    schedule = training_schedule(planned)
    step_lengths = [stage["episode_length"] for stage in schedule for _ in range(stage["steps"])]
    planned_writes = config["batch_size"] * sum(step_lengths)
    records_schedule = bool(config.get("curriculum", "")) or "training_schedule" in metadata
    if records_schedule:
        require(metadata["training_schedule"] == schedule, "Metadata training schedule differs")
        require(metadata["training_observation_writes_planned"] == planned_writes, "Planned training write count differs")
        require(metadata["training_observation_writes_completed"] == planned_writes, "Completed training write count differs")
    require(config["clip"] == planned["outer_gradient_clip"], "Outer gradient clip differs from design")
    require(config["eval_episodes"] == planned["heldout_episodes"], "Held-out episode count differs from design")
    require(config["max_seconds"] == planned["max_seconds_per_run"], "Time budget differs from design")
    seeds = [config[name] for name in ("init_seed", "train_seed", "eval_seed")]
    require(seeds in planned["seeds"], f"Unplanned seed tuple: {seeds}")
    for name in ("nkeys", "nvals", "dim", "depth", "expansion_factor", "max_inner_lr"):
        require(metadata["model_config"][name] == config[name], f"Model and run configuration disagree: {name}")
    require(metadata["nkeys"] == config["nkeys"] and metadata["nvals"] == config["nvals"], "Vocabulary metadata disagrees")
    require(metadata["expected_uniform_chance"] == 1 / config["nvals"], "Incorrect chance level")
    require(metadata["inner_update"]["max_inner_lr"] == config["max_inner_lr"], "Inner-step cap disagrees")
    require(metadata["inner_update"]["loss"] == "mean squared error over features", "Unexpected inner loss")
    require(metadata["inner_update"]["momentum"] is False, "Unexpected inner momentum")
    require(metadata["inner_update"]["write_chunk"] == metadata["inner_update"]["gradient_refresh"] == 1, "Unexpected write/refresh cadence")

    sources = {}
    for relative, expected in metadata["sources_sha256"].items():
        path = PurePosixPath(relative.replace("\\", "/"))
        require(not path.is_absolute() and ".." not in path.parts, "Source manifest must contain relative paths")
        source = run / "source" / Path(*path.parts)
        actual = sha256(source)
        require(actual == expected, f"Source hash differs: {relative}")
        sources[str(path)] = actual
    for name in ("scripts/train_memory_pilot.py", "scripts/research_memory.py",
                 "titans_pytorch/neural_memory.py", "titans_pytorch/memory_models.py"):
        require(name in sources, f"Missing source snapshot: {name}")

    checkpoint_hash = sha256(run / "checkpoint.pt")
    require(checkpoint_hash == metadata["checkpoint_sha256"], "Checkpoint hash differs")
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    require(checkpoint["config"] == metadata["model_config"], "Checkpoint configuration differs")
    require(checkpoint["steps"] == metadata["steps_completed"] == config["steps"], "Fixed-step training is incomplete")
    require(checkpoint["init_seed"] == config["init_seed"] and checkpoint["train_seed"] == config["train_seed"], "Checkpoint seed differs")
    if records_schedule:
        require(checkpoint["training_schedule"] == schedule, "Checkpoint training schedule differs")
        require(checkpoint["observation_writes_seen"] == planned_writes, "Checkpoint training write count differs")
    state = checkpoint["state_dict"]
    shapes = {"key.weight": (config["nkeys"], config["dim"]),
              "value.weight": (config["nvals"], config["dim"]),
              "head.weight": (config["nvals"], config["dim"]), "head.bias": (config["nvals"],)}
    for name, expected in shapes.items():
        require(tuple(state[name].shape) == expected, f"Unexpected checkpoint tensor shape: {name}")
    require(all(torch.isfinite(tensor).all() for tensor in state.values()), "Checkpoint contains nonfinite values")
    require(all(tensor.dtype == torch.float32 for tensor in state.values()), "Unexpected checkpoint dtype")
    registered = sum(tensor.numel() for tensor in state.values())
    fast = sum(tensor.numel() for name, tensor in state.items() if name.startswith("memory.memory_model_parameters."))
    require(registered == metadata["registered_parameters"], "Registered parameter count differs")
    require(fast == metadata["fast_state_parameters_per_episode"], "Fast-state parameter count differs")

    trace = list(csv_rows(run / "training.csv"))
    require([int(row["step"]) for row in trace] == list(range(1, config["steps"] + 1)), "Incomplete or duplicated training steps")
    previous_elapsed = -1.0
    observation_writes = 0
    for row in trace:
        step = int(row["step"])
        length = step_lengths[step - 1]
        observation_writes += config["batch_size"] * length
        if records_schedule:
            require(int(row["episode_length"]) == length, f"Incorrect curriculum length at step {step}")
            require(int(row["observation_writes_seen"]) == observation_writes, f"Incorrect cumulative writes at step {step}")
        require(int(row["episodes_seen"]) == step * config["batch_size"], "Episode counter differs")
        for name in ("loss", "accuracy", "gradient_norm", "mean_query_delay", "revised_query_fraction", "elapsed_seconds"):
            require(math.isfinite(float(row[name])), f"Nonfinite {name} at step {step}")
        require(float(row["loss"]) >= 0 and float(row["gradient_norm"]) >= 0, "Invalid loss or gradient norm")
        require(0 <= float(row["accuracy"]) <= 1 and 0 <= float(row["revised_query_fraction"]) <= 1, "Invalid training fraction")
        require(0 <= float(row["mean_query_delay"]) <= length - 1, "Invalid query delay for curriculum stage")
        require(float(row["elapsed_seconds"]) >= previous_elapsed, "Training elapsed time decreased")
        previous_elapsed = float(row["elapsed_seconds"])

    counts = Counter(episodes=0, matched=0, rolled=0, initial=0)
    by_delay, by_revision = defaultdict(Counter), defaultdict(Counter)
    observed_keys, queried_keys = set(), set()
    for index, row in enumerate(csv_rows(run / "heldout.csv")):
        require(int(row["episode"]) == index, "Held-out episodes are missing or duplicated")
        keys, values = json.loads(row["keys"]), json.loads(row["values"])
        require(len(keys) == len(values) == config["episode_length"], "Held-out episode length differs")
        require(all(isinstance(key, int) and 0 <= key < config["nkeys"] for key in keys), "Out-of-range held-out key")
        require(all(isinstance(value, int) and 0 <= value < config["nvals"] for value in values), "Out-of-range held-out value")
        query, target = int(row["query"]), int(row["target"])
        positions = [position for position, key in enumerate(keys) if key == query]
        require(bool(positions), "Query key was never observed")
        require(target == values[positions[-1]], "Target is not the latest observed value")
        delay = len(keys) - 1 - positions[-1]
        revised = any(values[position] != target for position in positions)
        require(int(row["delay"]) == delay and row["revised"] == str(revised), "Query delay/revision label differs")
        counts["episodes"] += 1
        by_delay[delay]["episodes"] += 1
        by_revision[str(revised)]["episodes"] += 1
        for name in ("matched", "rolled", "initial"):
            prediction = int(row[name])
            require(0 <= prediction < config["nvals"], f"Out-of-range {name} prediction")
            correct = int(prediction == target)
            counts[name] += correct
            by_delay[delay][name] += correct
            by_revision[str(revised)][name] += correct
        observed_keys.update(keys)
        queried_keys.add(query)
    require(counts["episodes"] == config["eval_episodes"], "Held-out evaluation is incomplete")
    require(dict(counts) == metadata["evaluation"]["counts"], "Held-out counts differ from metadata")
    require(metadata["evaluation"]["complete"] is True, "Evaluation marked incomplete")
    require({str(delay): {"episodes": cell["episodes"], "matched": cell["matched"]}
             for delay, cell in by_delay.items()} == metadata["evaluation"]["by_delay"], "Per-delay evaluation counts differ")
    batch_sizes = [min(config["batch_size"], config["eval_episodes"] - start)
                   for start in range(0, config["eval_episodes"], config["batch_size"])]
    require(min(batch_sizes) >= 2, "A single-episode batch cannot provide a distinct rolled state")
    environment = metadata["environment"]
    for name in ("python", "platform", "torch", "device", "device_name", "dtype", "packages"):
        require(name in environment, f"Missing environment field: {name}")

    return {
        "status": "passed", "run": str(run), "repository_commit": metadata["repository_commit"],
        "configuration": config, "model_config": metadata["model_config"], "training_environment": environment,
        "training_steps": len(trace), "training_writes": observation_writes, "training_schedule": schedule,
        "first25_mean_loss": sum(float(row["loss"]) for row in trace[:25]) / len(trace[:25]),
        "last25_mean_loss": sum(float(row["loss"]) for row in trace[-25:]) / len(trace[-25:]),
        "training_seconds": metadata["training_seconds"], "elapsed_seconds": metadata["elapsed_seconds"],
        "registered_parameters": registered, "fast_state_parameters_per_episode": fast,
        "heldout_counts": dict(counts), "heldout_accuracy_percent": {name: counts[name] * 100 / counts["episodes"] for name in ("matched", "rolled", "initial")},
        "heldout_by_delay": dict(by_delay), "heldout_by_revised_query": dict(by_revision),
        "observed_key_coverage": len(observed_keys), "queried_key_coverage": len(queried_keys),
        "source_snapshot_sha256": sources,
        "input_sha256": {name: sha256(run / name) for name in ("metadata.json", "training.csv", "heldout.csv", "checkpoint.pt")},
        "model_inference_reexecuted": False,
        "checks": ["Prospective design and seed tuple match", "Source and checkpoint hashes match",
                   "Checkpoint configuration, shapes, finite tensors and parameter counts match",
                   "Complete finite training trace and explicit curriculum/write-count agreement", "Latest-value targets, delays and revision flags reconstructed",
                   "Own, rolled and initial-state prediction counts recomputed"],
        "limits": ["Saved control predictions are rescored; their forward passes are not rerun.",
                   "Passing the artifact audit establishes internal consistency, not adequate task learning or replay efficacy."]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=Path(__file__).resolve().parents[1] / "design.json")
    parser.add_argument("--design-amendment", type=Path,
                        help="Explicit recovery amendment; never inferred from run metadata or nearby files")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    provenance = {}
    try:
        design, provenance = load_design(args.design, args.design_amendment)
        result = audit(args.run.resolve(), design)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        result = {"status": "failed", "run": str(args.run.resolve()), "error": f"{type(error).__name__}: {error}"}
    result.update(provenance)
    result["audit_script_sha256"] = sha256(Path(__file__))
    result["audit_command"] = [sys.executable, *sys.argv]
    serialized = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
