"""Independent CPU-only audit and prespecified analysis of persisted artifacts.

This never imports the paired-acquisition runner and never executes a model.
The frozen v1 fact-stream module is loaded only after its source hash is checked.
All writes go to a new directory under this file's analysis directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


RUNS = ((68101, "duplicate", 32), (68101, "paired", 32),
        (68105, "duplicate", 32), (68105, "paired", 32), (68101, "paired", 128))
STEPS = (200, 1200, 1600)
CELLS = {"acquire_k1": (1, 0.), "acquire_k2": (2, 0.),
         "revise_k1": (1, 1.), "counterfactual_k2": (2, 0.)}
FAMILIES = ("seen", "heldout")
CONDITIONS = ("online", "reset_memory")
SCRIPT_DIR = Path(__file__).resolve().parent
FROZEN_COMMIT = "d82030974596207a37044ba3951fbbbe242e2d00"


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jread(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def jwrite(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def csvread(path):
    with Path(path).open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def csvwrite(path, rows):
    require(bool(rows), f"Refusing empty output table {path}")
    with Path(path).open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def near(actual, expected, label, *, atol=5e-5, rtol=5e-6):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected),
            rel_tol=rtol, abs_tol=atol), f"{label}: {actual} != {expected}")


def unique_index(rows, fields, label):
    result = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        require(key not in result, f"{label}: duplicate identity {key}")
        result[key] = row
    return result


def score(logits, targets):
    logits = np.asarray(logits)
    targets = np.asarray(targets, dtype=np.int64)
    require(logits.ndim == 2 and logits.shape == (len(targets), 256), "Logit/target shape mismatch")
    require(np.isfinite(logits).all(), "Nonfinite saved logits")
    require(np.all((97 <= targets) & (targets <= 112)), "Illegal target byte")
    work = logits.astype(np.float64)
    maximum = work.max(axis=1)
    log_z = maximum + np.log(np.exp(work - maximum[:, None]).sum(axis=1))
    legal = work[:, 97:113]
    legal_max = legal.max(axis=1)
    legal_log_z = legal_max + np.log(np.exp(legal - legal_max[:, None]).sum(axis=1))
    predictions = work.argmax(axis=1)
    forced = legal.argmax(axis=1) + 97
    return dict(prediction=predictions, forced_prediction=forced,
                correct=(predictions == targets).astype(np.int64),
                forced_correct=(forced == targets).astype(np.int64),
                nll=log_z - work[np.arange(len(targets)), targets],
                forced_nll=legal_log_z - work[np.arange(len(targets)), targets],
                legal_probability_mass=np.exp(legal_log_z - log_z))


def validate_prefix_target(prefix, target, episode, query, streams, label):
    expected = b"".join(event.encoded for event in episode.events) + streams.render_query(query)
    require(prefix == expected, f"{label}: prefix reconstruction differs")
    require(int(target) == query.answer_byte, f"{label}: answer target differs")
    return expected


def validate_scores(rows, logits, *, strict=False):
    computed = score(logits, [int(row["target"]) for row in rows])
    exact_fields = ("prediction", "correct") if strict else (
        "prediction", "forced_prediction", "correct", "forced_correct")
    for i, row in enumerate(rows):
        for field in exact_fields:
            require(int(row[field]) == int(computed[field][i]), f"Score mismatch row {i} {field}")
        if not strict:
            for field in ("nll", "forced_nll", "legal_probability_mass"):
                near(row[field], computed[field][i], f"row {i} {field}")
            obsolete = int(int(row["old_target"]) >= 0 and int(row["prediction"]) == int(row["old_target"]))
            require(int(row["obsolete"]) == obsolete, f"Obsolete score mismatch row {i}")
    return computed


def bootstrap_difference(left, right, *, draws=5000, seed=89104):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    require(left.ndim == right.ndim == 1 and left.shape == right.shape and len(left) > 0,
            "Bootstrap requires equally sized nonempty paired episode vectors")
    require(np.isfinite(left).all() and np.isfinite(right).all(), "Nonfinite bootstrap vector")
    delta = left - right
    indices = np.random.default_rng(seed).integers(0, len(delta), size=(draws, len(delta)))
    estimates = delta[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [.025, .975])
    return dict(episodes=len(delta), left=float(left.mean()), right=float(right.mean()),
                difference=float(delta.mean()), lower=float(lower), upper=float(upper),
                bootstrap_draws=draws, bootstrap_seed=seed)


def paired_vectors(left, right, value):
    """Pair by explicit semantic identity, never by incidental row position."""
    require(set(left) == set(right) and bool(left), "Episode identity sets differ or are empty")
    a, b = [], []
    for key in sorted(left):
        require(left[key]["identity"] == right[key]["identity"], f"Semantic pairing mismatch {key}")
        a.append(left[key][value])
        b.append(right[key][value])
    return a, b


def strict_pairs(rows, logits):
    computed = validate_scores(rows, logits, strict=True)
    indexed = unique_index(rows, ("family", "condition", "episode", "rank"), "strict rows")
    groups = defaultdict(dict)
    for i, row in enumerate(rows):
        key = (row["family"], row["condition"], int(row["episode"]))
        rank = int(row["rank"])
        require(rank in (0, 1), f"Invalid strict rank {rank}")
        groups[key][rank] = (row, i)
    result = {}
    for key, group in groups.items():
        require(set(group) == {0, 1}, f"Incomplete strict pair {key}")
        a, ia = group[0]
        b, ib = group[1]
        require(a["key"] != b["key"] and int(a["target"]) != int(b["target"]),
                f"Strict pair lacks distinct keys/values {key}")
        pa, pb = a["prefix"].encode("ascii"), b["prefix"].encode("ascii")
        require(len(pa) == len(pb) == 95 and pa[:-31] == pb[:-31], f"Strict contexts differ {key}")
        require(pa[-31:] != pb[-31:] and pa[-31:].replace(a["key"].encode(), b"###") ==
                pb[-31:].replace(b["key"].encode(), b"###"), f"Query switch changes wording {key}")
        ca, cb = int(computed["correct"][ia]), int(computed["correct"][ib])
        result[key] = dict(identity=(a["key"], b["key"], pa, pb, int(a["target"]), int(b["target"])),
                           accuracy=(ca + cb) / 2, first_accuracy=ca, second_accuracy=cb,
                           both_correct=int(ca and cb),
                           prediction_changes=int(computed["prediction"][ia] != computed["prediction"][ib]),
                           always_first_accuracy=.5, always_last_accuracy=.5,
                           always_first_both_correct=0., always_last_both_correct=0.)
    return result


def gate_episode_vectors(rows):
    unique_index(rows, ("cell", "family", "condition", "episode", "query_index"), "gate rows")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cell"], row["family"], row["condition"], int(row["episode"]))].append(row)
    result = {}
    for key, group in groups.items():
        group.sort(key=lambda row: int(row["query_index"]))
        result[key] = dict(identity=tuple((r["key"], int(r["target"]), r["prefix_sha256"]) for r in group),
                           accuracy=float(np.mean([int(r["correct"]) for r in group])))
    return result


def require_noop(rows, logits, fields, label):
    indices = {}
    for i, row in enumerate(rows):
        key = tuple(str(row[field]) for field in fields)
        condition = row["condition"]
        require(condition in CONDITIONS, f"{label}: unknown condition")
        require((key, condition) not in indices, f"{label}: duplicate no-op row")
        indices[(key, condition)] = i
    base_keys = {key for key, condition in indices if condition == "online"}
    require(base_keys == {key for key, condition in indices if condition == "reset_memory"},
            f"{label}: reset identity mismatch")
    for key in base_keys:
        ia, ib = indices[(key, "online")], indices[(key, "reset_memory")]
        require(rows[ia]["target"] == rows[ib]["target"], f"{label}: reset target mismatch")
        require(np.array_equal(logits[ia], logits[ib]), f"{label}: 128 reset is not exact no-op {key}")
    return len(base_keys)


def load_stream_module(path):
    name = "_frozen_fact_streams_for_independent_audit"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot load frozen fact-stream module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_audit(folder, meta, expected_commit=None, repo=None):
    require(meta["status"] == "succeeded", f"Run not succeeded: {folder}")
    require(isinstance(meta.get("source_sha256"), dict) and meta["source_sha256"], "Missing source manifest")
    if expected_commit:
        require(meta["repo_commit"] == expected_commit, f"Unexpected source commit {folder}")
    for name, digest in meta["source_sha256"].items():
        source = (folder / "sources" / name).resolve()
        require(source.is_relative_to((folder / "sources").resolve()), "Unsafe source manifest path")
        require(source.is_file() and sha(source) == digest, f"Source digest mismatch {folder.name}:{name}")
        if repo:
            data = subprocess.check_output(["git", "show", f"{meta['repo_commit']}:{name}"], cwd=repo)
            require(hashlib.sha256(data).hexdigest() == digest, f"Source differs from Git commit: {name}")
    return dict(meta["source_sha256"])


def tensor_state_equal(left, right, label):
    import torch
    require(isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right),
            f"{label}: state names differ")
    for name in left:
        require(isinstance(left[name], torch.Tensor) and isinstance(right[name], torch.Tensor),
                f"{label}: non-tensor {name}")
        require(left[name].shape == right[name].shape and left[name].dtype == right[name].dtype and
                torch.equal(left[name], right[name]), f"{label}: tensor differs {name}")


def safe_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def locate_parents(parent_root, design):
    root = Path(parent_root).resolve()
    candidates = [root] if root.is_file() else sorted(root.rglob("*.pt"))
    wanted = {digest: int(seed) for seed, digest in design["parents"].items()}
    found = {}
    for path in candidates:
        if path.name in ("optimizer.pt", "partial_state.pt", "initial_state.pt"):
            continue
        digest = sha(path)
        if digest in wanted:
            found.setdefault(wanted[digest], path)
    require(set(found) == {68101, 68105}, "Original parents not found by design SHA-256 hashes")
    result = {}
    for seed, path in found.items():
        payload = safe_load(path)
        require(payload["format_version"] == 1 and payload["architecture"] == "clmm.vendored_mac.byte_lm.v1",
                f"Parent format differs {seed}")
        info, config = payload["training_info"], payload["config"]
        require(info.get("init_seed") == seed and info.get("task") is None and
                info.get("parent_checkpoint_sha256") is None, f"Parent is not original {seed}")
        require((config["segment_len"], config["memory_segment_len"], config["memory_batch_size"]) == (32, 8, 32),
                f"Parent geometry differs {seed}")
        result[seed] = dict(path=str(path), sha256=sha(path), payload=payload)
    return result


def checkpoint_audit(folder, meta, parent, design, arm, segment):
    import torch
    initial = safe_load(folder / "initial_state.pt")
    tensor_state_equal(initial, parent["payload"]["model_state"], f"{folder.name} initial parent")
    require(meta["checkpoint_sha256"] == parent["sha256"] and meta["parent_init_seed"] ==
            parent["payload"]["training_info"]["init_seed"], "Parent hash/seed metadata mismatch")
    expected_config = dict(parent["payload"]["config"], segment_len=segment)
    checkpoints = {}
    records = {int(row["step"]): row for row in meta["checkpoints"]}
    require(set(records) == set(STEPS) and len(meta["checkpoints"]) == 3, "Stage metadata differs")
    for step in STEPS:
        stage = folder / f"step-{step:04d}"
        path = stage / "checkpoint.pt"
        digest = sha(path)
        require(digest == records[step]["checkpoint_sha256"], "Stage hash mismatch in run metadata")
        evaluation_meta = jread(stage / "evaluation_metadata.json")
        require(evaluation_meta["checkpoint_sha256"] == digest and evaluation_meta["step"] == step,
                "Stage hash/step mismatch in evaluation metadata")
        require(records[step]["evaluation"] == {field: evaluation_meta[field]
                for field in records[step]["evaluation"]}, "Stage evaluation metadata copy differs")
        require(records[step]["strict_summary"] == evaluation_meta["strict_summary"],
                "Stage strict metadata copy differs")
        payload = safe_load(path)
        require(payload["config"] == expected_config, f"Unexpected checkpoint config {folder.name}:{step}")
        require(payload["format_version"] == 1 and payload["architecture"] == parent["payload"]["architecture"],
                "Stage checkpoint format differs")
        state = payload["model_state"]
        require(set(state) == set(initial), "Checkpoint tensor names differ")
        for name, value in state.items():
            require(isinstance(value, torch.Tensor) and torch.isfinite(value).all().item() and
                    value.shape == initial[name].shape and value.dtype == initial[name].dtype,
                    f"Invalid stage tensor {name}")
        info = payload["training_info"]
        expected_info = dict(task="fact-paired-acquisition", parent_checkpoint_sha256=parent["sha256"],
            parent_init_seed=meta["parent_init_seed"], steps=step, arm=arm,
            train_seed=design["training"]["train_seed"], lr=design["training"]["lr"],
            optimizer="AdamW", weight_decay=.01, gradient_clip=1., batch_size=16,
            fixed_final_checkpoint=step == STEPS[-1])
        require(info == expected_info, f"Checkpoint training metadata mismatch {folder.name}:{step}")
        provenance = payload["source_info"]
        require(provenance["repo_commit"] == meta["repo_commit"] and
                provenance["source_sha256"] == meta["source_sha256"], "Checkpoint source provenance differs")
        optimizer = safe_load(stage / "optimizer.pt")
        all_ids = []
        for group in optimizer["param_groups"]:
            require(group["lr"] == design["training"]["lr"] and group["weight_decay"] == .01,
                    "Optimizer hyperparameters differ")
            all_ids.extend(group["params"])
        require(len(all_ids) == len(set(all_ids)), "Optimizer duplicates parameter identities")
        for state_info in optimizer["state"].values():
            require(int(state_info["step"]) == step, "Optimizer saved at wrong step")
            for key in ("exp_avg", "exp_avg_sq"):
                require(torch.isfinite(state_info[key]).all().item(), "Nonfinite optimizer state")
        checkpoints[step] = dict(state=state, evaluation_meta=evaluation_meta, sha256=digest)
    return checkpoints


def accepted_episode(streams, candidate, facts, revision, split, family):
    """Independent reconstruction of the declared rejection conditioning."""
    for attempt in range(100):
        episode = streams.make_episode(int(candidate) + attempt * 10_000_000_019,
            facts, 0, split, family, revision, include_recent=False)
        if len({event.value for event in episode.events[:facts]}) == facts:
            template = episode.queries[0].template_index
            return replace(episode, queries=tuple(replace(query, template_index=template)
                                                 for query in episode.queries))
    raise AuditError("Independent rejection limit exceeded")


def training_audit(folder, arm, streams, design):
    data = np.load(folder / "training_inputs.npz", allow_pickle=False)
    require(set(data.files) == {"input_bytes", "step_offsets", "targets"}, "Unexpected training arrays")
    raw, offsets, targets = data["input_bytes"], data["step_offsets"], data["targets"]
    require(raw.dtype == np.uint8 and raw.ndim == 1 and offsets.dtype == np.int64 and
            offsets.shape == (1601,) and targets.dtype == np.uint8 and targets.shape == (1600, 16),
            "Training NPZ types/shapes differ")
    require(offsets[0] == 0 and offsets[-1] == len(raw) and np.all(np.diff(offsets) > 0),
            "Training offsets are invalid")
    plans, logs = jread(folder / "training_plan.json"), csvread(folder / "training.csv")
    require(len(plans) == len(logs) == 1600, "Missing or duplicated training steps")
    contexts = []
    signatures = []
    for index, (plan, log) in enumerate(zip(plans, logs)):
        step = index + 1
        require(plan["step"] == step and int(log["step"]) == step, "Training steps out of order")
        rng = np.random.default_rng(np.random.SeedSequence([design["training"]["train_seed"], index]))
        if index < 200:
            stage, facts, revision = "k1", 1, 0.
        elif index < 1200:
            stage, facts, revision = "k2", 2, 0.
        else:
            stage = "mixed_revision"
            facts, revision = ((1, 0.), (2, 0.), (1, 1.), (2, .5))[int(rng.integers(4))]
        candidates = rng.integers(1, 1_000_000_000, size=8).tolist()
        choices = rng.integers(0, facts, size=8).tolist()
        require(plan["stage"] == stage and plan["facts"] == facts and plan["revision_fraction"] == revision,
                f"Curriculum mismatch step {step}")
        require(plan["candidate_seeds"] == candidates and plan["duplicate_choices"] == choices,
                f"Sampling plan mismatch step {step}")
        require(len(plan["episodes"]) == 8 and len(plan["query_indices"]) == 16,
                "Training context/answer counts differ")
        length = int(plan["input_length"])
        require(offsets[index + 1] - offsets[index] == 16 * length, "Training byte budget mismatch")
        batch = raw[offsets[index]:offsets[index + 1]].reshape(16, length)
        episode_contexts = []
        for pair_index, (saved, candidate, choice) in enumerate(zip(plan["episodes"], candidates, choices)):
            episode = streams.Episode.from_dict(saved)
            expected_episode = accepted_episode(streams, candidate, facts, revision, "train", "seen")
            require(episode == expected_episode, f"Persisted episode differs step {step} pair {pair_index}")
            require(episode.split == "train" and episode.delay == 0 and not episode.include_recent,
                    "Training namespace/distractor violation")
            context = b"".join(record.encoded for record in episode.events)
            episode_contexts.append(hashlib.sha256(context).hexdigest())
            expected_indices = [0, 1] if arm == "paired" and facts == 2 else [choice, choice]
            require(plan["query_indices"][pair_index * 2:pair_index * 2 + 2] == expected_indices,
                    f"Arm query supervision mismatch step {step}")
            for within, query_index in enumerate(expected_indices):
                row = 2 * pair_index + within
                query = episode.queries[query_index]
                validate_prefix_target(batch[row].tobytes(), targets[index, row], episode, query, streams,
                                       f"Training step {step} row {row}")
        require(log["stage"] == stage and int(log["facts"]) == facts and
                float(log["revision_fraction"]) == revision and int(log["input_length"]) == length and
                int(log["input_bytes"]) == length * 16 and int(log["supervised_answers"]) == 16,
                f"Training log budget mismatch step {step}")
        require(math.isfinite(float(log["loss"])) and math.isfinite(float(log["gradient_norm"])) and
                0 <= float(log["accuracy"]) <= 1, "Invalid training diagnostic")
        contexts.append(tuple(episode_contexts))
        signatures.append((stage, facts, revision, tuple(candidates), tuple(choices), length))
    return dict(input_bytes=int(len(raw)), supervised_answers=int(targets.size), steps=1600,
                contexts=contexts, signatures=signatures, plans=plans,
                initial_input_bytes=raw[:offsets[200]].copy(), initial_targets=targets[:200].copy(),
                training_array_sha256=sha(folder / "training_inputs.npz"),
                training_plan_sha256=sha(folder / "training_plan.json"))


def check_gate_artifacts(evaluation, evaluation_meta, streams, design):
    raw = gzip.decompress((evaluation / "evaluation_data.jsonl.gz").read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    require((evaluation / "evaluation_data_canonical.sha256").read_text().strip() == digest,
            "Gate canonical data hash mismatch")
    for filename, field in (("evaluation_data.jsonl.gz", "evaluation_data_sha256"),
                            ("answer_logits.npy", "answer_logits_sha256"),
                            ("predictions.csv", "predictions_sha256")):
        require(sha(evaluation / filename) == evaluation_meta[field], f"Gate artifact digest differs {filename}")
    require(evaluation_meta["evaluation_data_canonical_sha256"] == digest and
            evaluation_meta["checkpoint_unchanged"] is True, "Gate evaluation metadata mismatch")
    datasets, samples = {}, []
    for line in raw.decode("utf8").splitlines():
        entry = json.loads(line)
        if entry["type"] == "episodes":
            spec = entry["spec"]
            key = (spec["cell"], spec["family"])
            require(key not in datasets and key[0] in CELLS and key[1] in FAMILIES, "Duplicate/unknown gate cell")
            facts, revision = CELLS[key[0]]
            require(spec["facts"] == facts and spec["revision_fraction"] == revision and spec["delay"] == 0 and
                    spec["episodes"] == 64 and spec["conditions"] == list(CONDITIONS) and spec["all_queries"] is True,
                    "Gate cell specification differs")
            group = 1 if key[0] == "revise_k1" else 0
            seed_rng = np.random.default_rng(design["evaluation"]["dev_seed"] + facts * 101 + group * 10007)
            seeds = seed_rng.integers(2_000_000_000, 3_000_000_000, 64)
            require(len(entry["episodes"]) == 64 and len(set(int(s) for s in seeds)) == 64,
                    "Gate episode count or distinctness differs")
            episodes = []
            for saved, seed in zip(entry["episodes"], seeds):
                episode = streams.Episode.from_dict(saved)
                expected = streams.make_episode(int(seed), facts, 0, "dev", key[1], revision)
                if key[0] == "counterfactual_k2":
                    mapping = dict(zip(streams.VALUES, streams.VALUES[1:] + streams.VALUES[:1]))
                    expected = replace(expected,
                        events=tuple(replace(event, value=mapping[event.value]) for event in expected.events),
                        queries=tuple(replace(query, target=mapping[query.target],
                            old_value=mapping[query.old_value] if query.old_value else None)
                                      for query in expected.queries))
                require(episode == expected, f"Gate episode reconstruction differs {key}")
                episodes.append(episode)
            datasets[key] = episodes
        elif entry["type"] == "sample":
            samples.append(entry)
        else:
            raise AuditError("Unexpected gate JSONL record")
    require(set(datasets) == {(cell, family) for cell in CELLS for family in FAMILIES}, "Missing gate datasets")
    identities = ("cell", "family", "condition", "episode", "query_index")
    sample_index = unique_index([sample["row"] for sample in samples], identities, "gate samples")
    by_id = {tuple(str(sample["row"][field]) for field in identities): sample for sample in samples}
    rows = csvread(evaluation / "predictions.csv")
    row_index = unique_index(rows, identities, "gate predictions")
    require(set(row_index) == set(sample_index), "Gate row/sample identities differ")
    expected_count = sum(len(ep.queries) * 2 for eps in datasets.values() for ep in eps)
    require(len(rows) == expected_count == int(evaluation_meta["evaluation_queries"]), "Gate answer budget differs")
    for key, row in row_index.items():
        sample = by_id[key]
        saved = sample["row"]
        for field, value in saved.items():
            actual = row[field]
            if isinstance(value, bool):
                require(actual == str(value), f"Gate metadata differs {field}")
            else:
                require(str(actual) == str(value), f"Gate metadata differs {field}")
        episode = datasets[(row["cell"], row["family"])][int(row["episode"])]
        query_index = int(row["query_index"])
        require(0 <= query_index < len(episode.queries), "Gate query index invalid")
        query = episode.queries[query_index]
        prefix = validate_prefix_target(sample["prefix"].encode("ascii"), row["target"], episode, query, streams, "Gate")
        require(hashlib.sha256(prefix).hexdigest() == row["prefix_sha256"], "Gate prefix hash differs")
        require(int(row["target"]) == query.answer_byte and row["key"] == query.key and
                int(row["old_target"]) == (ord(query.old_value) if query.old_value else -1), "Gate labels differ")
        require(row["category"] == query.category and int(row["episode_seed"]) == episode.seed and
                int(row["input_length"]) == len(prefix) and int(row["online_records"]) == len(episode.events),
                "Gate semantic/budget fields differ")
        require(all(int(row[field]) == 0 for field in ("extra_records", "replay_records", "stale_replays",
                "target_replays", "target_stale_replays")), "Acquisition gate contains replay")
    logits = np.load(evaluation / "answer_logits.npy", allow_pickle=False)
    computed = validate_scores(rows, logits)
    # Independently recompute every aggregate retained by the original runner.
    summaries = unique_index(csvread(evaluation / "summary.csv"),
        ("cell", "family", "category", "condition"), "saved gate summary")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row["cell"], row["family"], row["category"], row["condition"])].append(i)
    require(set(summaries) == set(groups), "Saved gate summary identities differ")
    for key, indices in groups.items():
        saved = summaries[key]
        require(int(saved["queries"]) == len(indices) and int(saved["episodes"]) ==
                len({rows[i]["episode"] for i in indices}), "Saved gate summary counts differ")
        for field, array in (("accuracy", computed["correct"]), ("forced_accuracy", computed["forced_correct"]),
                             ("nll", computed["nll"]), ("legal_probability_mass", computed["legal_probability_mass"])):
            near(saved[field], array[indices].mean(), f"Saved gate summary {key} {field}")
        near(saved["obsolete_rate"], np.mean([int(rows[i]["obsolete"]) for i in indices]),
             "Saved obsolete summary")
    return rows, logits, computed, gate_episode_vectors(rows)


def check_strict_artifacts(evaluation, streams, design):
    semantic = jread(evaluation / "strict_episodes.json")
    mapping = unique_index(semantic, ("family", "episode"), "strict episode semantics")
    require(len(mapping) == 256, "Strict semantic count differs")
    candidates = np.random.default_rng(design["evaluation"]["dev_seed"] + 500_021).integers(
        3_100_000_000, 3_200_000_000, 128)
    require(len(set(int(s) for s in candidates)) == 128, "Strict candidate identities duplicate")
    episodes = {}
    for family in FAMILIES:
        for index, candidate in enumerate(candidates):
            saved = mapping[(family, str(index))]
            episode = streams.Episode.from_dict(saved["semantic"])
            expected = accepted_episode(streams, int(candidate), 2, 0., "dev", family)
            require(episode == expected, "Strict episode reconstruction differs")
            require(episode.split == "dev" and episode.delay == 0 and not episode.include_recent and
                    len({query.target for query in episode.queries}) == 2, "Strict episode specification differs")
            episodes[(family, index)] = episode
    rows = csvread(evaluation / "strict_predictions.csv")
    require(len(rows) == 1024, "Strict answer budget differs")
    for row in rows:
        require(row["family"] in FAMILIES and row["condition"] in CONDITIONS and int(row["rank"]) in (0, 1),
                "Strict categorical identity differs")
        episode = episodes[(row["family"], int(row["episode"]))]
        query = episode.queries[int(row["rank"])]
        validate_prefix_target(row["prefix"].encode("ascii"), row["target"], episode, query, streams, "Strict")
        require(row["key"] == query.key, "Strict query key differs")
    logits = np.load(evaluation / "strict_logits.npy", allow_pickle=False)
    pairs = strict_pairs(rows, logits)
    require(len(pairs) == 512, "Strict pair count differs")
    saved_summary = unique_index(jread(evaluation / "strict_summary.json"), ("family", "condition"), "strict summary")
    require(set(saved_summary) == {(family, condition) for family in FAMILIES for condition in CONDITIONS},
            "Strict summary cells differ")
    for (family, condition), saved in saved_summary.items():
        subset = [pair for key, pair in pairs.items() if key[:2] == (family, condition)]
        require(saved["queries"] == 256 and saved["episodes"] == 128, "Strict summary budgets differ")
        for field in ("accuracy", "first_accuracy", "second_accuracy", "both_correct", "prediction_changes"):
            near(saved[field], np.mean([pair[field] for pair in subset]), f"Strict summary {field}", atol=1e-12)
    return rows, logits, pairs


def strict_table(name, step, pairs):
    result = []
    fields = ("accuracy", "first_accuracy", "second_accuracy", "both_correct", "prediction_changes",
              "always_first_accuracy", "always_last_accuracy", "always_first_both_correct", "always_last_both_correct")
    for family in FAMILIES:
        for condition in CONDITIONS:
            subset = [value for key, value in pairs.items() if key[:2] == (family, condition)]
            result.append(dict(run=name, step=step, family=family, condition=condition,
                episodes=len(subset), queries=len(subset) * 2,
                **{field: float(np.mean([pair[field] for pair in subset])) for field in fields}))
    return result


def gate_table(name, step, rows, computed):
    result = []
    for cell in CELLS:
        for family in FAMILIES:
            for condition in CONDITIONS:
                indices = [i for i, row in enumerate(rows) if (row["cell"], row["family"], row["condition"]) ==
                           (cell, family, condition)]
                result.append(dict(run=name, step=step, cell=cell, family=family, condition=condition,
                    queries=len(indices), episodes=len({rows[i]["episode"] for i in indices}),
                    accuracy=float(computed["correct"][indices].mean()),
                    forced_accuracy=float(computed["forced_correct"][indices].mean()),
                    nll=float(computed["nll"][indices].mean()),
                    forced_nll=float(computed["forced_nll"][indices].mean()),
                    legal_probability_mass=float(computed["legal_probability_mass"][indices].mean()),
                    obsolete_rate=float(np.mean([int(rows[i]["obsolete"]) for i in indices]))))
    return result


def verify_design(design):
    require([(r["parent"], r["arm"], r["segment_len"]) for r in design["runs"]] == list(RUNS), "Run design differs")
    training, evaluation, analysis = design["training"], design["evaluation"], design["analysis"]
    for field, value in dict(steps=1600, answers_per_batch=16, unique_contexts_per_batch=8,
                              train_seed=89102, lr=.0003, optimizer="AdamW", weight_decay=.01, gradient_clip=1).items():
        require(training[field] == value, f"Design differs: training {field}")
    require(training["checkpoints"] == list(STEPS), "Checkpoint design differs")
    for field, value in dict(dev_seed=89103, gate_episodes=64, strict_episodes=128).items():
        require(evaluation[field] == value, f"Design differs: evaluation {field}")
    require(analysis["bootstrap_draws"] == 5000 and analysis["bootstrap_seed"] == 89104,
            "Bootstrap design differs")


def run_audit(args):
    root, output = args.root.resolve(), args.out.resolve()
    require(output.is_relative_to(SCRIPT_DIR) and output != SCRIPT_DIR, "--out must be a new subdirectory of analysis")
    require(not output.exists(), "--out must not already exist")
    design = jread(root / "design.json")
    verify_design(design)
    folders = {f"{arm}{segment}-{seed}": root / f"{arm}{segment}-{seed}" for seed, arm, segment in RUNS}
    require(all((folder / "metadata.json").is_file() for folder in folders.values()), "Download incomplete: missing run metadata")
    parents = locate_parents(args.parents, design)
    metas, source_manifest = {}, None
    for name, folder in folders.items():
        meta = jread(folder / "metadata.json")
        manifest = source_audit(folder, meta, args.expected_commit, args.repo)
        require(source_manifest is None or manifest == source_manifest, "Run source manifests differ")
        source_manifest = manifest
        require((folder / "design.json").is_file() and sha(folder / "design.json") == sha(root / "design.json"),
                "Run design snapshot differs")
        require(meta.get("design_sha256") == sha(root / "design.json"), "Run design hash metadata differs")
        metas[name] = meta
    streams = load_stream_module(next(iter(folders.values())) / "sources/clmm/pretrain/fact_streams.py")
    require(streams.RECORD_BYTES == 32 and streams.QUERY_PREFIX_BYTES == 31 and streams.VALUES == "abcdefghijklmnop",
            "Frozen stream constants differ")
    training, checkpoints, gate_vectors, strict_vectors = {}, {}, {}, {}
    strict_metrics, gate_metrics, ablations, budgets, noop_counts = [], [], [], [], []
    reference_data_hashes = {}
    for seed, arm, segment in RUNS:
        name = f"{arm}{segment}-{seed}"
        folder, meta = folders[name], metas[name]
        print(f"Auditing {name}", flush=True)
        arguments = meta["args"]
        expected_args = dict(steps=1600, batch_size=16, train_seed=89102, dev_seed=89103,
            gate_episodes=64, strict_episodes=128, lr=.0003, segment_len=segment, arm=arm,
            mode="paired_acquisition", suite="gate", block_size=4, buffer_size=8, max_seconds=600)
        require(all(arguments.get(key) == value for key, value in expected_args.items()), "Run arguments differ from design")
        require(meta["completed_steps"] == 1600, "Run did not complete fixed endpoint")
        require(0 < float(meta["elapsed_seconds"]) <= design["resources"]["max_seconds_per_run"],
                "Run exceeded its declared wall-clock limit")
        environment = meta["environment"]
        require(environment["dtype"] == "float32" and environment["deterministic_algorithms"] is True and
                environment["tf32"] is False and str(environment["device"]).startswith("cuda"),
                "Main run is not the declared deterministic float32 GPU environment")
        checkpoints[name] = checkpoint_audit(folder, meta, parents[seed], design, arm, segment)
        training[name] = training_audit(folder, arm, streams, design)
        record = training[name]
        require(meta["input_bytes"] == record["input_bytes"] and meta["supervised_answers"] == 25600,
                "Run total training budget differs")
        budgets.append(dict(run=name, steps=1600, supervised_answers=25600, input_bytes=record["input_bytes"],
                            training_array_sha256=record["training_array_sha256"],
                            training_plan_sha256=record["training_plan_sha256"]))
        for step in STEPS:
            evaluation = folder / f"step-{step:04d}/evaluation"
            stage_meta = checkpoints[name][step]["evaluation_meta"]
            rows, logits, computed, vectors = check_gate_artifacts(evaluation, stage_meta, streams, design)
            srows, slogits, spairs = check_strict_artifacts(evaluation, streams, design)
            require(stage_meta["strict_summary"] == jread(evaluation / "strict_summary.json"),
                    "Stage strict summary copy differs")
            gate_vectors[(name, step)], strict_vectors[(name, step)] = vectors, spairs
            strict_metrics.extend(strict_table(name, step, spairs))
            gate_metrics.extend(gate_table(name, step, rows, computed))
            # Inputs must remain identical across all fixed stages/runs.
            for filename in ("evaluation_data_canonical.sha256", "strict_episodes.json"):
                digest = sha(evaluation / filename)
                require(filename not in reference_data_hashes or reference_data_hashes[filename] == digest,
                        f"Evaluation population changed between checkpoints: {filename}")
                reference_data_hashes[filename] = digest
            if segment == 128:
                noop_counts.append(dict(run=name, step=step,
                    gate_query_pairs=require_noop(rows, logits, ("cell", "family", "episode", "query_index"), "gate"),
                    strict_query_pairs=require_noop(srows, slogits, ("family", "episode", "rank"), "strict")))
            for family in FAMILIES:
                normal = {key[3]: value for key, value in vectors.items()
                          if key[:3] == ("acquire_k2", family, "online")}
                reset = {key[3]: value for key, value in vectors.items()
                         if key[:3] == ("acquire_k2", family, "reset_memory")}
                a, b = paired_vectors(normal, reset, "accuracy")
                ablations.append(dict(run=name, step=step, cell="acquire_k2", family=family,
                    comparison="online_minus_reset_memory", **bootstrap_difference(a, b)))
    # Explicit context/byte/answer matching, all parents and attention geometries.
    reference_training = training["duplicate32-68101"]
    for name, record in training.items():
        require(record["contexts"] == reference_training["contexts"] and
                record["signatures"] == reference_training["signatures"] and
                record["input_bytes"] == reference_training["input_bytes"], f"Base context/budget matching failed {name}")
    for seed in (68101, 68105):
        left, right = f"paired32-{seed}", f"duplicate32-{seed}"
        require(np.array_equal(training[left]["initial_input_bytes"], training[right]["initial_input_bytes"]) and
                np.array_equal(training[left]["initial_targets"], training[right]["initial_targets"]), "K1 training arrays differ")
        tensor_state_equal(checkpoints[left][200]["state"], checkpoints[right][200]["state"],
                           f"Strong invariant K1 step200 paired/duplicate parent {seed}")
    primary = []
    for seed in (68101, 68105):
        for step in STEPS:
            left = {key[2]: value for key, value in strict_vectors[(f"paired32-{seed}", step)].items()
                    if key[:2] == ("seen", "online")}
            right = {key[2]: value for key, value in strict_vectors[(f"duplicate32-{seed}", step)].items()
                     if key[:2] == ("seen", "online")}
            a, b = paired_vectors(left, right, "both_correct")
            primary.append(dict(parent=seed, step=step, primary_endpoint=step == 1600,
                metric="strict_seen_both_correct", comparison="paired_minus_duplicate", **bootstrap_difference(a, b)))
    decisions = []
    for seed in (68101, 68105):
        name = f"paired32-{seed}"
        natural = next(row for row in gate_metrics if (row["run"], row["step"], row["cell"], row["family"], row["condition"]) ==
                       (name, 1600, "acquire_k2", "seen", "online"))
        revision = next(row for row in gate_metrics if (row["run"], row["step"], row["cell"], row["family"], row["condition"]) ==
                        (name, 1600, "revise_k1", "seen", "online"))
        strict = next(row for row in strict_metrics if (row["run"], row["step"], row["family"], row["condition"]) ==
                      (name, 1600, "seen", "online"))
        ablation = next(row for row in ablations if (row["run"], row["step"], row["family"]) == (name, 1600, "seen"))
        decision = dict(parent=seed, run=name, step=1600, natural_k2_accuracy=natural["accuracy"],
            strict_seen_both_correct=strict["both_correct"], k2_ablation_lower=ablation["lower"],
            revision_seen_accuracy=revision["accuracy"], natural_k2_pass=natural["accuracy"] >= .8,
            strict_pass=strict["both_correct"] >= .8, ablation_pass=ablation["lower"] > 0,
            revision_pass=revision["accuracy"] >= .8)
        decision["all_pass"] = all(decision[field] for field in ("natural_k2_pass", "strict_pass", "ablation_pass", "revision_pass"))
        decisions.append(decision)
    finals = [row for row in primary if row["step"] == 1600]
    conclusion = dict(pressure_continuation_qualified=all(row["all_pass"] for row in decisions),
        useful_pairing_lead=all(row["difference"] >= .05 and row["lower"] > 0 for row in finals),
        auxiliary128_is_memory_baseline=False, final_step_primary=1600,
        inference_scope="Fixed parents and sampled dev episodes only; no pooled training-seed uncertainty.")
    output.mkdir(parents=True)
    for filename, table in (("strict_metrics.csv", strict_metrics), ("gate_metrics.csv", gate_metrics),
                            ("primary_differences.csv", primary), ("k2_ablation_differences.csv", ablations),
                            ("gate_decisions.csv", decisions), ("training_budgets.csv", budgets),
                            ("reset128_exact_noop.csv", noop_counts)):
        csvwrite(output / filename, table)
    jwrite(output / "decisions.json", conclusion)
    inventory = {str(path.relative_to(root)): sha(path) for folder in folders.values()
        for path in folder.rglob("*") if path.is_file()}
    jwrite(output / "artifact_inventory_sha256.json", inventory)
    jwrite(output / "audit.json", dict(status="passed", design_sha256=sha(root / "design.json"),
        expected_commit=args.expected_commit, source_commits=sorted({meta["repo_commit"] for meta in metas.values()}),
        source_manifest=source_manifest, runs=5, stages_per_run=3, total_training_answers=5 * 25600,
        gate_queries=sum(row["queries"] for row in gate_metrics), strict_queries=sum(row["queries"] for row in strict_metrics),
        strong_k1_tensor_equality_parents=[68101, 68105], parent_paths={str(k): v["path"] for k, v in parents.items()},
        reset128_noop_checks=noop_counts, analysis_sha256=sha(__file__),
        limitations="Saved-artifact integrity and independent scoring; no model forward, training rerun, or reproduction claim."))
    # Packaging alias: written only after the complete five-run audit succeeds.
    jwrite(output / "verification.json", dict(status="verified", scope="saved artifacts only",
        main_run_count=5, stages_per_run=3, source_commit=args.expected_commit,
        audit_sha256=sha(output / "audit.json"), artifact_inventory_sha256=sha(output / "artifact_inventory_sha256.json"),
        model_execution_performed=False, training_reproduced=False))
    lines = ["Saved-artifact audit passed.", "", "Final strict seen both-correct, paired minus duplicate:"]
    for row in finals:
        lines.append(f"- Parent {row['parent']}: {100*row['left']:.2f}% vs {100*row['right']:.2f}%; "
                     f"difference {100*row['difference']:+.2f} pp [95% {100*row['lower']:+.2f}, {100*row['upper']:+.2f}].")
    lines.extend(["", f"Both paired32 final acquisition gates pass: {conclusion['pressure_continuation_qualified']}.",
        f"Prespecified useful pairing lead: {conclusion['useful_pairing_lead']}.", "",
        "Intervals resample complete paired episodes separately for each fixed parent. Stage scores are descriptive.",
        "The 128-byte run is an attention-access diagnostic; its exact reset no-op cannot establish memory-dependent acquisition.",
        "This audit does not rerun training or execute checkpoint inference."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps(dict(status="passed", output=str(output), **conclusion), indent=2), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True, help="Root containing design and five downloaded run directories")
    result.add_argument("--out", type=Path, required=True, help="NEW child directory of this analysis directory")
    result.add_argument("--parents", type=Path, required=True, help="Original parent file tree, searched by design hashes")
    result.add_argument("--expected-commit", default=FROZEN_COMMIT,
                        help="Require every run to report this exact source commit (defaults to frozen GPU commit)")
    result.add_argument("--repo", type=Path, help="Optional Git object database for exact source-to-commit checks")
    return result


if __name__ == "__main__":
    try:
        run_audit(parser().parse_args())
    except AuditError as error:
        print(f"AUDIT FAILED: {error}", file=sys.stderr)
        raise SystemExit(2)
