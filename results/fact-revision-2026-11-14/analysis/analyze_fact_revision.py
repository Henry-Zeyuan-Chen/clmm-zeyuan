"""Independent saved-evidence audit; never imports experiment/model generators.

Main: --root ROOT --out ANALYSIS/NEW --expected-commit FULL_SHA
Self checks: --selfcheck --out ANALYSIS/NEW
Smoke only: --root ROOT --smoke-run RUN --out ANALYSIS/NEW

The audit reconstructs serialized inputs and statistics, not neural inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
VALUES = "abcdefghijklmnop"
SEEDS = ("68101", "68105")
SHORT = ("acquire_k1", "acquire_k2", "revise_k1", "counterfactual_k2")
ARMS = ("online", "repeat_current", "continuous", "periodic")
RECORDS = {
    "seen": ("fact {key} has value {value}.\n", "key {key} has code {value}.\n"),
    "heldout": ("value for {key} is {value}.\n", "{key} maps to value {value}.\n"),
}
QUERIES = {
    "seen": ("value of key {key}: ", "code for key {key} = "),
    "heldout": ("recall value for {key}: ", "key {key} now has value = "),
}
ORDER = sorted((f"{k:03d}" for k in range(1000)),
               key=lambda k: (hashlib.sha256(("clmm.fact.keys.v1:" + k).encode()).digest(), k))
POOLS = {"train": ORDER[:600], "dev": ORDER[600:800], "test": ORDER[800:]}
POOL_HASHES = {k: hashlib.sha256("\n".join(v).encode()).hexdigest() for k, v in POOLS.items()}
KEY_SCHEME = "sha256(clmm.fact.keys.v1:+key), key tie-break; ranked 600/200/200"
SCOPE = "Saved-artifact reconstruction and independent statistics; no neural inference or training rerun."
NUM_TOL = 3e-5  # CPU float64 reconstruction versus saved float32 torch reductions.


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def equal(actual, expected, what):
    require(actual == expected, f"{what}: expected {expected!r}, got {actual!r}")


def near(actual, expected, what, tolerance=NUM_TOL):
    a, b = float(actual), float(expected)
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance,
            f"{what}: {a} != {b} within {tolerance}")


def checked_hash(path, expected, what):
    actual = sha(path)
    equal(actual, expected, what)
    return actual


def domain_rng(seed, domain):
    raw = hashlib.sha256(f"clmm-facts-v1:{domain}:{seed}".encode("ascii")).digest()
    return random.Random(int.from_bytes(raw, "big"))


def record_bytes(event):
    family, index = event["template_family"], event["template_index"]
    require(family in RECORDS and type(index) is int and index in (0, 1), "Bad record template")
    key, value = event["key"], event["value"]
    require(re.fullmatch(r"[0-9]{3}", key) is not None and value in VALUES and len(value) == 1,
            "Malformed record key/value")
    text = RECORDS[family][index].format(key=key, value=value).encode("ascii")
    require(len(text) <= 32, "Record is longer than one attention segment")
    return text.rjust(32, b" ")


def query_bytes(query):
    family, index = query["template_family"], query["template_index"]
    require(family in QUERIES and index in (0, 1), "Bad query template")
    text = QUERIES[family][index].format(key=query["key"]).encode("ascii")
    require(len(text) <= 31, "Query longer than 31 bytes")
    return text.rjust(31, b" ")


def validate_episode(ep, split="dev"):
    equal(ep["format_version"], 1, "episode format")
    equal(ep["split"], split, "episode split")
    equal(ep["key_pool_scheme"], KEY_SCHEME, "key namespace specification")
    equal(ep["key_pool_sha256"], POOL_HASHES[split], "key namespace hash")
    family, k, d, fraction = ep["template_family"], ep["facts"], ep["delay"], ep["revision_fraction"]
    require(family in RECORDS and (split != "train" or family == "seen"), "Template split violation")
    require(type(k) is int and k >= 1 and type(d) is int and d >= 0, "Invalid K/D")
    require(0 <= fraction <= 1 and math.isfinite(fraction), "Invalid revision fraction")
    nrev = math.floor(k * fraction)
    events, queries = ep["events"], ep["queries"]
    equal(len(events), k + nrev + d, "online event count")
    latest, initial, revised = {}, {}, set()
    for i, event in enumerate(events):
        equal(event["event_id"], i, "source event index")
        require(event["key"] in POOLS[split], "Event key outside declared namespace")
        equal(event["template_family"], family, "record family")
        expected_kind = "fact" if i < k else "revision" if i < k + nrev else "distractor"
        equal(event["kind"], expected_kind, "online event kind")
        equal(event["version"], 2 if expected_kind == "revision" else 1, "event version")
        record_bytes(event)
        if expected_kind == "revision":
            require(event["key"] in initial and event["key"] not in revised, "Invalid revised key")
            require(event["value"] != initial[event["key"]]["value"], "Revision repeats old value")
            revised.add(event["key"])
        else:
            require(event["key"] not in latest, "Repeated initial/distractor key")
            if expected_kind == "fact":
                initial[event["key"]] = event
        latest[event["key"]] = event
    source_events = events[:k] + ([events[-1]] if d and ep["include_recent"] else [])
    equal(len(queries), len(source_events), "query count")
    for query, source in zip(queries, source_events):
        key = source["key"]
        final = latest[key]
        category = "recent" if source["kind"] == "distractor" else "revised" if key in revised else "stable"
        for field, value in {"key": key, "target": final["value"], "category": category,
                             "old_value": source["value"] if category == "revised" else None,
                             "initial_event_id": source["event_id"], "latest_event_id": final["event_id"],
                             "template_family": family}.items():
            equal(query[field], value, f"online-only query {field}")
        query_bytes(query)
    return latest


def reference_schedule(ep, condition, block_size=4, capacity=8):
    """Independent reconstruction from online history and the specified RNG seed."""
    require(condition in ARMS, "Unknown schedule condition")
    buffer, latest, result, blocks, reference = {}, {}, [], [], []
    rng = domain_rng(ep["seed"], f"replay:{ep['seed']}")
    events = ep["events"]

    def buffer_rows():
        return [{"key": e["key"], "version": e["version"], "event_id": e["event_id"]}
                for e in buffer.values()]

    def add(event, kind, block, after):
        if kind == "online":
            latest[event["key"]] = event
            buffer.pop(event["key"], None)
            buffer[event["key"]] = event
            if len(buffer) > capacity:
                del buffer[next(iter(buffer))]
        require(event["event_id"] <= after, "Future source in replay")
        result.append(dict(execution_index=len(result), source_event_id=event["event_id"],
                           key=event["key"], value=event["value"], version=event["version"], kind=kind,
                           block_index=block, after_online_event_id=after,
                           stale=kind == "replay" and latest[event["key"]]["version"] > event["version"]))

    for number, start in enumerate(range(0, len(events), block_size)):
        online = events[start:start + block_size]
        snapshot = list(buffer.values())
        selection = [rng.choice(snapshot) for _ in online] if snapshot else []
        require(all(e["event_id"] < start for e in selection), "Noncausal snapshot")
        ids = [e["event_id"] for e in selection]
        reference.extend(ids)
        block = dict(block_index=number, online_event_ids=[e["event_id"] for e in online],
                     buffer_at_start=buffer_rows(), reference_replay_event_ids=ids,
                     selected_replay_event_ids=[], extra_event_ids=[])
        for rank, event in enumerate(online):
            add(event, "online", number, event["event_id"])
            if selection and condition in ("continuous", "repeat_current"):
                chosen = selection[rank] if condition == "continuous" else event
                kind = "replay" if condition == "continuous" else "repeat_current"
                add(chosen, kind, number, event["event_id"])
                block["extra_event_ids"].append(chosen["event_id"])
                if kind == "replay":
                    block["selected_replay_event_ids"].append(chosen["event_id"])
        if condition == "periodic":
            for chosen in selection:
                add(chosen, "replay", number, online[-1]["event_id"])
                block["selected_replay_event_ids"].append(chosen["event_id"])
                block["extra_event_ids"].append(chosen["event_id"])
        block["buffer_at_end"] = buffer_rows()
        blocks.append(block)
    return result, blocks, buffer_rows(), reference


def verify_sample(sample, ep, block_size=4, capacity=8):
    row, audit = sample["row"], sample["audit"]
    qi = row["query_index"]
    require(type(qi) is int and 0 <= qi < len(ep["queries"]), "Invalid query index")
    query = ep["queries"][qi]
    condition = "online" if row["condition"] == "reset_memory" else row["condition"]
    plan, blocks, buffer, reference = reference_schedule(ep, condition, block_size, capacity)
    for name, expected in {"execution_plan": plan, "blocks": blocks, "final_buffer": buffer,
                           "reference_replay_event_ids": reference, "query": query,
                           "schedule_seed": ep["seed"], "episode_seed": ep["seed"], "split": "dev",
                           "condition": condition, "block_size": block_size, "buffer_size": capacity,
                           "key_pool_sha256": POOL_HASHES["dev"], "key_pool_hashes": POOL_HASHES}.items():
        equal(audit[name], expected, f"sample audit {name}")
    encoded = b"".join(record_bytes(ep["events"][event["source_event_id"]]) for event in plan)
    prefix = encoded + query_bytes(query)
    equal(sample["prefix"].encode("ascii"), prefix, "reconstructed prefix (answer excluded)")
    require(len(prefix) % 32 == 31 and len(encoded) % 32 == 0, "Query not isolated at boundary")
    equal(audit["query_prefix"].encode("ascii"), query_bytes(query), "label-independent query prefix")
    n_online = len(ep["events"])
    replay = [e for e in plan if e["kind"] == "replay"]
    repeats = [e for e in plan if e["kind"] == "repeat_current"]
    extra = len(replay) + len(repeats)
    matching = [e for e in plan if e["key"] == query["key"]]
    target_replays = [e for e in matching if e["kind"] == "replay"]
    age = len(plan) - 1 - matching[-1]["execution_index"]
    counts = dict(online_record_count=n_online, replay_record_count=len(replay),
                  repeat_current_record_count=len(repeats), extra_record_count=extra,
                  total_record_count=len(plan), online_token_count=32*n_online,
                  replay_token_count=32*len(replay), repeat_current_token_count=32*len(repeats),
                  extra_token_count=32*extra, record_token_count=32*len(plan),
                  input_token_count=len(prefix), query_prefix_token_count=31, target_token_count=1,
                  training_sequence_token_count=len(prefix)+1, query_start_token=len(encoded),
                  query_attention_segment=len(plan), answer_byte=ord(query["target"]),
                  prefix_sha256=digest_bytes(prefix), record_bytes=32, query_prefix_bytes=31,
                  selected_replay_event_ids=[e["source_event_id"] for e in replay],
                  extra_event_ids=[e["source_event_id"] for e in plan if e["kind"] != "online"],
                  stale_replay_count=sum(e["stale"] for e in plan),
                  online_events_since_latest_target=n_online-1-query["latest_event_id"],
                  records_since_last_target_key_record=age,
                  last_target_key_record_source_event_id=matching[-1]["source_event_id"],
                  last_target_key_record_version=matching[-1]["version"],
                  target_in_final_buffer=any(e["key"] == query["key"] for e in buffer))
    for name, expected in counts.items():
        equal(audit[name], expected, f"sample audit {name}")
    expected_row = dict(facts=ep["facts"], delay=ep["delay"], episode_seed=ep["seed"],
                        family=ep["template_family"], key=query["key"], category=query["category"],
                        target=ord(query["target"]), old_target=ord(query["old_value"]) if query["old_value"] else -1,
                        input_length=len(prefix), prefix_sha256=digest_bytes(prefix), online_records=n_online,
                        extra_records=extra, replay_records=len(replay), stale_replays=sum(e["stale"] for e in plan),
                        target_replays=len(target_replays), target_stale_replays=sum(e["stale"] for e in target_replays),
                        target_resident=counts["target_in_final_buffer"], online_age=counts["online_events_since_latest_target"],
                        last_occurrence_age=age, last_observed_value=ord(ep["events"][-1]["value"]),
                        last_presented_value=ord(plan[-1]["value"]), last_key_presented_value=ord(matching[-1]["value"]))
    for name, expected in expected_row.items():
        equal(row[name], expected, f"sample row {name}")
    return row


def row_key(row):
    return (row["cell"], row["family"], int(row["episode"]), int(row["query_index"]), row["condition"])


def expected_specs(args):
    result = {}
    for family in ("seen", "heldout"):
        for cell, k, rev in (("acquire_k1", 1, 0.), ("acquire_k2", 2, 0.),
                             ("revise_k1", 1, 1.), ("counterfactual_k2", 2, 0.)):
            result[cell, family] = dict(cell=cell, facts=k, delay=0, revision_fraction=rev, family=family,
                                        episodes=args["gate_episodes"], conditions=["online", "reset_memory"], all_queries=True)
    if args["suite"] == "full":
        for k in (4, 8, 16):
            for d in (0, 8, 32):
                cell = f"pressure_k{k}_d{d}"
                result[cell, "seen"] = dict(cell=cell, facts=k, delay=d, revision_fraction=.5, family="seen",
                    episodes=args["primary_episodes"] if (k, d) == (8, 8) else args["grid_episodes"],
                    conditions=list(ARMS), all_queries=False)
    return result


def independent_scores(logits, targets):
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    require(values.ndim == 2 and values.shape[1] == 256 and values.shape[0] == len(targets), "Answer-logit shape")
    require(np.isfinite(values).all(), "Nonfinite answer logits")
    require(((targets >= 97) & (targets <= 112)).all(), "Target outside a-p")
    shifted = values - values.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    legal = values[:, 97:113]
    shifted_legal = legal - legal.max(axis=1, keepdims=True)
    legal_log_probs = shifted_legal - np.log(np.exp(shifted_legal).sum(axis=1, keepdims=True))
    indices = np.arange(len(targets))
    return dict(prediction=values.argmax(axis=1), forced_prediction=legal.argmax(axis=1)+97,
                nll=-log_probs[indices, targets], forced_nll=-legal_log_probs[indices, targets-97],
                legal_probability_mass=np.exp(log_probs[:, 97:113]).sum(axis=1))


def finite_state(state):
    import torch
    require(isinstance(state, dict) and bool(state), "Empty checkpoint state")
    require(all(isinstance(t, torch.Tensor) and bool(torch.isfinite(t).all()) for t in state.values()),
            "Invalid or nonfinite checkpoint tensor")


def load_checkpoint_evidence(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True)
    equal(obj["format_version"], 1, "checkpoint format")
    equal(obj["architecture"], "clmm.vendored_mac.byte_lm.v1", "checkpoint architecture")
    for key, value in dict(num_tokens=256, segment_len=32, memory_segment_len=8, memory_batch_size=32).items():
        equal(obj["config"][key], value, "checkpoint geometry " + key)
    finite_state(obj["model_state"])
    return obj


def check_common(run, checkpoint, expected_commit, design_hash=None, strict=True):
    meta = read_json(run / "metadata.json")
    equal(meta["status"], "succeeded", f"{run.name} completion")
    if expected_commit:
        equal(meta["repo_commit"], expected_commit, f"{run.name} code commit")
    checked_hash(checkpoint, meta["checkpoint_sha256"], f"{run.name} input checkpoint")
    equal(meta["key_pool_sha256"], POOL_HASHES, "metadata key pools")
    require(meta["source_sha256"], "Missing source snapshot hashes")
    for name, expected in meta["source_sha256"].items():
        path = (run / "sources" / name).resolve()
        require(path.is_relative_to((run / "sources").resolve()), "Source path escapes snapshot")
        checked_hash(path, expected, f"{run.name} source {name}")
    if "design_sha256" in meta:
        checked_hash(run / "design.json", meta["design_sha256"], "saved design")
    if strict:
        equal(meta.get("design_sha256"), design_hash, "prospective design hash")
        equal(meta["parameter_count"], 839248, "reported shared parameter count")
        env = meta["environment"]
        equal(env["dtype"], "float32", "precision")
        require(env["deterministic_algorithms"] and not env["tf32"], "Numerical protocol changed")
    return meta


def validate_eval(run, checkpoint, expected_commit, design, stage, strict=True):
    meta = check_common(run, checkpoint, expected_commit, design.get("_sha256"), strict)
    load_checkpoint_evidence(checkpoint)
    equal(meta["mode"], "evaluate", "evaluation mode")
    args = meta["args"]
    if strict:
        for key in ("dev_seed", "gate_episodes", "grid_episodes", "primary_episodes"):
            equal(args[key], design["evaluation"][key], f"eval argument {key}")
        equal(args["suite"], "gate" if stage == "before" else "full", "evaluation suite")
        equal(args["block_size"], design["replay"]["block_records"], "replay block")
        equal(args["buffer_size"], design["replay"]["buffer_keys"], "replay buffer")
    require(meta["checkpoint_unchanged"], "Reported outer-state mutation during evaluation")
    for file, field in (("evaluation_data.jsonl.gz", "evaluation_data_sha256"),
                         ("answer_logits.npy", "answer_logits_sha256"), ("predictions.csv", "predictions_sha256")):
        checked_hash(run / file, meta[field], file)
    raw = gzip.decompress((run / "evaluation_data.jsonl.gz").read_bytes())
    canonical = digest_bytes(raw)
    equal(canonical, meta["evaluation_data_canonical_sha256"], "canonical dataset hash")
    equal((run / "evaluation_data_canonical.sha256").read_text().strip(), canonical, "canonical hash sidecar")
    specs, actual_specs, episodes, ledger = expected_specs(args), {}, {}, {}
    current = None
    for line in raw.decode("utf8").splitlines():
        item = json.loads(line)
        if item["type"] == "episodes":
            spec = item["spec"]
            current = (spec["cell"], spec["family"])
            require(current in specs and current not in actual_specs, "Unexpected or duplicate dataset cell")
            equal(spec, specs[current], "dataset specification")
            actual_specs[current] = spec
            equal(len(item["episodes"]), spec["episodes"], "episode count")
            group = 1 if spec["cell"] == "revise_k1" else 2 if spec["cell"].startswith("pressure") else 0
            rng = np.random.default_rng(args["dev_seed"] + spec["facts"]*101 + group*10007)
            seeds = rng.integers(2_000_000_000, 3_000_000_000, size=spec["episodes"])
            require(len(set(map(int, seeds))) == len(seeds), "Repeated episode seed")
            for i, ep in enumerate(item["episodes"]):
                equal(ep["seed"], int(seeds[i]), "prespecified semantic episode seed")
                for name, val in (("facts", spec["facts"]), ("delay", spec["delay"]),
                                  ("revision_fraction", spec["revision_fraction"]), ("template_family", spec["family"])):
                    equal(ep[name], val, "episode cell agreement")
                validate_episode(ep)
                episodes[current + (i,)] = ep
        elif item["type"] == "sample":
            row = item["row"]
            require(current == (row["cell"], row["family"]), "Sample outside its dataset")
            ep = episodes[current + (row["episode"],)]
            verify_sample(item, ep, args["block_size"], args["buffer_size"])
            key = row_key(row)
            require(key not in ledger, "Duplicate semantic sample")
            ledger[key] = row
        else:
            raise AuditError("Unknown dataset line type")
    equal(set(actual_specs), set(specs), "complete cell coverage")
    expected_keys = set()
    for (cell, family, index), ep in episodes.items():
        spec = specs[cell, family]
        if spec["all_queries"]:
            indices = list(range(len(ep["queries"])))
        else:
            rng = np.random.default_rng(ep["seed"] + 97)
            indices = []
            for category in ("stable", "revised", "recent"):
                candidates = [i for i, q in enumerate(ep["queries"]) if q["category"] == category]
                if candidates:
                    indices.append(int(rng.choice(candidates)))
        expected_keys.update((cell, family, index, qi, arm) for qi in indices for arm in spec["conditions"])
    equal(set(ledger), expected_keys, "complete predetermined query/arm coverage")
    # Strong content/counterfactual/delay pairing checks, independent of metric rows.
    for family in ("seen", "heldout"):
        for i in range(args["gate_episodes"]):
            original, changed = episodes["acquire_k2", family, i], episodes["counterfactual_k2", family, i]
            equal(original["seed"], changed["seed"], "counterfactual seed")
            for a, b in zip(original["events"], changed["events"]):
                equal({k: v for k, v in a.items() if k != "value"},
                      {k: v for k, v in b.items() if k != "value"}, "counterfactual unchanged event fields")
                equal(b["value"], VALUES[(VALUES.index(a["value"])+1) % 16], "counterfactual value rotation")
            for a, b in zip(original["queries"], changed["queries"]):
                equal(query_bytes(a), query_bytes(b), "counterfactual query unchanged")
                equal(b["target"], VALUES[(VALUES.index(a["target"])+1) % 16], "counterfactual label rotation")
    for cell in SHORT:
        for i in range(args["gate_episodes"]):
            seen, held = episodes[cell, "seen", i], episodes[cell, "heldout", i]
            for field in ("events", "queries"):
                equal([{k: v for k, v in x.items() if k != "template_family"} for x in seen[field]],
                      [{k: v for k, v in x.items() if k != "template_family"} for x in held[field]], "template semantic pairing")
    if args["suite"] == "full":
        for k in (4, 8, 16):
            for i in range(min(args["grid_episodes"], args["primary_episodes"])):
                previous = episodes[f"pressure_k{k}_d0", "seen", i]
                for d in (8, 32):
                    following = episodes[f"pressure_k{k}_d{d}", "seen", i]
                    equal(previous["events"], following["events"][:len(previous["events"])], "paired online delay prefixes")
                    equal(previous["queries"][:k], following["queries"][:k], "paired initial queries across delay")
                    previous = following
    predictions = csv_rows(run / "predictions.csv")
    logits = np.load(run / "answer_logits.npy", allow_pickle=False)
    equal(logits.shape, (len(predictions), 256), "CSV/logit row count")
    equal(len(predictions), meta["evaluation_queries"], "metadata evaluation query count")
    equal(len(predictions), len(ledger), "dataset/CSV count")
    require(len({row_key(r) for r in predictions}) == len(predictions), "Duplicate CSV sample")
    require({row_key(r) for r in predictions} == set(ledger), "CSV semantic keys differ from dataset")
    metrics = independent_scores(logits, [int(r["target"]) for r in predictions])
    clean = []
    for i, record in enumerate(predictions):
        expected = ledger[row_key(record)]
        for name, value in expected.items():
            equal(record[name], str(value), f"CSV/dataset row {i} {name}")
        row = dict(expected)
        for name, values in metrics.items():
            value = values[i].item()
            if name in ("prediction", "forced_prediction"):
                equal(int(record[name]), value, f"logits/CSV {name}")
            else:
                near(record[name], value, f"logits/CSV {name}")
            row[name] = value
        row.update(correct=int(row["prediction"] == row["target"]),
                   forced_correct=int(row["forced_prediction"] == row["target"]),
                   obsolete=int(row["old_target"] >= 0 and row["prediction"] == row["old_target"]),
                   last_observed_correct=int(row["last_observed_value"] == row["target"]),
                   last_presented_correct=int(row["last_presented_value"] == row["target"]),
                   last_key_presented_correct=int(row["last_key_presented_value"] == row["target"]))
        for name in ("correct", "forced_correct", "obsolete"):
            equal(int(record[name]), row[name], "recomputed " + name)
        clean.append(row)
    groups = defaultdict(list)
    for row in clean:
        groups[row["cell"], row["family"], row["category"], row["condition"]].append(row)
    summaries = csv_rows(run / "summary.csv")
    equal(len(summaries), len(groups), "summary group count")
    require(len({(r["cell"], r["family"], r["category"], r["condition"]) for r in summaries}) == len(summaries), "Duplicate summary")
    for summary in summaries:
        group = groups[summary["cell"], summary["family"], summary["category"], summary["condition"]]
        equal(int(summary["queries"]), len(group), "summary denominator")
        equal(int(summary["episodes"]), len({r["episode_seed"] for r in group}), "summary independent units")
        for name, metric in dict(accuracy="correct", forced_accuracy="forced_correct", obsolete_rate="obsolete",
                                 nll="nll", legal_probability_mass="legal_probability_mass").items():
            near(summary[name], np.mean([r[metric] for r in group]), "summary " + name)
    return dict(meta=meta, rows=clean, ledger=ledger, episodes=episodes, canonical_hash=canonical)


RECORD_RE = re.compile(r" *(?:fact ([0-9]{3}) has value ([a-p])|key ([0-9]{3}) has code ([a-p]))\.\n")
QUERY_RE = re.compile(r" *(?:value of key ([0-9]{3}): |code for key ([0-9]{3}) = )")


def parse_training_prefix(raw, target, facts, delay, revision_fraction, query_index):
    require(len(raw) % 32 == 31, "Training prefix includes answer or wrong query length")
    chunks, tail = [raw[i:i+32].decode("ascii") for i in range(0, len(raw)-31, 32)], raw[-31:].decode("ascii")
    nrev = math.floor(facts * revision_fraction)
    equal(len(chunks), facts+nrev+delay, "Training online record count")
    decoded = []
    for text in chunks:
        match = RECORD_RE.fullmatch(text)
        require(match is not None, "Training record does not match seen template")
        groups = match.groups()
        key, value = (groups[0], groups[1]) if groups[0] else (groups[2], groups[3])
        require(key in POOLS["train"], "Training namespace leak")
        decoded.append((key, value))
    keys = [key for key, _ in decoded[:facts]]
    require(len(set(keys)) == facts, "Repeated initial training key")
    revisions = decoded[facts:facts+nrev]
    require(len({key for key, _ in revisions}) == nrev, "Repeated training revision key")
    initial = dict(decoded[:facts])
    for key, value in revisions:
        require(key in initial and value != initial[key], "Invalid training revision")
    distractors = decoded[facts+nrev:]
    require(len({key for key, _ in distractors}) == delay and not set(keys) & {key for key, _ in distractors}, "Training distractor key collision")
    match = QUERY_RE.fullmatch(tail)
    require(match is not None, "Training query contains answer or unknown template")
    key = next(value for value in match.groups() if value)
    candidates = keys + ([decoded[-1][0]] if delay else [])
    require(0 <= query_index < len(candidates), "Training query index outside candidates")
    equal(key, candidates[query_index], "Training query matches recorded index")
    equal(int(target), ord(dict(decoded)[key]), "Training target from latest online record")


def next_training_spec(step, steps, rng):
    if step / steps < .2:
        return 1, 0, 0.
    if step / steps < .4:
        return int(rng.choice([1, 2])), int(rng.choice([0, 2])), float(rng.choice([0., .5]))
    return int(rng.choice([1, 2, 4, 8], p=[.15, .25, .35, .25])), int(rng.choice([0, 4, 8])), .5


def validate_training(run, parent_path, expected_commit, design, strict=True):
    meta = check_common(run, parent_path, expected_commit, design.get("_sha256"), strict)
    equal(meta["mode"], "train", "training mode")
    args = meta["args"]
    if strict:
        for key in ("steps", "batch_size", "train_seed", "lr"):
            equal(args[key], design["adaptation"][key], "training argument " + key)
    steps, batch = args["steps"], args["batch_size"]
    equal(meta["completed_steps"], steps, "completed fixed training budget")
    equal(meta["supervised_answers"], steps*batch, "answer budget")
    equal(meta["reload_max_abs_delta"], 0., "saved reload equality")
    checked_hash(run / "training_inputs.npz", meta["training_input_sha256"], "training inputs")
    checked_hash(run / "training_plan.json", meta["training_plan_sha256"], "training plan")
    checked_hash(run / "checkpoint.pt", meta["output_checkpoint_sha256"], "adapted checkpoint")
    parent = load_checkpoint_evidence(parent_path)
    output = load_checkpoint_evidence(run / "checkpoint.pt")
    equal(output["config"], parent["config"], "unchanged model configuration")
    equal(set(output["model_state"]), set(parent["model_state"]), "checkpoint state keys")
    for key, value in output["model_state"].items():
        equal(tuple(value.shape), tuple(parent["model_state"][key].shape), "unchanged checkpoint shape")
    info = output["training_info"]
    equal(info["parent_checkpoint_sha256"], sha(parent_path), "checkpoint parent provenance")
    equal(info["steps"], steps, "checkpoint training steps")
    equal(info["train_seed"], args["train_seed"], "checkpoint training seed")
    require(info["fixed_final_checkpoint"], "Checkpoint is not declared fixed final")
    for name, expected in output["source_info"]["files_sha256"].items():
        # The shared Windows checkpoint used native separators; runner snapshots
        # use POSIX keys on every platform. They identify the same saved source.
        equal(meta["source_sha256"][name.replace("\\", "/")], expected, "checkpoint source provenance")
    plan, trace = read_json(run / "training_plan.json"), csv_rows(run / "training.csv")
    equal(len(plan), steps, "training plan length")
    equal(len(trace), steps, "training trace length")
    with np.load(run / "training_inputs.npz", allow_pickle=False) as archive:
        equal(set(archive.files), {"input_bytes", "step_offsets", "targets"}, "training array names")
        arrays = {key: archive[key] for key in archive.files}
    data, offsets, targets = arrays["input_bytes"], arrays["step_offsets"], arrays["targets"]
    require(data.dtype == np.uint8 and data.ndim == 1, "Training input byte format")
    require(np.issubdtype(offsets.dtype, np.integer), "Training offset format")
    equal(offsets.shape, (steps+1,), "training offsets length")
    equal(targets.shape, (steps, batch), "training target shape")
    require(targets.dtype == np.uint8 and ((targets >= 97) & (targets <= 112)).all(), "Training answer vocabulary")
    equal(int(offsets[0]), 0, "first training offset")
    equal(int(offsets[-1]), len(data), "terminal training offset")
    require((np.diff(offsets) > 0).all(), "Non-increasing training offsets")
    equal(meta["input_bytes"], len(data), "total input byte budget")
    rng = np.random.default_rng(args["train_seed"])
    for i, (step, log) in enumerate(zip(plan, trace)):
        equal(step["step"], i+1, "sequential training plan step")
        k, d, rev = next_training_spec(i, steps, rng)
        equal((step["facts"], step["delay"], step["revision_fraction"]), (k, d, rev), "prespecified curriculum")
        expected_seeds = rng.integers(1, 1_000_000_000, size=batch).tolist()
        expected_queries = [int(rng.integers(k + int(d > 0))) for _ in range(batch)]
        equal(step["episode_seeds"], expected_seeds, "training episode seed plan")
        equal(step["query_indices"], expected_queries, "training query selection plan")
        length = (k + math.floor(k*rev) + d)*32 + 31
        equal(step["input_length"], length, "training prefix length")
        equal(int(offsets[i+1]-offsets[i]), batch*length, "step byte offsets")
        matrix = data[offsets[i]:offsets[i+1]].reshape(batch, length)
        for j, raw in enumerate(matrix):
            parse_training_prefix(raw.tobytes(), targets[i, j], k, d, rev, expected_queries[j])
        for name, expected in dict(step=i+1, facts=k, delay=d, batch=batch,
                                   input_bytes=batch*length, supervised_answers=batch).items():
            equal(int(log[name]), expected, "training trace " + name)
        near(log["revision_fraction"], rev, "training trace revision fraction", 0.)
        for field in ("loss", "gradient_norm", "elapsed_seconds", "accuracy"):
            require(math.isfinite(float(log[field])), "Nonfinite training trace")
        require(0 <= float(log["accuracy"]) <= 1 and float(log["gradient_norm"]) >= 0, "Invalid training trace metric")
    canonical = digest_bytes(data.tobytes() + offsets.astype("<i8").tobytes() + targets.tobytes())
    return dict(meta=meta, plan=plan, arrays=arrays, canonical_hash=canonical)


def boot_ratio(sums, counts, draws, seed):
    sums, counts = np.asarray(sums, dtype=float), np.asarray(counts, dtype=float)
    require(sums.ndim == counts.ndim == 1 and len(sums) == len(counts) and len(sums) > 0,
            "Invalid bootstrap units")
    require(np.isfinite(sums).all() and np.isfinite(counts).all() and (counts > 0).all(), "Invalid bootstrap values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sums), size=(draws, len(sums)))
    samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return float(sums.sum()/counts.sum()), float(low), float(high)


def cluster_estimate(rows, metric, draws, seed):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["episode_seed"]].append(float(row[metric]))
    ordered = [clusters[key] for key in sorted(clusters)]
    point, low, high = boot_ratio([sum(x) for x in ordered], [len(x) for x in ordered], draws, seed)
    return dict(estimate=point, ci_low=low, ci_high=high, episodes=len(ordered), queries=len(rows))


def pair_rows(left, right):
    def keyed(rows):
        result = {}
        for row in rows:
            key = (row["episode_seed"], row["query_index"])
            require(key not in result, "Duplicate paired question")
            result[key] = row
        return result
    a, b = keyed(left), keyed(right)
    equal(set(a), set(b), "paired question sets")
    for key in sorted(a):
        equal((a[key]["key"], a[key]["target"], a[key]["category"]),
              (b[key]["key"], b[key]["target"], b[key]["category"]), "paired semantic labels")
    return [(a[key], b[key]) for key in sorted(a)]


def paired_estimate(left, right, metric, draws, seed):
    differences = [dict(episode_seed=a["episode_seed"], difference=float(a[metric])-float(b[metric]))
                   for a, b in pair_rows(left, right)]
    return cluster_estimate(differences, "difference", draws, seed)


def analysis_seed(base, name):
    return int.from_bytes(hashlib.sha256(f"{base}:{name}".encode()).digest()[:8], "big")


def analyze(evaluations, design):
    draws, base = design["analysis"]["bootstrap_draws"], design["analysis"]["bootstrap_seed"]
    estimates, contrasts, groups, gates, primary = [], [], {}, {}, {}
    metrics = ("correct", "forced_correct", "nll", "forced_nll", "legal_probability_mass",
               "last_observed_correct", "last_presented_correct", "last_key_presented_correct")
    for (seed, stage), data in evaluations.items():
        for row in data["rows"]:
            key = (seed, stage, row["cell"], row["family"], row["category"], row["condition"])
            groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        seed, stage, cell, family, category, condition = key
        for metric in metrics + (("obsolete",) if category == "revised" else ()):
            stats = cluster_estimate(rows, metric, draws, analysis_seed(base, repr((key, metric))))
            estimates.append(dict(checkpoint_seed=seed, stage=stage, cell=cell, family=family,
                                  category=category, condition=condition, metric=metric, **stats,
                                  uncertainty="paired-capable episode bootstrap; fixed checkpoint conditional"))
    def contrast(seed, stage, cell, family, category, left, right, metric="correct", role="exploratory"):
        a = groups[seed, stage, cell, family, category, left]
        b = groups[seed, stage, cell, family, category, right]
        stats = paired_estimate(a, b, metric, draws, analysis_seed(base, repr((seed, stage, cell, family, category, left, right, metric))))
        result = dict(checkpoint_seed=seed, stage=stage, cell=cell, family=family, category=category,
                      left=left, right=right, metric=metric, role=role, **stats)
        contrasts.append(result)
        return result
    for seed in SEEDS:
        gates[seed] = {}
        for stage in ("before", "after"):
            for family in ("seen", "heldout"):
                for cell in SHORT:
                    category = "revised" if cell == "revise_k1" else "stable"
                    difference = contrast(seed, stage, cell, family, category, "online", "reset_memory",
                                          role="acquisition_gate" if (cell, family) == ("acquire_k2", "seen") else "diagnostic")
                    contrast(seed, stage, cell, family, category, "online", "reset_memory", "nll", "diagnostic")
                    if (cell, family) == ("acquire_k2", "seen"):
                        rows = groups[seed, stage, cell, family, category, "online"]
                        accuracy = float(np.mean([r["correct"] for r in rows]))
                        gates[seed][stage] = dict(accuracy=accuracy, accuracy_threshold=.8,
                            normal_minus_reset=difference, passed=accuracy >= .8 and difference["ci_low"] > 0)
            for family in ("seen", "heldout"):
                for condition in ("online", "reset_memory"):
                    original = groups[seed, stage, "acquire_k2", family, "stable", condition]
                    changed = groups[seed, stage, "counterfactual_k2", family, "stable", condition]
                    index = {(r["episode_seed"], r["query_index"]): r for r in changed}
                    paired = []
                    for old in original:
                        new = index[old["episode_seed"], old["query_index"]]
                        equal(new["target"], 97+(old["target"]-97+1) % 16, "counterfactual target pair")
                        old_legal = 97 <= old["prediction"] <= 112
                        paired.append(dict(episode_seed=old["episode_seed"],
                            both_correct=old["correct"]*new["correct"], changed_prediction=int(old["prediction"] != new["prediction"]),
                            followed_relabeling=int(old_legal and new["prediction"] == 97+(old["prediction"]-97+1) % 16)))
                    for metric in ("both_correct", "changed_prediction", "followed_relabeling"):
                        stats = cluster_estimate(paired, metric, draws, analysis_seed(base, repr((seed, stage, family, condition, metric))))
                        estimates.append(dict(checkpoint_seed=seed, stage=stage, cell="counterfactual_pair", family=family,
                                              category="stable", condition=condition, metric=metric, **stats,
                                              uncertainty="episode bootstrap; paired original and relabeled inputs"))
        for k in (4, 8, 16):
            for d in (0, 8, 32):
                cell = f"pressure_k{k}_d{d}"
                for category in ("stable", "revised") + (("recent",) if d else ()):
                    for left, right in (("periodic", "continuous"), ("periodic", "repeat_current"),
                                        ("continuous", "repeat_current"), ("periodic", "online")):
                        is_primary = (k, d, category, left, right) == (8, 8, "stable", "periodic", "continuous")
                        item = contrast(seed, "after", cell, "seen", category, left, right,
                                        role="primary" if is_primary else "exploratory")
                        if is_primary:
                            primary[seed] = item
                        if category == "revised":
                            contrast(seed, "after", cell, "seen", category, left, right, "obsolete", "stale-input diagnostic")
        for cell in SHORT:
            category = "revised" if cell == "revise_k1" else "stable"
            for family in ("seen", "heldout"):
                for condition in ("online", "reset_memory"):
                    a = groups[seed, "after", cell, family, category, condition]
                    b = groups[seed, "before", cell, family, category, condition]
                    for metric in ("correct", "nll"):
                        stats = paired_estimate(a, b, metric, draws, analysis_seed(base, repr((seed, cell, family, condition, "adaptation", metric))))
                        contrasts.append(dict(checkpoint_seed=seed, stage="after_minus_before", cell=cell, family=family,
                            category=category, left=condition, right=condition, metric=metric, role="adaptation diagnostic", **stats))
    gate_pass = all(gates[s]["after"]["passed"] for s in SEEDS)
    lead = gate_pass and all(primary[s]["ci_low"] > 0 and primary[s]["estimate"] >= .05 for s in SEEDS)
    decisions = dict(main_certified=True, evidence_scope=SCOPE, acquisition=gates, primary=primary,
        both_acquisition_pass=gate_pass, useful_timing_pilot_lead=lead,
        conclusion="useful conditional pilot lead" if lead else
                   "acquisition gate failed; pressure results are floor diagnostics" if not gate_pass else
                   "continuation rule not met; no confirmed timing lead",
        interpretation_limits=["Intervals are conditional on each fixed checkpoint and sampled development episodes.",
            "Seeds are not pooled and two parents do not estimate training-procedure variability.",
            "Unmarked stale replays are conflicting-input diagnostics, not isolated consolidation failures.",
            "Raw-record replay is a reference schedule, not a sleep-specific module.",
            "K4 periodic partial blocks prevent literal serialized-prefix comparisons across delays.",
            "Development results are not untouched test results; reserved key namespace was not evaluated.",
            "Saved logits and tensors were checked; model inference and training were not independently rerun."])
    return estimates, contrasts, decisions


def manifest(root, run_names):
    result = {}
    for name in run_names:
        for path in sorted((root/name).rglob("*")):
            if path.is_file():
                result[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def run_main(args, out):
    require(args.root is not None and args.expected_commit is not None, "Main mode requires --root and --expected-commit")
    require(re.fullmatch("[0-9a-f]{40}", args.expected_commit) is not None, "Expected commit must be a full lowercase SHA")
    root = args.root.resolve()
    design = read_json(root / "design.json")
    design["_sha256"] = sha(root / "design.json")
    # Freeze this analyzer to the prospectively requested main protocol.
    for field, value in dict(steps=1600, batch_size=16, train_seed=78102, lr=.0005).items():
        equal(design["adaptation"][field], value, "main design " + field)
    for field, value in dict(dev_seed=78103, gate_episodes=64, grid_episodes=32, primary_episodes=128).items():
        equal(design["evaluation"][field], value, "main design " + field)
    equal(design["analysis"]["bootstrap_draws"], 5000, "main bootstrap draws")
    equal(design["analysis"]["bootstrap_seed"], 78104, "main bootstrap seed")
    equal(design["replay"]["block_records"], 4, "main replay block")
    equal(design["replay"]["buffer_keys"], 8, "main replay buffer")
    parents = {seed: getattr(args, "parent_"+seed) or root.parent / "clmm-shared-pretrain-2026-11-14" / ("lm-"+seed) / "checkpoint.pt" for seed in SEEDS}
    evaluations, trainings, all_meta, names = {}, {}, [], []
    for seed in SEEDS:
        checked_hash(parents[seed], design["parents"][seed], "fixed parent " + seed)
        load_checkpoint_evidence(parents[seed])
        for phase in ("before", "adapt", "after"):
            name = phase+"-"+seed
            require((root/name/"metadata.json").is_file(), "Missing canonical run: " + name)
            names.append(name)
        print(f"Auditing adaptation {seed}", flush=True)
        trainings[seed] = validate_training(root/("adapt-"+seed), parents[seed], args.expected_commit, design)
        all_meta.append(trainings[seed]["meta"])
        for phase in ("before", "after"):
            print(f"Auditing evaluation {phase}-{seed}", flush=True)
            checkpoint = parents[seed] if phase == "before" else root/("adapt-"+seed)/"checkpoint.pt"
            evaluations[seed, phase] = validate_eval(root/(phase+"-"+seed), checkpoint, args.expected_commit, design, phase)
            all_meta.append(evaluations[seed, phase]["meta"])
        # All short inputs remain exactly paired before and after adaptation.
        before = evaluations[seed, "before"]["ledger"]
        after = evaluations[seed, "after"]["ledger"]
        for key, value in before.items():
            equal(after[key], value, "before/after exact short sample pairing")
    left, right = (trainings[s] for s in SEEDS)
    equal(left["plan"], right["plan"], "identical training plans across parents")
    for key in left["arrays"]:
        require(np.array_equal(left["arrays"][key], right["arrays"][key]), "Different training arrays: " + key)
    equal(left["canonical_hash"], right["canonical_hash"], "canonical training input identity")
    for meta in all_meta[1:]:
        equal(meta["source_sha256"], all_meta[0]["source_sha256"], "common complete source snapshot")
    for phase in ("before", "after"):
        a, b = (evaluations[s, phase] for s in SEEDS)
        equal(a["canonical_hash"], b["canonical_hash"], "same complete evaluation dataset across parents")
        equal(a["ledger"], b["ledger"], "same paired evaluation ledger across parents")
    print("Computing episode-cluster intervals", flush=True)
    estimates, contrasts, decisions = analyze(evaluations, design)
    write_csv(out/"estimates.csv", estimates)
    write_csv(out/"contrasts.csv", contrasts)
    write_json(out/"decisions.json", decisions)
    write_json(out/"source-manifest.json", dict(design_sha256=design["_sha256"], expected_commit=args.expected_commit,
        analyzer_sha256=sha(__file__), parents={s: dict(path=str(parents[s]), sha256=sha(parents[s])) for s in SEEDS},
        artifacts=manifest(root, names)))
    return dict(status="verified", main_certified=True, evidence_scope=SCOPE,
        canonical_runs=names, expected_commit=args.expected_commit,
        training_steps_per_checkpoint=1600, supervised_answers_per_checkpoint=25600,
        training_input_canonical_sha256=left["canonical_hash"], estimates=len(estimates), contrasts=len(contrasts),
        bootstrap_draws=5000, bootstrap_unit="episode, paired questions and arms", tolerances={"float32_metric_absolute": NUM_TOL},
        output_sha256={name: sha(out/name) for name in ("estimates.csv", "contrasts.csv", "decisions.json", "source-manifest.json")},
        checks=["complete fixed budgets and final checkpoint lineage", "finite checkpoint tensors",
            "source and artifact hashes", "disjoint hashed key namespaces", "training bytes/targets/plan identical",
            "online-only answer reconstruction", "label-independent isolated query prefixes", "causal replay source and schedule reconstruction",
            "matched continuous/periodic content and repeat counts", "semantic CSV/logit correspondence",
            "independently recomputed full/forced metrics and summaries", "within/between-checkpoint data pairing",
            "counterfactual relabeling", "whole-episode paired bootstrap and prospective continuation rule"])


def smoke(args, out):
    require(args.smoke_run is not None, "Missing smoke run")
    run = args.smoke_run.resolve()
    meta = read_json(run/"metadata.json")
    checkpoint = Path(meta["checkpoint_path"])
    require(checkpoint.is_file(), "Smoke input checkpoint path unavailable locally")
    design = read_json(run/"design.json")
    if meta["mode"] == "train":
        result = validate_training(run, checkpoint, args.expected_commit, design, strict=False)
        detail = dict(steps=result["meta"]["completed_steps"], canonical_input_sha256=result["canonical_hash"])
    else:
        result = validate_eval(run, checkpoint, args.expected_commit, design, "before", strict=False)
        detail = dict(queries=len(result["rows"]), canonical_data_sha256=result["canonical_hash"])
    write_json(out/"decisions.json", dict(main_certified=False, conclusion="Smoke artifacts only; no main gate or efficacy certification."))
    return dict(status="smoke_verified", main_certified=False, smoke_run=str(run), evidence_scope=SCOPE, **detail)


def selfchecks(out, data_run=None):
    passed = []
    def rejects(name, operation):
        try:
            operation()
        except (AuditError, KeyError, ValueError):
            passed.append(name)
        else:
            raise AuditError("Selfcheck did not reject " + name)
    basic = [dict(episode_seed=e, query_index=q, key=f"{q:03d}", target=97, category="stable", correct=e % 2)
             for e in range(8) for q in range(2)]
    stats = paired_estimate(basic, list(reversed(basic)), "correct", 5000, 123)
    equal((stats["estimate"], stats["ci_low"], stats["ci_high"], stats["episodes"], stats["queries"]),
          (0., 0., 0., 8, 16), "paired identity interval")
    passed.append("paired identical arms have exactly zero interval despite varied outcomes")
    single = [row for row in basic if row["query_index"] == 0]
    equal(cluster_estimate(basic, "correct", 5000, 1)["ci_low"], cluster_estimate(single, "correct", 5000, 1)["ci_low"],
          "duplicating correlated questions must not narrow interval")
    equal(cluster_estimate(basic, "correct", 5000, 1)["ci_high"], cluster_estimate(single, "correct", 5000, 1)["ci_high"], "cluster upper interval")
    passed.append("two correlated queries cluster together rather than increasing independent sample size")
    require(boot_ratio([1., 0.], [1., 3.], 100, 1)[0] == .25, "ratio of sums, not mean of episode ratios")
    passed.append("unequal-denominator bootstrap estimates ratio of sums")
    rejects("missing paired question", lambda: pair_rows(basic, basic[:-1]))
    rejects("duplicate paired question", lambda: pair_rows(basic, basic+[basic[0]]))
    changed = [dict(row, target=98) if i == 0 else row for i, row in enumerate(basic)]
    rejects("altered paired target", lambda: pair_rows(basic, changed))
    logits = np.zeros((2, 256), dtype=np.float32)
    scores = independent_scores(logits, [97, 112])
    near(scores["nll"][0], math.log(256), "uniform full NLL", 1e-12)
    near(scores["forced_nll"][0], math.log(16), "uniform forced NLL", 1e-12)
    near(scores["legal_probability_mass"][0], 1/16, "uniform legal mass", 1e-12)
    passed.append("independent full/forced softmax metrics match analytic uniform probabilities")
    logits[0, 0] = np.nan
    rejects("nonfinite saved logits", lambda: independent_scores(logits, [97, 112]))
    rejects("illegal answer byte", lambda: independent_scores(np.zeros((1, 256)), [32]))
    k = POOLS["train"][0]
    raw = RECORDS["seen"][0].format(key=k, value="a").encode().rjust(32, b" ") + QUERIES["seen"][0].format(key=k).encode().rjust(31, b" ")
    parse_training_prefix(raw, 97, 1, 0, 0., 0)
    rejects("answer byte appended to model prefix", lambda: parse_training_prefix(raw+b"a", 97, 1, 0, 0., 0))
    rejects("wrong saved training target", lambda: parse_training_prefix(raw, 98, 1, 0, 0., 0))
    leaked = raw.replace(k.encode(), POOLS["dev"][0].encode())
    rejects("dev key leaked into training", lambda: parse_training_prefix(leaked, 97, 1, 0, 0., 0))
    import torch
    finite_state({"weight": torch.tensor([1.])})
    rejects("nonfinite checkpoint tensor", lambda: finite_state({"weight": torch.tensor([float("nan")])}))
    probe = out/"hash-fixture.txt"
    probe.write_text("original", encoding="utf8")
    original = sha(probe)
    probe.write_text("changed", encoding="utf8")
    rejects("corrupted source/artifact hash", lambda: checked_hash(probe, original, "fixture"))
    if data_run is not None:
        # Mutate detached in-memory copies only; preserve all real run artifacts.
        raw = gzip.decompress((data_run/"evaluation_data.jsonl.gz").read_bytes()).decode("utf8")
        current = None
        chosen = None
        for line in raw.splitlines():
            item = json.loads(line)
            if item["type"] == "episodes":
                current = item["episodes"]
            elif item["row"]["condition"] == "periodic" and item["row"]["replay_records"]:
                chosen = item, current[item["row"]["episode"]]
                break
        require(chosen is not None, "Selfcheck data needs a periodic replay sample")
        sample, ep = chosen
        validate_episode(ep)
        verify_sample(sample, ep)
        passed.append("real saved periodic sample independently reconstructs before mutation")
        bad = copy.deepcopy(sample)
        next(e for e in bad["audit"]["execution_plan"] if e["kind"] == "replay")["source_event_id"] = len(ep["events"])
        rejects("future replay source index in saved execution plan", lambda: verify_sample(bad, ep))
        bad = copy.deepcopy(sample)
        bad["audit"]["extra_record_count"] += 1
        rejects("corrupted replay update budget", lambda: verify_sample(bad, ep))
        bad = copy.deepcopy(sample)
        bad["prefix"] += chr(bad["row"]["target"])
        rejects("answer appended to saved evaluation input", lambda: verify_sample(bad, ep))
        bad = copy.deepcopy(sample)
        bad["row"]["target"] = 97+(bad["row"]["target"]-97+1) % 16
        rejects("corrupted saved evaluation target", lambda: verify_sample(bad, ep))
        bad_ep = copy.deepcopy(ep)
        bad_ep["events"][0]["key"] = POOLS["test"][0]
        rejects("reserved test key in development history", lambda: validate_episode(bad_ep))
        # Exercise the complete decision/aggregation path with deliberately
        # invented outcomes on saved sample identities. Never save those tables.
        fixture_rows = [json.loads(line)["row"] for line in raw.splitlines()
                        if json.loads(line)["type"] == "sample"]
        synthetic = {}
        for seed in SEEDS:
            for stage in ("before", "after"):
                fixture = []
                for source in fixture_rows:
                    if stage == "before" and source["cell"] not in SHORT:
                        continue
                    is_correct = int(stage == "after" and source["condition"] in ("online", "periodic"))
                    target = source["target"]
                    fixture.append(dict(source, correct=is_correct, forced_correct=is_correct,
                        prediction=target if is_correct else 0,
                        forced_prediction=target if is_correct else 97+(target-97+1) % 16,
                        nll=0. if is_correct else 5., forced_nll=0. if is_correct else 3.,
                        legal_probability_mass=.8, obsolete=0, last_observed_correct=0,
                        last_presented_correct=0, last_key_presented_correct=0))
                synthetic[seed, stage] = dict(rows=fixture)
        fixture_design = dict(analysis=dict(bootstrap_draws=100, bootstrap_seed=78104))
        _, _, verdict = analyze(synthetic, fixture_design)
        require(verdict["both_acquisition_pass"] and verdict["useful_timing_pilot_lead"],
                "Known perfect synthetic paired lead rejected")
        for row in synthetic["68105", "after"]["rows"]:
            if row["cell"] == "acquire_k2" and row["family"] == "seen" and row["condition"] == "online":
                row["correct"] = 0
        _, _, verdict = analyze(synthetic, fixture_design)
        require(not verdict["both_acquisition_pass"] and not verdict["useful_timing_pilot_lead"],
                "One passing checkpoint improperly rescued failed acquisition")
        passed.append("full synthetic aggregation and continuation rule; one weak checkpoint cannot be rescued")
    result = dict(status="passed", main_certified=False, fixture_scope="Synthetic analyzer fixtures; no model result certification",
                  checks=passed, count=len(passed))
    write_json(out/"selfchecks.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--parent-68101", dest="parent_68101", type=Path)
    parser.add_argument("--parent-68105", dest="parent_68105", type=Path)
    parser.add_argument("--selfcheck-data", type=Path, help="Optional saved evaluation run for in-memory corruption checks")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--selfcheck", action="store_true")
    mode.add_argument("--smoke-run", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(HERE) and out != HERE, "Output must be a NEW subdirectory of this authorized analysis directory")
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", main_certified=False, evidence_scope=SCOPE)
    try:
        if args.selfcheck:
            result = selfchecks(out, args.selfcheck_data)
        elif args.smoke_run:
            result = smoke(args, out)
        else:
            result = run_main(args, out)
    except BaseException as error:
        result.update(status="failed", main_certified=False, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        write_json(out/"decisions.json", dict(main_certified=False, conclusion="Artifact audit failed; no scientific decision is certified.",
                                            error_type=type(error).__name__, error=str(error)))
        print(result["traceback"], file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    result.update(analyzer_sha256=sha(__file__), audited_utc=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv])
    write_json(out/"verification.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
