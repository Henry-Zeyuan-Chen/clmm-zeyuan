"""Independent CPU artifact audit; never executes training or changes the repo."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_signature(keys, values):
    return hashlib.sha256(bytes(keys) + b"|" + bytes(values)).hexdigest()


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def near(actual, expected, tolerance=1e-10):
    assert np.isfinite(float(actual)) and abs(float(actual) - expected) <= tolerance, (actual, expected)


def audit_plan(ep, arm, plan):
    keys, values, boundary = ep["keys"], ep["values"], ep["boundary"]
    working = OrderedDict()

    def arrive(index):
        key = keys[index]
        working.pop(key, None)
        working[key] = index
        while len(working) > 8:
            working.popitem(last=False)

    events = plan["events"]
    for index in range(boundary):
        event = events[index]
        assert event == {"source": index, "key": keys[index], "value": values[index],
                         "kind": "online", "after_online": index}
        arrive(index)
    cursor = boundary
    selected = plan["selected"]
    for block_number, start in enumerate(range(boundary, len(keys), 4)):
        stop = min(start + 4, len(keys))
        sources = selected[block_number * 4:block_number * 4 + stop - start]
        assert len(sources) == stop - start
        assert all(source in working.values() and source < start for source in sources)
        expected = []
        for offset, index in enumerate(range(start, stop)):
            expected.append((index, "online", index))
            if arm == "continuous":
                expected.append((sources[offset], "extra", index))
            elif arm == "repeat":
                expected.append((index, "extra", index))
        if arm == "periodic":
            expected.extend((source, "extra", stop - 1) for source in sources)
        for source, kind, after in expected:
            event = events[cursor]
            assert event == {"source": source, "key": keys[source], "value": values[source],
                             "kind": kind, "after_online": after}
            assert source <= after
            cursor += 1
        for index in range(start, stop):
            arrive(index)
    assert cursor == len(events) == plan["writes"]
    assert plan["writes"] == boundary + ep["delay"] * (2 if arm in ("continuous", "periodic", "repeat") else 1)
    assert list(working) == plan["final_buffer_keys"]
    return len(events)


def audit_run(run, evaluation_name="evaluation"):
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    result = {"run": run.name, "evaluation_stage": evaluation_name, "status": metadata["status"], "source_files": 0}
    for name, digest in metadata["source_sha256"].items():
        assert sha(run / "sources" / name) == digest, (run, name)
        result["source_files"] += 1
    if metadata.get("design_sha256"):
        assert sha(run / "design.json") == metadata["design_sha256"]
    if metadata.get("initial_checkpoint_sha256"):
        assert sha(run / "initial_checkpoint.pt") == metadata["initial_checkpoint_sha256"]
    if metadata["status"] != "succeeded":
        result["failure"] = metadata.get("failure")
        return result, []

    args = metadata["args"]
    with np.load(run / "training_data.npz") as archive:
        data = {name: archive[name] for name in archive.files}
    assert sha(run / "training_data.npz") == metadata["training_data_sha256"]
    train_plan = json.loads((run / "training_plan.json").read_text(encoding="utf-8"))
    assert len(train_plan) == args["steps"]
    seen = set()
    observed = answers = 0
    key_ids = set()
    atomic_bindings = set()
    for step, (length_raw, count_raw) in enumerate(zip(data["lengths"], data["counts"], strict=True)):
        length, count = int(length_raw), int(count_raw)
        assert count == train_plan[step]["facts"] in (2, 4, 8)
        assert length == count + train_plan[step]["revisions"] + train_plan[step]["delay"]
        assert np.all(data["keys"][step, :, length:] == 255)
        assert np.all(data["values"][step, :, length:] == 255)
        assert np.all(data["queries"][step, :, count:] == 255)
        assert np.all(data["targets"][step, :, count:] == 255)
        for sample in range(args["batch_size"]):
            keys = data["keys"][step, sample, :length].tolist()
            values = data["values"][step, sample, :length].tolist()
            assert all(0 <= key < 64 for key in keys)
            assert all(0 <= value < 16 for value in values)
            key_ids.update(keys)
            atomic_bindings.update(zip(keys, values, strict=True))
            latest = dict(zip(keys, values, strict=True))
            queries = data["queries"][step, sample, :count].tolist()
            targets = data["targets"][step, sample, :count].tolist()
            assert queries == keys[:count]
            assert targets == [latest[key] for key in queries]
            assert len(set(targets)) == count
            assert not set(keys[:count]).intersection(keys[count + train_plan[step]["revisions"]:])
            seen.add(history_signature(keys, values))
        observed += args["batch_size"] * length
        answers += args["batch_size"] * count
    if args["mode"] == "train":
        training = csv_rows(run / "training.csv")
        assert len(training) == args["steps"]
        assert [int(row["step"]) for row in training] == list(range(1, args["steps"] + 1))
        assert sum(int(row["observations"]) for row in training) == observed == metadata["training_observations"]
        assert sum(int(row["answers"]) for row in training) == answers == metadata["training_answers"]
        assert all(np.isfinite(float(row[field])) for row in training for field in ("loss", "accuracy", "gradient_norm"))
        assert sha(run / "checkpoint.pt") == metadata["checkpoint_sha256"]
    elif metadata.get("checkpoint_sha256"):
        assert sha(run / "checkpoint.pt") == metadata["checkpoint_sha256"]
    result.update(training_trace_observations=observed, training_trace_answers=answers,
                  training_unique_histories=len(seen), training_key_ids=len(key_ids),
                  training_atomic_bindings=len(atomic_bindings), offline_updated=args["mode"] == "train")

    evaluation = run / evaluation_name
    evaluation_metadata = metadata["initial_evaluation" if evaluation_name == "initial-evaluation" else "evaluation"]
    episodes = json.loads((evaluation / "episodes.json").read_text(encoding="utf-8"))
    plans = json.loads((evaluation / "plans.json").read_text(encoding="utf-8"))
    plan_map = {(item["cell"], item["arm"], item["episode"]): item["plan"] for item in plans}
    assert len(plan_map) == len(plans)
    event_count = 0
    for cell, items in episodes.items():
        for episode_index, ep in enumerate(items):
            assert history_signature(ep["keys"], ep["values"]) == ep["signature"] not in seen
            assert ep["boundary"] == ep["facts"] + ep["revisions"]
            assert len(ep["keys"]) == len(ep["values"]) == ep["boundary"] + ep["delay"]
            assert len(set(ep["keys"][:ep["facts"]])) == ep["facts"]
            assert not set(ep["keys"][:ep["facts"]]).intersection(ep["keys"][ep["boundary"]:])
            latest = dict(zip(ep["keys"], ep["values"], strict=True))
            protected = ep["queries"][:4] if ep["pressure"] else ep["queries"]
            assert len({query["target"] for query in protected}) == len(protected)
            for query in ep["queries"]:
                assert latest[query["key"]] == query["target"]
                if query["category"] == "revised":
                    prior = [value for key, value in zip(ep["keys"], ep["values"], strict=True) if key == query["key"]]
                    assert len(prior) == 2 and prior[0] == query["old"] != prior[1] == query["target"]
            for (pcell, arm, peid), plan in plan_map.items():
                if pcell == cell and peid == episode_index:
                    schedule = arm if arm in ("continuous", "periodic", "repeat") else "online"
                    event_count += audit_plan(ep, schedule, plan)
            if ep["pressure"]:
                left = plan_map[cell, "continuous", episode_index]
                right = plan_map[cell, "periodic", episode_index]
                assert left["selected"] == right["selected"]
    for facts in (8, 16, 32):
        if f"pressure_k{facts}_d0" not in episodes:
            continue
        groups = [episodes[f"pressure_k{facts}_d{delay}"] for delay in (0, 16, 64)]
        for small, medium, large in zip(*groups, strict=True):
            for prefix in (small, medium):
                assert large["keys"][:len(prefix["keys"])] == prefix["keys"]
                assert large["values"][:len(prefix["values"])] == prefix["values"]

    predictions = csv_rows(evaluation / "predictions.csv")
    scores = np.load(evaluation / "scores.npy")
    assert scores.shape == (len(predictions), 16) and np.isfinite(scores).all()
    assert sha(evaluation / "predictions.csv") == evaluation_metadata["predictions_sha256"]
    assert sha(evaluation / "scores.npy") == evaluation_metadata["scores_sha256"]
    records = []
    for raw, vector in zip(predictions, scores, strict=True):
        cell, arm, eid, rank = raw["cell"], raw["arm"], int(raw["episode"]), int(raw["rank"])
        ep = episodes[cell][eid]
        query = ep["queries"][rank]
        for field in ("key", "target", "old"):
            assert int(raw[field]) == query[field]
        assert raw["category"] == query["category"]
        prediction = int(vector.argmax())
        assert prediction == int(raw["prediction"])
        assert int(raw["correct"]) == (prediction == query["target"])
        assert int(raw["obsolete"]) == (query["old"] >= 0 and prediction == query["old"])
        near(raw["mse"], np.mean((vector - np.eye(16)[query["target"]]) ** 2))
        plan = plan_map[cell, arm, eid]
        assert int(raw["actual_writes"]) == plan["writes"]
        expected_count = 0 if arm == "reset" else ep["delay"] if arm == "boundary_reset" else plan["writes"]
        assert int(raw["state_write_count"]) == expected_count
        batch_start = (eid // args["eval_batch_size"]) * args["eval_batch_size"]
        batch_count = min(args["eval_batch_size"], len(episodes[cell]) - batch_start)
        source = batch_start + (eid - batch_start - 1) % batch_count if arm == "rolled" else -1 if arm == "reset" else eid
        assert int(raw["state_source_episode"]) == source
        records.append({**raw, "correct": int(raw["correct"]), "obsolete": int(raw["obsolete"]),
                        "mse": float(raw["mse"]), "episode": eid, "run": run.name,
                        "evaluation_stage": evaluation_name})

    summary = csv_rows(evaluation / "summary.csv")
    for row in summary:
        subset = [record for record in records if record["cell"] == row["cell"] and record["arm"] == row["arm"]
                  and (row["category"] == "all" or row["category"] == record["category"])]
        by_episode = defaultdict(list)
        for record in subset:
            by_episode[record["episode"]].append(record)
        assert int(row["queries"]) == len(subset)
        assert int(row["episodes"]) == len(by_episode)
        near(row["accuracy"], np.mean([record["correct"] for record in subset]))
        near(row["all_correct"], np.mean([all(record["correct"] for record in group) for group in by_episode.values()]))
        near(row["mse"], np.mean([record["mse"] for record in subset]))
        if row["category"] == "revised":
            near(row["obsolete_rate"], np.mean([record["obsolete"] for record in subset]))
        else:
            assert row["obsolete_rate"] == ""
    resources = csv_rows(evaluation / "resources.csv")
    assert all(float(row["roundtrip_max_abs_delta"]) == 0 for row in resources)
    assert evaluation_metadata["outer_parameters_unchanged"] is True
    result.update(evaluation_cells=len(episodes), audited_plan_writes=event_count,
                  prediction_rows=len(records), summary_rows=len(summary),
                  state_roundtrip_batches=len(resources), status="passed")
    return result, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    records = []
    pairing_groups = {}
    episode_groups = {}
    main_sources = None
    for metadata_path in sorted(args.root.rglob("metadata.json")):
        if "sources" in metadata_path.parts or "analysis" in metadata_path.parts:
            continue
        if metadata_path.parent.name == "dense-calibration":
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "source_sha256" not in metadata or "train_seed" not in metadata.get("args", {}):
            continue
        if metadata["status"] == "succeeded":
            settings = metadata["args"]
            pairing = (settings["train_seed"], settings["steps"], settings["batch_size"])
            digest = metadata["training_data_sha256"]
            assert pairing not in pairing_groups or pairing_groups[pairing] == digest
            pairing_groups[pairing] = digest
            eval_pairing = (digest, settings["dev_seed"], settings["eval_episodes"], settings["pressure_episodes"], settings["suite"])
            episode_digest = sha(metadata_path.parent / "evaluation/episodes.json")
            assert eval_pairing not in episode_groups or episode_groups[eval_pairing] == episode_digest
            episode_groups[eval_pairing] = episode_digest
            if not metadata_path.parent.name.startswith("smoke"):
                frozen = (metadata["source_commit"], metadata["source_sha256"])
                assert main_sources is None or frozen == main_sources
                main_sources = frozen
        result, rows = audit_run(metadata_path.parent)
        runs.append(result)
        records.extend(rows)
        if (metadata_path.parent / "initial-evaluation").is_dir():
            initial_result, initial_rows = audit_run(metadata_path.parent, "initial-evaluation")
            runs.append(initial_result)
            records.extend(initial_rows)
    assert runs, "No completed KV run metadata found"
    main_runs = [run for run in runs if not run["run"].startswith("smoke")]
    offline_runs = [run for run in main_runs if run.get("offline_updated") and run["evaluation_stage"] == "evaluation"]
    report = {"status": "passed", "audit_scope": "Saved traces/scores/provenance and internal consistency; no retraining or learned-checkpoint inference", "runs": runs,
              "total_prediction_rows": len(records), "total_audited_plan_writes": sum(run.get("audited_plan_writes", 0) for run in runs),
              "shared_training_data_groups": len(pairing_groups), "shared_evaluation_episode_groups": len(episode_groups),
              "main_source_commit": main_sources[0] if main_sources else None,
              "main_prediction_rows": sum(run.get("prediction_rows", 0) for run in main_runs),
              "main_audited_plan_writes": sum(run.get("audited_plan_writes", 0) for run in main_runs),
              "main_serialization_batches": sum(run.get("state_roundtrip_batches", 0) for run in main_runs),
              "main_actual_offline_training_runs": len(offline_runs),
              "main_actual_offline_training_observations": sum(run["training_trace_observations"] for run in offline_runs),
              "main_actual_offline_supervised_answers": sum(run["training_trace_answers"] for run in offline_runs)}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
