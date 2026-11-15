"""CPU-only checkpoint reload against a bounded saved-prediction subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    checkpoints = {}
    for path in args.root.glob("*/metadata.json"):
        if path.parent.name.startswith("smoke") or path.parent.name == "dense-calibration":
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["status"] != "succeeded":
            continue
        for name in ("checkpoint.pt", "initial_checkpoint.pt"):
            checkpoint = path.parent / name
            if checkpoint.exists():
                checkpoints[sha(checkpoint)] = checkpoint
        candidates.append((path.parent, metadata))
    assert candidates, "No main-protocol successful run artifacts"
    source = candidates[0][0] / "sources"
    sys.path.insert(0, str(source))
    from clmm.pretrain.kv_control import KVConfig, DeltaAdapter, NeuralAdapter

    torch.set_num_threads(1)
    reports = []
    for folder, metadata in candidates:
        # Reload each trained checkpoint's short gate once and each pressure checkpoint once.
        if folder.name.startswith("confirm"):
            continue
        checkpoint = folder / "checkpoint.pt"
        if not checkpoint.exists():
            checkpoint = checkpoints[metadata["parent_sha256"]]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        offline_changed = None
        if (folder / "initial_checkpoint.pt").exists():
            initial = torch.load(folder / "initial_checkpoint.pt", map_location="cpu", weights_only=True)
            assert initial["config"] == payload["config"]
            assert initial["model_state"].keys() == payload["model_state"].keys()
            offline_changed = [name for name in initial["model_state"]
                               if not torch.equal(initial["model_state"][name], payload["model_state"][name])]
            assert set(offline_changed) == {"engine.memory_model_parameters.0", "engine.memory_model_parameters.1"}
            assert torch.count_nonzero(initial["model_state"]["engine.memory_model_parameters.1"]) == 0
        model = (DeltaAdapter if payload["method"] == "delta" else NeuralAdapter)(KVConfig(**payload["config"]))
        model.load_state_dict(payload["model_state"], strict=True)
        model.requires_grad_(False)
        model.eval()
        evaluation = folder / "evaluation"
        episodes = json.loads((evaluation / "episodes.json").read_text(encoding="utf-8"))
        plan_items = json.loads((evaluation / "plans.json").read_text(encoding="utf-8"))
        plans = {(item["cell"], item["arm"], item["episode"]): item["plan"] for item in plan_items}
        rows = read(evaluation / "predictions.csv")
        scores = np.load(evaluation / "scores.npy")
        index = {(row["cell"], row["arm"], int(row["episode"]), int(row["rank"])): i for i, row in enumerate(rows)}
        selected_cells = [cell for cell in ("acquire2", "acquire4", "revise2", "pressure_k8_d64", "pressure_k32_d64") if cell in episodes]
        answers = 0
        max_error = 0.0
        batches = 0
        with torch.no_grad():
            for cell in selected_cells:
                arms = ("online", "repeat", "continuous", "periodic", "boundary_reset") if cell.startswith("pressure") else ("online", "reset", "rolled")
                for arm in arms:
                    # Recipient episodes are 0/1. Rolled states are rebuilt from their saved donor episodes.
                    recipient = [0, 1]
                    source_ids = [int(rows[index[cell, arm, eid, 0]]["state_source_episode"]) for eid in recipient]
                    build_ids = source_ids if arm == "rolled" else recipient
                    state = model.initial_state(2)
                    chosen_plans = [plans[cell, arm, eid] for eid in build_ids]
                    if arm != "reset":
                        for event_index in range(chosen_plans[0]["writes"]):
                            keys = torch.tensor([plan["events"][event_index]["key"] for plan in chosen_plans])
                            values = torch.tensor([plan["events"][event_index]["value"] for plan in chosen_plans])
                            state = model.write(keys, values, state)
                            if event_index + 1 == episodes[cell][build_ids[0]]["boundary"]:
                                stream = io.BytesIO()
                                torch.save(model.export_state(state), stream)
                                stream.seek(0)
                                state = model.import_state(torch.load(stream, weights_only=True))
                                if arm == "boundary_reset":
                                    state = model.initial_state(2)
                    queries = torch.tensor([[query["key"] for query in episodes[cell][eid]["queries"]] for eid in recipient])
                    actual = model.read(queries, state).numpy()
                    expected = np.stack([[scores[index[cell, arm, eid, rank]] for rank in range(len(episodes[cell][eid]["queries"]))] for eid in recipient])
                    error = float(np.abs(actual - expected).max())
                    max_error = max(max_error, error)
                    assert np.allclose(actual, expected, atol=2e-6, rtol=2e-5), (folder.name, cell, arm, error)
                    assert np.array_equal(actual.argmax(-1), expected.argmax(-1)), (folder.name, cell, arm)
                    answers += actual.shape[0] * actual.shape[1]
                    batches += 1
        reports.append({"run": folder.name, "checkpoint_sha256": sha(checkpoint), "device": "cpu",
                        "answers_reloaded": answers, "batches": batches,
                        "max_absolute_score_error": max_error, "argmax_matches": True,
                        "offline_changed_state_dict_entries": offline_changed})
    result = {"status": "passed", "scope": "Selected final-checkpoint answers only; CPU versus saved runtime scores, not retraining", "torch": torch.__version__, "runs": reports,
              "total_answers": sum(report["answers_reloaded"] for report in reports)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
