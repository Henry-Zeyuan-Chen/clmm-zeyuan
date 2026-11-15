"""Separate full-trace and CPU numerical replay audit of the dense instrument."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import argparse

import numpy as np
import torch

import audit_artifacts as base
import analyze_results as analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folder = args.root / "dense-calibration"
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "succeeded"
    assert metadata["offline_training"] is False and metadata["slow_learned_parameters"] == 0
    assert metadata["source_commit"] == "ff62df0f85ff7cbf62f3209755ea066959aa7085"
    assert metadata["environment"]["device"] == "cpu"
    assert metadata["elapsed_seconds"] < metadata["args"]["max_seconds"] <= 300
    for name, digest in metadata["source_sha256"].items():
        assert base.sha(folder / "sources" / name) == digest
    assert base.sha(folder / "calibrate_dense.py") == metadata["adapter_source_sha256"]
    assert base.sha(folder / "checkpoint.pt") == metadata["checkpoint_sha256"]
    assert base.sha(folder / "planned_training_exclusion.json") == metadata["planned_training_exclusion"]["sha256"]
    exclusion = json.loads((folder / "planned_training_exclusion.json").read_text(encoding="utf-8"))
    assert exclusion["used_for_offline_training"] is False
    seen = set(exclusion["signatures"])

    torch.set_num_threads(1)
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["offline_training"] is False
    assert checkpoint["config"] == metadata["config"]
    state = checkpoint["model_state"]
    assert set(state) == {"keys", "values"}
    keys, values = state["keys"], state["values"]
    assert tuple(keys.shape) == (64, 32) and tuple(values.shape) == (16, 32)
    assert keys.dtype == values.dtype == torch.float32
    assert torch.allclose(keys.square().sum(1), torch.ones(64), atol=2e-7, rtol=0)
    assert torch.equal(values, torch.eye(32)[:16])
    generator = torch.Generator().manual_seed(90401)
    expected_keys = torch.nn.functional.normalize(torch.randn(64, 32, generator=generator), dim=-1)
    assert torch.equal(expected_keys, keys)
    codebook_hash = hashlib.sha256(keys.numpy().tobytes() + values.numpy().tobytes()).hexdigest()
    assert codebook_hash == metadata["config"]["codebook_sha256"]

    evaluation = folder / "evaluation"
    episodes = json.loads((evaluation / "episodes.json").read_text(encoding="utf-8"))
    plan_items = json.loads((evaluation / "plans.json").read_text(encoding="utf-8"))
    plans = {(item["cell"], item["arm"], item["episode"]): item["plan"] for item in plan_items}
    assert len(plans) == len(plan_items)
    writes = 0
    for cell, items in episodes.items():
        for episode_id, episode in enumerate(items):
            assert base.history_signature(episode["keys"], episode["values"]) == episode["signature"] not in seen
            assert not set(episode["keys"][:episode["facts"]]).intersection(episode["keys"][episode["boundary"]:])
            latest = dict(zip(episode["keys"], episode["values"], strict=True))
            protected = episode["queries"][:4] if episode["pressure"] else episode["queries"]
            assert len({query["target"] for query in protected}) == len(protected)
            for query in episode["queries"]:
                assert latest[query["key"]] == query["target"]
                if query["category"] == "revised":
                    prior = [value for key, value in zip(episode["keys"], episode["values"], strict=True) if key == query["key"]]
                    assert prior == [query["old"], query["target"]] and prior[0] != prior[1]
            for (pcell, arm, eid), plan in plans.items():
                if pcell == cell and eid == episode_id:
                    writes += base.audit_plan(episode, arm if arm in ("repeat", "continuous", "periodic") else "online", plan)
            if episode["pressure"]:
                assert plans[cell, "continuous", episode_id]["selected"] == plans[cell, "periodic", episode_id]["selected"]
    for facts in (8, 16, 32):
        groups = [episodes[f"pressure_k{facts}_d{delay}"] for delay in (0, 16, 64)]
        for early, middle, late in zip(*groups, strict=True):
            for prefix in (early, middle):
                assert prefix["keys"] == late["keys"][:len(prefix["keys"])]
                assert prefix["values"] == late["values"][:len(prefix["values"])]

    rows = base.csv_rows(evaluation / "predictions.csv")
    scores = np.load(evaluation / "scores.npy")
    assert scores.shape == (len(rows), 16) and np.isfinite(scores).all()
    assert base.sha(evaluation / "scores.npy") == metadata["evaluation"]["scores_sha256"]
    assert base.sha(evaluation / "predictions.csv") == metadata["evaluation"]["predictions_sha256"]
    lookup = {(row["cell"], row["arm"], int(row["episode"]), int(row["rank"])): index for index, row in enumerate(rows)}
    for row, vector in zip(rows, scores, strict=True):
        cell, arm, eid, rank = row["cell"], row["arm"], int(row["episode"]), int(row["rank"])
        episode, query = episodes[cell][eid], episodes[cell][eid]["queries"][rank]
        assert all(int(row[field]) == query[field] for field in ("key", "target", "old"))
        assert row["category"] == query["category"]
        prediction = int(vector.argmax())
        assert int(row["prediction"]) == prediction
        assert int(row["correct"]) == (prediction == query["target"])
        assert int(row["obsolete"]) == (query["old"] >= 0 and prediction == query["old"])
        base.near(row["mse"], np.mean((vector - np.eye(16)[query["target"]]) ** 2))
        assert int(row["actual_writes"]) == plans[cell, arm, eid]["writes"]
        expected_count = 0 if arm == "reset" else episode["delay"] if arm == "boundary_reset" else plans[cell, arm, eid]["writes"]
        assert int(row["state_write_count"]) == expected_count
        start, count = (eid // 16) * 16, min(16, len(episodes[cell]) - (eid // 16) * 16)
        donor = start + (eid - start - 1) % count if arm == "rolled" else -1 if arm == "reset" else eid
        assert int(row["state_source_episode"]) == donor

    for declared in base.csv_rows(evaluation / "summary.csv"):
        chosen = [row for row in rows if row["cell"] == declared["cell"] and row["arm"] == declared["arm"]
                  and (declared["category"] == "all" or row["category"] == declared["category"])]
        assert len(chosen) == int(declared["queries"])
        base.near(declared["accuracy"], np.mean([int(row["correct"]) for row in chosen]))
        base.near(declared["all_correct"], analysis.all_correct(chosen).mean())
        base.near(declared["mse"], np.mean([float(row["mse"]) for row in chosen]))
        if declared["category"] == "revised":
            base.near(declared["obsolete_rate"], np.mean([int(row["obsolete"]) for row in chosen]))
    resources = base.csv_rows(evaluation / "resources.csv")
    for row in resources:
        count = int(row["batch"])
        assert all(int(row[field]) == 4096 * count for field in ("boundary_before_bytes", "boundary_restored_bytes", "final_bytes", "peak_post_write_bytes"))
        assert int(row["per_episode_final_bytes"]) == 4096
        assert float(row["roundtrip_max_abs_delta"]) == 0
    assert metadata["evaluation"]["outer_parameters_unchanged"] is True

    # Independent implementation: consume only each current pair and fast matrix.
    # Reproduce all scores directly, without importing the supplied dense adapter.
    largest_error = 0.0
    with torch.no_grad():
        for cell, items in episodes.items():
            arms = ("online", "repeat", "continuous", "periodic", "boundary_reset") if cell.startswith("pressure") else ("online", "reset", "rolled")
            for arm in arms:
                for start in range(0, len(items), 16):
                    batch = items[start:start + 16]
                    weight = torch.zeros(len(batch), 32, 32)
                    selected = [plans[cell, arm, start + index] for index in range(len(batch))]
                    for event_index in range(selected[0]["writes"]):
                        current_keys = keys[torch.tensor([plan["events"][event_index]["key"] for plan in selected])]
                        current_values = values[torch.tensor([plan["events"][event_index]["value"] for plan in selected])]
                        predicted_value = torch.bmm(current_keys[:, None], weight)[:, 0]
                        weight = weight + current_keys[:, :, None] * (current_values - predicted_value)[:, None, :]
                        if arm == "boundary_reset" and event_index + 1 == batch[0]["boundary"]:
                            weight.zero_()
                    if arm == "reset":
                        weight.zero_()
                    elif arm == "rolled":
                        weight = weight.roll(1, 0)
                    query_keys = keys[torch.tensor([[query["key"] for query in episode["queries"]] for episode in batch])]
                    actual = torch.bmm(query_keys, weight)[..., :16].numpy()
                    expected = np.stack([[scores[lookup[cell, arm, start + index, rank]] for rank in range(len(episode["queries"]))] for index, episode in enumerate(batch)])
                    largest_error = max(largest_error, float(np.abs(actual - expected).max()))
                    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-6)
                    assert np.array_equal(actual.argmax(-1), expected.argmax(-1))

    gate_rates = {cell: float(analysis.all_correct([row for row in rows if row["cell"] == cell and row["arm"] == "online"]).mean())
                  for cell in ("acquire2", "acquire4", "revise2")}
    k8_initial = analysis.all_correct([row for row in rows if row["cell"] == "pressure_k8_d0" and row["arm"] == "online"]).mean()
    eligible = gate_rates["acquire2"] >= 0.9 and gate_rates["revise2"] >= 0.9 and gate_rates["acquire4"] >= 0.8 and k8_initial >= 0.8
    before = [row for row in rows if row["cell"] == "pressure_k8_d64" and row["arm"] == "continuous" and row["category"] == "stable"]
    after = [row for row in rows if row["cell"] == "pressure_k8_d64" and row["arm"] == "periodic" and row["category"] == "stable"]
    before_groups, after_groups = analysis.by_episode(before), analysis.by_episode(after)
    differences = np.array([np.mean([int(row["correct"]) for row in after_groups[eid]])
                            - np.mean([int(row["correct"]) for row in before_groups[eid]]) for eid in sorted(before_groups)])
    mean, low, high = analysis.interval(differences, seed=90404)
    summaries, contrasts = analysis.pressure(folder, bootstrap_seed=90404)
    analysis.write(args.out / "dense_pressure_fixed_targets.csv", summaries)
    analysis.write(args.out / "dense_pressure_paired_contrasts.csv", contrasts)
    report = {"status": "passed", "scope": "Separate untrained single-codebook instrument; no neural selection or training claim",
              "prediction_rows": len(rows), "audited_plan_writes": writes, "serialization_batches": len(resources),
              "all_predictions_independently_replayed": True, "max_absolute_replayed_score_error": largest_error,
              "codebook_sha256": codebook_hash, "adapter_source_sha256": metadata["adapter_source_sha256"],
              "offline_training": False, "exclusion_only_histories": len(seen), "gate_all_correct": gate_rates,
              "k8_d0_four_target_all_correct": float(k8_initial), "eligible": bool(eligible),
              "primary": {"cell": "pressure_k8_d64", "category": "stable", "contrast": "periodic_minus_continuous",
                          "paired_episodes": len(differences), "difference": mean, "ci_low": low, "ci_high": high,
                          "continuous_correct": sum(int(row["correct"]) for row in before),
                          "periodic_correct": sum(int(row["correct"]) for row in after), "queries_per_arm": len(before),
                          "bootstrap_seed": 90404, "bootstrap_draws": 5000,
                          "uncertainty_scope": "Sampled episodes for one fixed key codebook; not codebook-seed variability"},
              "warnings": ["Representation geometry and state dimension both differ from the orthogonal control.",
                           "Calibration was specified after the main ceiling; it is a supplemental diagnostic.",
                           "Low-D0 larger-K cells cannot isolate delay retention."]}
    (args.out / "dense_calibration_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
