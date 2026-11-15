"""Recompute qualification and fixed-target pressure summaries from raw rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def by_episode(rows):
    result = defaultdict(list)
    for row in rows:
        result[int(row["episode"])].append(row)
    return result


def all_correct(rows):
    return np.array([all(int(row["correct"]) for row in group)
                     for _, group in sorted(by_episode(rows).items())], dtype=float)


def gate(folder, stage="evaluation"):
    rows = read(folder / stage / "predictions.csv")
    rates = {}
    for cell in ("acquire2", "acquire4", "revise2"):
        rates[cell] = float(all_correct([row for row in rows if row["cell"] == cell and row["arm"] == "online"]).mean())
    normal = all_correct([row for row in rows if row["cell"] == "acquire2" and row["arm"] == "online"])
    rolled = all_correct([row for row in rows if row["cell"] == "acquire2" and row["arm"] == "rolled"])
    assert normal.shape == rolled.shape == (128,)
    delta = (normal - rolled).reshape(8, 16).mean(axis=1)
    indices = np.random.default_rng(90304).integers(0, 8, size=(5000, 8))
    low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
    passed = rates["acquire2"] >= 0.9 and rates["acquire4"] >= 0.8 and rates["revise2"] >= 0.9 and low > 0
    return {"run": folder.name, "stage": stage, **rates, "normal_minus_rolled": float(delta.mean()),
            "ci_low": float(low), "ci_high": float(high), "passed": bool(passed)}


def interval(values, seed=90324):
    values = np.asarray(values, dtype=float)
    index = np.random.default_rng(seed).integers(0, len(values), (5000, len(values)))
    low, high = np.quantile(values[index].mean(axis=1), [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def pressure(folder, bootstrap_seed=90324):
    records = [row for row in read(folder / "evaluation/predictions.csv") if row["cell"].startswith("pressure")]
    summaries, contrasts = [], []
    arrays = {}
    for cell in sorted({row["cell"] for row in records}):
        for arm in ("online", "repeat", "continuous", "periodic", "boundary_reset"):
            for category in ("targets", "stable", "revised", "recent"):
                group = [row for row in records if row["cell"] == cell and row["arm"] == arm
                         and ((category == "targets" and row["category"] in ("stable", "revised")) or row["category"] == category)]
                if not group:
                    continue
                episodes = sorted(by_episode(group).items())
                accuracy = np.array([np.mean([int(row["correct"]) for row in rows]) for _, rows in episodes])
                mse = np.array([np.mean([float(row["mse"]) for row in rows]) for _, rows in episodes])
                obsolete = np.array([np.mean([int(row["obsolete"]) for row in rows]) for _, rows in episodes])
                mean, low, high = interval(accuracy, seed=bootstrap_seed)
                arrays[cell, arm, category] = accuracy
                summaries.append({"run": folder.name, "cell": cell, "arm": arm, "category": category,
                                  "episodes": len(episodes), "queries": len(group), "accuracy": mean,
                                  "ci_low": low, "ci_high": high, "all_correct": float(all_correct(group).mean()),
                                  "mse": float(mse.mean()), "obsolete_rate": float(obsolete.mean()) if category == "revised" else ""})
        for category in ("targets", "stable", "revised", "recent"):
            for left, right in (("periodic", "continuous"), ("periodic", "repeat"), ("continuous", "repeat"), ("online", "boundary_reset")):
                if (cell, left, category) not in arrays:
                    continue
                left_values, right_values = arrays[cell, left, category], arrays[cell, right, category]
                mean, low, high = interval(left_values - right_values, seed=bootstrap_seed)
                contrasts.append({"run": folder.name, "cell": cell, "category": category,
                                  "contrast": f"{left}_minus_{right}", "episodes": len(left_values),
                                  "difference": mean, "ci_low": low, "ci_high": high,
                                  "left_accuracy": float(left_values.mean()), "right_accuracy": float(right_values.mean())})
    return summaries, contrasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    gates, summaries, contrasts, training_changes = [], [], [], []
    for metadata_path in sorted(args.root.glob("*/metadata.json")):
        folder = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if folder.name.startswith("smoke") or folder.name == "dense-calibration" or metadata["status"] != "succeeded":
            continue
        gates.append(gate(folder))
        if (folder / "initial-evaluation").is_dir():
            gates.append(gate(folder, "initial-evaluation"))
            before = read(folder / "initial-evaluation/predictions.csv")
            after = read(folder / "evaluation/predictions.csv")
            assert (folder / "initial-evaluation/episodes.json").read_bytes() == (folder / "evaluation/episodes.json").read_bytes()
            for cell in ("acquire2", "acquire4", "revise2"):
                initial = [row for row in before if row["cell"] == cell and row["arm"] == "online"]
                final = [row for row in after if row["cell"] == cell and row["arm"] == "online"]
                initial_groups, final_groups = by_episode(initial), by_episode(final)
                mse_difference = np.array([np.mean([float(row["mse"]) for row in final_groups[eid]])
                                           - np.mean([float(row["mse"]) for row in initial_groups[eid]])
                                           for eid in sorted(initial_groups)])
                mse_change, low, high = interval(mse_difference)
                training_changes.append({"run": folder.name, "cell": cell, "episodes": len(initial_groups),
                                         "initial_accuracy": np.mean([int(row["correct"]) for row in initial]),
                                         "final_accuracy": np.mean([int(row["correct"]) for row in final]),
                                         "initial_all_correct": float(all_correct(initial).mean()),
                                         "final_all_correct": float(all_correct(final).mean()),
                                         "initial_mse": np.mean([float(row["mse"]) for row in initial]),
                                         "final_mse": np.mean([float(row["mse"]) for row in final]),
                                         "mse_final_minus_initial": mse_change, "mse_change_ci_low": low,
                                         "mse_change_ci_high": high})
        if metadata["args"]["suite"] == "full":
            summary, contrast = pressure(folder)
            summaries.extend(summary)
            contrasts.extend(contrast)
    write(args.out / "independent_gates.csv", gates)
    write(args.out / "pressure_fixed_targets.csv", summaries)
    write(args.out / "pressure_paired_contrasts.csv", contrasts)
    write(args.out / "training_initial_final.csv", training_changes)

    selection_path = args.root / "selection.json"
    qualification = {"status": "no_selection_record", "selected": None, "confirmed": False}
    if selection_path.exists():
        saved = json.loads(selection_path.read_text(encoding="utf-8"))
        reconstructed = []
        chosen = None
        assert 1 <= len(saved["candidates"]) <= 2
        for candidate_index, candidate in enumerate(saved["candidates"]):
            expected_tag, expected_eta = (("s05", 0.5), ("s10", 1.0))[candidate_index]
            assert candidate["tag"] == expected_tag and candidate["step_size"] == expected_eta
            assert candidate["runs"] == [f"{expected_tag}-90301", f"{expected_tag}-90305"]
            independently = [gate(args.root / run) for run in candidate["runs"]]
            assert len(independently) == len(candidate["gates"])
            for independent, declared in zip(independently, candidate["gates"], strict=True):
                for field in ("acquire2", "acquire4", "revise2", "normal_minus_rolled", "ci_low", "ci_high"):
                    assert abs(independent[field] - declared[field]) < 1e-12
                assert independent["passed"] == declared["passed"]
            reconstructed.append({"tag": candidate["tag"], "gates": independently})
            if len(independently) == 2 and all(item["passed"] for item in independently):
                chosen = candidate["tag"]
                break
        assert chosen == saved["selected"]
        confirmation = [gate(args.root / item["run"]) for item in saved.get("confirmation", [])]
        confirmed = len(confirmation) == 2 and all(item["passed"] for item in confirmation)
        assert confirmed == saved["confirmed"]
        qualification = {"status": "reconstructed_from_raw_predictions", "selected": chosen, "confirmed": confirmed,
                         "candidates": reconstructed, "confirmation": confirmation,
                         "interval_scope": "Fixed checkpoint(s) and sampled episodes; not training-procedure uncertainty"}
    (args.out / "qualification_audit.json").write_text(json.dumps(qualification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate_rows": len(gates), "pressure_summary_rows": len(summaries),
                      "paired_pressure_contrasts": len(contrasts), "qualification": qualification}, indent=2))


if __name__ == "__main__":
    main()
