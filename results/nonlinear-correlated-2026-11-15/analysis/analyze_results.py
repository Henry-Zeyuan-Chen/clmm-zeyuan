"""Reconstruct calibration, qualification and paired fixed-seed contrasts."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
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


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["episode"])].append(row)
    return [group for _, group in sorted(groups.items())]


def values(rows, metric="accuracy"):
    if metric == "all_correct":
        return np.array([all(int(row["correct"]) for row in group) for group in grouped(rows)], dtype=float)
    field = {"accuracy": "correct", "mse": "mse", "obsolete": "obsolete"}[metric]
    return np.array([np.mean([float(row[field]) for row in group]) for group in grouped(rows)])


def chosen(rows, cell, arm="online", category="all"):
    return [row for row in rows if row["cell"] == cell and row["arm"] == arm
            and (category == "all" or row["category"] == category
                 or category == "targets" and row["category"] in ("stable", "revised"))]


def interval(array):
    array = np.asarray(array, dtype=float)
    index = np.random.default_rng(90504).integers(0, len(array), size=(5000, len(array)))
    low, high = np.quantile(array[index].mean(1), [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def gate(folder, stage="evaluation"):
    rows = read(folder / stage / "predictions.csv")
    short = {cell: values(chosen(rows, cell), "all_correct") for cell in ("acquire2", "acquire4", "revise2")}
    assert all(len(array) == 128 for array in short.values())
    delta = short["acquire2"] - values(chosen(rows, "acquire2", "rolled"), "all_correct")
    mean, low, high = interval(delta.reshape(8, 16).mean(1))
    d0 = values(chosen(rows, "pressure_k8_d0", category="targets"), "all_correct").mean()
    stable0 = values(chosen(rows, "pressure_k8_d0", category="stable")).mean()
    stable64 = values(chosen(rows, "pressure_k8_d64", category="stable")).mean()
    result = {cell: float(array.mean()) for cell, array in short.items()}
    result.update(normal_minus_rolled=mean, ci_low=low, ci_high=high, k8_d0_all_correct=float(d0),
                  stable_d0=float(stable0), stable_d64=float(stable64), stable_drop=float(stable0 - stable64))
    result["short_pass"] = bool(result["acquire2"] >= .9 and result["acquire4"] >= .8 and result["revise2"] >= .9 and low > 0)
    result["acquisition_pass"] = bool(result["short_pass"] and d0 >= .8)
    result["headroom_pass"] = bool(.15 <= stable64 <= .85 and stable0 - stable64 >= .1)
    result["eligible"] = bool(result["acquisition_pass"] and result["headroom_pass"])
    return result


def verify_declared(actual, declared):
    for key, value in actual.items():
        if isinstance(value, bool):
            assert value is declared[key], (key, value, declared[key])
        else:
            assert abs(value - declared[key]) < 1e-12, (key, value, declared[key])


def pressure_tables(folder):
    rows = read(folder / "evaluation/predictions.csv")
    summaries, contrasts = [], []
    arrays = {}
    for cell in sorted({row["cell"] for row in rows if row["cell"].startswith("pressure")}):
        for arm in sorted({row["arm"] for row in rows if row["cell"] == cell}):
            for category in ("targets", "stable", "revised", "recent"):
                group = chosen(rows, cell, arm, category)
                if not group:
                    continue
                array = values(group)
                mean, low, high = interval(array)
                arrays[cell, arm, category] = array
                summaries.append(dict(run=folder.name, cell=cell, arm=arm, category=category,
                    episodes=len(array), queries=len(group), accuracy=mean, ci_low=low, ci_high=high,
                    all_correct=float(values(group, "all_correct").mean()), mse=float(values(group, "mse").mean()),
                    obsolete_rate=float(values(group, "obsolete").mean()) if category == "revised" else ""))
        for category in ("targets", "stable", "revised", "recent"):
            for left, right in (("periodic", "continuous"), ("continuous", "online"), ("periodic", "online"),
                                ("repeat", "online"), ("online", "boundary_reset")):
                if (cell, left, category) not in arrays or (cell, right, category) not in arrays:
                    continue
                mean, low, high = interval(arrays[cell, left, category] - arrays[cell, right, category])
                contrasts.append(dict(run=folder.name, cell=cell, category=category, contrast=f"{left}_minus_{right}",
                    episodes=len(arrays[cell, left, category]), difference=mean, ci_low=low, ci_high=high,
                    left_accuracy=float(arrays[cell, left, category].mean()), right_accuracy=float(arrays[cell, right, category].mean())))
    return summaries, contrasts


def replay_coverage(folder):
    episodes = json.loads((folder / "evaluation/episodes.json").read_text())
    plans = json.loads((folder / "evaluation/plans.json").read_text())
    plan_map = {(item["cell"], item["arm"], item["episode"]): item["plan"] for item in plans}
    rows = read(folder / "evaluation/predictions.csv")
    result = []
    for cell in sorted(name for name in episodes if name.startswith("pressure")):
        for arm in ("continuous", "periodic"):
            for category in ("targets", "stable", "revised", "recent"):
                group = chosen(rows, cell, arm, category)
                if not group:
                    continue
                details = []
                for row in group:
                    plan = plan_map[cell, arm, int(row["episode"])]
                    key = int(row["key"])
                    replayed = [event for event in plan["events"] if event["kind"] == "extra" and event["key"] == key]
                    details.append(dict(correct=int(row["correct"]), replayed=bool(replayed),
                                        replay_count=len(replayed), resident=key in plan["final_buffer_keys"]))
                evicted = [item for item in details if not item["resident"]]
                seen = [item for item in details if item["replayed"]]
                result.append(dict(run=folder.name, cell=cell, arm=arm, category=category, queries=len(details),
                    ever_replayed=sum(item["replayed"] for item in details), replay_writes=sum(item["replay_count"] for item in details),
                    final_resident=sum(item["resident"] for item in details), final_evicted=len(evicted),
                    evicted_accuracy=float(np.mean([item["correct"] for item in evicted])) if evicted else "",
                    ever_replayed_accuracy=float(np.mean([item["correct"] for item in seen])) if seen else ""))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = {}
    for path in args.root.glob("*/metadata.json"):
        info = json.loads(path.read_text())
        if path.parent.name.startswith("smoke") or info["status"] != "succeeded":
            continue
        metadata[path.parent.name] = info
    gates, pressure, contrasts, calibration_changes, coverage = [], [], [], [], []
    for name, info in metadata.items():
        folder = args.root / name
        gates.append(dict(run=name, stage="evaluation", rank=info["config"]["rank"], init_seed=info["args"]["init_seed"], **gate(folder)))
        summary, contrast = pressure_tables(folder)
        pressure.extend(summary)
        contrasts.extend(contrast)
        if info["args"]["phase"] == "comparison":
            coverage.extend(replay_coverage(folder))
        if (folder / "initial-evaluation").is_dir():
            gates.append(dict(run=name, stage="initial-evaluation", rank=info["config"]["rank"], init_seed=info["args"]["init_seed"], **gate(folder, "initial-evaluation")))
            assert (folder / "initial-evaluation/episodes.json").read_bytes() == (folder / "evaluation/episodes.json").read_bytes()
            before, after = read(folder / "initial-evaluation/predictions.csv"), read(folder / "evaluation/predictions.csv")
            for cell in sorted({row["cell"] for row in after}):
                category = "targets" if cell.startswith("pressure") else "all"
                initial, final = chosen(before, cell, category=category), chosen(after, cell, category=category)
                change, low, high = interval(values(final, "mse") - values(initial, "mse"))
                calibration_changes.append(dict(run=name, rank=info["config"]["rank"], init_seed=info["args"]["init_seed"], cell=cell,
                    category=category, episodes=len(values(final)), initial_accuracy=float(values(initial).mean()),
                    final_accuracy=float(values(final).mean()), initial_all_correct=float(values(initial, "all_correct").mean()),
                    final_all_correct=float(values(final, "all_correct").mean()), initial_mse=float(values(initial, "mse").mean()),
                    final_mse=float(values(final, "mse").mean()), mse_final_minus_initial=change, mse_change_ci_low=low, mse_change_ci_high=high))
    write(args.out / "independent_gates.csv", gates)
    write(args.out / "pressure_fixed_targets.csv", pressure)
    write(args.out / "pressure_paired_contrasts.csv", contrasts)
    write(args.out / "calibration_initial_final.csv", calibration_changes)
    write(args.out / "replay_residency.csv", coverage)

    saved = json.loads((args.root / "selection.json").read_text())
    candidates, acquisition, first_eligible = [], [], None
    for index, candidate in enumerate(saved["candidates"]):
        rank = (32, 16, 8)[index]
        assert candidate["rank"] == rank and candidate["runs"] == [f"r{rank}-90511", f"r{rank}-90515"]
        actual = [gate(args.root / name) for name in candidate["runs"]]
        assert len(candidate["gates"]) == 2
        for result, declared in zip(actual, candidate["gates"], strict=True):
            verify_declared(result, declared)
        if all(item["acquisition_pass"] for item in actual):
            acquisition.append(rank)
        if all(item["eligible"] for item in actual):
            first_eligible = rank
            assert index == len(saved["candidates"]) - 1
        candidates.append(dict(rank=rank, runs=candidate["runs"], gates=actual))
    selected = first_eligible if first_eligible is not None else acquisition[-1] if acquisition else 8
    if first_eligible is None:
        assert len(candidates) == 3
    assert saved["selected_rank"] == selected and saved["calibration_eligible"] is (first_eligible is not None)
    confirmation = []
    for declared, seed in zip(saved["confirmation"], (90511, 90515), strict=True):
        assert declared["run"] == f"confirm-r{selected}-{seed}"
        actual = gate(args.root / declared["run"])
        verify_declared(actual, declared)
        confirmation.append(dict(run=declared["run"], **actual))
    confirmed = all(item["acquisition_pass"] for item in confirmation)
    assert len(confirmation) == 2 and saved["confirmed"] is confirmed
    assert saved["fresh_headroom_both"] is all(item["headroom_pass"] for item in confirmation)
    freeze = json.loads((args.root / "comparison-freeze.json").read_text())
    assert freeze["selection"] == saved and freeze["no_replay_outcomes_inspected"] is True
    assert freeze["selection_sha256"] == hashlib.sha256((args.root / "selection.json").read_bytes()).hexdigest()
    assert freeze["design_sha256"] == hashlib.sha256((args.root / "design.json").read_bytes()).hexdigest()

    eligibility, primary, training_comparison = [], [], []
    comparison_rows = {}
    for stage in ("trained", "initial"):
        for seed in (90511, 90515):
            name = f"comparison-{stage}-r{selected}-{seed}"
            assert datetime.fromisoformat(metadata[name]["started_utc"]) >= datetime.fromisoformat(freeze["frozen_utc"])
            rows = read(args.root / name / "evaluation/predictions.csv")
            comparison_rows[stage, seed] = rows
            for facts in (8, 16, 32):
                d0 = chosen(rows, f"pressure_k{facts}_d0", category="targets")
                rate = float(values(d0, "all_correct").mean())
                eligibility.append(dict(run=name, stage=stage, init_seed=seed, rank=selected, facts=facts,
                    d0_accuracy=float(values(d0).mean()), d0_four_target_all_correct=rate, d0_pass=bool(rate >= .8),
                    calibration_eligible=first_eligible is not None, confirmed=confirmed,
                    qualified_trained_retention=bool(stage == "trained" and first_eligible is not None and confirmed and rate >= .8)))
        differences, lefts, rights = [], [], []
        for seed in (90511, 90515):
            rows = comparison_rows[stage, seed]
            left = values(chosen(rows, "pressure_k8_d64", "periodic", "stable"))
            right = values(chosen(rows, "pressure_k8_d64", "continuous", "stable"))
            assert len(left) == len(right) == 128
            difference = left - right
            differences.append(difference)
            lefts.append(left)
            rights.append(right)
            mean, low, high = interval(difference)
            primary.append(dict(stage=stage, initializer=str(seed), rank=selected, episodes=128, fixed_initializers=1,
                                periodic_accuracy=float(left.mean()), continuous_accuracy=float(right.mean()),
                                difference=mean, ci_low=low, ci_high=high))
        mean, low, high = interval(np.mean(differences, axis=0))
        primary.append(dict(stage=stage, initializer="mean_of_fixed_initializers", rank=selected, episodes=128, fixed_initializers=2,
                            periodic_accuracy=float(np.mean(lefts)), continuous_accuracy=float(np.mean(rights)),
                            difference=mean, ci_low=low, ci_high=high))
    episode_hashes = {hashlib.sha256((args.root / f"comparison-{stage}-r{selected}-{seed}" / "evaluation/episodes.json").read_bytes()).hexdigest()
                      for stage in ("trained", "initial") for seed in (90511, 90515)}
    assert len(episode_hashes) == 1
    for seed in (90511, 90515):
        initial, trained = comparison_rows["initial", seed], comparison_rows["trained", seed]
        for cell in sorted({row["cell"] for row in trained}):
            categories = ("targets", "stable", "revised", "recent") if cell.startswith("pressure") else ("all",)
            for arm in sorted({row["arm"] for row in trained if row["cell"] == cell}):
                if not cell.startswith("pressure") and arm != "online":
                    continue
                for category in categories:
                    before, after = chosen(initial, cell, arm, category), chosen(trained, cell, arm, category)
                    if not before:
                        continue
                    for metric in ("accuracy", "mse"):
                        a, b = values(after, metric), values(before, metric)
                        mean, low, high = interval(a - b)
                        training_comparison.append(dict(rank=selected, init_seed=seed, cell=cell, arm=arm, category=category,
                            metric=metric, episodes=len(a), initial_mean=float(b.mean()), trained_mean=float(a.mean()),
                            trained_minus_initial=mean, ci_low=low, ci_high=high))
    paired_d0 = {facts: all(item["d0_pass"] for item in eligibility if item["stage"] == "trained" and item["facts"] == facts)
                 for facts in (8, 16, 32)}
    for item in eligibility:
        item["both_trained_d0_pass"] = paired_d0[item["facts"]]
        item["qualified_trained_retention"] = bool(item["stage"] == "trained" and first_eligible is not None and confirmed and paired_d0[item["facts"]])
    paired_k8_pass = bool(first_eligible is not None and confirmed and paired_d0[8])
    for item in primary:
        item["scope"] = "untrained diagnostic anchor" if item["stage"] == "initial" else "qualified within-setting timing comparison" if paired_k8_pass else "ineligible diagnostic"
    result = dict(status="reconstructed_from_raw_predictions", selected_rank=selected, calibration_eligible=first_eligible is not None,
                  confirmed=confirmed, fixed_two_seed_primary_scope="qualified within-setting timing comparison" if paired_k8_pass else "ineligible diagnostic",
                  candidates=candidates, confirmation=confirmation,
                  bootstrap=dict(seed=90504, draws=5000, unit="same128 whole-history IDs across both fixed initializers; no training/codebook population CI"))
    write(args.out / "retention_eligibility.csv", eligibility)
    write(args.out / "primary.csv", primary)
    write(args.out / "initial_final.csv", training_comparison)
    (args.out / "selection_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(selection=result, primary=primary), indent=2))


if __name__ == "__main__":
    main()
