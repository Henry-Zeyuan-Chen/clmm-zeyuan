"""Figures from independently reconstructed source tables, without model calls."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, out, name):
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    out = args.analysis / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    selection = json.loads((args.analysis / "selection_audit.json").read_text())
    primary = read(args.analysis / "primary.csv")
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    labels = []
    for index, row in enumerate(primary):
        point, low, high = [100 * float(row[key]) for key in ("difference", "ci_low", "ci_high")]
        mean = row["initializer"] == "mean_of_fixed_initializers"
        color = "#0072B2" if row["stage"] == "trained" else "#777777"
        ax.errorbar(point, index, xerr=[[point - low], [high - point]], fmt="D" if mean else "o", color=color,
                    capsize=4, markersize=7 if mean else 5, linewidth=2 if mean else 1.3)
        initializer = "mean of 2 fixed initializers" if mean else f"initializer {row['initializer']}"
        labels.append(f"{row['stage'].capitalize()}, {initializer}")
        ax.annotate(f"{point:+.2f} [{low:+.2f}, {high:+.2f}]", (high, index), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=9)
    low = min(100 * float(row["ci_low"]) for row in primary)
    high = max(100 * float(row["ci_high"]) for row in primary)
    width = max(2.5, high - low)
    ax.set_xlim(min(0, low) - width * .2, max(0, high) + width * .75)
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
    ax.axhline(2.5, color="#cccccc", linewidth=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Periodic minus continuous stable-query accuracy (percentage points)")
    ax.grid(axis="x", alpha=.15)
    fig.suptitle(f"Prespecified K8, D64 timing contrast at key rank {selection['selected_rank']}", fontsize=13)
    ax.set_title(selection["fixed_two_seed_primary_scope"] + "; initial checkpoints are diagnostic anchors", fontsize=9)
    fig.text(.5, -.045, "128 paired histories; one fixed codebook; 5,000 bootstrap draws, seed 90504. Fixed-seed means preserve pairing across both models.", ha="center", fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "primary_timing_contrast")

    rows = [row for row in read(args.analysis / "pressure_fixed_targets.csv") if row["run"].startswith("comparison-") and row["category"] == "stable"]
    eligibility = read(args.analysis / "retention_eligibility.csv")
    colors = {"online": "#555555", "repeat": "#009E73", "continuous": "#0072B2", "periodic": "#D55E00", "boundary_reset": "#CC79A7"}
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), sharex=True, sharey=True)
    delays = [0, 16, 64]
    rank = selection["selected_rank"]
    for row_index, stage in enumerate(("trained", "initial")):
        for column, facts in enumerate((8, 16, 32)):
            axis = axes[row_index, column]
            d0 = [next(row for row in eligibility if row["stage"] == stage and int(row["init_seed"]) == seed and int(row["facts"]) == facts) for seed in (90511, 90515)]
            rates = "/".join(f"{float(row['d0_four_target_all_correct']) * 100:.1f}" for row in d0)
            scope = "qualified" if stage == "trained" and all(row["qualified_trained_retention"] == "True" for row in d0) else "diagnostic"
            axis.set_title(f"{stage.capitalize()}, K={facts} ({scope})\nD0 four-target all correct: {rates}%", fontsize=9)
            for arm in ("online", "repeat", "continuous", "periodic", "boundary_reset"):
                values = np.array([[float(next(row["accuracy"] for row in rows if row["run"] == f"comparison-{stage}-r{rank}-{seed}"
                                               and row["cell"] == f"pressure_k{facts}_d{delay}" and row["arm"] == arm))
                                    for delay in delays] for seed in (90511, 90515)]) * 100
                axis.plot(delays, values.mean(0), marker="s" if arm == "periodic" else "o", markersize=4,
                          linestyle="--" if arm == "periodic" else ":" if arm == "boundary_reset" else "-",
                          linewidth=1.5, color=colors[arm], label=arm.replace("_", " ").capitalize())
                axis.fill_between(delays, values.min(0), values.max(0), color=colors[arm], alpha=.08)
            axis.set_ylim(0, 105)
            axis.set_yticks([0, 25, 50, 75, 100])
            axis.set_xticks(delays)
            axis.grid(axis="y", alpha=.17)
            if row_index == 1:
                axis.set_xlabel("Outside-key distractor observations")
    axes[0, 0].set_ylabel("Stable-query accuracy (%)")
    axes[1, 0].set_ylabel("Stable-query accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .985), ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"Nonlinear correlated-key memory, rank {rank}: trained and exact initial checkpoints", fontsize=13, y=1.025)
    fig.text(.5, -.025, "128 paired histories per cell. Lines average two fixed initializers; shading is their range, not a confidence interval.", ha="center", fontsize=8.5)
    fig.text(.5, -.064, "D0 labels list initializers 90511/90515. The same initial targets form each delay curve; compare schedules within the stable stratum.", ha="center", fontsize=8.4)
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, out, "pressure_and_initializer")
    print(str(out))


if __name__ == "__main__":
    main()
