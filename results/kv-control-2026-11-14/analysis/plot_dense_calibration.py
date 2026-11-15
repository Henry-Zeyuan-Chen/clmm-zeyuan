"""Separate figure for the one-codebook supplemental interference instrument."""

import argparse
import csv
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    with (args.analysis / "dense_pressure_fixed_targets.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    audit = json.loads((args.analysis / "dense_calibration_audit.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.1), sharex=True, sharey=True)
    delays = [0, 16, 64]
    styles = {"online": ("#555555", "o", "-", "Online / repeat"),
              "continuous": ("#0072B2", "o", "-", "Continuous"),
              "periodic": ("#D55E00", "s", "--", "Periodic"),
              "boundary_reset": ("#CC79A7", ".", ":", "Boundary reset")}
    def number(facts, delay, arm, category, field="accuracy"):
        return float(next(row[field] for row in rows if row["cell"] == f"pressure_k{facts}_d{delay}"
                          and row["arm"] == arm and row["category"] == category))
    for column, facts in enumerate((8, 16, 32)):
        all_correct = int(round(32 * number(facts, 0, "online", "targets", "all_correct")))
        axes[0, column].set_title(f"K = {facts}\nD0 four-target all correct: {all_correct}/32", fontsize=10)
        for row_index, category in enumerate(("stable", "revised")):
            axis = axes[row_index, column]
            for arm, (color, marker, style, label) in styles.items():
                accuracy = [number(facts, delay, arm, category) for delay in delays]
                if arm == "online":
                    assert accuracy == [number(facts, delay, "repeat", category) for delay in delays]
                axis.plot(delays, np.array(accuracy) * 100, color=color, marker=marker,
                          linestyle=style, linewidth=1.6, label=label)
            axis.grid(axis="y", alpha=0.18)
            axis.set_ylim(0, 105)
            axis.set_yticks([0, 25, 50, 75, 100])
            axis.set_xticks(delays)
            if row_index == 1:
                axis.set_xlabel("Outside-key distractor observations")
    axes[0, 0].set_ylabel("Stable-query accuracy (%)")
    axes[1, 0].set_ylabel("Revised-query accuracy (%)")
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)
    figure.suptitle("One fixed dense-key linear control detects interference", y=1.01, fontsize=13)
    primary = audit["primary"]
    figure.text(0.5, -0.025,
                f"Prespecified K8, D64 stable periodic − continuous: +{primary['difference'] * 100:.2f} pp; "
                f"95% paired-episode bootstrap [{primary['ci_low'] * 100:.2f}, {primary['ci_high'] * 100:.2f}] pp (includes zero).",
                ha="center", fontsize=8.7)
    figure.text(0.5, -0.067,
                "32 paired episodes per cell; one untrained codebook. K32 begins with weak acquisition. Geometry and state dimension both differ from the main control.",
                ha="center", fontsize=8.3)
    figure.tight_layout()
    out = args.analysis / "figures"
    out.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(out / f"dense_interference_calibration.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(str(out / "dense_interference_calibration.png"))


if __name__ == "__main__":
    main()
