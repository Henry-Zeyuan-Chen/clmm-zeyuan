"""Numerical figures from independently recomputed result tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    directory = args.analysis / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    training = rows(args.analysis / "training_initial_final.csv")
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), sharey=True)
    for axis, cell, title in zip(axes, ("acquire2", "acquire4", "revise2"), ("Acquire K2", "Acquire K4", "Revise K2"), strict=True):
        for seed, color in (("90301", "#0072B2"), ("90305", "#D55E00")):
            row = next(row for row in training if row["cell"] == cell and row["run"].endswith(seed))
            axis.plot([0, 1], [float(row["initial_mse"]), float(row["final_mse"])], marker="o", color=color,
                      linewidth=1.7, label=f"Initializer {seed}")
        axis.set_title(title)
        axis.set_xticks([0, 1], ["Initial", "After 1,200 updates"])
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.18)
        axis.text(0.5, 0.93, "All correct: 128/128\nat both stages, both initializers", ha="center", va="top",
                  transform=axis.transAxes, fontsize=9)
    axes[0].set_ylabel("Mean squared read-coordinate error")
    axes[0].set_ylim(5e-6, 0.25)
    axes[-1].legend(loc="lower left", frameon=False, fontsize=9)
    figure.suptitle("Short-task accuracy is already at ceiling before outer training", y=1.02, fontsize=13)
    figure.text(0.5, -0.04, "Same 128 sampled development episodes per cell; MSE improves, while observed all-correct accuracy stays at 100%.", ha="center", fontsize=9)
    figure.tight_layout()
    save(figure, directory, "initial_final_control")

    pressure = rows(args.analysis / "pressure_fixed_targets.csv")
    pressure = [row for row in pressure if row["run"].startswith("pressure-") and row["category"] == "targets"]
    figure, axes = plt.subplots(2, 3, figsize=(10.6, 6.3), sharex=True, sharey="row")
    delays = np.array([0, 16, 64])
    colors = {"online": "#555555", "repeat": "#009E73", "continuous": "#0072B2", "periodic": "#D55E00"}
    for column, facts in enumerate((8, 16, 32)):
        top, lower = axes[:, column]
        top.set_title(f"K = {facts}")
        top.plot(delays, np.ones(3) * 100, "o-", color="#222222", label="All four ordinary arms")
        for arm in ("boundary_reset",):
            values = np.array([[float(next(row["accuracy"] for row in pressure if row["run"].endswith(seed)
                                              and row["cell"] == f"pressure_k{facts}_d{delay}" and row["arm"] == arm))
                                for delay in delays] for seed in ("90301", "90305")]) * 100
            top.plot(delays, values.mean(0), "s--", color="#CC79A7", label="Boundary reset")
            top.fill_between(delays, values.min(0), values.max(0), color="#CC79A7", alpha=0.15)
        for arm in ("online", "repeat", "continuous", "periodic"):
            values = np.array([[float(next(row["mse"] for row in pressure if row["run"].endswith(seed)
                                              and row["cell"] == f"pressure_k{facts}_d{delay}" and row["arm"] == arm))
                                for delay in delays] for seed in ("90301", "90305")])
            lower.plot(delays, values.mean(0), marker="o", linewidth=1.5, linestyle="--" if arm == "periodic" else "-",
                       color=colors[arm], label=arm.capitalize())
            lower.fill_between(delays, values.min(0), values.max(0), color=colors[arm], alpha=0.10)
        top.set_ylim(0, 105)
        top.set_yticks([0, 25, 50, 75, 100])
        top.grid(axis="y", alpha=0.18)
        lower.set_yscale("log")
        lower.grid(axis="y", alpha=0.18)
        lower.set_xticks(delays)
        lower.set_xlabel("Outside-key distractor observations")
    axes[0, 0].set_ylabel("Four-target accuracy (%)")
    axes[1, 0].set_ylabel("Four-target mean squared error")
    axes[0, 0].legend(loc="center left", frameon=False, fontsize=8)
    axes[1, 0].legend(loc="upper left", frameon=False, fontsize=8)
    figure.suptitle("Qualified neural memory: no accuracy headroom in the orthogonal pressure grid", y=1.01, fontsize=13)
    figure.text(0.5, -0.025, "32 episodes per cell and initializer; the same four initial targets at every delay. Lines average two fixed initializers; shading is their range.", ha="center", fontsize=8.5)
    figure.tight_layout()
    save(figure, directory, "orthogonal_pressure_ceiling")
    print(str(directory))


if __name__ == "__main__":
    main()
