"""Plot only independently audited tables; no model or raw-logit execution."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
COLORS = {"duplicate": "#D97932", "paired": "#1F6E9C", "auxiliary": "#7657A5"}


def table(path):
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def one(rows, **wanted):
    found = [row for row in rows if all(str(row[key]) == str(value) for key, value in wanted.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one audited row for {wanted}, found {len(found)}")
    return found[0]


def main(args):
    report, out = args.report.resolve(), args.out.resolve()
    if not out.is_relative_to(BASE) or out == BASE or out.exists():
        raise ValueError("--out must be a NEW child directory of analysis")
    verification = json.loads((report / "verification.json").read_text(encoding="utf8"))
    if verification["status"] != "verified" or verification["scope"] != "saved artifacts only":
        raise ValueError("Plotting requires a completed saved-artifact audit")
    strict = table(report / "strict_metrics.csv")
    gate = table(report / "gate_metrics.csv")
    differences = table(report / "primary_differences.csv")
    decisions = json.loads((report / "decisions.json").read_text(encoding="utf8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.labelsize": 10, "savefig.dpi": 180})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    stages = [200, 1200, 1600]
    for ax, seed in zip(axes[0], (68101, 68105)):
        for arm in ("duplicate", "paired"):
            values = [100 * float(one(strict, run=f"{arm}32-{seed}", step=step,
                family="seen", condition="online")["both_correct"]) for step in stages]
            ax.plot(stages, values, marker="o", color=COLORS[arm], label=f"{arm}, 32 bytes", lw=2)
        if seed == 68101:
            values = [100 * float(one(strict, run="paired128-68101", step=step,
                family="seen", condition="online")["both_correct"]) for step in stages]
            ax.plot(stages, values, marker="s", linestyle="--", color=COLORS["auxiliary"],
                label="paired, 128 bytes (auxiliary)", lw=1.5)
        ax.axhline(80, color="#5B6268", ls=":", lw=1)
        ax.set(title=f"Strict key-conditioned acquisition · parent {seed}",
               ylabel="Both queries correct (%)", xlabel="Fixed saved training step", ylim=(-2, 103))
        ax.set_xticks(stages)
        ax.grid(axis="y", alpha=.18)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax = axes[1, 0]
    positions, labels = [], []
    for group, seed in enumerate((68101, 68105)):
        for offset, arm in enumerate(("duplicate", "paired")):
            position = group * 3 + offset
            normal = one(gate, run=f"{arm}32-{seed}", step=1600,
                cell="acquire_k2", family="seen", condition="online")
            reset = one(gate, run=f"{arm}32-{seed}", step=1600,
                cell="acquire_k2", family="seen", condition="reset_memory")
            ax.bar(position, 100 * float(normal["accuracy"]), width=.65,
                   color=COLORS[arm], alpha=.9)
            ax.plot(position, 100 * float(reset["accuracy"]), marker="D", ms=6,
                    color="#252A30", linestyle="", label="reset memory" if position == 0 else None)
            positions.append(position); labels.append(f"{arm}\n{seed}")
    ax.axhline(80, color="#5B6268", ls=":", lw=1)
    ax.set(title="Final natural K2 acquisition · seen templates", ylabel="Per-query accuracy (%)", ylim=(-2, 103))
    ax.set_xticks(positions, labels, fontsize=9)
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    for y, seed in enumerate((68101, 68105)):
        row = one(differences, parent=seed, step=1600)
        mean, lower, upper = [100 * float(row[key]) for key in ("difference", "lower", "upper")]
        ax.errorbar(mean, y, xerr=[[max(0, mean-lower)], [max(0, upper-mean)]], fmt="o",
            color=COLORS["paired"], capsize=5, ms=7, lw=2)
        ax.annotate(f"{mean:+.2f} pp [{lower:+.2f}, {upper:+.2f}]", (mean, y),
                    xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0, color="#252A30", lw=1)
    ax.axvline(5, color="#5B6268", ls=":", lw=1)
    ax.set(title="Primary final endpoint · paired minus duplicate",
           xlabel="Strict seen both-correct difference (percentage points)", ylim=(-.65, 1.65))
    ax.set_yticks((0, 1), ("Parent 68101", "Parent 68105"))
    ax.grid(axis="x", alpha=.18)
    fig.suptitle("CLMM paired-key acquisition diagnostic", fontsize=15, fontweight="bold")
    fig.supxlabel("Fixed stage endpoints; final step 1600 is primary. Intervals: 5000 paired-episode bootstrap draws per parent.",
                  fontsize=9)
    out.mkdir(parents=True)
    fig.savefig(out / "paired_acquisition.png", bbox_inches="tight")
    fig.savefig(out / "paired_acquisition.pdf", bbox_inches="tight")
    plt.close(fig)
    (out / "figure_scope.json").write_text(json.dumps(dict(
        report=str(report), verification_sha256=__import__("hashlib").sha256((report / "verification.json").read_bytes()).hexdigest(),
        pressure_continuation_qualified=decisions["pressure_continuation_qualified"],
        useful_pairing_lead=decisions["useful_pairing_lead"],
        scope="All plotted values come from completed independent audit tables; stage points are descriptive.",
        source_tables=["strict_metrics.csv", "gate_metrics.csv", "primary_differences.csv"]), indent=2) + "\n", encoding="utf8")
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
