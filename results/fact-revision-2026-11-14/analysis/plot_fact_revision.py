"""Plot only independently certified tables; no inference or invented values."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def rows(path):
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def choose(table, **where):
    selected = [row for row in table if all(row.get(key) == str(value) for key, value in where.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one verified row: {where}, got {len(selected)}")
    row = selected[0]
    return np.array([float(row[key])*100 for key in ("estimate", "ci_low", "ci_high")])


def draw(axis, values, labels, colors, ylabel, ylim=None):
    points = np.array(values)
    x = np.arange(len(values))
    for i, (mean, low, high) in enumerate(points):
        axis.errorbar(i, mean, yerr=[[mean-low], [high-mean]], fmt="o", color=colors[i], capsize=4, ms=6)
    axis.set_xticks(x, labels)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=.2)
    axis.set_xlim(-.6, len(values)-.4)
    if ylim is not None:
        axis.set_ylim(*ylim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace these plot outputs in an existing authorized analysis directory")
    args = parser.parse_args()
    source, out = args.analysis.resolve(), args.out.resolve()
    if not out.is_relative_to(HERE) or out == HERE:
        raise ValueError("Plot output must be a new subdirectory of the authorized analysis directory")
    verified = json.loads((source/"verification.json").read_text())
    if verified.get("status") != "verified" or verified.get("main_certified") is not True:
        raise ValueError("Refusing to plot uncertified or smoke-only main results")
    for name in ("estimates.csv", "contrasts.csv", "decisions.json"):
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != verified["output_sha256"][name]:
            raise ValueError("Verified table changed after analysis: " + name)
    estimates, contrasts = rows(source/"estimates.csv"), rows(source/"contrasts.csv")
    decisions = json.loads((source/"decisions.json").read_text())
    acquisition_failed = not decisions["both_acquisition_pass"]
    out.mkdir(parents=True, exist_ok=args.overwrite)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.3), constrained_layout=True)
    palette = ["#6b7280", "#b2b8c2", "#20639b", "#8bb8da"]
    for col, seed in enumerate(("68101", "68105")):
        values = [choose(estimates, checkpoint_seed=seed, stage=stage, cell="acquire_k2", family="seen",
                         category="stable", condition=condition, metric="correct")
                  for stage in ("before", "after") for condition in ("online", "reset_memory")]
        draw(axes[0, col], values, ["Before\nnormal", "Before\nreset", "After\nnormal", "After\nreset"],
             palette, "Full-vocabulary answer accuracy (%)", (0, 100))
        axes[0, col].axhline(80, color="#999999", ls="--", lw=1, label="Acquisition threshold")
        gate_label = "gate passed" if decisions["acquisition"][seed]["after"]["passed"] else "gate failed"
        axes[0, col].set_title(f"Checkpoint {seed}: K=2 acquisition ({gate_label})")
        values = [choose(contrasts, checkpoint_seed=seed, stage="after", cell="pressure_k8_d8", family="seen",
                         category="stable", left=left, right=right, metric="correct")
                  for left, right in (("periodic", "continuous"), ("periodic", "repeat_current"), ("continuous", "repeat_current"))]
        draw(axes[1, col], values, ["Periodic −\ncontinuous", "Periodic −\nrepeat", "Continuous −\nrepeat"],
             ["#20639b", "#3b966b", "#7b6aa8"], "Paired accuracy difference (percentage points)", (-7, 4))
        axes[1, col].axhline(0, color="black", lw=.8)
        axes[1, col].set_title("K=8, D=8: stable facts (floor diagnostic)" if acquisition_failed else "K=8, D=8: stable facts")
    fig.suptitle("Main experiment: acquisition failed" if acquisition_failed else "Fact acquisition and reference replay timing", fontsize=15)
    fig.supxlabel("95% episode-bootstrap intervals, conditional on each fixed checkpoint.\nReplay contrasts are floor diagnostics after failed acquisition; both panels use the same scale."
                  if acquisition_failed else
                  "95% episode-bootstrap intervals, conditional on each fixed checkpoint.\nFirst replay contrast is primary; repeat-control contrasts are exploratory.", fontsize=9)
    fig.savefig(out/"fact_revision_pilot.png", dpi=180)
    fig.savefig(out/"fact_revision_pilot.pdf")
    plt.close(fig)
    manifest = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                for name in ("verification.json", "estimates.csv", "contrasts.csv", "decisions.json")}
    (out/"manifest.json").write_text(json.dumps(dict(source_analysis=str(source), source_sha256=manifest,
          script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          outputs=["fact_revision_pilot.png", "fact_revision_pilot.pdf"]), indent=2)+"\n")


if __name__ == "__main__":
    main()
