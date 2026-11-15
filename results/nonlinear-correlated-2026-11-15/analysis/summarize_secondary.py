"""Fixed-seed secondary contrasts on the prespecified primary histories."""

import argparse
import json
from pathlib import Path

from analyze_results import chosen, interval, read, values, write
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank = json.loads((args.root / "selection.json").read_text())["selected_rank"]
    rows = {(stage, seed): read(args.root / f"comparison-{stage}-r{rank}-{seed}" / "evaluation/predictions.csv")
            for stage in ("initial", "trained") for seed in (90511, 90515)}
    def array(stage, arm, metric="accuracy"):
        return np.mean([values(chosen(rows[stage, seed], "pressure_k8_d64", arm, "stable"), metric)
                        for seed in (90511, 90515)], axis=0)
    secondary, training = [], []
    for stage in ("initial", "trained"):
        for left, right in (("continuous", "online"), ("periodic", "online"), ("repeat", "online"), ("online", "boundary_reset")):
            a, b = array(stage, left), array(stage, right)
            mean, low, high = interval(a - b)
            secondary.append(dict(stage=stage, cell="pressure_k8_d64", category="stable", contrast=f"{left}_minus_{right}",
                episodes=128, fixed_initializers=2, left_accuracy=float(a.mean()), right_accuracy=float(b.mean()),
                difference=mean, ci_low=low, ci_high=high, scope="secondary; fixed checkpoints/codebook; online has fewer writes"))
    for arm in ("online", "repeat", "continuous", "periodic"):
        for metric in ("accuracy", "mse"):
            after, before = array("trained", arm, metric), array("initial", arm, metric)
            mean, low, high = interval(after - before)
            training.append(dict(cell="pressure_k8_d64", category="stable", arm=arm, metric=metric,
                episodes=128, fixed_initializers=2, initial_mean=float(before.mean()), trained_mean=float(after.mean()),
                trained_minus_initial=mean, ci_low=low, ci_high=high, scope="secondary; same histories/exact initial checkpoints"))
    write(args.out / "fixed_seed_secondary.csv", secondary)
    write(args.out / "fixed_seed_initial_final.csv", training)
    print(json.dumps(dict(secondary=secondary, initial_final=training), indent=2))


if __name__ == "__main__":
    main()
