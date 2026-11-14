"""Post-hoc descriptive slices of saved training logs; never validation scores."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not any(args.out.iterdir()), 'Output directory must be empty'
    records, sources = [], []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, seed in zip(axes, (68101, 68105)):
        source = args.root / f'adapt-{seed}/training.csv'
        sources.append(dict(path=str(source.relative_to(args.root)),
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        with source.open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert [int(r['step']) for r in rows] == list(range(1, 1601))
        simple = [r for r in rows if int(r['facts']) == 1 and int(r['delay']) == 0
                  and math.floor(int(r['facts']) * float(r['revision_fraction'])) == 0]
        for phase, selected in (
            ('warmup_last100', [r for r in simple if 221 <= int(r['step']) <= 320]),
            ('later_k1d0', [r for r in simple if int(r['step']) >= 641]),
        ):
            records.append(dict(seed=seed, phase=phase, batches=len(selected),
                mean_accuracy=float(np.mean([float(r['accuracy']) for r in selected])),
                mean_loss=float(np.mean([float(r['loss']) for r in selected])),
                max_gradient_norm=max(float(r['gradient_norm']) for r in selected)))
        with (args.out / f'k1d0_training_{seed}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(simple)
        steps = [int(r['step']) for r in simple]
        accuracy = np.array([float(r['accuracy']) for r in simple]) * 100
        trailing = [accuracy[max(0, i-24):i+1].mean() for i in range(len(accuracy))]
        ax.scatter(steps, accuracy, s=12, color='#3975a2', alpha=.23, label='Each K1 / D0 batch')
        ax.plot(steps, trailing, lw=2, color='#175886', label='Trailing 25 K1 / D0 batches')
        for x in (320, 640):
            ax.axvline(x, color='#888888', ls='--', lw=1)
        ax.set(xlabel='Optimizer step', title=f'Parent checkpoint {seed}', ylim=(-3, 103))
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('One-fact training accuracy (%)')
    axes[0].legend(loc='lower left', fontsize=9)
    fig.suptitle('One-fact performance falls during the harder mixed curriculum', fontsize=15)
    fig.text(.5, .015, 'Fresh training batches, not held-out validation. Post hoc view; dashed lines mark curriculum changes.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .93))
    fig.savefig(args.out / 'training_trajectory.png', dpi=170)
    fig.savefig(args.out / 'training_trajectory.svg')
    plt.close(fig)
    prior = json.loads((args.root / 'training_diagnostic.json').read_text())['rows']
    assert len(prior) == len(records)
    for expected, actual in zip(prior, records):
        assert expected.keys() == actual.keys()
        for key in expected:
            if isinstance(expected[key], float):
                assert abs(expected[key] - actual[key]) < 1e-12
            else:
                assert expected[key] == actual[key]
    report = dict(scope='Post-hoc training-log description; no intermediate validation or causal attribution.',
                  compared_to_original_diagnostic=True, sources=sources, rows=records,
                  plot_smoothing='Trailing 25 observed K1/D0 batches, not 25 optimizer steps.')
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    with (args.out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    shutil.copy2(__file__, args.out / Path(__file__).name)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
