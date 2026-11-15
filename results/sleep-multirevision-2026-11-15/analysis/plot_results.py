"""Render numerical figures only from the independent saved-artifact tables."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from independent_tools import ARMS, read_csv

COLORS = dict(online='#737373', repeat='#bf8d22', continuous='#2876ac', periodic='#b44b4b')
LABELS = dict(online='Online', repeat='Repeat current', continuous='Continuous', periodic='Periodic')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    rows = read_csv(analysis/'aggregate.csv')
    contrasts = read_csv(analysis/'contrasts.csv')
    diagnostics = read_csv(analysis/'diagnostics.csv')
    telemetry = read_csv(analysis/'replay_telemetry.csv')
    out = analysis/'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 190, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})
    files = []

    def save(fig, name):
        for extension in ('png', 'pdf', 'svg'):
            path = out/f'{name}.{extension}'
            fig.savefig(path, bbox_inches='tight')
            files.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        plt.close(fig)

    def curve(ax, family, category, title):
        for offset, arm in enumerate(ARMS):
            subset = sorted((r for r in rows if r['route_family'] == family and r['phase'] == 'mean'
                             and r['arm'] == arm and r['category'] == category),
                            key=lambda r: int(r['retention_tail']))
            assert len(subset) == 3
            x = np.asarray([int(r['retention_tail']) for r in subset], dtype=float) + (offset-1.5)*1.8
            y = np.asarray([float(r['accuracy']) for r in subset])*100
            lo = np.asarray([float(r['lower95']) for r in subset])*100
            hi = np.asarray([float(r['upper95']) for r in subset])*100
            ax.errorbar(x, y, yerr=np.stack([y-lo, hi-y]), marker='o', markersize=4,
                        capsize=2.5, linewidth=1.4, color=COLORS[arm], label=LABELS[arm])
        ax.set(title=title, xlabel='Final distractor tail', ylabel='Accuracy (%)',
               xticks=[0, 64, 128], xlim=(-8, 136), ylim=(-2, 102))
        ax.grid(axis='y', color='#dedede', linewidth=.5)

    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.15), constrained_layout=True)
    for ax, category, title in zip(axes, ('stable', 'revised'), ('Four stable targets', 'Two revised targets')):
        curve(ax, 'balanced', category, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Matched writes: fixed-checkpoint replay timing', fontsize=11)
    save(fig, 'balanced_accuracy')

    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.8), constrained_layout=True)
    for row_index, (family, family_label) in enumerate((('content', 'Content map'), ('mean5random', 'Mean of all five random maps'))):
        for col, category in enumerate(('stable', 'revised')):
            curve(axes[row_index, col], family, category, f'{family_label}: {category}')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    save(fig, 'routing_families')

    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.15), constrained_layout=True)
    for ax, category, title in zip(axes, ('early', 'eligible'), ('Stable early: positions 0/1', 'Stable initially eligible: positions 12/13')):
        curve(ax, 'balanced', category, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    save(fig, 'stable_query_groups')

    primary = sorted((r for r in contrasts if r['role'] == 'co-primary'), key=lambda r: r['category'], reverse=True)
    assert len(primary) == 2
    fig, ax = plt.subplots(figsize=(6.5, 2.65), constrained_layout=True)
    for y, row in enumerate(primary):
        estimate, lower, upper = (100*float(row[key]) for key in ('estimate', 'lower', 'upper'))
        ax.errorbar(estimate, y, xerr=[[estimate-lower], [upper-estimate]], fmt='o', color=COLORS['periodic'], capsize=4)
        ax.annotate(f'{estimate:+.2f} pp [{lower:+.2f}, {upper:+.2f}]', (upper, y), xytext=(7, 0),
                    textcoords='offset points', va='center', fontsize=9)
    ax.axvline(0, color='#777777', linewidth=.8)
    ax.set(yticks=range(2), yticklabels=[r['category'].capitalize() for r in primary],
           xlabel='Periodic minus continuous accuracy (percentage points)',
           title='Co-primary final-tail D128 contrasts', ylim=(-.6, 1.6))
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1]+max(xlim[1]-xlim[0], .2)*1.0)
    ax.text(.02, .02, '97.5% intervals; 128 paired histories, fixed parents/maps', transform=ax.transAxes,
            va='bottom', fontsize=8, color='#555555')
    save(fig, 'co_primary_contrasts')

    def balanced_diagnostic(cell, arm, field):
        selected = [r for r in diagnostics if r['cell'] == cell and r['arm'] == arm and r['category'] == 'revised']
        assert len(selected) == 12
        content = np.mean([float(r[field]) for r in selected if r['route'] == 'content'])
        random = np.mean([float(r[field]) for r in selected if r['route'] != 'content'])
        return .5*content+.5*random

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
    positions, labels = np.arange(4), ['C, phase 0', 'P, phase 0', 'C, phase 2', 'P, phase 2']
    penultimate, earlier, other, stale = [], [], [], []
    for phase in (0, 2):
        cell = f'pressure_k16_p{phase}_d128'
        for arm in ('continuous', 'periodic'):
            pen = balanced_diagnostic(cell, arm, 'penultimate_error')*100
            any_old = balanced_diagnostic(cell, arm, 'obsolete_any_error')*100
            accuracy = next(float(r['accuracy']) for r in rows if r['route_family'] == 'balanced'
                            and r['phase'] == str(phase) and r['retention_tail'] == '128'
                            and r['arm'] == arm and r['category'] == 'revised')*100
            assert pen <= any_old+1e-9 and any_old <= 100-accuracy+1e-9
            penultimate.append(pen)
            earlier.append(max(0, any_old-pen))
            other.append(max(0, 100-accuracy-any_old))
            stale.append(next(int(r['stale_extra_writes']) for r in telemetry
                              if r['cell'] == cell and r['arm'] == arm))
    axes[0].bar(positions, penultimate, label='Penultimate version', color='#b44b4b')
    axes[0].bar(positions, earlier, bottom=penultimate, label='Earlier obsolete version', color='#dda37b')
    axes[0].bar(positions, other, bottom=np.asarray(penultimate)+earlier, label='Other error', color='#b5bec5')
    axes[0].set(title='Revised-target errors at D128', ylabel='Error rate (%)', xticks=positions, xticklabels=labels)
    axes[0].set_ylim(0, 1.5*max(np.asarray(penultimate)+earlier+other))
    axes[0].legend(frameon=False, fontsize=7, loc='best')
    bars = axes[1].bar(positions, stale, color=[COLORS[a] for a in ('continuous', 'periodic')]*2)
    axes[1].bar_label(bars, padding=3)
    axes[1].set(title='Same literal replay, changed application time', ylabel='Stale extra writes / 128 streams',
                xticks=positions, xticklabels=labels, ylim=(0, max(stale)*1.22))
    for ax in axes:
        ax.tick_params(axis='x', labelrotation=20, labelsize=8)
        ax.grid(axis='y', linewidth=.4, alpha=.4)
        ax.set_axisbelow(True)
    save(fig, 'revision_error_and_stale_replay')

    manifest = dict(files=files,
                    inputs={name: hashlib.sha256((analysis/name).read_bytes()).hexdigest()
                            for name in ('aggregate.csv', 'contrasts.csv', 'diagnostics.csv', 'replay_telemetry.csv')},
                    uncertainty='Nominal95% history-cluster bootstrap on curves; Bonferroni97.5% on two co-primary contrasts',
                    weighting='.5 content + .5 mean(all5random), average2fixedparents/2phases unless family panel',
                    visual_inspection='pending')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(figures=len(files)//3, files=len(files), out=str(out))))


if __name__ == '__main__':
    main()
