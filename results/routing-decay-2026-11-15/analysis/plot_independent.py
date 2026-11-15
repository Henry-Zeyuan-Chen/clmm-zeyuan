"""Static scientific figures from independently recomputed CSV results."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def read(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    destination = args.analysis/'figures'
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#263442', 'text.color': '#263442',
                         'axes.titleweight': 'bold', 'savefig.dpi': 180, 'figure.facecolor': 'white'})
    aggregate = read(args.analysis/'aggregate_pressure.csv')
    admission = read(args.analysis/'family_admission.csv')
    groups = [('content', 'protected', '#b56515', '-', 'Content / protected'),
              ('content', 'global', '#b56515', '--', 'Content / global decay'),
              ('mean5random', 'protected', '#236a94', '-', 'Mean 5 fixed maps / protected'),
              ('mean5random', 'global', '#236a94', '--', 'Mean 5 fixed maps / global decay')]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.7), sharex=True, sharey=True, constrained_layout=True)
    for row, rule in enumerate(('hebbian', 'gradient')):
        for column, k in enumerate((8, 16)):
            ax = axes[row, column]
            for route, decay, color, line, label in groups:
                values = [next(r for r in aggregate if r['rule'] == rule and r['cell'] == f'pressure_k{k}_d{d}'
                               and r['category'] == 'stable' and r['route'] == route and r['inactive_decay'] == decay)
                          for d in (0, 16, 64)]
                y = np.array([float(r['accuracy'])*100 for r in values])
                low = np.array([float(r['lower95'])*100 for r in values])
                high = np.array([float(r['upper95'])*100 for r in values])
                ax.plot((0, 16, 64), y, marker='o', markersize=4, linestyle=line, color=color, label=label)
                ax.fill_between((0, 16, 64), low, high, color=color, alpha=.07)
            ax.set_title(f'{rule.capitalize()} dense parents · K{k}', loc='left')
            ax.set_ylim(0, 102)
            ax.set_xlim(-2, 67)
            ax.set_xticks((0, 16, 64))
            ax.grid(axis='y', alpha=.2)
            relevant = [r for r in admission if r['rule'] == rule and int(r['K']) == k]
            passed = sum(r['both_parent_own_admission'] == 'True' for r in relevant)
            ax.text(.02, .03, f'Own admission: {passed}/{len(relevant)} table × decay families',
                    transform=ax.transAxes, fontsize=8, va='bottom',
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8, 'pad': 2})
            if column == 0:
                ax.set_ylabel('Stable-query accuracy (%)')
            if row == 1:
                ax.set_xlabel('Distractor observations')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Frozen routing × inactive decay\nFixed two parents and five random maps; 95% history intervals', fontsize=14)
    fig.savefig(destination/'factorial_stable_recall.png')
    fig.savefig(destination/'factorial_stable_recall.pdf')
    plt.close(fig)

    contrasts = read(args.analysis/'contrasts.csv')
    wanted = [('content_protected_minus_mean5random_protected', 'Content − fixed random\n(protected traces)'),
              ('content_protected_minus_content_global', 'Protected − global decay\n(content allocation)')]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, constrained_layout=True)
    for ax, rule in zip(axes, ('hebbian', 'gradient')):
        ylabels = []
        for y, (key, label) in enumerate(wanted):
            r = next(r for r in contrasts if r['rule'] == rule and r['contrast'] == key)
            value, lo, hi = [float(r[k])*100 for k in ('estimate', 'lower', 'upper')]
            primary = r['co_primary'] == 'True'
            ylabels.append(label+('\n[admission failed: diagnostic]' if r['admitted'] != 'True' else ''))
            ax.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt='o', capsize=5,
                        color='#b56515' if primary else '#236a94', markersize=7, linewidth=2)
            ax.text(.98, y-.14, f'{value:+.2f} [{lo:+.2f}, {hi:+.2f}] pp',
                    transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=9)
        ax.axvline(0, color='#9daab5', linewidth=1)
        ax.set_yticks((0, 1), ylabels)
        ax.set_ylim(1.35, -.45)
        ax.set_xlim(-17, 48)
        ax.set_title(f'{rule.capitalize()}: '+('co-primary, 97.5% intervals' if rule == 'hebbian' else 'secondary, 95% intervals'), loc='left')
        ax.grid(axis='x', alpha=.2)
    axes[-1].set_xlabel('Paired accuracy difference (percentage points)')
    fig.suptitle('K8 · D64 · stable queries\nIntervals resample shared histories; parents and maps are fixed', fontsize=14)
    fig.savefig(destination/'paired_mechanism_contrasts.png')
    fig.savefig(destination/'paired_mechanism_contrasts.pdf')
    plt.close(fig)

    permap = read(args.analysis/'k8_d64_stable_per_map_parent.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), sharex=True, sharey=True, constrained_layout=True)
    xlabels = ['Content', '90701', '90702', '90703', '90704', '90705']
    routes = ['content', *(f'random-{s}' for s in range(90701, 90706))]
    for row, rule in enumerate(('hebbian', 'gradient')):
        for column, decay in enumerate(('protected', 'global')):
            ax = axes[row, column]
            for seed, offset, marker, color in ((90511, -.08, 'o', '#236a94'), (90515, .08, 's', '#a85620')):
                selected = [next(r for r in permap if r['rule'] == rule and int(r['parent_seed']) == seed
                                 and r['route'] == route and r['inactive_decay'] == decay) for route in routes]
                y = [float(r['accuracy'])*100 for r in selected]
                ax.scatter(np.arange(6)+offset, y, marker=marker, color=color, s=43, label=f'Parent {seed}', zorder=3)
                # Hollow red rings retain and identify weak admission cases.
                weak = [i for i, r in enumerate(selected) if r['own_admission'] != 'True']
                if weak:
                    ax.scatter(np.asarray(weak)+offset, np.asarray(y)[weak], marker='o', facecolors='none',
                               edgecolors='#bd303c', linewidths=1.5, s=100, zorder=4)
            ax.set_title(f'{rule.capitalize()} · '+('inactive traces protected' if decay == 'protected' else 'global decay'), loc='left')
            ax.set_xticks(range(6), xlabels)
            ax.set_ylim(-2, 102)
            ax.grid(axis='y', alpha=.2)
            ax.axvline(.5, color='#b5bec7', linewidth=1, alpha=.7)
            if column == 0:
                ax.set_ylabel('Stable-query accuracy (%)')
            if row == 1:
                ax.set_xlabel('Fixed assignment table')
    legend = [Line2D([0], [0], marker='o', linestyle='', color='#236a94', label='Parent 90511'),
              Line2D([0], [0], marker='s', linestyle='', color='#a85620', label='Parent 90515'),
              Line2D([0], [0], marker='o', linestyle='', markerfacecolor='none', markeredgecolor='#bd303c', label='Own admission failed')]
    fig.legend(handles=legend, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Every fixed map and parent retained · K8 D64\nNo routing-map or checkpoint selection', fontsize=14)
    fig.savefig(destination/'every_map_parent.png')
    fig.savefig(destination/'every_map_parent.pdf')
    plt.close(fig)

    short = read(args.analysis/'short_gates.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    for column, ax in enumerate(axes):
        parent_values = []
        for seed, offset, marker, color in ((90511, -.045, 'o', '#236a94'), (90515, .045, 's', '#a85620')):
            if column == 0:
                selected = [next(r for r in permap if r['rule'] == 'hebbian' and int(r['parent_seed']) == seed
                                 and r['route'] == 'content' and r['inactive_decay'] == decay) for decay in ('protected', 'global')]
                values = np.array([float(r['accuracy'])*100 for r in selected])
            else:
                selected = [next(r for r in short if r['rule'] == 'hebbian' and int(r['parent_seed']) == seed
                                 and r['route'] == 'content' and r['inactive_decay'] == decay) for decay in ('protected', 'global')]
                values = np.array([float(r['revise2'])*100 for r in selected])
            parent_values.append(values)
            ax.plot(np.arange(2)+offset, values, marker=marker, color=color, markersize=6,
                    alpha=.9, linewidth=1, label=f'Parent {seed}')
        average = np.mean(parent_values, axis=0)
        ax.plot((0, 1), average, marker='D', color='#263442', markersize=6, linewidth=1.5,
                linestyle=':', label='Fixed-parent mean')
        for x, value in enumerate(average):
            ax.annotate(f'{value:.2f}%', (x, value), xytext=(0, 11 if column == 0 else -18),
                        textcoords='offset points', ha='center', fontsize=10, weight='bold')
        ax.set_xlim(-.25, 1.25)
        ax.set_xticks((0, 1), ('Protected', 'Global decay'))
        ax.set_ylabel('Accuracy (%)')
        ax.grid(axis='y', alpha=.2)
        if column == 0:
            ax.set_title('Late stable recall · K8 D64\nOwn admission failed: diagnostic', loc='left')
            ax.set_ylim(0, 48)
        else:
            ax.set_title('Short revision · revise2\nBoth answers correct; zoomed axis', loc='left')
            ax.set_ylim(80, 102)
            ax.axhline(90, color='#bd303c', linewidth=1, linestyle='--')
            ax.text(.95, 90.3, '90% gate per parent', color='#bd303c', fontsize=9, ha='right')
            ax.scatter([.045], [float(next(r['revise2'] for r in short if r['case'] == 'hebbian-90515-content-protected'))*100],
                       marker='o', facecolors='none', edgecolors='#bd303c', s=105, linewidths=1.5, zorder=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Frozen Hebbian content router: stability and revision\nRevision comparison examined after outcomes', fontsize=13)
    fig.savefig(destination/'stability_revision_tradeoff.png')
    fig.savefig(destination/'stability_revision_tradeoff.pdf')
    plt.close(fig)
    print(str(destination.resolve()))


if __name__ == '__main__':
    main()
