"""Deterministic scientific figures from saved independent aggregate tables."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from independent_tools import ARMS, read_csv, sha

ROOT = Path(__file__).resolve().parent
FIG = ROOT/'figures'
LABEL = dict(online='Online', repeat='Repeat current', continuous='Literal continuous', periodic='Literal periodic',
             continuous_bounded='Bounded continuous', periodic_bounded='Bounded periodic')
COLOR = dict(online='#8b8b8b', repeat='#292929', continuous='#ba3c46', periodic='#db8731',
             continuous_bounded='#246ba8', periodic_bounded='#249c89')
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'svg.fonttype':'none', 'pdf.fonttype':42, 'savefig.dpi':180})


def save(fig, name, outputs):
    fig.savefig(FIG/f'{name}.png', bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG/f'{name}.svg', bbox_inches='tight')
    outputs.append(name)
    plt.close(fig)


def curves(rows, categories, name, outputs):
    fig, axes = plt.subplots(1,len(categories),figsize=(10,3.6),sharey=True,layout='constrained')
    axes = np.atleast_1d(axes)
    for ax, category in zip(axes,categories):
        for offset, arm in enumerate(ARMS):
            selected = sorted((r for r in rows if r['route_family']=='balanced' and r['phase']=='mean'
                               and r['category']==category and r['arm']==arm), key=lambda r:int(r['retention_tail']))
            x=np.asarray([int(r['retention_tail']) for r in selected])
            y=np.asarray([float(r['accuracy']) for r in selected])*100
            low=np.asarray([float(r['lower95']) for r in selected])*100
            high=np.asarray([float(r['upper95']) for r in selected])*100
            ax.errorbar(x,y,yerr=[y-low,high-y],color=COLOR[arm],label=LABEL[arm],
                        marker='o',markersize=4,capsize=2,lw=1.5,alpha=.95,
                        linestyle='--' if arm in ('online','repeat') else '-')
        ax.set(title=category.capitalize(),xlabel='Final retention-tail arrivals',xticks=[0,64,128],ylim=(-2,102))
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Accuracy (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Balanced routes; average of two fixed parents and both phases. 95% paired-history CIs.',fontsize=10)
    save(fig,name,outputs)


def main():
    FIG.mkdir(exist_ok=True)
    summary=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
    aggregate, contrasts=read_csv(ROOT/'aggregate.csv'),read_csv(ROOT/'contrasts.csv')
    outputs=[]
    fig,ax=plt.subplots(figsize=(7.5,3.1),layout='constrained')
    primary=sorted(summary['primary'],key=lambda r:r['category'])
    for i,row in enumerate(primary):
        value,low,high=[row[k]*100 for k in ('estimate','lower','upper')]
        ax.errorbar(value,i,xerr=[[value-low],[high-value]],fmt='o',color='#246ba8',capsize=5)
        ax.text(high+.12,i,f'{value:+.2f} [{low:+.2f}, {high:+.2f}]',va='center',fontsize=9)
    ax.axvline(0,color='black',lw=.8,linestyle='--')
    ax.set(yticks=range(2),yticklabels=[r['category'].capitalize() for r in primary],
           xlabel='Bounded minus literal accuracy (percentage points)',ylim=(-.6,1.6))
    ax.margins(x=.3)
    ax.set_title('D128 co-primary effects; average both schedules and phases\n97.5% paired-history CIs; n=128; fixed checkpoints/maps',fontsize=10)
    ax.grid(axis='x',alpha=.2)
    save(fig,'co_primary_freshness_effects',outputs)
    curves(aggregate,('stable','revised'),'balanced_accuracy',outputs)
    curves(aggregate,('early','eligible'),'stable_query_groups',outputs)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained',sharey=True)
    for ax,category in zip(axes,('stable','revised')):
        for schedule,color in (('continuous','#246ba8'),('periodic','#249c89'),('interaction','#7a49a5')):
            rows=sorted((r for r in contrasts if r['route_family']=='balanced' and r['phase']=='mean'
                         and r['category']==category and r['schedule']==schedule
                         and r['contrast_kind']==('timing_x_freshness' if schedule=='interaction' else 'freshness')),
                        key=lambda r:int(r['retention_tail']))
            x=np.asarray([int(r['retention_tail']) for r in rows])
            y=np.asarray([float(r['estimate']) for r in rows])*100
            lo=np.asarray([float(r['lower']) for r in rows])*100
            hi=np.asarray([float(r['upper']) for r in rows])*100
            label='Timing × freshness' if schedule=='interaction' else schedule.capitalize()+' freshness'
            ax.errorbar(x,y,yerr=[y-lo,hi-y],color=color,marker='o',capsize=3,label=label)
        ax.axhline(0,color='black',lw=.8,linestyle='--')
        ax.set(title=category.capitalize(),xlabel='Final retention-tail arrivals',xticks=[0,64,128])
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Accuracy contrast (percentage points)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Freshness effects by schedule and their difference; nominal 95% paired-history CIs',fontsize=10)
    save(fig,'timing_freshness_interaction',outputs)
    telemetry=read_csv(ROOT/'replay_telemetry.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    wanted=ARMS[2:]
    for phase,shift in ((0,-.18),(2,.18)):
        selected=[next(r for r in telemetry if r['arm']==arm and int(r['phase'])==phase and int(r['retention_tail'])==128) for arm in wanted]
        axes[0].bar(np.arange(4)+shift,[float(r['stale_extra_writes']) for r in selected],width=.35,label=f'Phase {phase}',color='#386f9d' if phase==0 else '#91bddb')
    axes[0].set(xticks=np.arange(4),xticklabels=['Literal\ncontinuous','Literal\nperiodic','Bounded\ncontinuous','Bounded\nperiodic'],ylabel='Mean stale extra writes / history',title='Stale at application time (D128)')
    axes[0].legend(frameon=False)
    for metric,shift,color,label in (('refreshed_records',-.22,'#246ba8','Refreshed source'),('changed_values',0,'#249c89','Changed value'),('fallback_writes',.22,'#a2a2a2','Eviction fallback')):
        values=[np.mean([float(r[metric]) for r in telemetry if r['arm']==arm and int(r['retention_tail'])==128]) for arm in ARMS[4:]]
        axes[1].bar(np.arange(2)+shift,values,width=.22,color=color,label=label)
    axes[1].set(xticks=np.arange(2),xticklabels=['Bounded continuous','Bounded periodic'],ylabel='Mean extra writes / history',title='Refresh and fallback (D128; both phases)')
    axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    save(fig,'replay_staleness_refresh_fallback',outputs)
    manifest=dict(source_tables={name:sha(ROOT/name) for name in ('summary.json','aggregate.csv','contrasts.csv','replay_telemetry.csv')},
                  script_sha256=sha(__file__),figures=[dict(name=name,files={ext:sha(FIG/f'{name}.{ext}') for ext in ('png','pdf','svg')}) for name in outputs],
                  uncertainty='Paired history bootstrap; primary97.5% (Bonferroni2), secondary95%; fixed checkpoints/maps',
                  telemetry_unit='One generated history set; not12 independent checkpoint replications')
    (FIG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(figures=len(outputs),formats=['png','pdf','svg'])))


if __name__=='__main__':
    main()
