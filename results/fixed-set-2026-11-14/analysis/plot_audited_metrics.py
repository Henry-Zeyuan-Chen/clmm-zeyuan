"""Produce publication-exportable plots using only verified audit tables."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={32:'#31688e',128:'#35b779'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    verification=json.loads((a.report/'verification.json').read_text());assert verification['status']=='verified'
    assert not a.out.exists(),'Figure directory must be new'
    with (a.report/'metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(11.6,7.3),sharey=True)
    groups=('train','novel_values','heldout_keys','heldout_keys_templates')
    titles=('Exact fixed train bank','New values on fixed ordered key pairs','Heldout key identities','Heldout keys and wording')
    for ax,group,title in zip(axes.flat,groups,titles):
        for parent,style in ((68101,'-'),(68105,'--')):
            for segment in (32,128):
                data=[r for r in rows if r['run']==f'fixed{segment}-{parent}' and r['group']==group and r['condition']=='online'];data.sort(key=lambda r:int(r['step']))
                x=[int(r['step']) for r in data];y=[100*float(r['both_correct']) for r in data]
                ax.plot(x,y,style,color=COLORS[segment],marker='o',markersize=4,label=f'{segment}-byte, parent {parent}')
        if group in ('train','novel_values'):ax.axhline(95 if group=='train' else 80,color='#6b7280',lw=.8,ls=':')
        ax.set_title(title,fontsize=11);ax.set_ylim(-3,103);ax.set_xlabel('Training updates');ax.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('Both questions correct (%)');axes[1,0].set_ylabel('Both questions correct (%)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.005))
    fig.suptitle('Fixed-set fitting and transfer: all declared stage endpoints',fontsize=13)
    fig.tight_layout(rect=(0,.085,1,.955));a.out.mkdir(parents=True)
    fig.savefig(a.out/'fixed_set_trajectories.png',dpi=200,bbox_inches='tight');fig.savefig(a.out/'fixed_set_trajectories.pdf',bbox_inches='tight');plt.close(fig)
    final=[r for r in rows if r['condition']=='online' and int(r['step']) in (1600,2000)]
    run_names=verification['runs'];fig,ax=plt.subplots(figsize=(11.6,5.6))
    labels=[];positions=[]
    groupnames=('train','reference_bank','novel_values','heldout_keys','heldout_keys_templates','train_query_only')
    offsets={'train':-.28,'reference_bank':-.28,'novel_values':-.14,'heldout_keys':0,'heldout_keys_templates':.14,'train_query_only':.28}
    color={'train':'#443983','reference_bank':'#443983','novel_values':'#31688e','heldout_keys':'#21918c','heldout_keys_templates':'#35b779','train_query_only':'#b8de29'}
    for run_index,name in enumerate(run_names):
        data=[r for r in final if r['run']==name]
        for r in data:
            g=r['group'];x=run_index+offsets[g];y=100*float(r['both_correct']);lo=100*float(r['both_lower']);hi=100*float(r['both_upper'])
            label=('exact/reference bank' if g=='train' else g) if run_index==0 else None
            ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt='o',color=color[g],ms=5,capsize=3,label=label)
        positions.append(run_index);labels.append(name.replace('-681','\n681'))
    ax.set_xticks(positions,labels);ax.set_ylim(-3,103);ax.set_ylabel('Both questions correct (%)');ax.grid(axis='y',alpha=.2)
    ax.set_title('Final checkpoints; empirical 95% paired-episode/cluster bootstrap intervals',fontsize=11)
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.16))
    fig.tight_layout();fig.savefig(a.out/'final_checkpoints.png',dpi=200,bbox_inches='tight');fig.savefig(a.out/'final_checkpoints.pdf',bbox_inches='tight');plt.close(fig)
    if (a.report/'disjoint_metrics.csv').is_file():
        assert verification['disjoint_probe'] is not None
        with (a.report/'disjoint_metrics.csv').open(newline='') as f:disjoint=list(csv.DictReader(f))
        names=('fixed32-68101','fixed128-68101','fresh32-68101','fresh128-68101')
        fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),sharey=True)
        for i,name in enumerate(names):
            for cond,offset,color in (('online',-.08,'#31688e'),('reset_memory',.08,'#b8de29')):
                r=next(r for r in disjoint if r['run']==name and r['condition']==cond)
                y=100*float(r['both_correct']);lo=100*float(r['both_lower']);hi=100*float(r['both_upper'])
                axes[0].errorbar(i+offset,y,yerr=[[y-lo],[hi-y]],fmt='o',color=color,ms=6,capsize=3,label=cond if i==0 else None)
            for source,offset,color in (('original',-.08,'#443983'),('disjoint',.08,'#35b779')):
                collection=rows if source=='original' else disjoint
                r=next(r for r in collection if r['run']==name and r['condition']=='online' and (source=='disjoint' or (r['group']=='novel_values' and int(r['step'])==2000)))
                axes[1].plot(i+offset,100*float(r['posthoc_old_assignment_accuracy']),'o',color=color,ms=6,label=source if i==0 else None)
        for ax in axes:
            ax.set_xticks(range(4),[n.replace('-68101','') for n in names],rotation=20);ax.set_ylim(-3,103);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.21),ncol=2)
        axes[0].axhline(80,color='#6b7280',lw=.8,ls=':');axes[0].set_ylabel('Percent')
        axes[0].set_title('Training-disjoint assignments: both questions correct',fontsize=10)
        axes[1].set_title('Posthoc: original-bank target answers per query',fontsize=10)
        fig.suptitle('Primary parent 68101; all four final step-2000 checkpoints',fontsize=12)
        fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(a.out/'disjoint_transfer.png',dpi=200,bbox_inches='tight');fig.savefig(a.out/'disjoint_transfer.pdf',bbox_inches='tight');plt.close(fig)
    (a.out/'provenance.json').write_text(json.dumps(dict(source='Verified metrics.csv; no model inference or invented values',verification=verification['status'],source_commit=verification['source_commit'],bootstrap=verification['novel_value_unit']),indent=2)+'\n')

if __name__=='__main__':main()
