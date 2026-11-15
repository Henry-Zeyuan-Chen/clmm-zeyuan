"""Numerical figures from audited source CSVs; all incomplete/failed gates stay labeled."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SYSTEMS=(('gradient','dense'),('gradient','top1'),('hebbian','dense'),('hebbian','top1'),('neural','none'))
LABELS={('gradient','dense'):'Gradient dense',('gradient','top1'):'Gradient top-one',
        ('hebbian','dense'):'Hebbian dense',('hebbian','top1'):'Hebbian top-one',('neural','none'):'Neural descriptive anchor'}
COLORS={('gradient','dense'):'#0072B2',('gradient','top1'):'#56B4E9',('hebbian','dense'):'#D55E00',('hebbian','top1'):'#009E73',('neural','none'):'#555555'}
def rows(path):
    if not path.exists():return []
    with path.open(newline='') as handle:return list(csv.DictReader(handle))
def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight');fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True)
    args=p.parse_args();out=args.analysis/'figures';out.mkdir(exist_ok=True)
    pressure=rows(args.analysis/'pressure_fixed_targets.csv');admission=rows(args.analysis/'admission.csv');primary=rows(args.analysis/'primary.csv')
    if not pressure:
        print('No completed main pressure rows; no efficacy figures generated.')
        return
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.7,4.6),sharey=True)
    for ax,k in zip(axes,(8,16)):
        for system in SYSTEMS:
            family='neural' if system[0]=='neural' else 'trained'
            group=[r for r in pressure if r['family']==family and (r['rule'],r['routing'])==system and r['arm']=='online' and r['category']=='stable' and r['cell'].startswith(f'pressure_k{k}_')]
            if not group:continue
            arrays=[];delays=[]
            for delay in (0,16,64):
                samples=[float(r['accuracy'])*100 for r in group if r['cell']==f'pressure_k{k}_d{delay}']
                if len(samples)!=2:continue
                arrays.append(samples);delays.append(delay)
            if not arrays:continue
            arr=np.asarray(arrays)
            gates=[r for r in admission if r['family']==family and (r['rule'],r['routing'])==system and int(r['facts'])==k]
            admitted=len(gates)==2 and all(r['both_seed_admission']=='True' for r in gates)
            label=LABELS[system]+(' (diagnostic)' if not admitted else '')
            ax.plot(delays,arr.mean(1),color=COLORS[system],marker='o',linestyle='-' if admitted else '--',label=label)
            ax.fill_between(delays,arr.min(1),arr.max(1),color=COLORS[system],alpha=.10)
        ax.set(title=f'K={k}: stable facts, online only',xlabel='Outside-key distractor observations',xticks=[0,16,64],ylim=(-2,103))
        ax.grid(axis='y',alpha=.20)
    axes[0].set_ylabel('Stable-query accuracy (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.98),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Reference adapters on the shared correlated-key stream',y=1.04,fontsize=15)
    fig.text(.5,-.02,'64 paired histories per pressure cell; lines average two fixed initializers, shading is their range. Dashed systems fail own admission.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.87));save(fig,out/'online_retention')
    if primary:
        fig,ax=plt.subplots(figsize=(10,3.7))
        ordered=sorted(primary,key=lambda r:2 if r['fixed_initializers']=='2' else int(r['initializer']))
        for i,row in enumerate(ordered):
            mean=100*float(row['difference']);lo=100*float(row['ci_low']);hi=100*float(row['ci_high'])
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='D' if row['fixed_initializers']=='2' else 'o',color='#009E73',capsize=4)
            ax.text(hi+.25,i,f'{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]',va='center',fontsize=10)
        labels=['Mean of 2 fixed initializers' if r['fixed_initializers']=='2' else f'Initializer {r["initializer"]}' for r in ordered]
        ax.set_yticks(range(len(ordered)),labels);ax.invert_yaxis();ax.axvline(0,color='#444444',linestyle='--')
        ax.set_xlabel('Hebbian top-one minus dense stable accuracy (percentage points)')
        admitted=all(r['both_systems_admitted']=='True' for r in ordered)
        ax.set_title('Prespecified ONLINE K8,D64 development-pilot contrast\n'+('Both systems pass own admission' if admitted else 'Acquisition/interference diagnostic: own admission fails'))
        ax.grid(axis='x',alpha=.15)
        xlo=min(0,min(float(r['ci_low'])*100 for r in ordered));xhi=max(0,max(float(r['ci_high'])*100 for r in ordered))
        ax.set_xlim(xlo-1,xhi+max(6,.25*(xhi-xlo)))
        fig.text(.5,-.02,'64 shared histories; 5,000 paired bootstrap draws, seed 90604. Fixed-checkpoint/codebook uncertainty; intervals unadjusted.',ha='center',fontsize=9)
        fig.tight_layout();save(fig,out/'primary_access_contrast')
    fork=[r for r in pressure if r['family']=='fork' and r['category']=='stable' and r['arm']=='online' and r['cell']=='pressure_k8_d64']
    if fork:
        fig,ax=plt.subplots(figsize=(9.5,4))
        for i,system in enumerate(SYSTEMS[:4]):
            samples=[100*float(r['accuracy']) for r in fork if (r['rule'],r['routing'])==system]
            if len(samples)!=2:continue
            mean=float(np.mean(samples));ax.bar(i,mean,color=COLORS[system],alpha=.85)
            ax.vlines(i,min(samples),max(samples),color='black',lw=2);ax.scatter([i,i],samples,color='black',s=18)
            ax.text(i,mean+2,f'{mean:.1f}%',ha='center')
        ax.set(xticks=range(4),xticklabels=[LABELS[s] for s in SYSTEMS[:4]],ylabel='Stable-query accuracy (%)',ylim=(0,105),
               title='Frozen-transfer diagnostic: identical gradient/dense slow tensors\nK8,D64 online; new zero traces, no adaptation')
        ax.grid(axis='y',alpha=.15)
        fig.text(.5,-.02,'Two fixed parent checkpoints, 64 shared histories; black marks are the two parent-specific results. Own gates are in admission.csv.',ha='center',fontsize=9)
        fig.tight_layout();save(fig,out/'frozen_transfer')
    print(str(out))

if __name__=='__main__':main()
