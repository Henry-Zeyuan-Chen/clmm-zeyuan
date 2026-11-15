"""Deterministic stress curves and exploratory focus from audited tables."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from raw_eval_tools import read_csv
p=argparse.ArgumentParser()
p.add_argument('--analysis',type=Path,required=True)
a=p.parse_args()
rows=read_csv(a.analysis/'aggregate.csv')
contrasts=read_csv(a.analysis/'contrasts.csv')
out=a.analysis/'figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
colors={'ema':'#2866a4','local_delta':'#d46c2c'}
names={'ema':'EMA','local_delta':'Local delta'}
families={'content':'content','mean5random':'mean of 5 random maps'}
fig,axes=plt.subplots(2,2,figsize=(12.3,7))
fig.subplots_adjust(left=.07,right=.76,bottom=.09,top=.88,hspace=.34,wspace=.28)
for row,d in enumerate((0,64)):
    for col,category in enumerate(('revised','stable')):
        ax=axes[row,col]
        for rule in ('ema','local_delta'):
            for family in ('content','mean5random'):
                selected=[next(x for x in rows if x['rule']==rule and x['route_family']==family and
                               int(x['R'])==r and int(x['D'])==d and x['category']==category) for r in (1,4,16)]
                mean=np.asarray([float(x['accuracy']) for x in selected])*100
                lo=np.asarray([float(x['lower95']) for x in selected])*100
                hi=np.asarray([float(x['upper95']) for x in selected])*100
                ax.errorbar(range(3),mean,yerr=np.vstack((mean-lo,hi-mean)),color=colors[rule],
                            linestyle='-' if family=='content' else '--',marker='o',markersize=4,capsize=3,
                            label=f'{names[rule]}, {families[family]}')
        ax.set_xticks(range(3),['1','4','16'])
        ax.set_xlabel('Old observations per revised key (R)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0,105)
        ax.set_title(f'{category.capitalize()} queries; delay {d}')
        ax.grid(alpha=.2)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.77,.88),frameon=False,fontsize=9)
fig.suptitle('Exploratory old-value reinforcement stress; fixed trained checkpoints',y=.97)
for suffix in ('png','pdf','svg'):
    fig.savefig(out/f'reinforcement_curves.{suffix}',dpi=180)
plt.close(fig)

focus=[x for x in contrasts if x['prespecified_exploratory_focus']=='True']
fig,axes=plt.subplots(1,2,figsize=(9.8,3.3),constrained_layout=True)
for ax,category,threshold in zip(axes,('revised','stable'),(0,-3)):
    for i,family in enumerate(('content','mean5random')):
        x=next(x for x in focus if x['route_family']==family and x['category']==category)
        mean,lo,hi=[float(x[k])*100 for k in ('estimate','lower95','upper95')]
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color='#2866a4' if family=='content' else '#d46c2c',capsize=4)
    ax.axvline(threshold,color='#444444',linestyle='--',linewidth=1,label=f'Threshold {threshold:+g} pp')
    ax.set_yticks([0,1],['Content','Mean of 5 random maps'])
    ax.set_ylim(-.6,1.6)
    ax.set_xlabel('Local delta − EMA (percentage points)')
    ax.set_title(category.capitalize()+' query change')
    ax.grid(axis='x',alpha=.2)
    ax.legend(frameon=False,loc='lower left',fontsize=8)
fig.suptitle('Exploratory focus R16 D64; nominal 95% paired-history intervals')
for suffix in ('png','pdf','svg'):
    fig.savefig(out/f'focus_contrasts.{suffix}',dpi=180)
plt.close(fig)
(out/'captions.txt').write_text(
    'Reinforcement curves: All three fixed old-reinforcement counts and both delays, all24 trained checkpoints retained. Means average the two fixed initializers and, for random, all five fixed maps. Error bars are nominal95 history-cluster bootstrap intervals,5000PCG64seed91004 draws, conditional on these checkpoints/codebook/maps. Old values are repeated before the true conflicting observations; observations remain identical across methods.\n'
    'Focus contrasts: R16D64 was declared before stress inference, after seeing the original trained revision ceiling. These nominal95 exploratory intervals do not amend or rescue the original failed joint primary. Revised-query lower>0 and stable lower>-3points are exploratory criteria. D0 and short gates remain visible in accompanying tables.\n')
print(out)
