"""Plot secondary audited initial/final endpoints and paired changes."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from raw_eval_tools import read_csv

p=argparse.ArgumentParser()
p.add_argument('--analysis',type=Path,required=True)
a=p.parse_args()
rows=read_csv(a.analysis/'initial_to_final_contrasts.csv')
rows=[r for r in rows if r['metric']=='K8D64_stable']
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
colors={'ema':'#2866a4','local_delta':'#d46c2c'}
labels={'ema':'EMA','local_delta':'Local delta'}
family={'content':'content','mean5random':'mean of 5 random'}
fig,axes=plt.subplots(1,2,figsize=(10.4,3.8),constrained_layout=True)
for i,row in enumerate(rows):
    label=f"{labels[row['rule']]}, {family[row['route_family']]}"
    color=colors[row['rule']]
    axes[0].plot([0,1],[float(row['initial'])*100,float(row['final'])*100],
                 color=color,marker='o',linestyle='-' if row['route_family']=='content' else '--',label=label)
    point,lo,hi=[float(row[k])*100 for k in ('final_minus_initial','lower95','upper95')]
    axes[1].errorbar(point,i,xerr=[[point-lo],[hi-point]],color=color,fmt='o',capsize=4)
axes[0].set_xticks([0,1],['Initial','After 1200 steps'])
axes[0].set_xlim(-.12,1.12)
axes[0].set_ylim(0,105)
axes[0].set_ylabel('K8 D64 stable accuracy (%)')
axes[0].set_title('Fixed initial and terminal checkpoints')
axes[0].grid(axis='y',alpha=.2)
axes[0].legend(frameon=False,fontsize=8,loc='center right')
axes[1].set_yticks(range(len(rows)),[f"{labels[r['rule']]}, {family[r['route_family']]}" for r in rows])
axes[1].axvline(0,color='#444444',linestyle='--',linewidth=1)
axes[1].set_ylim(-.5,3.5)
axes[1].set_xlabel('Final − initial (percentage points)')
axes[1].set_title('Paired change; nominal 95% intervals')
axes[1].grid(axis='x',alpha=.2)
fig.suptitle('Training effect on late stable recall; fixed parents and maps')
out=a.analysis/'figures'
out.mkdir(exist_ok=True)
for suffix in ('png','pdf','svg'):
    fig.savefig(out/f'initial_to_final.{suffix}',dpi=180)
plt.close(fig)
(out/'initial_to_final_caption.txt').write_text(
    'Secondary initial/final paired analysis on the same128 confirmation histories;5000paired-history bootstrap draws PCG64seed90804, nominal95 intervals. Means condition on the two fixed initializers; random averages all five fixed maps. Connected points are the two measured endpoints, not an interpolated learning curve. Both rules have100% observed short revised-query accuracy initially and finally, so this figure focuses on late stable recall. All primary tests and gates remain unchanged.\n')
print(out)
