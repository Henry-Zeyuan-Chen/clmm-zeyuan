"""Plots audited exported tables; no manually supplied performance values."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from raw_eval_tools import read_csv

p=argparse.ArgumentParser()
p.add_argument('--analysis',type=Path,required=True)
p.add_argument('--title',required=True)
a=p.parse_args()
out=a.analysis/'figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
rows=read_csv(a.analysis/'final_aggregate.csv')
present={r['rule'] for r in rows}
rules=[r for r in ('ema','additive','erase_ema','local_delta') if r in present]
labels={'ema':'EMA','additive':'Additive','erase_ema':'Erasure + EMA','local_delta':'Local delta'}
colors={'content':'#2866a4','mean5random':'#d46c2c'}
family_labels={'content':'Content','mean5random':'Mean of 5 random maps'}
fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),constrained_layout=True)
for ax,cell,category,title in zip(axes,('revise2','pressure_k8_d64'),('revised','stable'),
                                 ('Short revised-query accuracy','K8 D64 stable-query accuracy')):
    for family,offset in [('content',-.18),('mean5random',.18)]:
        selected=[next(r for r in rows if r['rule']==rule and r['route_family']==family and r['cell']==cell and r['category']==category) for rule in rules]
        mean=np.asarray([float(r['accuracy']) for r in selected])*100
        lo=np.asarray([float(r['lower95']) for r in selected])*100
        hi=np.asarray([float(r['upper95']) for r in selected])*100
        ax.bar(np.arange(len(rules))+offset,mean,width=.34,color=colors[family],label=family_labels[family],
               yerr=np.vstack((mean-lo,hi-mean)),error_kw={'capsize':3,'linewidth':1})
    ax.set_xticks(np.arange(len(rules)),[labels[r] for r in rules],rotation=15,ha='right')
    ax.set_ylim(0,105)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False,loc='lower left')
fig.suptitle(a.title+'; fixed parents and maps')
for suffix in ('png','pdf','svg'):
    fig.savefig(out/f'rule_accuracy.{suffix}',dpi=180)
plt.close(fig)

contrasts=[r for r in read_csv(a.analysis/'final_contrasts.csv') if r['contrast'] in ('revision_superiority','stable_noninferiority')]
fig,axes=plt.subplots(1,2,figsize=(9.2,3.1),constrained_layout=True)
for ax,label,title,threshold in zip(axes,('revision_superiority','stable_noninferiority'),
                                    ('Revised-query change','Stable-query change'),(0,-3)):
    endpoints=[]
    for i,family in enumerate(('content','mean5random')):
        r=next(r for r in contrasts if r['route_family']==family and r['contrast']==label)
        mean,lo,hi=[float(r[k])*100 for k in ('estimate','lower','upper')]
        endpoints.extend((mean,lo,hi))
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[family],capsize=4,markersize=6)
    ax.axvline(threshold,color='#333333',linestyle='--',linewidth=1,label=f'Threshold {threshold:+g} pp')
    ax.set_yticks([0,1],['Content','Mean of 5 random maps'])
    ax.set_ylim(-.6,1.6)
    if all(value==0 for value in endpoints):
        ax.set_xlim(-1,1)
        ax.set_xticks([-1,0,1])
    ax.set_xlabel('Local delta − EMA (percentage points)')
    ax.set_title(title)
    ax.grid(axis='x',alpha=.2)
    ax.legend(frameon=False,loc='lower left',fontsize=9)
fig.suptitle(a.title+'; adjusted paired-history intervals')
for suffix in ('png','pdf','svg'):
    fig.savefig(out/f'primary_contrasts.{suffix}',dpi=180)
plt.close(fig)
(out/'captions.txt').write_text(
    'rule_accuracy: Accuracy means over the two fixed parents; random additionally averages all five fixed maps. Error bars are nominal95% paired-history bootstrap intervals (5000draws); conditional on checkpoints/codebook/maps.\n'
    'primary_contrasts: Candidate local_delta minus EMA. Error bars are97.5% two-sided paired-history percentile intervals, with lower.0125 for four Bonferroni one-sided inequalities. Superiority requires revision lower>0; noninferiority requires stable lower>-3percentagepoints. Gate admission is reported in tables and must additionally pass. No population uncertainty across training seeds/maps is implied.\n')
print(out)
