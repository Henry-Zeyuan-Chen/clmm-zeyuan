"""Coordinator recomputation/figures from compact saved answer bytes, no models."""
import argparse,csv,gzip,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=argparse.ArgumentParser();P.add_argument('--input',required=True,type=Path);P.add_argument('--out',required=True,type=Path);A=P.parse_args()
A.out.mkdir(parents=True,exist_ok=False)
variants=['original','order','value','order_value'];worlds={};identities={};rowsread=0
with gzip.open(A.input,'rt',encoding='utf8',newline='') as f:
    for r in csv.DictReader(f):
        rowsread+=1;assert int(r['correct'])==int(int(r['target'])==int(r['prediction']))
        if r['condition']!='normal':continue
        identity=tuple(r[k] for k in ['evaluation','dataset','phase','training_task','step','task','encoding','parent','compose'])
        ident=identities.setdefault(identity,len(identities));key=(ident,r['group'],int(r['episode']))
        bit=1<<(2*variants.index(r['variant'])+int(r['rank']))
        seen,correct=worlds.get(key,(0,0));assert seen&bit==0
        worlds[key]=(seen|bit,correct|(bit if int(r['correct']) else 0))
inverse={i:k for k,i in identities.items()};cells=defaultdict(list)
for (ident,group,episode),(seen,correct) in worlds.items():
    assert seen==255,(ident,group,episode,seen)
    cells[ident,group].append(dict(episode=episode,joint4=int(correct&15==15),joint8=int(correct==255),
        **{v:int(correct&(3<<(i*2))==(3<<(i*2))) for i,v in enumerate(variants)}))
metrics=[]
for (ident,group),values in cells.items():
    record=dict(zip(['evaluation','dataset','phase','training_task','step','task','encoding','parent','compose'],inverse[ident]))
    n=len(values);metrics.append(dict(**record,group=group,worlds=n,joint4=sum(x['joint4'] for x in values),
        joint8=sum(x['joint8'] for x in values),**{v+'_both':sum(x[v] for x in values) for v in variants}))
def savecsv(name,rows):
    with (A.out/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
savecsv('normal_metrics.csv',metrics)
main=[r for r in metrics if r['dataset']=='final' and r['phase']=='main' and r['step']=='20000' and r['training_task']==r['task'] and r['compose']=='0']
assert len(main)==48
assert all(r['worlds']==256 for r in main)
admission=[]
for evaluation in sorted({r['evaluation'] for r in main}):
    m=[r for r in main if r['evaluation']==evaluation];assert len(m)==3
    passed=all(r['joint4']>=205 and all(r[v+'_both']>=205 for v in variants) for r in m)
    admission.append(dict(evaluation=evaluation,admitted=passed,minimum_joint4=min(r['joint4'] for r in m),
        minimum_both=min(r[v+'_both'] for r in m for v in variants),worlds=256))
savecsv('admission.csv',admission)
primary=[]
for pos,neg in [('select_aligned','select'),('decode','value')]:
    bycase=[]
    for parent in ['68101','68105']:
        for enc in ['digits','atomic']:
            vectors=[]
            for task in [pos,neg]:
                ids=[i for k,i in identities.items() if k[1]=='final' and k[2]=='main' and k[3]==task and k[4]=='20000' and k[5]==task and k[6]==enc and k[7]==parent]
                assert len(ids)==1
                ordered=sorted(cells[ids[0],'dev_seen'],key=lambda x:x['episode'])
                assert [r['episode'] for r in ordered]==list(range(256))
                vectors.append(np.array([r['joint4'] for r in ordered]))
            bycase.append(vectors[0]-vectors[1])
    paired=np.asarray(bycase).mean(0);rng=np.random.Generator(np.random.PCG64(91604))
    draws=paired[rng.integers(0,256,(5000,256))].mean(1)
    primary.append(dict(positive=pos,negative=neg,point=float(paired.mean()),interval_97_5=np.quantile(draws,[.0125,.9875]).tolist(),
        per_world_effect=paired.tolist(),conditional_fixed_models=True,empirical_degeneracy_not_equivalence=True))
(A.out/'coordinator_primaries.json').write_text(json.dumps(primary,indent=2)+'\n')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
tasks=['select','select_aligned','decode','value'];labels=['Select','Aligned\nselect','Given-index\ndecode','Value\nrecall']
colors={'digits':'#176b9a','atomic':'#cb6431'}
fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),sharey=True,layout='constrained')
for ax,group,title in zip(axes,['dev_seen','dev_heldout'],['Familiar formats','Held-out fact + query formats']):
    for parent,marker in [('68101','o'),('68105','^')]:
        for enc,dx in [('digits',-.13),('atomic',.13)]:
            vals=[next(r for r in main if r['group']==group and r['task']==task and r['parent']==parent and r['encoding']==enc) for task in tasks]
            ax.scatter(np.arange(4)+dx+(int(parent)-68103)*.015,[100*r['joint4']/r['worlds'] for r in vals],
                color=colors[enc],marker=marker,s=36,label=f'{enc}, parent {parent}')
    ax.axhline(80,color='.65',ls='--',lw=.8);ax.set_xticks(range(4),labels);ax.set_ylim(-3,103);ax.set_title(title)
    ax.set_ylabel('Joint original + order correct (%)');ax.grid(axis='y',alpha=.15)
handles,leglabels=axes[0].get_legend_handles_labels();fig.legend(handles,leglabels,loc='outside lower center',ncol=4,frameon=False)
for ext in ['png','pdf','svg']:fig.savefig(A.out/f'final_capabilities.{ext}',dpi=180)
plt.close(fig)
gate=[r for r in metrics if r['dataset']=='gate' and r['phase']=='main' and r['training_task']==r['task']]
fig,axes=plt.subplots(2,4,figsize=(10.5,5.5),sharex=True,sharey=True,layout='constrained')
for row,group in enumerate(['dev_seen','dev_heldout']):
    for col,task in enumerate(tasks):
        ax=axes[row,col]
        for parent,ls in [('68101','-'),('68105','--')]:
            for enc in ['digits','atomic']:
                points=sorted([r for r in gate if r['group']==group and r['task']==task and r['parent']==parent and r['encoding']==enc],key=lambda x:int(x['step']))
                ax.plot([int(r['step'])/1000 for r in points],[100*r['joint4']/r['worlds'] for r in points],color=colors[enc],ls=ls,marker='.',lw=1,label=f'{enc}, parent {parent}')
        ax.set_ylim(-3,103);ax.set_xticks([2,10,20]);ax.grid(alpha=.15)
        if row==0:ax.set_title(labels[col].replace('\n',' '))
        if row==1:ax.set_xlabel('Updates (thousands)')
        if col==0:ax.set_ylabel(('Familiar' if row==0 else 'Held-out')+'\njoint correct (%)')
handles,leglabels=axes[0,0].get_legend_handles_labels();fig.legend(handles,leglabels,loc='outside lower center',ncol=4,frameon=False)
for ext in ['png','pdf','svg']:fig.savefig(A.out/f'development_trajectories.{ext}',dpi=180)
plt.close(fig)
bridge=[r for r in metrics if r['dataset']=='final' and r['phase']=='bridge']
if bridge:
    points=[]
    for r in bridge:
        if r['compose']=='1' and r['training_task']=='sd':points.append(dict(**r,method='SD: two forwards'))
        if r['compose']=='0' and r['training_task']=='value' and r['task']=='value':points.append(dict(**r,method='V: one forward'))
    savecsv('bridge_comparison.csv',points)
    fig,axes=plt.subplots(1,2,figsize=(8.8,4.3),sharey=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.29,top=.88,wspace=.22)
    for ax,group,title in zip(axes,['dev_seen','dev_heldout'],['Familiar formats','Held-out fact + query formats']):
        for parent,marker in [('68101','o'),('68105','^')]:
            for enc,dx in [('digits',-.12),('atomic',.12)]:
                p=[r for r in points if r['group']==group and r['parent']==parent and r['encoding']==enc]
                if not p:continue
                p=sorted(p,key=lambda r:r['method']);ax.plot([i+dx for i in range(len(p))],[100*r['joint4']/r['worlds'] for r in p],color=colors[enc],alpha=.35,lw=.8)
                ax.scatter([0+dx,1+dx],[100*r['joint4']/r['worlds'] for r in p],color=colors[enc],marker=marker,s=35,label=f'{enc}, parent {parent}')
        ax.set_xticks([0,1],['SD curriculum\n2 forwards','Value continuation\n1 forward']);ax.set_ylim(-3,103);ax.set_title(title);ax.grid(axis='y',alpha=.15)
        ax.set_ylabel('Joint original + order correct (%)')
    handles,leglabels=axes[0].get_legend_handles_labels();fig.legend(handles,leglabels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=2,frameon=False)
    for ext in ['png','pdf','svg']:fig.savefig(A.out/f'bridge_comparison.{ext}',dpi=180)
    plt.close(fig)
report=dict(status='passed',prediction_rows=rowsread,normal_worlds=len(worlds),main_cases=16,admitted=sum(r['admitted'] for r in admission),
    source_prediction_sha256=hashlib.sha256(A.input.read_bytes()).hexdigest(),analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(A.out/'coordinator_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
