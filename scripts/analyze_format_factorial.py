"""Coordinator byte-level joins and fixed-checkpoint paired format contrasts."""
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=argparse.ArgumentParser();P.add_argument('--packet',required=True,type=Path);P.add_argument('--runs',required=True,type=Path);P.add_argument('--out',required=True,type=Path)
A=P.parse_args();A.out.mkdir(parents=True,exist_ok=False)
cases=json.loads((A.packet/'cases.json').read_text());variants=['original','order','value','order_value'];cells=['F0Q0','F0Q1','F1Q0','F1Q1']
normal={};controls={};total=0
for c in cases:
    meta=json.loads((A.runs/c['id']/'metadata.json').read_text());assert meta['status']=='succeeded' and meta['weights_and_RNG_pure']
    with (A.runs/c['id']/'predictions.csv').open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            total+=1;correct=int(r['prediction'])==int(r['target']);assert int(correct)==int(r['correct'])
            if r['condition']=='normal':
                key=(c['id'],r['group'],r['cell'],int(r['episode']))
                bit=1<<(2*variants.index(r['variant'])+int(r['rank']));seen,ok=normal.get(key,(0,0));assert not seen&bit
                normal[key]=(seen|bit,ok|(bit if correct else 0))
            else:
                key=(c['id'],r['group'],r['condition'],int(r['episode']))
                bit=1<<int(r['rank']);seen,ok=controls.get(key,(0,0));assert not seen&bit
                controls[key]=(seen|bit,ok|(bit if correct else 0))
assert total==138240 and all(v[0]==255 for v in normal.values()) and all(v[0]==3 for v in controls.values())
world_counts={'train_keys':32,'dev_keys':128};rows=[];vectors={}
for c in cases:
    for group,n in world_counts.items():
        for cell in cells:
            b=np.array([normal[c['id'],group,cell,i][1] for i in range(n)],dtype=np.uint8)
            joint=((b&15)==15).astype(float);vectors[c['id'],group,cell]=joint
            rows.append(dict(case=c['id'],stage=c['stage'],parent=c['parent'],task=c['task'],encoding=c['encoding'],group=group,cell=cell,worlds=n,
                joint4=int(joint.sum()),joint4_swapped=int(((b&240)==240).sum()),joint8=int((b==255).sum()),
                **{v+'_both':int(((b&(3<<(2*j)))==(3<<(2*j))).sum()) for j,v in enumerate(variants)}))
def csvsave(name,records):
    with (A.out/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
csvsave('case_cells.csv',rows)
control_rows=[]
for c in cases:
    for group,n in world_counts.items():
        for condition in ['no_facts','cue_erased']:
            b=[controls[c['id'],group,condition,i][1] for i in range(n)]
            both=sum(v==3 for v in b)
            if condition=='cue_erased':assert both==0
            control_rows.append(dict(case=c['id'],group=group,condition=condition,worlds=n,both_correct=both))
csvsave('control_cells.csv',control_rows)
statistics={};draws={g:np.random.Generator(np.random.PCG64(91804)).integers(0,n,(5000,n)) for g,n in world_counts.items()}
for stage in ['trained','initial']:
    for task in ['select','decode','value']:
        selected=[c for c in cases if c['stage']==stage and c['task']==task];assert len(selected)==4
        for group,n in world_counts.items():
            values={cell:np.stack([vectors[c['id'],group,cell] for c in selected]) for cell in cells}
            effects={'record_loss':values['F0Q0']-values['F1Q0'],
                'query_loss':values['F0Q0']-values['F0Q1'],
                'interaction':values['F1Q1']-values['F1Q0']-values['F0Q1']+values['F0Q0']}
            records={}
            for name,effect in effects.items():
                per_world=effect.mean(axis=0);bootstrap=per_world[draws[group]].mean(axis=1)
                primary=stage=='trained' and task=='value' and group=='dev_keys' and name!='interaction'
                ci=np.quantile(bootstrap,[.0125,.9875] if primary else [.025,.975])*100
                records[name]=dict(effect_pp=float(per_world.mean()*100),interval_pp=ci.tolist(),interval_percent=97.5 if primary else 95,
                    per_world_effect=per_world.tolist(),per_case_effect_pp={c['id']:float(v.mean()*100) for c,v in zip(selected,effect)},
                    conditional_fixed_checkpoints=True,primary=primary,degenerate_empirical_interval=bool(ci[0]==ci[1]))
            statistics[f'{stage}/{task}/{group}']=records
(A.out/'coordinator_statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
means=[]
for task in ['select','decode','value']:
    for group in world_counts:
        for cell in cells:
            r=[r for r in rows if r['stage']=='trained' and r['task']==task and r['group']==group and r['cell']==cell]
            means.append(dict(task=task,group=group,cell=cell,worlds_per_checkpoint=r[0]['worlds'],fixed_checkpoints=4,
                joint4_percent=100*sum(v['joint4'] for v in r)/(4*r[0]['worlds'])))
csvsave('fixed_model_means.csv',means)
plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
order=['F0Q0','F1Q0','F0Q1','F1Q1'];labels=['Original\nF0Q0','Fact shift\nF1Q0','Query shift\nF0Q1','Both\nF1Q1']
matrix=np.array([[next(r['joint4_percent'] for r in means if r['task']==t and r['group']=='dev_keys' and r['cell']==cell) for cell in order]
    for t in ['select','decode','value']])
fig,ax=plt.subplots(figsize=(6.8,3.4),layout='constrained');im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto')
ax.set_xticks(range(4),labels);ax.set_yticks(range(3),['Record selection','Given-index decoding','Value recall']);ax.set_title('Fresh development keys: four-answer joint success')
for i in range(3):
    for j in range(4):ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',color='black' if matrix[i,j]>60 else 'white')
fig.colorbar(im,ax=ax,label='Mean of four fixed checkpoints (%)')
for ext in ['png','pdf','svg']:fig.savefig(A.out/f'format_components.{ext}',dpi=180)
plt.close(fig)
selected=[c for c in cases if c['stage']=='trained' and c['task']=='value'];matrix=np.array([[vectors[c['id'],'dev_keys',cell].sum() for cell in order] for c in selected])
fig,ax=plt.subplots(figsize=(6.8,3.5),layout='constrained');im=ax.imshow(matrix,vmin=0,vmax=128,cmap='viridis',aspect='auto')
ax.set_xticks(range(4),labels);ax.set_yticks(range(4),[f"{c['parent']} / {c['encoding']}" for c in selected]);ax.set_title('Value checkpoints: keep all four cases')
for i in range(4):
    for j in range(4):ax.text(j,i,f'{int(matrix[i,j])}/128',ha='center',va='center',color='black' if matrix[i,j]>77 else 'white')
fig.colorbar(im,ax=ax,label='Worlds with all four answers correct')
for ext in ['png','pdf','svg']:fig.savefig(A.out/f'value_case_matrix.{ext}',dpi=180)
plt.close(fig)
primary=statistics['trained/value/dev_keys'];fig,ax=plt.subplots(figsize=(5.8,3.4),layout='constrained')
for i,name in enumerate(['record_loss','query_loss']):
    p=primary[name];lo,hi=p['interval_pp'];ax.errorbar(i,p['effect_pp'],yerr=[[p['effect_pp']-lo],[hi-p['effect_pp']]],fmt='ko',capsize=5,label='Fixed-model mean,97.5%CI' if i==0 else None)
    ax.scatter(np.full(4,i)+np.linspace(-.08,.08,4),list(p['per_case_effect_pp'].values()),s=20,color='#d56a32',alpha=.75,label='Each checkpoint' if i==0 else None)
ax.axhline(0,color='.6',lw=.8);ax.set_xticks([0,1],['Fact-format loss\nF0Q0−F1Q0','Query-format loss\nF0Q0−F0Q1']);ax.set_ylabel('Loss in joint success (percentage points)');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
for ext in ['png','pdf','svg']:fig.savefig(A.out/f'primary_losses.{ext}',dpi=180)
plt.close(fig)
(A.out/'analysis_metadata.json').write_text(json.dumps(dict(status='passed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),vectors=total,
    cases=len(cases),normal_cells=len(rows),bootstrap_seed=91804,resamples=5000,scope='Saved answer-byte joins; no neural inference. Conditional fixed-checkpoint effects; model/training variability not estimated.'),indent=2)+'\n')
print(json.dumps(primary))
