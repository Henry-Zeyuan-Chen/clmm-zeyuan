"""Saved-answer joins and conditional fixed-pair LR-recipe contrasts."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=argparse.ArgumentParser();P.add_argument('--packet',type=Path,required=True);P.add_argument('--runs',type=Path,required=True);P.add_argument('--out',type=Path,required=True)
A=P.parse_args();A.out.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def table(p):
    with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def save(name,rows):
    with (A.out/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
initializers=read(A.packet/'initializers.json');cells=[f'F{f}Q{q}' for f in (0,1,3) for q in (0,1,3)];views=['original','order','value','order_value'];counts={'train_keys':256,'dev_keys':256}
cases=[dict(id=f"{i['id']}-{arm}",pair=i['id'],parent=i['parent'],encoding=i['encoding'],arm=arm) for i in initializers for arm in ['initial','constant','warmup']]
vectors={};rows=[];controls=[];qual=[];total=0
for c in cases:
    root=A.runs/'confirmation'/c['id'];meta=read(root/'metadata.json');assert meta['status']=='succeeded' and meta['weights_and_RNG_pure'] and meta['registered_parameters']==839248
    bits={};erased={}
    for r in table(root/'predictions.csv'):
        total+=1;correct=int(r['prediction'])==int(r['target']);assert int(correct)==int(r['correct'])
        if r['condition']=='normal':
            key=(r['group'],r['cell'],int(r['episode']));bit=1<<(2*views.index(r['variant'])+int(r['rank']))
            seen,value=bits.get(key,(0,0));assert not seen&bit;bits[key]=(seen|bit,value|(bit if correct else 0))
        else:
            key=(r['group'],r['condition'],int(r['episode']));bit=1<<int(r['rank']);seen,value=erased.get(key,(0,0));assert not seen&bit;erased[key]=(seen|bit,value|(bit if correct else 0))
    assert all(b[0]==255 for b in bits.values()) and all(b[0]==3 for b in erased.values())
    failures=[]
    for group,n in counts.items():
        threshold=205
        for cell in cells:
            b=np.array([bits[group,cell,j][1] for j in range(n)],dtype=np.uint8);joint=((b&15)==15).astype(float);vectors[c['id'],group,cell]=joint
            perview={v+'_both':int(((b&(3<<(2*j)))==(3<<(2*j))).sum()) for j,v in enumerate(views)}
            metrics=dict(joint4=int(joint.sum()),joint4_swapped=int(((b&240)==240).sum()),joint8=int((b==255).sum()),**perview)
            rows.append(dict(**c,group=group,cell=cell,worlds=n,**metrics))
            for metric in ['joint4',*[v+'_both' for v in views]]:
                if metrics[metric]<threshold:failures.append(dict(group=group,cell=cell,metric=metric,count=metrics[metric],required=threshold))
        for condition in ['no_facts','cue_erased']:
            both=sum(erased[group,condition,j][1]==3 for j in range(n))
            if condition=='cue_erased':assert both==0
            controls.append(dict(**c,group=group,condition=condition,worlds=n,both_correct=both))
    if c['arm']!='initial':
        qual.append(dict(**c,admitted=not failures,failed_endpoints=len(failures),tested_endpoints=90,failures=failures,scope='Visible attention128 acquisition only'))
assert total==466944
assert len(qual)==8 and sum(q['tested_endpoints'] for q in qual)==720
save('case_cells.csv',rows);save('control_cells.csv',controls)
statistics={}
for group,n in counts.items():
    indices=np.random.Generator(np.random.PCG64(92205 if group=='train_keys' else 92204)).integers(0,n,(5000,n))
    for cell in cells:
        effects=np.stack([vectors[i['id']+'-warmup',group,cell]-vectors[i['id']+'-constant',group,cell] for i in initializers]);perworld=effects.mean(0)
        primary=(group,cell) in [('train_keys','F0Q0'),('dev_keys','F0Q0'),('dev_keys','F3Q3')];q=[.05/6,1-.05/6] if primary else [.025,.975]
        ci=np.quantile(perworld[indices].mean(1),q,method='linear')*100
        statistics[group+'/'+cell]=dict(primary=primary,effect_pp=float(perworld.mean()*100),interval_pp=ci.tolist(),interval_percent=100*(1-.05/3) if primary else 95,
            per_world_effect=perworld.tolist(),per_pair_effect_pp={i['id']:float(e.mean()*100) for i,e in zip(initializers,effects)},endpoint_supported=bool(primary and ci[0]>0),conditional_fixed_pairs=True)
summary=[]
for arm in ['initial','constant','warmup']:
    for group,n in counts.items():
        for cell in cells:
            subset=[r for r in rows if r['arm']==arm and r['group']==group and r['cell']==cell]
            summary.append(dict(arm=arm,group=group,cell=cell,worlds_per_case=n,fixed_cases=4,joint4_percent=100*sum(r['joint4'] for r in subset)/(4*n),joint8_percent=100*sum(r['joint8'] for r in subset)/(4*n)))
save('fixed_pair_means.csv',summary)
(A.out/'statistics.json').write_text(json.dumps(statistics,indent=2)+'\n',encoding='utf8');(A.out/'qualification.json').write_text(json.dumps(qual,indent=2)+'\n',encoding='utf8')
plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
def export(fig,name):
    for ext in ['png','pdf','svg']:
        p=A.out/f'{name}.{ext}';fig.savefig(p,dpi=180)
        if ext=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    plt.close(fig)
matrix=np.array([[next(r['joint4_percent'] for r in summary if r['arm']==arm and r['group']=='dev_keys' and r['cell']==cell) for cell in cells] for arm in ['initial','constant','warmup']])
fig,ax=plt.subplots(figsize=(10.8,3.1),layout='constrained');im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto')
ax.set_xticks(range(9),cells);ax.set_yticks(range(3),['Before V training','Constant LR','500-step warm-up']);ax.set_title('Fresh development keys: four-answer joint recall')
for i in range(3):
    for j in range(9):ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',color='black' if matrix[i,j]>60 else 'white',fontsize=8)
fig.colorbar(im,ax=ax,label='Mean of four fixed cases (%)');export(fig,'format_transfer')
selected=[c for c in cases if c['arm']!='initial'];matrix=np.array([[vectors[c['id'],'dev_keys',cell].sum() for cell in cells] for c in selected])
fig,ax=plt.subplots(figsize=(10.8,5),layout='constrained');im=ax.imshow(matrix,vmin=0,vmax=256,cmap='viridis',aspect='auto');ax.set_xticks(range(9),cells);ax.set_yticks(range(8),[c['id'] for c in selected]);ax.set_title('All eight terminal checkpoints: no case selection')
for i in range(8):
    for j in range(9):ax.text(j,i,f'{int(matrix[i,j])}',ha='center',va='center',fontsize=8,color='black' if matrix[i,j]>153 else 'white')
fig.colorbar(im,ax=ax,label='Joint-correct worlds / 256');export(fig,'terminal_cases')
fig,ax=plt.subplots(figsize=(6.8,3.8),layout='constrained')
for j,(group,cell) in enumerate([('train_keys','F0Q0'),('dev_keys','F0Q0'),('dev_keys','F3Q3')]):
    p=statistics[group+'/'+cell];lo,hi=p['interval_pp'];mean=p['effect_pp'];ax.errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='ko',capsize=4,label='Fixed-pair mean, adjusted CI' if j==0 else None)
    ax.scatter(j+np.linspace(-.09,.09,4),list(p['per_pair_effect_pp'].values()),s=25,c='#d56a32',alpha=.75,label='Each initializer / encoding' if j==0 else None)
ax.axhline(0,color='.6',lw=.8);ax.set_xticks(range(3),['Train keys\nF0Q0','Dev keys\nF0Q0','Dev keys, new wording\nF3Q3']);ax.set_ylabel('Warm-up − constant joint recall (pp)');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15);export(fig,'primary_effects')
curves=[];fig,axes=plt.subplots(2,2,figsize=(8,5),layout='constrained')
for init,ax in zip(initializers,axes.flat):
    for arm,color in [('constant','#31688e'),('warmup','#d56a32')]:
        train=table(A.runs/'main'/f"{init['id']}-{arm}"/'training.csv');assert len(train)==6000
        means=[np.mean([float(r['loss']) for r in train[j:j+100]]) for j in range(0,6000,100)]
        ax.plot(range(100,6001,100),means,label=arm,color=color)
        curves.extend(dict(pair=init['id'],arm=arm,block_end=j+100,mean_loss=float(loss)) for j,loss in zip(range(0,6000,100),means))
    ax.set_title(init['id']);ax.set_xlabel('Update');ax.set_ylabel('Training NLL (100-step mean)');ax.grid(alpha=.15)
axes.flat[0].legend(frameon=False);export(fig,'learning_curves');save('learning_curve_table.csv',curves)
lr_rows=[];fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
for arm,color in [('constant','#31688e'),('warmup','#d56a32')]:
    logs=[table(A.runs/'main'/f"{i['id']}-{arm}"/'training.csv') for i in initializers]
    rates=np.array([[float(r['learning_rate']) for r in log] for log in logs]);assert rates.shape==(4,6000)
    assert np.array_equal(rates,np.broadcast_to(rates[0],rates.shape))
    actual=rates[0];expected=np.full(6000,3e-4) if arm=='constant' else 3e-4*np.minimum(np.arange(1,6001)/500,1)
    assert np.allclose(actual,expected,rtol=0,atol=1e-18)
    for ax,limit in zip(axes,[1000,6000]):ax.plot(np.arange(1,limit+1),actual[:limit]*1e4,label=arm,color=color)
    lr_rows.extend(dict(arm=arm,update=j+1,learning_rate=float(rate)) for j,rate in enumerate(actual))
for ax in axes:ax.set_xlabel('Optimizer update');ax.set_ylabel('Logged LR (×10⁻⁴)');ax.grid(alpha=.15)
axes[0].set_title('Warm-up boundary');axes[1].set_title('Same 6k update budget');axes[0].legend(frameon=False)
export(fig,'learning_rates');save('learning_rate_table.csv',lr_rows)
meta=dict(status='passed',vectors=total,cases=12,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),bootstrap_seeds={'train_keys':92205,'dev_keys':92204},resamples=5000,all_three_primary_supported=all(statistics[g+'/'+c]['endpoint_supported'] for g,c in [('train_keys','F0Q0'),('dev_keys','F0Q0'),('dev_keys','F3Q3')]),scope='Independent answer-byte joins; fixed paired checkpoints/streams, not training-procedure variability or retention.')
(A.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8');print(json.dumps({g+'/'+cell:statistics[g+'/'+cell] for g,cell in [('train_keys','F0Q0'),('dev_keys','F0Q0'),('dev_keys','F3Q3')]}))
