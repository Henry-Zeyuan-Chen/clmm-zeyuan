"""Make numerical figures only from independently audited saved results."""
import argparse,csv,hashlib,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'digits':'#C76B27','atomic':'#236A9F'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
def csvwrite(path,rows):
 with path.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,out,name):
 for suffix in ('png','pdf','svg'):fig.savefig(out/(name+'.'+suffix),dpi=200,bbox_inches='tight')
 plt.close(fig)
def rolling(x,n=50):
 s=np.cumsum(np.r_[0,np.asarray(x,dtype=float)]);i=np.arange(1,len(x)+1);return (s[i]-s[np.maximum(0,i-n)])/np.minimum(i,n)
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
 data=json.loads(a.audit.read_text());assert data['status']=='passed' and data['vectors']==81920
 a.out.mkdir(parents=True,exist_ok=True);table=[]
 finals=sorted((c for c in data['cases'] if c['mode']=='train'),key=lambda c:(c['parent'],c['encoding']=='atomic'))
 assert len(finals)==4
 for c in data['cases']:
  for m in c['metrics']:table.append(dict(mode=c['mode'],parent=c['parent'],encoding=c['encoding'],**m))
 csvwrite(a.out/'metric_source.csv',table);csvwrite(a.out/'co_primary_source.csv',[{**r,'ci97_5_lower':r['ci97_5'][0],'ci97_5_upper':r['ci97_5'][1]} for r in data['co_primary']])
 metrics={(c['parent'],c['encoding'],m['group'],m['condition']):m for c in finals for m in c['metrics']}
 labels=[f"{c['parent']}\n{c['encoding']}" for c in finals]
 fig=plt.figure(figsize=(11.6,7.2));gs=fig.add_gridspec(2,3,height_ratios=(1.2,1),hspace=.55)
 for column,(title,original,swapped) in enumerate((('Training keys; seen wording','acquire_train_seen','acquire_train_swapped'),('Dev keys; seen wording','acquire_dev_seen','acquire_dev_swapped'),('Dev keys; heldout wording','acquire_dev_heldout','acquire_dev_heldout_swapped'))):
  ax=fig.add_subplot(gs[0,column]);ax.axhline(80,color='#8F9295',lw=1,ls='--',label='80% admission threshold')
  for i,c in enumerate(finals):
   for offset,group,marker in ((-.09,original,'o'),(.09,swapped,'^')):
    value=metrics[c['parent'],c['encoding'],group,'normal']['both_correct']*100
    ax.scatter(i+offset,value,color=COLORS[c['encoding']],marker=marker,s=38,zorder=3)
  max_value=max(metrics[c['parent'],c['encoding'],g,'normal']['both_correct']*100 for c in finals for g in (original,swapped))
  upper=1.05 if max_value<=1 else 103
  ax.set(xticks=range(4),xticklabels=labels,ylim=(-.04,upper),title=title,ylabel='Both answers correct (%)' if column==0 else '')
  if upper<80:ax.text(.02,.94,'80% admission threshold is outside this zoom',transform=ax.transAxes,fontsize=7.4,va='top',color='#5D6063')
  ax.grid(axis='y',alpha=.2)
 ax=fig.add_subplot(gs[1,:]);ax.axvline(0,color='#666',lw=1)
 for i,effect in enumerate(data['co_primary']):
  point=100*effect['atomic_minus_digits'];lo,hi=100*np.array(effect['ci97_5'])
  ax.errorbar(point,i,xerr=[[max(0,point-lo)],[max(0,hi-point)]],fmt='o',color='#173D60',capsize=4,ms=7,zorder=3)
  ax.scatter(np.array(effect['parent_effects'])*100,[i-.075,i+.075],color='#8A8D91',marker='x',s=35,zorder=2)
 ax.set(yticks=[0,1],yticklabels=['Training keys; seen wording','Dev keys; heldout wording'],xlabel='Atomic minus digits joint-four correctness (percentage points)',title='Co-primary effects: 97.5% paired-world intervals; two fixed parents')
 ax.grid(axis='x',alpha=.2);ax.set_ylim(-.45,1.45)
 handles=[plt.Line2D([],[],color='black',marker='o',ls='',label='Original world'),plt.Line2D([],[],color='black',marker='^',ls='',label='Value-swapped world'),plt.Line2D([],[],color='#8A8D91',marker='x',ls='',label='Individual fixed-parent contrast')]
 fig.legend(handles=handles,loc='upper center',ncol=3,bbox_to_anchor=(.5,.98),frameon=False)
 fig.suptitle('Fixed-size key/value binding: terminal checkpoints',y=1.02,fontsize=13)
 fig.text(.02,-.01,'Each point uses 256 paired worlds. Joint-four requires both requested-key answers in both original and swapped worlds.\nIntervals resample whole world IDs jointly across formats and fixed parents. All-zero joint-four outcomes yield degenerate intervals; these are not equivalence evidence.',fontsize=8)
 save(fig,a.out,'binding_acquisition_and_primary')
 fig,axes=plt.subplots(2,2,figsize=(10.5,7),sharex=True,sharey=True)
 condition_style=(('normal','#236A9F','o','Full context'),('retrieval_zero','#C76B27','^','Retrieved output zeroed'),('query_only','#73787D','s','Query only'))
 for row,split in enumerate(('train','dev')):
  for col,measure in enumerate(('accuracy','prediction_changes')):
   ax=axes[row,col]
   for condition,color,marker,label in condition_style:
    y=[]
    for c in finals:
     group=f'query_only_{split}_seen' if condition=='query_only' else f'acquire_{split}_seen';cond='normal' if condition=='query_only' else condition
     y.append(metrics[c['parent'],c['encoding'],group,cond][measure]*100)
    ax.plot(range(4),y,color=color,marker=marker,lw=1,label=label,ms=5)
   ax.set(xticks=range(4),xticklabels=labels,ylim=(-3,103),title=f"{'Training' if split=='train' else 'Dev'} keys: {'answer accuracy' if col==0 else 'requested-key prediction changes'}",ylabel='Percent')
   ax.grid(axis='y',alpha=.2)
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.01),frameon=False)
 fig.suptitle('Input and retrieved-output controls: terminal checkpoints',y=1.055,fontsize=13)
 fig.tight_layout();fig.text(.02,-.04,'Seen wording, 256 worlds per case. A prediction change alone does not establish correct key/value selection.\nAll normal/reset score vectors agree exactly by construction (95 < 128); this does not establish memory independence. Zeroing retrieval still permits ordinary within-prefix updates.',fontsize=8)
 save(fig,a.out,'binding_controls')
 fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True);curves=[]
 for c in finals:
  with (pathlib.Path(c['run'])/'training.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
  assert len(rows)==2000
  style='-' if c['parent']==68101 else '--';label=f"{c['encoding']} / {c['parent']}";steps=np.arange(1,2001)
  for ax,key,title in zip(axes.flat,('loss','first_accuracy','second_accuracy','prediction_changes'),('Answer cross-entropy','First query accuracy','Second query accuracy','Requested-key prediction changes')):
   values=rolling([float(r[key]) for r in rows]);scale=1 if key=='loss' else 100
   ax.plot(steps,values*scale,color=COLORS[c['encoding']],ls=style,lw=1.4,label=label);ax.axvline(256,color='#8A8D91',ls=':',lw=1);ax.set_title(title);ax.grid(alpha=.2)
   ax.set_ylabel('Nats' if key=='loss' else 'Percent')
   if key!='loss':ax.set_ylim(-3,103)
  for i,r in enumerate(rows):curves.append(dict(parent=c['parent'],encoding=c['encoding'],**r))
 for ax in axes[1,:]:ax.set_xlabel('Optimizer step')
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,bbox_to_anchor=(.5,1.01),frameon=False);fig.suptitle('Fixed 2,000-step acquisition training',y=1.055,fontsize=13);fig.tight_layout()
 fig.text(.02,-.035,'Curves show trailing 50-update means. Dotted line: 256 K1 updates followed by 1,744 K2 updates.\nK1 duplicates its one query; first and second query correspond to fact order in K2. Training traces are descriptive and did not select checkpoints.',fontsize=8)
 csvwrite(a.out/'training_curve_source.csv',curves);save(fig,a.out,'binding_training_curves')
 manifest={'status':'generated_needs_visual_QA','source_audit':str(a.audit),'source_audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'figures':['binding_acquisition_and_primary','binding_controls','binding_training_curves'],'formats':['png','pdf','svg'],'tables':['metric_source.csv','co_primary_source.csv','training_curve_source.csv'],'aggregation':'Raw saved-vector independent audit; training curves trailing50update means; acquisition 256worlds; fixedparent paired bootstrap97.5%'}
 (a.out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
