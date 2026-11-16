"""Create reproducible numerical figures only from independently audited tables."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_results import load_csv
from independent_helpers import paired_bootstrap


def export(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(out/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def run(audited,out):
    out.mkdir(parents=True,exist_ok=True)
    summary=load_csv(audited/'case_summary.csv');counter=load_csv(audited/'counterfactual_metrics.csv')
    summary=[r for r in summary if r['mode']=='train' and r['condition']=='normal' and r['arm']=='online']
    cases=sorted({(int(r['segment']),int(r['parent']),r['training_arm'],r['case']) for r in summary})
    if not cases:raise AssertionError('No final audited cases')
    columns=['acquire_train_seen','acquire_train_swapped','acquire_dev_seen','acquire_dev_heldout','joint_four_answer_correct']
    labels=['Fresh train keys','Value-swapped','Unseen dev keys','Dev + heldout wording','Joint 4 answers']
    values=np.zeros((len(cases),len(columns)));counts=[];rowlabels=[]
    for i,(segment,parent,training_arm,case) in enumerate(cases):
        rowlabels.append(f'{parent} / segment {segment} / '+('bounded' if training_arm.endswith('_bounded') else 'literal'))
        rowcounts=[]
        for j,group in enumerate(columns):
            if group=='joint_four_answer_correct':
                record=next(r for r in counter if r['case']==case);count=int(record['correct_count']);n=int(record['episodes']);value=float(record['estimate'])
            else:
                record=next(r for r in summary if r['case']==case and r['group']==group);count=int(record['both_correct_count']);n=int(record['episodes']);value=float(record['both_correct'])
            values[i,j]=value;rowcounts.append((count,n))
        counts.append(rowcounts)
    fig,ax=plt.subplots(figsize=(10.5,max(3.5,len(cases)*.45)))
    im=ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(len(cases)):
        for j in range(len(columns)):
            count,n=counts[i][j];ax.text(j,i,f'{count}/{n}',ha='center',va='center',fontsize=9,color='white' if values[i,j]>.55 else '#17212d')
    ax.set_xticks(range(len(columns)),labels,fontsize=9);ax.set_yticks(range(len(cases)),rowlabels,fontsize=9)
    ax.tick_params(axis='both',length=0);ax.set_title('Fresh-assignment acquisition at 839,248 registered parameters',fontsize=12,pad=12)
    fig.colorbar(im,ax=ax,fraction=.025,pad=.025,label='All required answers correct (full 256-way output)')
    fig.text(.01,-.025,'Counts retain complete episodes. Joint 4 requires both original and swapped pairs correct. Each case is a fixed terminal checkpoint.',fontsize=8)
    fig.tight_layout();export(fig,out,'acquisition_counts')
    primary_path=audited/'primary_estimates.csv'
    if primary_path.exists():
        primary=load_csv(primary_path)
        if primary:
            fig,ax=plt.subplots(figsize=(7.7,3.5))
            for i,row in enumerate(primary):
                mid,lo,hi=[100*float(row[k]) for k in ['estimate','lower','upper']]
                ax.errorbar(mid,i,xerr=np.array([[mid-lo],[hi-mid]]),fmt='o',color='#b45309',capsize=4,markersize=6)
                ax.annotate(f'{mid:+.2f} pp [{lo:+.2f}, {hi:+.2f}]',(mid,i),xytext=(0,14),textcoords='offset points',ha='center',fontsize=9)
            ax.axvline(0,color='#6b7280',lw=1);ax.set_yticks(range(len(primary)),['Fresh train keys / seen wording','Dev keys / heldout wording'])
            ax.set_xlabel('Bounded-training minus literal-training\nboth-correct accuracy (percentage points)')
            ax.set_title('Paired acquisition effects across four fixed parent / geometry cases',fontsize=11)
            ax.set_ylim(-.7,len(primary)-.3);ax.grid(axis='x',alpha=.2)
            fig.text(.01,-.025,'97.5% paired whole-episode percentile intervals; 5,000 PCG64 draws, seed 91304. Fixed cases averaged within each episode.',fontsize=8)
            fig.tight_layout();export(fig,out,'primary_paired_effects')
    rows=load_csv(audited/'episode_metrics.csv');rows=[r for r in rows if r['mode']=='train' and r['condition']=='normal' and r['group'].startswith('revision')]
    if rows:
        arms=['online','repeat','continuous','periodic','continuous_bounded','periodic_bounded']
        short=['Online','Repeat','Literal C','Literal P','Bounded C','Bounded P']
        fig,axes=plt.subplots(2,2,figsize=(11,6),sharey=True)
        source_rows=[]
        for rowindex,group in enumerate(['revision_dev_seen','revision_dev_heldout']):
            for colindex,segment in enumerate([32,128]):
                ax=axes[rowindex,colindex]
                for offset,(training_arm,color,label) in enumerate([('periodic','#2563eb','Literal-trained'),('periodic_bounded','#d97706','Bounded-trained')]):
                    means=[];lower=[];upper=[]
                    for arm in arms:
                        subset=[r for r in rows if r['group']==group and int(r['segment'])==segment and r['training_arm']==training_arm and r['arm']==arm]
                        episodes=sorted({int(r['episode']) for r in subset});values=[]
                        for episode in episodes:
                            pair=[float(r['accuracy']) for r in subset if int(r['episode'])==episode]
                            assert len(pair)==2,'Both fixed parents required';values.append(np.mean(pair))
                        interval=paired_bootstrap(values,seed=91304,confidence=.95)
                        mid,lo,hi=[100*float(interval[k][0]) for k in ['estimate','lower','upper']]
                        means.append(mid);lower.append(mid-lo);upper.append(hi-mid)
                        source_rows.append(dict(group=group,segment=segment,training_arm=training_arm,arm=arm,estimate=mid,lower=lo,upper=hi,metric='per_query_accuracy',unit='percent accuracy / paired episode / two fixed parents averaged'))
                    positions=np.arange(len(arms))+(offset-.5)*.12
                    ax.errorbar(positions,means,yerr=[lower,upper],fmt='o-',color=color,capsize=2,lw=1.2,ms=3,label=label)
                ax.set_title(f"Segment {segment} / {'seen' if rowindex==0 else 'heldout'} wording",fontsize=10)
                ax.set_xticks(range(len(arms)),short,rotation=25,ha='right',fontsize=8);ax.set_ylim(-3,103);ax.grid(axis='y',alpha=.2)
                if colindex==0:ax.set_ylabel('Revised per-question accuracy (%)')
        axes[0,0].legend(fontsize=8,loc='upper right');fig.suptitle('Partial revision accuracy: every two-question pair fails',fontsize=12)
        fig.text(.01,-.025,'All eight checkpoints score 0/64 both-correct under every normal replay arm. Nominal 95% whole-episode intervals; two fixed parents averaged.\nOnline has fewer input records. Per-question differences are diagnostics of unqualified models.',fontsize=8)
        fig.tight_layout();export(fig,out,'revision_diagnostics')
        with (out/'revision_figure_source.csv').open('w',newline='',encoding='utf8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(source_rows[0]));writer.writeheader();writer.writerows(source_rows)
    (out/'figure_manifest.json').write_text(json.dumps(dict(source=str(audited),figures=sorted(p.name for p in out.glob('*.png')),scope='actual audited saved data only'),indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audited',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.audited,args.out)

