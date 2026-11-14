"""Plot verified full-pilot training and paired-window ablation evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='',encoding='utf8') as handle:
        return list(csv.DictReader(handle))


def main(args):
    analysis,out=args.analysis.resolve(),args.out.resolve()
    verification=json.loads((analysis/'verification.json').read_text())
    assert verification['status']=='passed' and verification['mode']=='full pilot'
    runs=sorted(verification['runs'],key=lambda row:row['init_seed'])
    assert [row['init_seed'] for row in runs]==[68101,68105], 'Plot the two fixed pilots; no best-seed selection'
    assert verification['shared_arrays_status']=='verified identical across supplied runs'
    estimates=read_csv(analysis/'estimates.csv')
    contrasts=read_csv(analysis/'contrasts.csv')
    out.mkdir(parents=True,exist_ok=True)
    assert not any(out.iterdir()), 'Choose an empty figure output directory'
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10.6,7.6),gridspec_kw={'height_ratios':[1,1.25]})
    colors={'normal':'#1768AC','reset_memory_at_segments':'#C86624'}
    labels={'normal':'Normal memory','reset_memory_at_segments':'Reset each attention segment'}
    source_points,source_training=[],[]
    source_hashes={str(analysis/name):sha256(analysis/name) for name in
                   ('verification.json','estimates.csv','contrasts.csv','decisions.json')}
    for column,run in enumerate(runs):
        seed=run['init_seed']; run_path=Path(run['run'])
        training_path=run_path/'training.csv'
        assert sha256(training_path)==run['input_sha256'][str(training_path)]
        source_hashes[str(training_path)]=sha256(training_path)
        rows=read_csv(training_path)
        steps=np.array([int(row['step']) for row in rows]); loss=np.array([float(row['loss']) for row in rows])
        assert len(rows)==400 and np.isfinite(loss).all()
        for row in rows:
            source_training.append(dict(init_seed=seed,**row))
        ax=axes[0,column]
        ax.plot(steps,loss,color=colors['normal'],alpha=.27,linewidth=.75,label='Per-step loss')
        width=20
        ax.plot(steps[width-1:],np.convolve(loss,np.ones(width)/width,mode='valid'),
                color=colors['normal'],linewidth=1.7,label='20-step trailing mean')
        ax.set(xlabel='Optimizer step',ylabel='Training NLL (nats / byte)',xlim=(1,400))
        ax.set_title(f'Seed {seed} · '+('primary candidate' if seed==68101 else 'sensitivity run'),loc='left')
        ax.grid(axis='y',alpha=.16)
        if column==0: ax.legend(frameon=False,fontsize=8,loc='upper right')
        ax=axes[1,column]
        case_labels=['256 bytes\nall positions','1,024 bytes\npositions 256–1,023']
        delta_lines=[]
        for case,(length,region) in enumerate([(256,'full'),(1024,'late')]):
            for condition,shift in [('normal',-.085),('reset_memory_at_segments',.085)]:
                row=next(row for row in estimates if int(row['init_seed'])==seed and row['stage']=='final'
                    and row['condition']==condition and int(row['length'])==length and row['region']==region)
                assert row['run']==str(run_path) and row['checkpoint_sha256']==run['checkpoint_sha256']
                point,low,high=[float(row[name]) for name in ('bits_per_byte','bpb_ci_low','bpb_ci_high')]
                ax.errorbar(case+shift,point,yerr=[[point-low],[high-point]],fmt='o',ms=5,
                    color=colors[condition],capsize=3,elinewidth=1.2,
                    label=labels[condition] if case==0 else None)
                source_points.append(row)
            pair=next(row for row in contrasts if int(row['init_seed'])==seed and
                row['comparison']=='reset_memory_at_segments_minus_normal' and int(row['length'])==length and row['region']==region)
            d,lo,hi=[float(pair[name]) for name in ('difference_bits_per_byte','ci_low_bits','ci_high_bits')]
            delta_lines.append(f'{length:,} / {region}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        ax.set(xlim=(-.4,1.4),xticks=[0,1],xticklabels=case_labels,ylabel='Development bits / byte (lower is better)')
        ax.grid(axis='y',alpha=.16)
        ax.set_title('Paired memory intervention',loc='left')
        ax.text(0,-.28,'Reset − normal, paired 95% CI (bits / byte)\n'+'\n'.join(delta_lines),
                transform=ax.transAxes,fontsize=8.5,va='top',linespacing=1.4)
        if column==0: ax.legend(frameon=False,fontsize=8,loc='best')
    # Common units and limits aid the seed comparison without pooling observations.
    for row in range(2):
        lows,highs=zip(*(ax.get_ylim() for ax in axes[row]))
        for ax in axes[row]: ax.set_ylim(min(lows),max(highs))
    fig.suptitle('Shared byte-LM pilot: training and memory-context check',x=.08,ha='left',fontsize=14)
    fig.subplots_adjust(left=.08,right=.98,top=.91,bottom=.23,hspace=.45,wspace=.30)
    fig.text(.08,.035,'95% intervals resample 128 paired development windows (5,000 draws), separately for each fixed checkpoint.\n'
        'No seed pooling; reserved test data are not evaluated. Reset is an inference intervention, not a separately trained baseline.',
        fontsize=8.5,ha='left',va='bottom')
    for suffix in ('png','svg','pdf'):
        fig.savefig(out/('shared_lm_pilot.'+suffix),dpi=200)
    plt.close(fig)
    for name,rows in [('source_estimates.csv',source_points),('source_training.csv',source_training)]:
        with (out/name).open('w',newline='',encoding='utf8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    (out/'source_contrasts.csv').write_bytes((analysis/'contrasts.csv').read_bytes())
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    manifest=dict(source_sha256=source_hashes,plot_script_sha256=sha256(Path(__file__)),
        figure_description='Separate fixed-seed panels; per-condition window CIs with explicitly paired reset-minus-normal differences.',
        outputs_sha256={path.name:sha256(path) for path in out.iterdir() if path.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(str(out/'shared_lm_pilot.png'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
