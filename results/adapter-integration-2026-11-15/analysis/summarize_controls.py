"""Fixed-two-seed paired initializer and timing summaries at K8,D64."""
import argparse
import numpy as np
import analyze_results as a

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=a.Path,required=True);p.add_argument('--out',type=a.Path,required=True)
    args=p.parse_args();initial_rows=[];timing=[];cell='pressure_k8_d64'
    for rule,route in (('gradient','dense'),('gradient','top1'),('hebbian','dense'),('hebbian','top1'),('neural','none')):
        current=[];before=[]
        for seed in a.SEEDS:
            folder=args.root/(f'neural-anchor-{seed}' if rule=='neural' else f'{rule}-{route}-{seed}')
            if not (folder/'evaluation/predictions.csv').exists():continue
            final=a.rows(folder/'evaluation/predictions.csv');current.append(final)
            if (folder/'initial-evaluation/predictions.csv').exists():before.append(a.rows(folder/'initial-evaluation/predictions.csv'))
        if len(current)!=2:continue
        for left,right in (('periodic','continuous'),('periodic','online'),('continuous','online'),('repeat','online')):
            left_arrays=[a.by_episode(a.chosen(r,cell,left,'stable')) for r in current]
            right_arrays=[a.by_episode(a.chosen(r,cell,right,'stable')) for r in current]
            diff=np.mean(np.asarray(left_arrays)-np.asarray(right_arrays),axis=0);m,lo,hi=a.interval(diff)
            timing.append(dict(rule=rule,routing=route,contrast=f'{left}_minus_{right}',episodes=len(diff),fixed_initializers=2,
                left_mean=float(np.mean(left_arrays)),right_mean=float(np.mean(right_arrays)),difference=m,ci_low=lo,ci_high=hi,
                scope=('Secondary unadjusted fixed-checkpoint/history comparison; identical P/C replay records and write opportunities' if right=='continuous'
                       else 'Secondary unadjusted fixed-checkpoint/history comparison; online has fewer writes')))
        if len(before)==2:
            for metric in ('accuracy','all_correct','mse'):
                aa=[a.by_episode(a.chosen(r,cell,category='stable'),metric) for r in before]
                bb=[a.by_episode(a.chosen(r,cell,category='stable'),metric) for r in current]
                diff=np.mean(np.asarray(bb)-np.asarray(aa),axis=0);m,lo,hi=a.interval(diff)
                initial_rows.append(dict(rule=rule,routing=route,cell=cell,category='stable',arm='online',metric=metric,
                    episodes=len(diff),fixed_initializers=2,initial_mean=float(np.mean(aa)),final_mean=float(np.mean(bb)),
                    difference=m,ci_low=lo,ci_high=hi,scope='Secondary paired exact initializer comparison'))
    a.write(args.out/'fixed_seed_initial_final.csv',initial_rows);a.write(args.out/'fixed_seed_timing.csv',timing)
    print(dict(initial_final_rows=len(initial_rows),timing_rows=len(timing)))

if __name__=='__main__':main()
