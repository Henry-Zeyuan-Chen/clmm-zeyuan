"""Descriptive key-conditioning behavior from audited complete K2 pairs."""
import collections,json
from pathlib import Path
import numpy as np
from audit_results import load_csv,write_csv
ROOT=Path(__file__).resolve().parent;rows=load_csv(ROOT/'audited/episode_metrics.csv')
output=[]
for case in sorted({r['case'] for r in rows if r['mode']=='train'}):
    for group in ['acquire_train_seen','acquire_train_swapped','acquire_dev_seen','acquire_dev_heldout']:
        subset=[r for r in rows if (r['case'],r['group'],r['arm'],r['condition'])==(case,group,'online','normal')]
        same=in_pair=first_value=second_value=correct=0
        for row in subset:
            a,b=int(row['first_target']),int(row['second_target']);p,q=int(row['first_prediction']),int(row['second_prediction']);assert a!=b
            same+=p==q;in_pair+=(p in (a,b))+(q in (a,b));first_value+=(p==a)+(q==a);second_value+=(p==b)+(q==b);correct+=int(row['first_correct'])+int(row['second_correct'])
        query_only=[r for r in rows if (r['case'],r['group'],r['condition'])==(case,'query_only_train_seen','normal')]
        output.append(dict(case=case,group=group,episodes=len(subset),identical_predictions=same,
            key_switch_changes=len(subset)-same,per_query_accuracy=correct/(2*len(subset)),
            predictions_in_current_value_pair=in_pair/(2*len(subset)),first_value_rate=first_value/(2*len(subset)),second_value_rate=second_value/(2*len(subset)),
            query_only_accuracy=float(np.mean([float(r['accuracy']) for r in query_only]))))
write_csv(ROOT/'audited/query_behavior.csv',output)
print(json.dumps([r for r in output if r['group']=='acquire_train_seen']))
