"""Exploratory old-target retention after 64 observations without shadow selection."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pressure_analysis import ARMS,BOOTSTRAP_SEED,DRAWS,bootstrap_ratios,interval,sha256,write_csv


def main(args):
    run,verified,out=args.run.resolve(),args.verified.resolve(),args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'contrasts.csv').exists() or (out/'verification.json').exists():
        raise FileExistsError('Choose a fresh secondary analysis directory')
    audit=json.loads((verified/'verification.json').read_text())
    assert audit['status']=='passed'
    entry=next(item for item in audit['runs'] if Path(item['run']).resolve()==run)
    for filename in ('metadata.json','predictions.csv'):
        assert sha256(run/filename)==entry['input_sha256'][str(run/filename)]
    metadata=json.loads((run/'metadata.json').read_text())
    n,block_size=metadata['episodes'],metadata['snapshot_capacity']
    cases=[]
    if 64 in metadata['facts'] and 256 in metadata['delays']:
        cases.append((64,256))
    if 1024 in metadata['delays']:
        cases.extend((facts,1024) for facts in metadata['facts'])
    design=dict(status='posthoc exploratory check requested after viewing original grid primary',
        recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),
        cases=[dict(facts=facts,delay=delay) for facts,delay in cases],
        eligibility='Old target; working_resident=False; shadow_ever_selected_for_replay=True; delay-(last_shadow_selection_block+1)*block_size >=64.',
        minimum_observations_since_last_selected_block=64,comparisons=['periodic-repeat','periodic-online'],
        bootstrap_draws=DRAWS,bootstrap_seed=BOOTSTRAP_SEED,unit='Paired episode ratio of correct/eligible query sums',
        original_primary='Original primary definition, results and continuation signal remain unchanged.')
    (out/'posthoc_design.json').write_text(json.dumps(design,indent=2)+'\n',encoding='utf8')
    cells=defaultdict(dict)
    with (run/'predictions.csv').open(newline='',encoding='utf8') as handle:
        for row in csv.DictReader(handle):
            facts,delay=int(row['facts']),int(row['delay'])
            if (facts,delay) not in cases or row['condition'] not in ('stable','revised'):
                continue
            last_block=int(row['last_shadow_selection_block'])
            gap=delay-(last_block+1)*block_size
            eligible=(row['working_resident']=='False' and row['shadow_ever_selected_for_replay']=='True' and gap>=64)
            if not eligible:
                continue
            assert last_block>=0 and gap>=64
            ci=int(row['checkpoint_index'])
            identity=(int(row['episode']),row['condition'],int(row['query_rank']))
            correct=int(int(row['prediction'])==int(row['target_value']))
            assert correct==int(row['correct'])
            cell=cells[ci,facts,delay,row['arm']]
            assert identity not in cell
            cell[identity]=(correct,gap,int(row['key']),int(row['target_value']))
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    indices=rng.integers(n,size=(DRAWS,n),dtype=np.int32)
    counts,contrasts,episode_rows=[],[],[]
    for checkpoint in metadata['checkpoints']:
        ci=checkpoint['index']
        for facts,delay in cases:
            reference=cells[ci,facts,delay,'online']
            denominators=np.zeros(n,dtype=np.int32)
            numerators=np.zeros((4,n),dtype=np.int32)
            for identity in reference:
                denominators[identity[0]]+=1
            for ai,arm in enumerate(ARMS):
                current=cells[ci,facts,delay,arm]
                assert set(current)==set(reference)
                for identity,values in current.items():
                    assert values[1:]==reference[identity][1:]
                    numerators[ai,identity[0]]+=values[0]
            total=int(denominators.sum())
            boot,valid=bootstrap_ratios(numerators,denominators,indices)
            point=100*numerators.sum(axis=1)/total if total else np.full(4,np.nan)
            shared=dict(run=str(run),checkpoint_index=ci,init_seed=checkpoint['init_seed'],train_seed=checkpoint['train_seed'],
                facts=facts,delay=delay,condition='old',stratum='nonresident_selected_gap64',
                episodes=n,eligible_episodes=int((denominators>0).sum()),eligible_queries=total,
                minimum_actual_observation_gap=min((v[1] for v in reference.values()),default=None),
                valid_bootstrap_draws=int(valid.sum()),empty_bootstrap_draws=int((~valid).sum()),
                analysis_status='posthoc exploratory')
            for ai,arm in enumerate(ARMS):
                low,high=interval(boot[ai])
                counts.append(dict(**shared,arm=arm,correct=int(numerators[ai].sum()),
                    accuracy_pct=float(point[ai]) if total else None,ci_low_pct=low,ci_high_pct=high))
            for right in ('repeat','online'):
                li,ri=ARMS.index('periodic'),ARMS.index(right)
                low,high=interval(boot[li]-boot[ri])
                contrasts.append(dict(**shared,left='periodic',right=right,
                    difference_pp=float(point[li]-point[ri]) if total else None,ci_low_pp=low,ci_high_pp=high))
            for episode in range(n):
                episode_rows.append(dict(run=str(run),checkpoint_index=ci,init_seed=checkpoint['init_seed'],facts=facts,delay=delay,
                    episode=episode,eligible_queries=int(denominators[episode]),
                    **{arm+'_correct':int(numerators[ai,episode]) for ai,arm in enumerate(ARMS)}))
    write_csv(out/'counts.csv',counts)
    write_csv(out/'contrasts.csv',contrasts)
    write_csv(out/'episode_counts.csv',episode_rows)
    verification=dict(status='passed',analysis_status='posthoc exploratory',run=str(run),
        source_audit_sha256=sha256(verified/'verification.json'),source_predictions_sha256=sha256(run/'predictions.csv'),
        analysis_script_sha256=sha256(Path(__file__)),posthoc_design_sha256=sha256(out/'posthoc_design.json'),
        bootstrap_draws=DRAWS,bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        same_eligible_query_set_and_gaps_across_arms=True,original_primary_untouched=True,
        interpretation='At least64 observations since the end of the last block selecting the target; this is a shared selection-history stratum.',
        limitations=['Subgroup was chosen after observing the original primary result.',
                    'Selection history is descriptive and correlated with target/episode properties.',
                    'Intervals are fixed-checkpoint paired episode percentiles and are unadjusted.'])
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf8')
    print(json.dumps(contrasts,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--verified',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    main(parser.parse_args())
