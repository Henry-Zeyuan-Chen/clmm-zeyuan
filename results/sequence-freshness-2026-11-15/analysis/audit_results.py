"""Audit saved sequence predictions and compute fixed-case paired estimates."""
from __future__ import annotations
import argparse,collections,csv,hashlib,json
from pathlib import Path
import numpy as np
from independent_helpers import score_logits,paired_bootstrap
from audit_packet import read_json,expected_samples,render_query,sha


def load_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))

def write_csv(path,rows):
    rows=list(rows)
    if not rows:return
    with path.open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def expected_evaluation(groups):
    expected=[]
    arms=('online','repeat','continuous','periodic','continuous_bounded','periodic_bounded')
    for group,episodes in groups.items():
        for arm in arms if group.startswith('revision') else ('online',):
            conditions=('normal','reset_memory') if not group.startswith('revision') or arm in ('online','periodic_bounded') else ('normal',)
            for condition in conditions:
                for index,ep in enumerate(episodes):
                    samples,plan=expected_samples(ep,arm)
                    # Stale means changed value relative to the latest real observation at application.
                    latest={};stale=0
                    for event in plan:
                        source=ep['events'][event['source']]
                        if event['kind']=='online':latest[source['key']]=source['value']
                        else:stale+=int(source['value']!=latest[source['key']])
                    for rank,(prefix,target) in enumerate(samples):
                        expected.append(dict(group=group,arm=arm,condition=condition,episode=index,rank=rank,
                            key=ep['queries'][rank]['key'],target=target,
                            old=ord(ep['queries'][rank]['old_value']) if ep['queries'][rank]['old_value'] else -1,
                            prefix_hex=prefix.hex(),input_bytes=len(prefix),real_records=len(ep['events']),
                            extra_records=len(plan)-len(ep['events']),stale_extras=stale))
    for index,ep in enumerate(groups['acquire_train_seen']):
        for rank,query in enumerate(ep['queries']):
            expected.append(dict(group='query_only_train_seen',arm='online',condition='normal',episode=index,rank=rank,
                key=query['key'],target=ord(query['target']),old=-1,prefix_hex=render_query(query).hex(),
                input_bytes=31,real_records=0,extra_records=0,stale_extras=0))
    return expected


def audit_case(folder,groups,packet_meta):
    meta=read_json(folder/'metadata.json');assert meta['status']=='succeeded'
    assert meta['parameter_count']==839248 and meta['reload_max_error']==0
    assert meta['source_commit']==packet_meta['source_commit']
    assert meta['data_sha256']==packet_meta['files_sha256']
    assert meta['initial_state_sha256']==meta['parent']['state_sha256']
    assert meta['evaluation']['pure_checkpoint'] and meta['evaluation']['RNG_unchanged']
    assert meta['checkpoint_sha256']==sha(folder/'checkpoint.pt')
    rows=load_csv(folder/'evaluation/predictions.csv');expected=expected_evaluation(groups)
    logits=np.load(folder/'evaluation/answer_logits.npy',allow_pickle=False)
    assert sha(folder/'evaluation/answer_logits.npy')==meta['evaluation']['logits_sha256']
    assert len(rows)==len(expected)==len(logits)==meta['evaluation']['queries']
    for index,(row,exp) in enumerate(zip(rows,expected)):
        assert int(row['score_index'])==index
        for key,value in exp.items():assert (int(row[key]) if isinstance(value,int) else row[key])==value,(folder,key,index)
    scores=score_logits(logits,[r['target'] for r in expected])
    max_score_error={}
    for key,values in scores.items():
        if key not in rows[0]:continue
        recorded=np.array([float(r[key]) for r in rows])
        error=float(np.max(np.abs(recorded-values)));max_score_error[key]=error
        assert np.allclose(recorded,values,rtol=1e-5,atol=2e-6),(folder,key,error)
    for i,row in enumerate(rows):
        row.update({key:float(value[i]) for key,value in scores.items()})
        for key in ['episode','rank','target','old','score_index','input_bytes','real_records','extra_records','stale_extras']:
            row[key]=int(row[key])
        row['prediction']=int(row['prediction']);row['forced_prediction']=int(row['forced_prediction']);row['correct']=int(row['correct'])
    episode_rows=[];grouped=collections.defaultdict(list)
    for row in rows:grouped[(row['group'],row['arm'],row['condition'],row['episode'])].append(row)
    for (group,arm,condition,episode),pair in grouped.items():
        pair=sorted(pair,key=lambda r:r['rank']);assert [r['rank'] for r in pair]==[0,1]
        episode_rows.append(dict(case=folder.name,mode=meta['mode'],parent=meta['parent_seed'],segment=meta['segment'],training_arm=meta['training_arm'],
            group=group,arm=arm,condition=condition,episode=episode,first_correct=pair[0]['correct'],second_correct=pair[1]['correct'],
            accuracy=sum(r['correct'] for r in pair)/2,both_correct=int(all(r['correct'] for r in pair)),
            prediction_changes=int(pair[0]['prediction']!=pair[1]['prediction']),
            first_prediction=pair[0]['prediction'],second_prediction=pair[1]['prediction'],
            first_target=pair[0]['target'],second_target=pair[1]['target'],
            nll=float(np.mean([r['nll'] for r in pair])),legal_probability_mass=float(np.mean([r['legal_probability_mass'] for r in pair])),
            obsolete_rate=float(np.mean([int(r['old']>=0 and r['prediction']==r['old']) for r in pair]))))
    summaries=[]
    for group,arm,condition in dict.fromkeys((r['group'],r['arm'],r['condition']) for r in episode_rows):
        subset=[r for r in episode_rows if (r['group'],r['arm'],r['condition'])==(group,arm,condition)]
        base=dict(case=folder.name,mode=meta['mode'],parent=meta['parent_seed'],segment=meta['segment'],training_arm=meta['training_arm'],group=group,arm=arm,condition=condition,episodes=len(subset))
        for metric in ['accuracy','both_correct','first_correct','second_correct','prediction_changes','nll','legal_probability_mass','obsolete_rate']:
            base[metric]=float(np.mean([r[metric] for r in subset]))
        base['both_correct_count']=sum(r['both_correct'] for r in subset)
        base['acquisition_gate_pass']=bool(base['both_correct']>=.8) if group in ['acquire_train_seen','acquire_dev_seen','acquire_dev_heldout'] and condition=='normal' else ''
        summaries.append(base)
    saved=load_csv(folder/'evaluation/summary.csv')
    assert len(saved)==len(summaries)
    for record,independent in zip(saved,summaries):
        for key in ['group','arm','condition']:assert record[key]==independent[key]
        for key in ['accuracy','both_correct','prediction_changes','nll','legal_probability_mass']:
            assert np.isclose(float(record[key]),independent[key],rtol=1e-5,atol=2e-6),(folder,key)
    # Acquisition at128 fits within one attention segment; reset should be an exact saved-logit no-op.
    if meta['segment']==128:
        paired=collections.defaultdict(dict)
        for row in rows:
            if row['group'].startswith('acquire'):
                paired[(row['group'],row['arm'],row['episode'],row['rank'])][row['condition']]=row['score_index']
        assert all(np.array_equal(logits[v['normal']],logits[v['reset_memory']]) for v in paired.values())
    train_audit=None
    if meta['mode'] in ('train','preflight'):
        trace=load_csv(folder/'training.csv');assert len(trace)==meta['completed_steps']==packet_meta['steps']
        assert [int(row['step']) for row in trace]==list(range(1,len(trace)+1))
        assert all(np.isfinite(float(row[key])) for row in trace for key in ['loss','accuracy','gradient_norm','elapsed_seconds'])
        answers=sum(int(row['answers']) for row in trace);tokens=sum(int(row['input_bytes']) for row in trace)
        assert answers==meta['supervised_answers']==packet_meta['training_answers']
        assert tokens==meta['training_input_bytes']==packet_meta['training_input_bytes']
        memory_steps=None
        if 'memory_gradient_norm' in trace[0]:
            assert all(np.isfinite(float(r['memory_gradient_norm'])) and float(r['memory_gradient_norm'])>=0 for r in trace)
            memory_steps=sum(float(r['memory_gradient_norm'])>0 for r in trace)
            assert memory_steps==meta['memory_nonzero_gradient_steps']
        train_audit=dict(steps=len(trace),answer_forks=answers,input_bytes=tokens,memory_nonzero_gradient_steps=memory_steps)
    else:assert meta['completed_steps']==meta['supervised_answers']==meta['training_input_bytes']==0
    resources=load_csv(folder/'evaluation/resources.csv')
    covered=[]
    for record in resources:
        indices=json.loads(record['row_indices'])
        assert len(indices)>0 and float(record['elapsed_seconds'])>=0
        for index in indices:
            assert rows[index]['condition']==record['condition'] and rows[index]['input_bytes']==int(record['input_length'])
        covered.extend(indices)
    assert sorted(covered)==list(range(len(rows)))
    return dict(folder=str(folder),mode=meta['mode'],parent=meta['parent_seed'],segment=meta['segment'],training_arm=meta['training_arm'],
        status='passed',logit_vectors=len(logits),max_score_error=max_score_error,training=train_audit),episode_rows,summaries


def estimate_primary(episode_rows,seed=91304,draws=5000):
    finals=[r for r in episode_rows if r['mode']=='train']
    cases={(r['parent'],r['segment']) for r in finals}
    expected_cases={(68101,32),(68101,128),(68105,32),(68105,128)}
    assert cases==expected_cases,'All four fixed parent/geometry pairs required for declared aggregate'
    estimates=[]
    for group in ['acquire_train_seen','acquire_dev_heldout']:
        index={}
        for row in finals:
            if (row['group'],row['arm'],row['condition'])==(group,'online','normal'):
                key=(row['parent'],row['segment'],row['training_arm'],row['episode']);assert key not in index
                index[key]=row['both_correct']
        units=sorted({key[-1] for key in index})
        differences=[]
        for episode in units:
            differences.append(np.mean([index[(parent,segment,'periodic_bounded',episode)]-index[(parent,segment,'periodic',episode)] for parent,segment in sorted(cases)]))
        interval=paired_bootstrap(differences,seed,draws,.975)
        estimates.append(dict(endpoint=group,metric='both_correct',contrast='bounded-minus-literal adaptation',fixed_cases=4,
            episodes=len(units),estimate=float(interval['estimate'][0]),lower=float(interval['lower'][0]),upper=float(interval['upper'][0]),
            confidence=.975,draws=draws,seed=seed,unit='whole episode after fixed-case averaging'))
    return estimates


def run(packet,folders,out,draws=5000,require_complete=False):
    out.mkdir(parents=True,exist_ok=True);groups=read_json(packet/'episodes.json');packet_meta=read_json(packet/'metadata.json')
    audits=[];all_episodes=[];all_summaries=[]
    for folder in folders:
        audit,episodes,summaries=audit_case(folder,groups,packet_meta);audits.append(audit);all_episodes.extend(episodes);all_summaries.extend(summaries)
    primary=[]
    final_keys={(r['parent'],r['segment'],r['training_arm']) for r in audits if r['mode']=='train'}
    if len(final_keys)==8:primary=estimate_primary(all_episodes,draws=draws)
    elif require_complete:raise AssertionError('Eight declared final cases not available')
    write_csv(out/'episode_metrics.csv',all_episodes);write_csv(out/'case_summary.csv',all_summaries);write_csv(out/'primary_estimates.csv',primary)
    result=dict(status='passed',scope='saved-output score/data analysis only; no model execution',cases=audits,
        total_logit_vectors=sum(row['logit_vectors'] for row in audits),complete_eight_final_cases=len(final_keys)==8,
        primary=primary,source_sha256={str(Path(__file__).name):sha(Path(__file__))})
    (out/'results_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--case',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'audited');parser.add_argument('--draws',type=int,default=5000);parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();run(args.packet,args.case,args.out,args.draws,args.require_complete)

