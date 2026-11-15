"""Own admission gates and paired development-pilot/frozen-transfer contrasts."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

SEEDS=(90511,90515)
BOOTSTRAP_SEED=90604

def read(path):return json.loads(path.read_text())
def rows(path):
    with path.open(newline='') as handle:return list(csv.DictReader(handle))
def write(path,data):
    if data:
        with path.open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def chosen(data,cell,arm='online',category='targets'):
    return [r for r in data if r['cell']==cell and r['arm']==arm and
            (r['category']!='recent' if category=='targets' else r['category']==category)]
def by_episode(data,metric='accuracy'):
    grouped=defaultdict(list)
    for row in data:grouped[int(row['episode'])].append(row)
    if metric=='all_correct':return np.array([all(int(r['correct']) for r in group) for _,group in sorted(grouped.items())],dtype=float)
    field={'accuracy':'correct','mse':'mse','obsolete':'obsolete'}[metric]
    return np.array([np.mean([float(r[field]) for r in group]) for _,group in sorted(grouped.items())])
def interval(array):
    array=np.asarray(array,dtype=float)
    ix=np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(array),size=(5000,len(array)))
    lo,hi=np.quantile(array[ix].mean(1),[.025,.975])
    return float(array.mean()),float(lo),float(hi)
def descriptors(meta):
    family='trained' if meta['offline_training'] else 'fork' if meta['args']['mode']=='fork' else 'neural'
    rule=meta['config'].get('update_rule','neural')
    route=meta['config'].get('routing','none')
    seed=int(meta['args']['init_seed'])
    return dict(family=family,rule=rule,routing=route,initializer=seed)
def engineering(name):return name.startswith(('smoke','throughput','preflight'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    records={};gates=[];pressure=[];timing=[];initial_final=[];dynamics=[];gate_map={};d0_map={};paired_episodes={}
    for run in sorted(args.root.iterdir()):
        if not (run/'metadata.json').exists() or engineering(run.name):continue
        meta=read(run/'metadata.json')
        if meta['status']!='succeeded':continue
        desc=descriptors(meta);key=(desc['family'],desc['rule'],desc['routing'],desc['initializer'])
        data=rows(run/'evaluation/predictions.csv');eps=read(run/'evaluation/episodes.json')
        records[key]=dict(run=run,meta=meta,desc=desc,data=data,eps=eps)
        short={cell:by_episode(chosen(data,cell),'all_correct') for cell in ('acquire2','acquire4','revise2')}
        assert all(len(v)==128 for v in short.values())
        delta=short['acquire2']-by_episode(chosen(data,'acquire2','rolled'),'all_correct')
        mean,lo,hi=interval(delta.reshape(8,16).mean(1))
        short_pass=bool(short['acquire2'].mean()>=.9 and short['acquire4'].mean()>=.8 and short['revise2'].mean()>=.9 and lo>0)
        gate=dict(run=run.name,**desc,acquire2=float(short['acquire2'].mean()),acquire4=float(short['acquire4'].mean()),
                  revise2=float(short['revise2'].mean()),normal_minus_rolled=mean,rolled_ci_low=lo,rolled_ci_high=hi,short_pass=short_pass)
        gate_map[key]=gate
        for facts in (8,16):
            cell=f'pressure_k{facts}_d0'
            group=chosen(data,cell)
            if not group:continue
            d0=float(by_episode(group,'all_correct').mean());d0_map[key+(facts,)]=d0
            gates.append(dict(**gate,facts=facts,d0_four_target_all_correct=d0,individual_admission=bool(short_pass and d0>=.8)))
        for cell in sorted(name for name in eps if name.startswith('pressure')):
            for arm in sorted({r['arm'] for r in data if r['cell']==cell}):
                for category in ('targets','stable','revised','recent'):
                    group=chosen(data,cell,arm,category)
                    if not group:continue
                    a=by_episode(group);m,lo,hi=interval(a)
                    pressure.append(dict(run=run.name,**desc,cell=cell,arm=arm,category=category,episodes=len(a),queries=len(group),
                        accuracy=m,ci_low=lo,ci_high=hi,all_correct=float(by_episode(group,'all_correct').mean()),
                        mse=float(by_episode(group,'mse').mean()),obsolete_rate=float(by_episode(group,'obsolete').mean()) if category=='revised' else ''))
            for category in ('targets','stable','revised','recent'):
                for left,right in (('periodic','continuous'),('periodic','online'),('continuous','online'),('repeat','online')):
                    a=chosen(data,cell,left,category);b=chosen(data,cell,right,category)
                    if not a or not b:continue
                    diff=by_episode(a)-by_episode(b);m,lo,hi=interval(diff)
                    timing.append(dict(run=run.name,**desc,cell=cell,category=category,contrast=f'{left}_minus_{right}',
                        episodes=len(diff),difference=m,ci_low=lo,ci_high=hi,scope='secondary unadjusted; fixed checkpoint/history'))
        if (run/'initial-evaluation').exists():
            initial=rows(run/'initial-evaluation/predictions.csv')
            initial_eps=read(run/'initial-evaluation/episodes.json')
            for cell in initial_eps:
                assert initial_eps[cell]==eps[cell]
                for category in ('targets','stable','revised','recent'):
                    before=chosen(initial,cell,category=category);after=chosen(data,cell,category=category)
                    if not before or not after:continue
                    for metric in ('accuracy','all_correct','mse'):
                        a,b=by_episode(before,metric),by_episode(after,metric);m,lo,hi=interval(b-a)
                        initial_final.append(dict(run=run.name,**desc,cell=cell,category=category,metric=metric,
                            episodes=len(a),initial_mean=float(a.mean()),final_mean=float(b.mean()),difference=m,ci_low=lo,ci_high=hi))
        learned=meta['evaluation'].get('learned') or {}
        resource=rows(run/'evaluation/resources.csv')
        dynamics.append(dict(run=run.name,**desc,parameters=meta['parameters']['registered'],trainable=meta['parameters']['trainable'],
            eta=learned.get('eta',''),alpha_mean=learned.get('alpha_mean',''),alpha_min=learned.get('alpha_min',''),alpha_max=learned.get('alpha_max',''),
            max_trace_saturation=max((float(r['trace_saturation']) for r in resource if r['trace_saturation']),default=0),
            all_batch_active_module_updates=sum(int(r['active_module_updates']) for r in resource if r['active_module_updates']),
            max_per_stream_adaptive_bytes=max(float(r['per_stream_final_bytes']) for r in resource),
            evaluation_batch_wall_seconds=sum(float(r['elapsed_seconds']) for r in resource)))
    def qualified(family,rule,route,facts):
        return all((family,rule,route,seed) in gate_map and gate_map[family,rule,route,seed]['short_pass']
                   and d0_map.get((family,rule,route,seed,facts),0)>=.8 for seed in SEEDS)
    for gate in gates:
        gate['both_seed_admission']=qualified(gate['family'],gate['rule'],gate['routing'],gate['facts'])
        gate['interpretation']='trained-system retention admitted' if gate['family']=='trained' and gate['both_seed_admission'] else 'frozen-transfer admitted' if gate['family']=='fork' and gate['both_seed_admission'] else 'descriptive neural anchor' if gate['family']=='neural' and gate['both_seed_admission'] else 'acquisition/interference diagnostic'
    contrasts=[];primary=[];incomplete_pairs=[]
    # All intervention contrasts use the same history IDs and average fixed-seed
    # differences within each history before bootstrap resampling.
    for family in ('trained','fork'):
        for cell in ('pressure_k8_d64','pressure_k16_d64'):
            facts=8 if 'k8_' in cell else 16
            for name,left_rule,left_route,right_rule,right_route in (
                ('hebbian_top1_minus_dense','hebbian','top1','hebbian','dense'),
                ('gradient_top1_minus_dense','gradient','top1','gradient','dense'),
                ('hebbian_minus_gradient_dense','hebbian','dense','gradient','dense'),
                ('hebbian_minus_gradient_top1','hebbian','top1','gradient','top1')):
                arrays=[]
                for seed in SEEDS:
                    left=records.get((family,left_rule,left_route,seed));right=records.get((family,right_rule,right_route,seed))
                    if not left or not right:
                        incomplete_pairs.append(dict(family=family,cell=cell,contrast=name,initializer=seed));continue
                    a=chosen(left['data'],cell,category='stable');b=chosen(right['data'],cell,category='stable')
                    if not a or not b:continue
                    assert left['eps'][cell]==right['eps'][cell]
                    aa,bb=by_episode(a),by_episode(b);diff=aa-bb;arrays.append(diff)
                    m,lo,hi=interval(diff)
                    admit=qualified(family,left_rule,left_route,facts) and qualified(family,right_rule,right_route,facts)
                    row=dict(family=family,cell=cell,contrast=name,initializer=seed,fixed_initializers=1,episodes=len(diff),
                        left_accuracy=float(aa.mean()),right_accuracy=float(bb.mean()),difference=m,ci_low=lo,ci_high=hi,
                        both_systems_admitted=admit,scope='trained-system development pilot' if family=='trained' and admit else 'fixed-slow-weight transfer diagnostic' if family=='fork' else 'acquisition/interference diagnostic')
                    contrasts.append(row)
                    if family=='trained' and cell=='pressure_k8_d64' and name=='hebbian_top1_minus_dense':primary.append(row)
                if len(arrays)==2:
                    assert arrays[0].shape==arrays[1].shape
                    m,lo,hi=interval(np.mean(arrays,axis=0))
                    previous=[r for r in contrasts if r['family']==family and r['cell']==cell and r['contrast']==name and r['fixed_initializers']==1]
                    row=dict(previous[0],initializer='mean_of_fixed_initializers',fixed_initializers=2,
                        left_accuracy=float(np.mean([r['left_accuracy'] for r in previous])),right_accuracy=float(np.mean([r['right_accuracy'] for r in previous])),
                        difference=m,ci_low=lo,ci_high=hi)
                    contrasts.append(row)
                    if family=='trained' and cell=='pressure_k8_d64' and name=='hebbian_top1_minus_dense':primary.append(row)
    sham=[]
    for seed in SEEDS:
        trained=records.get(('trained','gradient','dense',seed));fork=records.get(('fork','gradient','dense',seed))
        if not trained or not fork:continue
        before=np.load(trained['run']/'evaluation/scores.npy');after=np.load(fork['run']/'evaluation/scores.npy')
        index={(r['cell'],r['arm'],int(r['episode']),int(r['rank'])):i for i,r in enumerate(trained['data'])}
        delta=0.
        for i,r in enumerate(fork['data']):
            j=index[(r['cell'],r['arm'],int(r['episode']),int(r['rank']))]
            assert np.array_equal(before[j],after[i])
            delta=max(delta,float(np.max(np.abs(before[j]-after[i]))))
        sham.append(dict(initializer=seed,compared_queries=len(fork['data']),max_score_error=delta,all_scores_exact=True))
    write(args.out/'admission.csv',gates);write(args.out/'pressure_fixed_targets.csv',pressure)
    write(args.out/'timing_secondary.csv',timing);write(args.out/'initial_final.csv',initial_final)
    write(args.out/'learned_dynamics_resources.csv',dynamics);write(args.out/'system_contrasts.csv',contrasts);write(args.out/'primary.csv',primary)
    report=dict(status='reconstructed_from_saved_predictions',bootstrap_seed=BOOTSTRAP_SEED,bootstrap_draws=5000,
        primary='trained Hebbian top1 minus dense, ONLINE K8D64 stable accuracy',primary_rows=primary,
        primary_paired_seeds_complete=any(r['fixed_initializers']==2 for r in primary),
        interval_scope='Paired histories for two fixed trained checkpoints/codebook, not training-procedure uncertainty; pilot intervals unadjusted',
        frozen_sham_reloads=sham,incomplete_pairs=incomplete_pairs,successful_records=len(records))
    (args.out/'analysis_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(successful_records=len(records),primary=primary,sham=sham,incomplete_pairs=len(incomplete_pairs)),indent=2))

if __name__=='__main__':main()
