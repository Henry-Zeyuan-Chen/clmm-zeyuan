"""Posthoc auxiliary-decay diagnostic; verify original streams and online invariance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import pressure_analysis as a

PROTOCOL=dict(facts=[64],delays=[0,256],episodes=256,eval_seed=58111,aux_decay_scale=1)


def keyed_rows(path, subset=False):
    rows={}
    with path.open(newline='',encoding='utf8') as handle:
        for row in csv.DictReader(handle):
            if subset and (int(row['facts'])!=64 or int(row['delay']) not in (0,256)):
                continue
            key=tuple(row[name] for name in ('episode','facts','arm','delay','condition','query_rank'))
            assert key not in rows
            rows[key]=row
    return rows


def compare_prefix(original,diagnostic):
    assert set(original)==set(diagnostic)
    checked=[]
    for name,array in diagnostic.items():
        reference=original[name]
        if name in ('keys','values','reservoir_draw','sample_slots','sample_sources','sample_value_sources'):
            reference=reference[:,:array.shape[1]]
        elif name in ('working_sources_at_query','working_value_sources_at_query'):
            positions=[list(map(int,original['delays'])).index(int(delay)) for delay in diagnostic['delays']]
            reference=reference[:,positions]
        elif name=='delays':
            assert array.tolist()==[0,256]
            continue
        np.testing.assert_array_equal(array,reference,err_msg=name)
        checked.append(name)
    return checked


def main(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'verification.json').exists() and not (out/'counts.csv').exists(), 'Choose a new diagnostic output'
    run,baseline=args.run.resolve(),args.baseline.resolve()
    design=json.loads(args.design.read_text())
    diagnostic_design=json.loads(args.diagnostic_design.read_text())
    for field in ('facts','delays','episodes','eval_seed'):
        assert diagnostic_design[field]==PROTOCOL[field],field
    amendment=a.load_training_amendment(args.training_amendment,args.design,design)
    audit=json.loads((args.baseline_verified/'verification.json').read_text())
    assert audit['status']=='passed'
    old_audit=next(row for row in audit['runs'] if Path(row['run']).resolve()==baseline)
    for filename in ('metadata.json','predictions.csv','summary.csv','streams_K64.npz'):
        assert a.sha256(baseline/filename)==old_audit['input_sha256'][str(baseline/filename)]
    old_meta=json.loads((baseline/'metadata.json').read_text())
    assert old_meta['aux_decay_scale']==0 and old_meta['eval_seed']==58111 and old_meta['episodes']==256
    metadata,outcomes,expected,traces,checked=a.verify_run(run,design,'grid',
        training_amendment=amendment,expected_evaluation=PROTOCOL)
    assert metadata['design_sha256'] in (a.sha256(args.design),a.sha256(args.diagnostic_design))
    assert len(metadata['checkpoints'])==len(old_meta['checkpoints'])==1
    current=metadata['checkpoints'][0]
    previous=old_meta['checkpoints'][0]
    assert current['init_seed']==previous['init_seed']==48103
    assert current['train_seed']==previous['train_seed']==48104
    assert current['sha256']==previous['sha256']
    assert current['config']==previous['config']
    assert metadata['repo_commit']==old_meta['repo_commit']
    for field in ('torch','device','batch_size','deterministic_algorithms','cuda_tf32','cublas_workspace_config'):
        assert metadata[field]==old_meta[field],field
    def source_map(meta):
        prefix=meta['repo'].replace('\\','/').rstrip('/')+'/'
        return {path.replace('\\','/').removeprefix(prefix):digest for path,digest in meta['source_sha256'].items()}
    assert source_map(metadata)==source_map(old_meta), 'The diagnostic may only change auxiliary decay'
    with np.load(baseline/'streams_K64.npz',allow_pickle=False) as loaded:
        original_data={name:loaded[name] for name in loaded.files}
    with np.load(run/'streams_K64.npz',allow_pickle=False) as loaded:
        diagnostic_data={name:loaded[name] for name in loaded.files}
    prefix_fields=compare_prefix(original_data,diagnostic_data)
    old_rows=keyed_rows(baseline/'predictions.csv',subset=True)
    new_rows=keyed_rows(run/'predictions.csv')
    assert old_rows.keys()==new_rows.keys()
    expected_changes={'aux_decay_scale','prediction','correct','stale_old'}
    online_equal,prefill_equal=0,0
    reference_correct=np.full_like(outcomes[current['index']][64],-1)
    for key,row in new_rows.items():
        reference=old_rows[key]
        assert set(row)==set(reference)
        for field,value in row.items():
            if field not in expected_changes:
                assert value==reference[field],(key,field)
        if row['arm']=='online' or int(row['delay'])==0:
            for field in ('prediction','correct','stale_old'):
                assert row[field]==reference[field],(key,field,'Expected unchanged prediction')
        online_equal+=row['arm']=='online'
        prefill_equal+=int(row['delay'])==0
        ai=a.ARMS.index(row['arm'])
        di=metadata['delays'].index(int(row['delay']))
        col=int(row['query_rank'])+{'stable':0,'revised':2,'recent':4}[row['condition']]
        correct=int(int(reference['prediction'])==int(reference['target_value']))
        assert correct==int(reference['correct'])
        reference_correct[ai,di,int(row['episode']),col]=correct
    assert np.all(reference_correct>=0)
    cells,contrasts,episode_rows,bootstrap_hash=a.analyze_results(run,metadata,outcomes,expected)
    a.write_csv(out/'counts.csv',cells)
    a.write_csv(out/'contrasts.csv',contrasts)
    a.write_csv(out/'primary_episode_counts.csv',episode_rows)
    n=metadata['episodes']
    indices=np.random.default_rng(a.BOOTSTRAP_SEED).integers(n,size=(a.DRAWS,n),dtype=np.int32)
    assert hashlib.sha256(indices.tobytes()).hexdigest()==bootstrap_hash
    common,_=expected[64]
    paired,paired_episodes=[],[]
    for di,delay in enumerate(metadata['delays']):
        resident,selected=common['working_resident'][di],common['shadow_ever_selected_for_replay'][di]
        strata=dict(all=np.ones_like(resident,dtype=bool),resident=resident,
            nonresident_selected=(~resident)&selected,nonresident_never_selected=(~resident)&(~selected))
        for condition,columns in a.CONDITIONS.items():
            for stratum,eligibility in strata.items():
                eligible=eligibility[:,columns]
                denominator=eligible.sum(axis=1)
                total=int(denominator.sum())
                for ai,arm in enumerate(a.ARMS):
                    new_correct=outcomes[current['index']][64][ai,di][:,columns]
                    old_correct=reference_correct[ai,di][:,columns]
                    numerator=np.stack([(new_correct*eligible).sum(axis=1),(old_correct*eligible).sum(axis=1)])
                    boot,valid=a.bootstrap_ratios(numerator,denominator,indices)
                    low,high=a.interval(boot[0]-boot[1])
                    paired.append(dict(run=str(run),baseline=str(baseline),init_seed=48103,train_seed=48104,
                        facts=64,delay=delay,condition=condition,stratum=stratum,arm=arm,
                        episodes=n,eligible_episodes=int((denominator>0).sum()),eligible_queries=total,
                        new_aux_decay_scale=1,baseline_aux_decay_scale=0,
                        new_correct=int(numerator[0].sum()),baseline_correct=int(numerator[1].sum()),
                        difference_pp=float(100*(numerator[0].sum()-numerator[1].sum())/total) if total else None,
                        ci_low_pp=low,ci_high_pp=high,valid_bootstrap_draws=int(valid.sum()),
                        empty_bootstrap_draws=int((~valid).sum())))
                    if delay==256 and ((condition=='old' and stratum=='nonresident_selected') or (condition=='recent' and stratum=='all')):
                        for episode in range(n):
                            paired_episodes.append(dict(arm=arm,condition=condition,stratum=stratum,episode=episode,
                                eligible_queries=int(denominator[episode]),new_correct=int(numerator[0,episode]),
                                baseline_correct=int(numerator[1,episode])))
    a.write_csv(out/'decay_paired_contrasts.csv',paired)
    a.write_csv(out/'decay_paired_episode_counts.csv',paired_episodes)
    verification=dict(status='passed',analysis_status='Posthoc weak-checkpoint auxiliary-decay sensitivity diagnostic; no original continuation decision is altered.',
        run=str(run),baseline=str(baseline),protocol=PROTOCOL,init_seed=48103,train_seed=48104,
        prediction_rows_verified=checked,trace_audits=traces,identical_prefix_fields=prefix_fields,
        online_predictions_identical=online_equal,common_prefill_predictions_identical=prefill_equal,
        all_nonprediction_row_fields_identical_except_aux_decay=True,checkpoint_and_sources_identical=True,
        base_design_sha256=a.sha256(args.design),training_amendment_sha256=a.sha256(args.training_amendment),
        diagnostic_design_sha256=a.sha256(args.diagnostic_design),run_design_reference_sha256=metadata['design_sha256'],
        baseline_verification_sha256=a.sha256(args.baseline_verified/'verification.json'),
        run_input_sha256={str(path):a.sha256(path) for path in [run/'metadata.json',run/'predictions.csv',run/'summary.csv',run/'streams_K64.npz']},
        baseline_input_sha256=old_audit['input_sha256'],
        bootstrap_draws=a.DRAWS,bootstrap_seed=a.BOOTSTRAP_SEED,bootstrap_indices_sha256=bootstrap_hash,
        bootstrap_unit='Paired episodes; ratio of summed correct-query counts and summed eligible-query counts.',
        limitations=['One previously observed weak checkpoint and the same already-seen streams.',
            'Decayed auxiliary updates are a policy sensitivity test, not proof of a particular regularization mechanism.',
            'No fresh confirmation or best-policy selection; original two-seed outcome is retained.',
            'Independent reconstruction does not rerun neural inference.'])
    source_paths=[Path(__file__),Path(a.__file__)]
    manifest={path.name:a.sha256(path) for path in source_paths}
    for path in source_paths:
        (out/path.name).write_bytes(path.read_bytes())
    verification['source_sha256']=manifest
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(dict(status='passed',online_predictions_identical=online_equal,
        common_prefill_predictions_identical=prefill_equal,paired_contrast_rows=len(paired)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--baseline-verified',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--design',type=Path,default=Path(__file__).resolve().parent.parent/'design.json')
    parser.add_argument('--training-amendment',type=Path,default=Path(__file__).resolve().parent.parent/'training-amendment-2.json')
    parser.add_argument('--diagnostic-design',type=Path,default=Path(__file__).resolve().parent.parent/'decay-diagnostic-design.json')
    main(parser.parse_args())
