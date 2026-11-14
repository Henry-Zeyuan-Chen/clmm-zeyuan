"""Independent saved-artifact audit and paired-window analysis; no model inference.

Example: python analyze_shared_lm.py --run ../lm-68101 ../lm-68105 --out verified
The source runner and repository are never imported or modified.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

BOOTSTRAP_SEED, DRAWS = 68104, 5000
BOUNDS = [0, 90_000_000, 95_000_000, 100_000_000]
SEEDS = {68101, 68105}
CONDITIONS = ('normal', 'zero_memory_output', 'reset_memory_at_segments')
CONFIG = dict(num_tokens=256, dim=128, depth=2, dim_head=32, heads=4,
    ff_mult=4.0, num_residual_streams=1, segment_len=32, num_persist_mem_tokens=4,
    num_longterm_mem_tokens=0, memory_layers=[1,2], memory_segment_len=8,
    memory_batch_size=32, memory_dim_head=32, memory_heads=4, memory_depth=2,
    memory_expansion_factor=2.0, momentum=True, qk_rmsnorm=True, max_inner_lr=.01,
    memory_norm_residual=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def write_csv(path, rows):
    require(bool(rows), 'Cannot write an empty evidence table')
    with path.open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with path.open(newline='',encoding='utf8') as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), label+' nonfinite')
    require(math.isclose(float(actual),float(expected),rel_tol=1e-11,abs_tol=1e-8),
            f'{label}: {actual} != {expected}')


def load_corpus(path):
    with gzip.open(path,'rb') as handle:
        raw=handle.read(BOUNDS[-1]+1)
    require(len(raw)==BOUNDS[-1], 'Decoded corpus must contain exactly100,000,000 bytes')
    return np.frombuffer(raw,dtype=np.uint8), dict(path=str(path.resolve()),
        compressed_sha256=sha256(path),raw_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def validate_design(design):
    train,evaluation=design['training'],design['evaluation']
    require(set(train['initialization_seeds'])==SEEDS,'Unexpected planned initialization seeds')
    for name,value in dict(shared_training_seed=68102,steps=400,batch_size=16,
        sequence_length=256,predicted_training_bytes_each=1_638_400,lr=.0003,
        weight_decay=.01,gradient_clip=1.0).items():
        require(train[name]==value,'Design training '+name)
    require(evaluation['dev_seed']==68103 and evaluation['windows']==128 and
        evaluation['batch_size']==8 and evaluation['lengths']==[256,1024], 'Design evaluation differs')
    require(tuple(evaluation['conditions'])==CONDITIONS,'Design conditions differ')
    for name,expected in [('train_range',BOUNDS[:2]),('dev_range',BOUNDS[1:3]),
                          ('reserved_test_range',BOUNDS[2:])]:
        require(design['data'][name]==expected,'Design '+name)
    mapping={'neural_memory_layers':'memory_layers','neural_memory_segment_len':'memory_segment_len',
             'neural_memory_batch_size':'memory_batch_size'}
    for name,value in design['model'].items():
        target=mapping.get(name,name)
        if target in CONFIG:
            require(value==CONFIG[target], 'Design model '+name)


def validate_run(run, args, corpus, corpus_info, design):
    metadata=json.loads((run/'metadata.json').read_text())
    cfg=metadata['configuration']
    require(metadata['status']=='succeeded','Run did not complete successfully: '+str(run))
    require(metadata['design_sha256']==sha256(args.design)==sha256(run/'design.json'), 'Design hashes differ')
    require(metadata['model_config']==CONFIG,'Resolved model configuration differs')
    for key,value in dict(seq_len=256,train_seed=68102,dev_seed=68103,lr=.0003,weight_decay=.01).items():
        require(cfg[key]==value,'Run configuration '+key)
    if not args.allow_smoke:
        for key,value in dict(steps=400,batch_size=16,eval_windows=128,eval_batch_size=8,max_seconds=1200.).items():
            require(cfg[key]==value,'Main pilot configuration '+key)
        require(cfg['init_seed'] in SEEDS,'Unplanned initialization seed')
        require(metadata['repo_commit'].startswith(args.expected_commit),'Unexpected runner source commit')
    else:
        require(cfg['steps']<400 and cfg['eval_windows']<128,'--allow-smoke is only for smaller format checks')
    if not args.allow_smoke and cfg['device']=='cpu':
        require(args.execution_amendment is not None,'Full CPU pilot requires explicit --execution-amendment')
    steps,batch,n=cfg['steps'],cfg['batch_size'],cfg['eval_windows']
    require(metadata['completed_steps']==steps,'Incomplete optimizer budget')
    require(metadata['parameters']==839248,'Unexpected registered parameter count')
    require(metadata['memory_gradient_observed'] is True,'No reported memory gradient')
    require(metadata['checkpoint_reload_max_abs_difference']==0,'Reload check failed')
    require(0<=metadata['trained_suffix_causality_max_abs_difference']<=1e-6,'Causality check failed')
    require(metadata['outer_parameters_unchanged_during_evaluation'] is True,'Outer weights changed')
    require(math.isfinite(metadata['elapsed_seconds']) and metadata['elapsed_seconds']>=0,'Invalid duration')
    # Finally-block accounting can extend slightly past the final deadline check.
    require(metadata['elapsed_seconds']<=cfg['max_seconds']+1,'Reported duration exceeds deadline allowance')
    environment=metadata['environment']
    require(environment['dtype']=='float32' and environment['deterministic_algorithms'] is True
            and environment['tf32'] is False,'Unexpected numeric execution configuration')
    require(environment['cublas_workspace_config']==':4096:8','Unexpected CUBLAS configuration')
    command=metadata['command']
    for index,token in enumerate(command):
        if token.startswith('--'):
            key=token[2:].replace('-','_')
            require(key in cfg and index+1<len(command),'Unrecognized saved command flag')
            if key not in ('out','data','design'):
                require(float(cfg[key])==float(command[index+1])
                        if isinstance(cfg[key],(int,float)) else str(cfg[key])==command[index+1],
                        'Saved command/config mismatch: '+key)
    input_paths=[run/name for name in ('metadata.json','design.json','training.csv','training_offsets.npy',
        'dev_windows.npz','checkpoint.pt','optimizer.pt','evaluation.csv','summary.json')]
    for relative,digest in metadata['source_sha256'].items():
        path=(run/'sources'/relative).resolve()
        require(path.is_relative_to((run/'sources').resolve()),'Source path escapes snapshot')
        require(sha256(path)==digest,'Captured source hash mismatch: '+relative)
        input_paths.append(path)
    data_info=metadata['data']
    for key in ('compressed_sha256','raw_sha256','bytes'):
        require(data_info[key]==corpus_info[key],'Local corpus mismatch: '+key)
    require(data_info['split_bounds']==BOUNDS and data_info['test_evaluated'] is False,'Split/test declaration differs')
    require(data_info['dev_windows_sha256']==sha256(run/'dev_windows.npz'),'Dev artifact hash differs')
    with np.load(run/'dev_windows.npz',allow_pickle=False) as loaded:
        require(set(loaded.files)=={'tokens','offsets'},'Unexpected dev NPZ fields')
        tokens,offsets=loaded['tokens'],loaded['offsets']
    require(tokens.shape==(n,1025) and tokens.dtype==np.uint8,'Dev token shape/dtype')
    require(offsets.shape==(n,) and np.issubdtype(offsets.dtype,np.integer),'Dev offset shape/dtype')
    require(np.all(offsets>=BOUNDS[1]) and np.all(offsets+1025<=BOUNDS[2]),'Dev target crosses a split')
    require(len(np.unique(offsets))==n and np.all(np.diff(np.sort(offsets))>=1025),'Dev windows overlap')
    require(np.all((offsets-BOUNDS[1])%1025==0),'Dev starts are not on the declared grid')
    np.testing.assert_array_equal(tokens,corpus[offsets[:,None]+np.arange(1025)[None,:]])
    train_offsets=np.load(run/'training_offsets.npy',allow_pickle=False)
    require(train_offsets.shape==(steps,batch) and np.issubdtype(train_offsets.dtype,np.integer),'Training offset shape/dtype')
    require(np.all(train_offsets>=0) and np.all(train_offsets+257<=BOUNDS[1]),'Training target crosses a split')
    train=read_csv(run/'training.csv')
    require(len(train)==steps,'Training log row count')
    previous=-1.
    for step,row in enumerate(train,1):
        require(int(row['step'])==step and int(row['predicted_bytes'])==step*batch*256,'Training budget ledger mismatch')
        values=[float(row[name]) for name in ('loss','gradient_norm','elapsed_seconds')]
        require(all(math.isfinite(value) and value>=0 for value in values),'Nonfinite/negative training log value')
        require(values[-1]>=previous,'Training timestamps decrease'); previous=values[-1]
    require(sha256(run/'checkpoint.pt')==metadata['checkpoint_sha256'],'Checkpoint hash mismatch')
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
    require(checkpoint['format_version']==1 and checkpoint['architecture']=='clmm.vendored_mac.byte_lm.v1','Checkpoint architecture differs')
    require(checkpoint['config']==CONFIG,'Checkpoint config differs')
    state=checkpoint['model_state']
    require(isinstance(state,dict) and len(state)==80,'Checkpoint tensor dictionary differs')
    require(all(isinstance(tensor,torch.Tensor) and bool(torch.isfinite(tensor).all()) for tensor in state.values()),'Nonfinite checkpoint state')
    require(sum(tensor.numel() for tensor in state.values())==metadata['parameters'],'Checkpoint tensor count differs')
    info=checkpoint['training_info']
    for key,value in dict(init_seed=cfg['init_seed'],train_seed=cfg['train_seed'],dev_seed=cfg['dev_seed'],
        steps=steps,predicted_bytes=steps*batch*256,optimizer='AdamW',lr=cfg['lr'],weight_decay=cfg['weight_decay']).items():
        require(info[key]==value,'Checkpoint training information '+key)
    source=checkpoint['source_info']
    for key,value in dict(repo_commit=metadata['repo_commit'],source_sha256=metadata['source_sha256'],data=data_info).items():
        require(source[key]==value,'Checkpoint source information '+key)
    require(len(source['adapter_materialized_parameters'])==6,'Unexpected head-storage materialization')
    for name in source['adapter_materialized_parameters']:
        require(name in state and state[name].shape[0]==4 and state[name].is_contiguous(),'Invalid materialized parameter '+name)
    del checkpoint,state
    rows=read_csv(run/'evaluation.csv')
    keys=[('initial','normal',256)]+[('final',condition,length) for length in (256,1024) for condition in CONDITIONS]
    groups={key:{} for key in keys}
    require(len(rows)==len(keys)*n,'Evaluation row count differs')
    for row in rows:
        key=(row['stage'],row['condition'],int(row['length']))
        require(key in groups,'Unexpected evaluation condition')
        window=int(row['window']); length=key[-1]
        require(0<=window<n and window not in groups[key],'Duplicate/missing window index')
        require(int(row['offset'])==int(offsets[window]),'Evaluation rows are not paired by offset')
        require(int(row['bytes'])==length and int(row['late_bytes'])==max(0,length-256),'Per-window denominator differs')
        nll,late=float(row['nll_sum']),float(row['late_nll_sum'])
        require(math.isfinite(nll) and math.isfinite(late) and nll>=0 and 0<=late<=nll,'Invalid window NLL sum')
        require(length!=256 or late==0,'Short window has late loss')
        groups[key][window]=dict(nll=nll,late_nll=late,bytes=length,late_bytes=max(0,length-256))
    require(all(len(group)==n for group in groups.values()),'Incomplete evaluation cell')
    recomputed=[]
    for key,group in groups.items():
        sums={name:sum(row[name] for row in group.values()) for name in ('nll','late_nll','bytes','late_bytes')}
        recomputed.append(dict(stage=key[0],condition=key[1],length=key[2],windows=n,bytes=sums['bytes'],
            nll_sum=sums['nll'],late_bytes=sums['late_bytes'],late_nll_sum=sums['late_nll'],
            nll_per_byte=sums['nll']/sums['bytes'],bits_per_byte=sums['nll']/sums['bytes']/math.log(2),
            late_nll_per_byte=sums['late_nll']/sums['late_bytes'] if sums['late_bytes'] else None))
    for label,saved in [('summary.json',json.loads((run/'summary.json').read_text())),('metadata.summary',metadata['summary'])]:
        indexed={(row['stage'],row['condition'],row['length']):row for row in saved}
        require(len(saved)==len(indexed)==len(recomputed),label+' cells differ')
        for row in recomputed:
            other=indexed[row['stage'],row['condition'],row['length']]
            require(set(other)==set(row),label+' fields differ')
            for field,value in row.items():
                if isinstance(value,(int,float)):
                    close(other[field],value,label+'.'+field)
                else:
                    require(other[field]==value,label+'.'+field)
    audit=dict(run=str(run),init_seed=cfg['init_seed'],status='passed',analysis_mode='smoke' if args.allow_smoke else 'full pilot',
        completed_steps=steps,predicted_training_bytes=steps*batch*256,dev_windows=n,dev_rows=len(rows),
        checkpoint_sha256=metadata['checkpoint_sha256'],source_commit=metadata['repo_commit'],
        decoded_corpus_and_window_tokens_verified=True,all_saved_training_and_eval_offsets_exclude_test=True,
        saved_sources_and_checkpoint_tensors_verified=True,raw_summary_recomputation_passed=True,
        runtime_reported_checks=dict(exact_reload=metadata['checkpoint_reload_max_abs_difference'],
            suffix_causality=metadata['trained_suffix_causality_max_abs_difference'],
            outer_parameters_unchanged=metadata['outer_parameters_unchanged_during_evaluation'],
            memory_gradient_observed=metadata['memory_gradient_observed']),
        runtime_checks_independently_rerun=False,
        sampling_reconstruction='Offset bounds/grid/content verified; CPU RNG sequence not regenerated across Torch versions. Shared saved arrays are compared across runs.',
        training_environment=environment,input_sha256={str(path):sha256(path) for path in input_paths})
    return metadata,groups,train_offsets,offsets,tokens,audit


def ratio_bootstrap(numerator,denominator,indices):
    return numerator[indices].sum(axis=1)/denominator[indices].sum(axis=1)


def analyze(run,metadata,groups,smoke):
    n=metadata['configuration']['eval_windows']
    indices=np.random.default_rng(BOOTSTRAP_SEED).integers(n,size=(DRAWS,n),dtype=np.int32)
    identity=dict(run=str(run),init_seed=metadata['configuration']['init_seed'],
        checkpoint_sha256=metadata['checkpoint_sha256'],windows=n)
    estimates,contrasts=[],[]
    arrays={}
    for key,group in groups.items():
        for region in ('full','late'):
            if region=='late' and key[-1]==256:
                continue
            num=np.array([group[index]['nll' if region=='full' else 'late_nll'] for index in range(n)])
            den=np.array([group[index]['bytes' if region=='full' else 'late_bytes'] for index in range(n)])
            boot=ratio_bootstrap(num,den,indices)
            point=float(num.sum()/den.sum())
            low,high=map(float,np.quantile(boot,[.025,.975]))
            arrays[key,region]=(num,den,boot)
            estimates.append(dict(**identity,stage=key[0],condition=key[1],length=key[2],region=region,
                bytes=int(den.sum()),nll_sum=float(num.sum()),nll_per_byte=point,
                nll_ci_low=low,nll_ci_high=high,bits_per_byte=point/math.log(2),
                bpb_ci_low=low/math.log(2),bpb_ci_high=high/math.log(2)))
    comparisons=[(('initial','normal',256),('final','normal',256),'full','initial_minus_final')]
    comparisons += [(('final',condition,length),('final','normal',length),region,condition+'_minus_normal')
        for condition in CONDITIONS[1:] for length in (256,1024)
        for region in (('full',) if length==256 else ('full','late'))]
    for left,right,region,label in comparisons:
        num_l,den_l,boot_l=arrays[left,region]; num_r,den_r,boot_r=arrays[right,region]
        np.testing.assert_array_equal(den_l,den_r)
        point=float((num_l.sum()-num_r.sum())/den_l.sum())
        low,high=map(float,np.quantile(boot_l-boot_r,[.025,.975]))
        contrasts.append(dict(**identity,comparison=label,length=left[-1],region=region,
            bytes=int(den_l.sum()),left_stage=left[0],left_condition=left[1],right_stage=right[0],right_condition=right[1],
            difference_nats_per_byte=point,ci_low_nats=low,ci_high_nats=high,
            difference_bits_per_byte=point/math.log(2),ci_low_bits=low/math.log(2),ci_high_bits=high/math.log(2)))
    improvement=next(row for row in contrasts if row['comparison']=='initial_minus_final')
    context=next(row for row in contrasts if row['comparison']=='reset_memory_at_segments_minus_normal' and row['region']=='late')
    gain_met=improvement['difference_nats_per_byte']>=.2
    context_met=context['difference_nats_per_byte']>0 and context['ci_low_nats']>0
    decision=dict(**identity,execution_pass=True,pilot_decision_applicable=not smoke,
        baseline_gain_threshold_met=gain_met,baseline_accepted=not smoke and gain_met,
        memory_use_statistical_gate_met=context_met,
        baseline_with_useful_context_accepted=not smoke and gain_met and context_met,
        initial_to_final=improvement,memory_context_contrast=context,
        role='primary candidate' if identity['init_seed']==68101 else 'sensitivity run' if identity['init_seed']==68105 else 'format-only smoke',
        criterion='Execution valid; baseline requires initial256 minus final256 >=0.2nats/byte. Useful context additionally requires positive lower95% bound for reset-minus-normal late1024.',
        caveat='Conditional development-window intervals; no pooled training-seed uncertainty or equivalence conclusion.')
    return estimates,contrasts,decision,hashlib.sha256(indices.tobytes()).hexdigest()


def main(args):
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    require(not any(out.iterdir()),'Choose an empty analysis output directory')
    try:
        design=json.loads(args.design.read_text()); validate_design(design)
        amendment=None
        if args.execution_amendment:
            amendment=json.loads(args.execution_amendment.read_text())
            require((args.execution_amendment.resolve().parent/amendment['parent_design']).resolve()==args.design.resolve(),
                    'Execution amendment does not reference the supplied base design')
        corpus,corpus_info=load_corpus(args.data)
        estimates,contrasts,decisions,audits=[],[],[],[]
        previous=None; seen=set(); bootstrap_hashes={}
        for path in args.run:
            run=path.resolve()
            meta,groups,training,dev,tokens,audit=validate_run(run,args,corpus,corpus_info,design)
            seed=meta['configuration']['init_seed']
            require(seed not in seen,'Duplicate initialization seed in combined analysis'); seen.add(seed)
            if previous is not None:
                np.testing.assert_array_equal(training,previous[0],err_msg='Training offsets differ across seeds')
                np.testing.assert_array_equal(dev,previous[1],err_msg='Dev offsets differ across seeds')
                np.testing.assert_array_equal(tokens,previous[2],err_msg='Dev byte tokens differ across seeds')
                require(meta['source_sha256']==previous[3],'Shared-run source snapshots differ')
            previous=(training,dev,tokens,meta['source_sha256'])
            cells,pairs,decision,boot_hash=analyze(run,meta,groups,args.allow_smoke)
            estimates.extend(cells); contrasts.extend(pairs); decisions.append(decision); audits.append(audit)
            bootstrap_hashes[str(run)]=boot_hash
        write_csv(out/'estimates.csv',estimates); write_csv(out/'contrasts.csv',contrasts)
        save_json(out/'decisions.json',dict(decisions=decisions,selection='Seed68101 remains primary; never select the best seed.',
            missing_planned_seeds=sorted(SEEDS-seen) if not args.allow_smoke else [],
            shared_arrays_status='verified identical across supplied runs' if len(audits)>1 else 'pending second supplied run'))
        verification=dict(status='passed',mode='format-only smoke' if args.allow_smoke else 'full pilot',runs=audits,
            local_corpus=corpus_info,design_sha256=sha256(args.design),analyzer_sha256=sha256(Path(__file__)),
            execution_amendment_path=str(args.execution_amendment.resolve()) if amendment is not None else None,
            execution_amendment_sha256=sha256(args.execution_amendment) if amendment is not None else None,
            execution_amendment=amendment,
            analysis_environment=dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__)),
            bootstrap_seed=BOOTSTRAP_SEED,bootstrap_draws=DRAWS,bootstrap_indices_sha256=bootstrap_hashes,
            bootstrap_unit='Paired whole windows; ratio of summed loss and summed predicted-byte counts. No seed pooling.',
            shared_arrays_status='verified identical across supplied runs' if len(audits)>1 else 'pending second supplied run',
            limitations=['Per-byte logits/loss vectors are not saved: raw window-sum arithmetic is verified, neural NLL is not recomputed.',
                'Reload, suffix causality, nonzero memory gradients and evaluation weight immutability are runner-reported checks, not independent inference.',
                'Offset bounds prove saved training/evaluation windows exclude test; loading/hash checking reads the complete corpus.',
                'Development windows and fixed checkpoints do not establish unseen-corpus or training-seed generalization.',
                'Zero-read and segment-reset are inference interventions, not separately trained architecture baselines.',
                'A nonpositive interval is lack of this gate evidence, not equivalence.'])
        save_json(out/'verification.json',verification)
        (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        save_json(out/'source_manifest.json',{Path(__file__).name:sha256(Path(__file__))})
        print(json.dumps(dict(status='passed',runs=len(audits),decisions=[{k:row[k] for k in
            ('init_seed','execution_pass','pilot_decision_applicable','baseline_accepted','memory_use_statistical_gate_met','baseline_with_useful_context_accepted')} for row in decisions]),indent=2))
    except Exception as error:
        save_json(out/'audit_failure.json',dict(status='failed',error=str(error),traceback=traceback.format_exc(),
            analyzer_sha256=sha256(Path(__file__))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--design',type=Path,default=Path(__file__).resolve().parent.parent/'design.json')
    parser.add_argument('--data',type=Path,default=Path(r'C:/Users/henry/Documents/Codex/clmm-replay/data/enwik8.gz'))
    parser.add_argument('--expected-commit',default='8cc20ac')
    parser.add_argument('--execution-amendment',type=Path,help='Explicit environment amendment recorded without changing the model/data/budget design')
    parser.add_argument('--allow-smoke',action='store_true',help='Validate a tiny format run; cannot accept a full pilot')
    arguments=parser.parse_args()
    torch.set_num_threads(1)
    main(arguments)
