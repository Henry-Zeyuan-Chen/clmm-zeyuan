"""Independent pressure-trace audit and paired episode-ratio bootstrap analysis."""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ('online', 'repeat', 'continuous', 'periodic')
PAIRS = (('periodic', 'repeat'), ('continuous', 'repeat'), ('periodic', 'continuous'),
         ('repeat', 'online'), ('continuous', 'online'), ('periodic', 'online'))
STRATA = ('all', 'resident', 'nonresident_selected', 'nonresident_never_selected')
CONDITIONS = {'stable': (0, 1), 'revised': (2, 3), 'old': (0, 1, 2, 3), 'recent': (4, 5, 6, 7)}
BOOTSTRAP_SEED, DRAWS = 78111, 5000
ORIGINAL_INIT_SEEDS = frozenset((48101, 48103))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar(data, name):
    return int(data[name])


def verify_trace(data, episodes, facts, metadata):
    """Regenerate observations/draws and reconstruct reservoir using Python slots."""
    nkeys, nvals = metadata['nkeys'], metadata['nvals']
    capacity, block_size = metadata['working_capacity'], metadata['snapshot_capacity']
    delays = list(map(int, data['delays']))
    prefill, maximum = facts + 2, max(delays)
    length, blocks = prefill + maximum, maximum // block_size
    for name, expected in dict(facts=facts, nkeys=nkeys, nvals=nvals, seed=metadata['eval_seed'],
                                prefill=prefill, buffer_size=capacity, block_size=block_size).items():
        assert scalar(data, name) == expected, name
    assert delays == metadata['delays']
    shapes = dict(keys=(episodes,length), values=(episodes,length), active_keys=(episodes,facts),
        final_values=(episodes,facts), reservoir_draw=(episodes,length),
        sample_slots=(episodes,blocks,block_size), sample_sources=(episodes,blocks,block_size),
        sample_value_sources=(episodes,blocks,block_size),
        working_sources_at_query=(episodes,len(delays),capacity),
        working_value_sources_at_query=(episodes,len(delays),capacity))
    for name, shape in shapes.items():
        assert data[name].shape == shape, (name, data[name].shape, shape)
    for episode in range(episodes):
        seed = metadata['eval_seed']
        stream = np.random.default_rng(np.random.SeedSequence([seed,facts,episode,0]))
        admissions = np.random.default_rng(np.random.SeedSequence([seed,facts,episode,1]))
        samples = np.random.default_rng(np.random.SeedSequence([seed,facts,episode,2]))
        active = stream.choice(nkeys, facts, replace=False)
        initial = stream.integers(nvals, size=facts)
        final = initial.copy()
        final[-2:] = (initial[-2:] + stream.integers(1,nvals,size=2)) % nvals
        positions = stream.integers(facts-4,size=maximum)
        expected_keys = np.concatenate((active,active[-2:],active[positions]))
        expected_values = np.concatenate((initial,final[-2:],final[positions]))
        for name, expected in (('keys',expected_keys),('values',expected_values),
                               ('active_keys',active),('final_values',final)):
            np.testing.assert_array_equal(data[name][episode], expected)
        keys, values = data['keys'][episode], data['values'][episode]
        slots = [None] * capacity
        query_index = {delay:index for index,delay in enumerate(delays)}
        for observation in range(length):
            if observation >= prefill and (observation-prefill) % block_size == 0:
                block = (observation-prefill)//block_size
                occupied = [index for index,entry in enumerate(slots) if entry is not None]
                chosen = samples.choice(occupied,block_size,replace=False)
                np.testing.assert_array_equal(chosen,data['sample_slots'][episode,block])
                assert len(set(map(int,chosen))) == block_size
                for rank, slot in enumerate(chosen):
                    source,value_source = slots[int(slot)]
                    assert source < observation and value_source < observation
                    assert data['sample_sources'][episode,block,rank] == source
                    assert data['sample_value_sources'][episode,block,rank] == value_source
                    assert keys[source] == keys[value_source]
                    assert values[value_source] == final[np.flatnonzero(active==keys[source])[0]]
            current_key = keys[observation]
            for slot, entry in enumerate(slots):
                if entry is not None and keys[entry[0]] == current_key:
                    slots[slot] = (entry[0],observation)
            draw = observation if observation < capacity else int(admissions.integers(observation+1))
            assert data['reservoir_draw'][episode,observation] == draw
            if draw < capacity:
                slots[draw] = (observation,observation)
            delay = observation+1-prefill
            if delay in query_index:
                qi = query_index[delay]
                np.testing.assert_array_equal(data['working_sources_at_query'][episode,qi],
                    [-1 if item is None else item[0] for item in slots])
                np.testing.assert_array_equal(data['working_value_sources_at_query'][episode,qi],
                    [-1 if item is None else item[1] for item in slots])
    return dict(facts=facts,episodes=episodes,observations_verified=episodes*length,
                sampled_pairs_verified=episodes*blocks*block_size,maximum_logical_pairs=capacity+block_size)


def reconstruct_rows(data):
    """Vectorized event bookkeeping independent of evaluator/model code."""
    keys, values = data['keys'], data['values']
    n, length = keys.shape
    facts, prefill = scalar(data,'facts'),scalar(data,'prefill')
    nkeys, block_size = scalar(data,'nkeys'),scalar(data,'block_size')
    delays = list(map(int,data['delays']))
    nd = len(delays)
    episode_index = np.arange(n)
    query_index = np.empty((nd,n,8),dtype=np.int32)
    query_index[:,:,:4] = [facts-4,facts-3,facts,facts+1]
    for di, delay in enumerate(delays):
        observed = prefill+delay
        for episode in range(n):
            excluded = set(map(int,data['active_keys'][episode,-4:]))
            latest = {int(key):index for index,key in enumerate(keys[episode,:observed]) if int(key) not in excluded}
            positions = sorted(latest.values(),reverse=True)[:4]
            assert len(positions)==4
            query_index[di,episode,4:] = positions
    query = np.take_along_axis(keys[None,:,:],query_index,axis=2)
    answer = np.take_along_axis(values[None,:,:],query_index,axis=2)
    resident_count = np.empty((nd,n,8),dtype=np.int32)
    for di in range(nd):
        working = data['working_sources_at_query'][:,di]
        resident_keys = np.take_along_axis(keys,np.maximum(working,0),axis=1)
        resident_count[di] = ((resident_keys[:,:,None] == query[di,:,None,:]) & (working[:,:,None]>=0)).sum(axis=1)
    sample_keys = np.take_along_axis(keys[:,None,:],data['sample_sources'],axis=2)
    shadow_last = np.full((n,nkeys),-1,dtype=np.int32)
    shadow_count = np.zeros_like(shadow_last)
    shadows = {}
    shadows[0] = (np.zeros((n,8),dtype=np.int32),np.full((n,8),-1,dtype=np.int32))
    for block in range(sample_keys.shape[1]):
        for rank in range(block_size):
            current = sample_keys[:,block,rank]
            shadow_last[episode_index,current] = block
            shadow_count[episode_index,current] += 1
        delay = (block+1)*block_size
        if delay in delays:
            di = delays.index(delay)
            shadows[delay] = (shadow_count[episode_index[:,None],query[di]].copy(),
                              shadow_last[episode_index[:,None],query[di]].copy())
    common = dict(key=query,target_value=answer,last_observation_index=query_index,
                  working_resident_copies=resident_count,working_resident=resident_count>0,
                  shadow_selection_count=np.stack([shadows[d][0] for d in delays]),
                  last_shadow_selection_block=np.stack([shadows[d][1] for d in delays]))
    common['shadow_ever_selected_for_replay'] = common['shadow_selection_count']>0
    arm_values = {}
    for arm in ARMS:
        last_any = np.full((n,nkeys),-1,dtype=np.int32)
        last_aux,last_replay,last_source = last_any.copy(),last_any.copy(),last_any.copy()
        aux_count,replay_count = np.zeros_like(last_any),np.zeros_like(last_any)
        writes, observed, auxiliary = 0,0,0
        snapshots = []
        def write(kind, source, value_source):
            nonlocal writes, observed, auxiliary
            numeric_keys = keys[episode_index,source]
            last_any[episode_index,numeric_keys] = writes
            last_source[episode_index,numeric_keys] = value_source
            if kind != 'observation':
                last_aux[episode_index,numeric_keys] = writes
                aux_count[episode_index,numeric_keys] += 1
                auxiliary += 1
            if kind == 'replay':
                last_replay[episode_index,numeric_keys] = writes
                replay_count[episode_index,numeric_keys] += 1
            observed += kind == 'observation'
            writes += 1
        def snapshot(delay):
            di = delays.index(delay)
            q = query[di]
            fields = dict(last_any_write_index=last_any,last_aux_write_index=last_aux,
                last_replay_write_index=last_replay,last_write_value_source_index=last_source,
                auxiliary_writes_for_key=aux_count,replay_writes_for_key=replay_count)
            result={name:array[episode_index[:,None],q].copy() for name,array in fields.items()}
            result.update(total_observations=np.full((n,8),observed),
                          total_memory_writes=np.full((n,8),writes),
                          total_aux_writes=np.full((n,8),auxiliary))
            result['future_observations']=observed-1-query_index[di]
            result['future_memory_writes']=writes-1-result['last_any_write_index']
            result['ever_aux_written']=result['last_aux_write_index']>=0
            result['ever_replayed']=result['last_replay_write_index']>=0
            result['future_writes_since_aux']=np.where(result['ever_aux_written'],writes-1-result['last_aux_write_index'],-1)
            result['future_writes_since_replay']=np.where(result['ever_replayed'],writes-1-result['last_replay_write_index'],-1)
            snapshots.append(result)
        for index in range(prefill):
            write('observation',index,index)
        snapshot(0)
        for block in range((length-prefill)//block_size):
            start=prefill+block*block_size
            if arm=='periodic':
                for rank in range(block_size):
                    write('observation',start+rank,start+rank)
                for rank in range(block_size):
                    write('replay',data['sample_sources'][:,block,rank],data['sample_value_sources'][:,block,rank])
            else:
                for rank in range(block_size):
                    write('observation',start+rank,start+rank)
                    if arm=='repeat':
                        write('repeat',start+rank,start+rank)
                    elif arm=='continuous':
                        write('replay',data['sample_sources'][:,block,rank],data['sample_value_sources'][:,block,rank])
            delay=(block+1)*block_size
            if delay in delays:
                snapshot(delay)
        arm_values[arm]={name:np.stack([row[name] for row in snapshots]) for name in snapshots[0]}
    return common,arm_values


def parse_bool(value):
    assert value in ('True','False')
    return value=='True'


def load_training_amendment(path, design_path, design):
    """Validate an explicit extension without changing the base evaluation design."""
    amendment=json.loads(path.read_text())
    assert (path.resolve().parent/amendment['parent_design']).resolve()==design_path.resolve(), 'Amendment parent does not identify the base design'
    revised=amendment['revised_training']
    original={tuple(entry) for entry in design['training']['seeds']}
    proposed={tuple(entry) for entry in revised['seeds']}
    assert original.issubset(proposed), 'Amendment may not remove original seeds'
    assert len(proposed)==len(revised['seeds'])==len({entry[0] for entry in proposed})
    assert all(len(entry)==3 for entry in proposed)
    for name in ('steps','batch_size','nkeys','nvals','dim','depth'):
        assert revised[name]==design['training'][name], name
    assert revised['heldout_episode_length']==design['training']['episode_length']
    assert revised['max_inner_lr']==0.01 and revised['curriculum']=='16:100,64:100,256:200'
    return amendment


def verify_run(run, design, stage, training_amendment=None, expected_evaluation=None):
    metadata=json.loads((run/'metadata.json').read_text())
    assert metadata['status']=='complete','Run must finish before analysis'
    assert tuple(metadata['arms'])==ARMS
    assert metadata['working_capacity']==32 and metadata['snapshot_capacity']==8 and metadata['logical_pair_capacity']==40
    assert metadata['nkeys']==256 and metadata['nvals']==8
    protocol=dict(facts=design['evaluation']['facts'] if stage=='grid' else [64],
        delays=design['evaluation']['delays'] if stage=='grid' else [0,256],
        episodes=design['evaluation']['episodes_per_fact_count_per_checkpoint'] if stage=='grid' else 512,
        eval_seed=design['evaluation']['seed'] if stage=='grid' else 58121,
        aux_decay_scale=0)
    if expected_evaluation is not None:
        assert set(expected_evaluation)==set(protocol), 'A separate diagnostic must declare the full expected evaluation'
        protocol=dict(expected_evaluation)
    expected_facts,expected_delays=protocol['facts'],protocol['delays']
    expected_n,expected_seed=protocol['episodes'],protocol['eval_seed']
    assert metadata['aux_decay_scale']==protocol['aux_decay_scale']
    assert metadata['facts']==expected_facts and metadata['delays']==expected_delays
    assert metadata['episodes']==expected_n and metadata['eval_seed']==expected_seed
    for original,digest in metadata['source_sha256'].items():
        relative=original.replace('\\','/').removeprefix(metadata['repo'].replace('\\','/').rstrip('/')+'/')
        assert relative!=original.replace('\\','/')
        assert sha256(run/'sources'/relative)==digest
    n=metadata['episodes']
    data_by_k,expected,trace_reports={},{},[]
    for dataset in metadata['datasets']:
        facts=dataset['facts']
        path=run/f'streams_K{facts}.npz'
        assert sha256(path)==dataset['sha256']
        with np.load(path,allow_pickle=False) as loaded:
            data={name:loaded[name] for name in loaded.files}
        trace_reports.append(verify_trace(data,n,facts,metadata))
        data_by_k[facts]=data
        expected[facts]=reconstruct_rows(data)
    assert set(expected)==set(metadata['facts'])
    checkpoints={item['index']:item for item in metadata['checkpoints']}
    seed_entries=design['training']['seeds'] if training_amendment is None else training_amendment['revised_training']['seeds']
    planned_seeds={(entry[0],entry[1]) for entry in seed_entries}
    for item in checkpoints.values():
        assert item['steps']==design['training']['steps'] and item['checkpoint_parameters_and_buffers_unchanged']
        assert (item['init_seed'],item['train_seed']) in planned_seeds
        for name in ('nkeys','nvals','dim','depth','expansion_factor'):
            assert item['config'][name]==design['training'][name],name
        if training_amendment is not None:
            assert item['config']['max_inner_lr']==training_amendment['revised_training']['max_inner_lr']
    outcomes={ci:{facts:np.full((4,len(metadata['delays']),n,8),-1,dtype=np.int8)
                  for facts in metadata['facts']} for ci in checkpoints}
    predictions={ci:{facts:np.full_like(array,-1) for facts,array in cells.items()}
                 for ci,cells in outcomes.items()}
    counts=defaultdict(lambda:dict(n=0,correct=0,stale=0))
    boolean_fields={'working_resident','shadow_ever_selected_for_replay','ever_aux_written','ever_replayed'}
    optional_fields={'future_writes_since_aux','future_writes_since_replay'}
    rows_checked=0
    with (run/'predictions.csv').open(newline='',encoding='utf8') as handle:
        for row in csv.DictReader(handle):
            ci,facts,delay,episode,rank=(int(row[name]) for name in ('checkpoint_index','facts','delay','episode','query_rank'))
            arm,condition=row['arm'],row['condition']
            col=rank+({'stable':0,'revised':2,'recent':4}[condition])
            assert 0<=rank<(4 if condition=='recent' else 2)
            ai,di=ARMS.index(arm),metadata['delays'].index(delay)
            checkpoint=checkpoints[ci]
            assert row['checkpoint_sha256']==checkpoint['sha256']
            assert int(row['init_seed'])==checkpoint['init_seed'] and int(row['train_seed'])==checkpoint['train_seed']
            assert int(row['eval_seed'])==metadata['eval_seed'] and float(row['aux_decay_scale'])==protocol['aux_decay_scale']
            common,specific=expected[facts]
            for fields in (common,specific[arm]):
                for name,array in fields.items():
                    value=array[di,episode,col]
                    if name in boolean_fields:
                        assert parse_bool(row[name])==bool(value),(facts,arm,delay,episode,name)
                    elif name in optional_fields and value<0:
                        assert row[name]==''
                    else:
                        assert int(row[name])==int(value),(facts,arm,delay,episode,name)
            for field in ('snapshot_active_pairs','pending_auxiliary_writes','local_attention_tokens'):
                assert int(row[field])==0
            assert row['snapshot_resident']=='False'
            multiplicity='' if condition=='recent' else ('1' if condition=='stable' else '2')
            assert row['initial_event_multiplicity']==multiplicity
            prediction=int(row['prediction'])
            assert 0<=prediction<metadata['nvals']
            correct=int(prediction==int(row['target_value']))
            assert int(row['correct'])==correct
            stale=0
            if condition=='revised':
                old=int(data_by_k[facts]['values'][episode,facts-2+rank])
                assert int(row['old_value'])==old and old!=int(row['target_value'])
                stale=int(prediction==old)
                assert int(row['stale_old'])==stale
            else:
                assert row['old_value']==row['stale_old']==''
            assert outcomes[ci][facts][ai,di,episode,col]==-1,'Duplicate raw row'
            outcomes[ci][facts][ai,di,episode,col]=correct
            predictions[ci][facts][ai,di,episode,col]=prediction
            cell=counts[ci,facts,arm,delay,condition]
            cell['n']+=1
            cell['correct']+=correct
            cell['stale']+=stale
            rows_checked+=1
    assert all(np.all(array>=0) for cells in outcomes.values() for array in cells.values())
    for cells in predictions.values():
        for array in cells.values():
            assert all(np.array_equal(array[0,0],array[ai,0]) for ai in range(1,4)), 'Common prefill predictions differ'
    summary_count=0
    with (run/'summary.csv').open(newline='',encoding='utf8') as handle:
        for row in csv.DictReader(handle):
            key=(int(row['checkpoint_index']),int(row['facts']),row['arm'],int(row['delay']),row['condition'])
            cell=counts[key]
            assert int(row['n'])==cell['n'] and int(row['correct'])==cell['correct']
            assert float(row['accuracy'])==cell['correct']/cell['n']
            assert row['stale_old']==(str(cell['stale']) if row['condition']=='revised' else '')
            summary_count+=1
    assert summary_count==len(counts)
    for entry in metadata['runtime_by_arm']:
        total_observations=n*(entry['facts']+2+max(metadata['delays']))
        total_aux=0 if entry['arm']=='online' else n*max(metadata['delays'])
        assert entry['observation_writes']==total_observations
        assert entry['auxiliary_writes']==total_aux
        assert entry['total_memory_writes']==total_observations+total_aux
    return metadata,outcomes,expected,trace_reports,rows_checked


def bootstrap_ratios(numerator,denominator,indices):
    """Numerator [arms,episode], shared denominator [episode]."""
    boot_den=denominator[indices].sum(axis=1)
    valid=boot_den>0
    result=np.full((numerator.shape[0],len(indices)),np.nan)
    if valid.any():
        boot_num=numerator[:,indices[valid]].sum(axis=2)
        result[:,valid]=100*boot_num/boot_den[valid]
    return result,valid


def interval(values):
    finite=values[np.isfinite(values)]
    return (None,None) if len(finite)==0 else tuple(map(float,np.quantile(finite,[.025,.975])))


def analyze_results(run,metadata,outcomes,expected):
    n=metadata['episodes']
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    indices=rng.integers(n,size=(DRAWS,n),dtype=np.int32)
    cells,contrasts,episode_rows=[] ,[], []
    for ci,by_k in outcomes.items():
        checkpoint=next(item for item in metadata['checkpoints'] if item['index']==ci)
        for facts,correct in by_k.items():
            common,_=expected[facts]
            for di,delay in enumerate(metadata['delays']):
                resident=common['working_resident'][di]
                selected=common['shadow_ever_selected_for_replay'][di]
                strata=dict(all=np.ones_like(resident,dtype=bool),resident=resident,
                    nonresident_selected=(~resident)&selected,
                    nonresident_never_selected=(~resident)&(~selected))
                for condition,columns in CONDITIONS.items():
                    for stratum,eligible_all in strata.items():
                        eligible=eligible_all[:,columns]
                        denominator=eligible.sum(axis=1)
                        numerator=(correct[:,di][:,:,columns]*eligible[None,:,:]).sum(axis=2)
                        total=int(denominator.sum())
                        eligible_episodes=int((denominator>0).sum())
                        boot,valid=bootstrap_ratios(numerator,denominator,indices)
                        point=100*numerator.sum(axis=1)/total if total else np.full(4,np.nan)
                        shared=dict(run=str(run),checkpoint_index=ci,init_seed=checkpoint['init_seed'],
                            train_seed=checkpoint['train_seed'],facts=facts,delay=delay,condition=condition,
                            stratum=stratum,episodes=n,eligible_episodes=eligible_episodes,
                            eligible_queries=total,valid_bootstrap_draws=int(valid.sum()),
                            empty_bootstrap_draws=int((~valid).sum()))
                        for ai,arm in enumerate(ARMS):
                            low,high=interval(boot[ai])
                            cells.append(dict(**shared,arm=arm,correct=int(numerator[ai].sum()),
                                accuracy_pct=float(point[ai]) if total else None,ci_low_pct=low,ci_high_pct=high))
                        for left,right in PAIRS:
                            li,ri=ARMS.index(left),ARMS.index(right)
                            low,high=interval(boot[li]-boot[ri])
                            contrasts.append(dict(**shared,left=left,right=right,
                                difference_pp=float(point[li]-point[ri]) if total else None,
                                ci_low_pp=low,ci_high_pp=high))
                        if facts==64 and delay==256 and ((condition=='old' and stratum=='nonresident_selected') or (condition=='recent' and stratum=='all')):
                            for episode in range(n):
                                episode_rows.append(dict(run=str(run),checkpoint_index=ci,init_seed=checkpoint['init_seed'],
                                    condition=condition,stratum=stratum,episode=episode,eligible_queries=int(denominator[episode]),
                                    **{arm+'_correct':int(numerator[ai,episode]) for ai,arm in enumerate(ARMS)}))
    return cells,contrasts,episode_rows,hashlib.sha256(indices.tobytes()).hexdigest()


def continuation(contrasts,stage):
    records=[]
    for key in sorted({(row['run'],row['checkpoint_index'],row['init_seed']) for row in contrasts}):
        selected=[row for row in contrasts if (row['run'],row['checkpoint_index'],row['init_seed'])==key and row['facts']==64 and row['delay']==256]
        primary=next(row for row in selected if row['condition']=='old' and row['stratum']=='nonresident_selected' and row['left']=='periodic' and row['right']=='repeat')
        recent=next(row for row in selected if row['condition']=='recent' and row['stratum']=='all' and row['left']=='periodic' and row['right']=='repeat')
        anchors=[row for row in selected if row['left']=='periodic' and row['right']=='online' and
                 ((row['condition']=='old' and row['stratum']=='nonresident_selected') or (row['condition']=='recent' and row['stratum']=='all'))]
        present=primary['difference_pp'] is not None and primary['ci_low_pp'] is not None
        passed=bool(present and primary['difference_pp']>=5 and primary['ci_low_pp']>0 and recent['difference_pp']>=-2)
        records.append(dict(run=key[0],checkpoint_index=key[1],init_seed=key[2],primary=primary,recent=recent,
                            online_anchors=anchors,meets_prespecified_signal=passed,
                            eligible_for_original_continuation=key[2] in ORIGINAL_INIT_SEEDS))
    originals=[record for record in records if record['init_seed'] in ORIGINAL_INIT_SEEDS]
    original_set_complete=(len(originals)==2 and {record['init_seed'] for record in originals}==ORIGINAL_INIT_SEEDS)
    return dict(stage=stage,checkpoints=records,
        original_init_seeds=sorted(ORIGINAL_INIT_SEEDS),
        original_checkpoint_set_complete=original_set_complete,
        supplementary_checkpoint_init_seeds=sorted({record['init_seed'] for record in records}-ORIGINAL_INIT_SEEDS),
        both_checkpoint_signal=original_set_complete and all(record['meets_prespecified_signal'] for record in originals),
        fresh_confirmation_seed=58121,fresh_confirmation_episodes=512,
        criterion='Primary gain >=5pp, lower paired95% bound >0, and recent-key point loss <=2pp against repeat in both fixed checkpoints.',
        interpretation='A gain over repeat measures the use of extra updates. Online-anchor contrasts are reported separately.')


def main(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'verification.json').exists() or (out/'contrasts.csv').exists():
        raise FileExistsError('Choose a fresh analysis directory')
    design=json.loads(args.design.read_text())
    assert design['analysis']['primary_cell']==dict(facts=64,delay=256)
    assert {entry[0] for entry in design['training']['seeds']}==ORIGINAL_INIT_SEEDS, 'Original two-seed design must remain intact'
    amendment_path=getattr(args,'training_amendment',None)
    amendment=load_training_amendment(amendment_path,args.design,design) if amendment_path else None
    all_cells,all_contrasts,all_episodes=[],[],[]
    audits=[]
    for path in args.run:
        run=path.resolve()
        metadata,outcomes,expected,traces,checked=verify_run(run,design,args.stage,training_amendment=amendment)
        assert metadata['design_sha256']==sha256(args.design), 'Run must identify the unchanged base design'
        cells,contrasts,episode_rows,bootstrap_hash=analyze_results(run,metadata,outcomes,expected)
        all_cells.extend(cells);all_contrasts.extend(contrasts);all_episodes.extend(episode_rows)
        audits.append(dict(run=str(run),status='passed',prediction_rows_verified=checked,trace_audits=traces,
            bootstrap_indices_sha256=bootstrap_hash,repo_commit=metadata['repo_commit'],checkpoints=metadata['checkpoints'],
            input_sha256={str(file):sha256(file) for file in [run/'metadata.json',run/'predictions.csv',run/'summary.csv',*run.glob('streams_*.npz')]}))
    write_csv(out/'counts.csv',all_cells)
    write_csv(out/'contrasts.csv',all_contrasts)
    write_csv(out/'primary_episode_counts.csv',all_episodes)
    decision=continuation(all_contrasts,args.stage)
    (out/'primary_decision.json').write_text(json.dumps(decision,indent=2)+'\n',encoding='utf8')
    verification=dict(status='passed',runs=audits,design_sha256=sha256(args.design),
        training_amendment_path=str(amendment_path.resolve()) if amendment_path else None,
        training_amendment_sha256=sha256(amendment_path) if amendment_path else None,
        analysis_script_sha256=sha256(Path(__file__)),numpy=np.__version__,bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_draws=DRAWS,bootstrap_unit='Paired episode; resampled numerator sum divided by eligible-query denominator sum.',
        empty_denominator_handling='Point estimate undefined when no eligible queries. Bootstrap draws with zero eligibility remain undefined; counts of valid/empty draws are saved.',
        interval='Percentile95%, conditional on each checkpoint, unadjusted exploratory comparisons.',
        limitations=['Logical reservoir capacity40; evaluator traces are outside the model query path.',
                    'Residency and prior selection are descriptive shared strata.',
                    'Stable/revised initial ticket multiplicities differ.',
                    'Shared episodes are not pooled across checkpoints.',
                    'Trace and metric reconstruction does not rerun neural inference.'])
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'passed','counts':len(all_cells),'contrasts':len(all_contrasts),
                      'both_checkpoint_signal':decision['both_checkpoint_signal']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--design',type=Path,default=Path(__file__).resolve().parent.parent/'design.json')
    parser.add_argument('--stage',choices=('grid','confirmation'),default='grid')
    parser.add_argument('--training-amendment',type=Path,help='Explicit seed extension; original two-seed continuation remains fixed')
    main(parser.parse_args())
