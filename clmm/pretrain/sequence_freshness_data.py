"""Fresh byte-association curriculum and bounded literal-payload replay plans."""
from dataclasses import replace
import hashlib
import json
import numpy as np
from .fact_streams import Episode,make_episode,render_query,VALUES
from .sequence_replay import SequenceReplay

ARMS=('online','repeat','continuous','periodic','continuous_bounded','periodic_bounded')

def signature(records):
    pairs=[(r.key,r.value) if hasattr(r,'key') else (r['key'],r['value']) for r in records]
    return hashlib.sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest()

def collect_histories(value):
    """Conservative ordered string-key/value histories from old saved JSON."""
    found=set()
    if isinstance(value,list):
        if value and all(isinstance(r,dict) and isinstance(r.get('key'),str) and len(r['key'])==3
                         and r['key'].isascii() and r['key'].isdigit() and r.get('value') in tuple(VALUES)
                         for r in value):found.add(signature(value))
        for child in value:found.update(collect_histories(child))
    elif isinstance(value,dict):
        for child in value.values():found.update(collect_histories(child))
    return found

def distinct_episode(seed,facts=2,revision=False,split='train',family='seen'):
    for attempt in range(1000):
        ep=make_episode(int(seed)+attempt*100_000_000_019,facts,2 if revision else 0,split,family,
                        1. if revision else 0.,include_recent=False)
        latest={};targets={r.key for r in ep.events[:facts]};valid=True
        for r in ep.events:
            if r.key in targets:latest[r.key]=r.value
            if len(latest)==facts and len(set(latest.values()))!=facts:valid=False
        if not valid:continue
        qi=ep.queries[0].template_index
        return replace(ep,queries=tuple(replace(q,template_index=qi) for q in ep.queries))
    raise RuntimeError('Distinct current target values rejection exhausted')

def retemplate(ep,family):
    return replace(ep,template_family=family,
        events=tuple(replace(r,template_family=family) for r in ep.events),
        queries=tuple(replace(q,template_family=family) for q in ep.queries))

def swap_assignments(ep):
    if len(ep.events)!=2 or len(ep.queries)!=2:raise ValueError('Swap requires plain two-fact acquisition')
    a,b=ep.events;records=(replace(a,value=b.value),replace(b,value=a.value))
    return replace(ep,events=records,queries=tuple(replace(q,target=records[i].value) for i,q in enumerate(ep.queries)))

def replay_context(ep,arm):
    """Only literal observations enter the live policy; queries are not passed."""
    if arm not in ARMS:raise ValueError('Unknown replay arm')
    policy=SequenceReplay(arm.removesuffix('_bounded'),ep.seed,
                          freshness='bounded' if arm.endswith('_bounded') else 'literal')
    emitted=[];rest=transient=0
    for i,record in enumerate(ep.events):
        transient=max(transient,policy.logical_counts()['raw_payload_records']+1)
        emitted.extend(policy.prime(record.encoded) if i<min(2,len(ep.events)) else policy.observe(record.encoded))
        rest=max(rest,policy.logical_counts()['raw_payload_records'])
    emitted.extend(policy.finish())
    latest={};stale=0;rows=[]
    for event in emitted:
        row={k:v for k,v in event.items() if k!='record_bytes'}
        source=ep.events[event['source']]
        assert event['source']<=event['after_online'] and source.encoded==event['record_bytes']
        if event['kind']=='online':latest[event['key']]=event['value']
        row['stale']=event['kind']=='extra' and event['value']!=latest[event['key']]
        stale+=row['stale'];row['record_hex']=event['record_bytes'].hex();rows.append(row)
    raw=b''.join(event['record_bytes'] for event in emitted)
    plan=dict(arm=arm,events=rows,prime_count=min(2,len(ep.events)),real_records=len(ep.events),
              total_records=len(emitted),extra_records=len(emitted)-len(ep.events),stale_extras=stale,
              peak_retained_raw_records=rest,peak_retained_raw_bytes=rest*32,
              transition_raw_record_envelope=transient,transition_raw_byte_envelope=transient*32,
              final_budget=policy.budget(),context_sha256=hashlib.sha256(raw).hexdigest())
    assert rest<=12 and transient<=13
    return raw,plan

def examples(ep,arm='online'):
    context,plan=replay_context(ep,arm)
    queries=ep.queries if len(ep.queries)==2 else (ep.queries[0],ep.queries[0])
    return [(context+render_query(q),q.answer_byte) for q in queries],plan

def training_worlds(steps=2000,contexts=8,seed=91302):
    records=[]
    for step in range(steps):
        fraction=step/steps
        stage='k1' if fraction<.128 else 'k2' if fraction<.64 else ('revision' if (step-int(.64*steps))%4==3 else 'k2')
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,step])))
        eps=[distinct_episode(s,1 if stage=='k1' else 2,stage=='revision') for s in rng.integers(4_000_000_000,5_000_000_000,size=contexts)]
        records.append(dict(step=step+1,stage=stage,episodes=[ep.to_dict() for ep in eps]))
    return records

def training_arrays(worlds):
    steps=len(worlds);batch=len(worlds[0]['episodes'])*2
    arrays={arm:np.zeros((steps,batch,351),dtype=np.uint8) for arm in ('periodic','periodic_bounded')}
    targets=np.empty((steps,batch),dtype=np.uint8);lengths=np.empty(steps,dtype=np.int16)
    for i,world in enumerate(worlds):
        for arm in arrays:
            samples=[sample for row in world['episodes'] for sample in examples(Episode.from_dict(row),arm)[0]]
            n=len(samples[0][0]);assert all(len(x)==n for x,y in samples)
            arrays[arm][i,:,:n]=np.array([list(x) for x,y in samples],dtype=np.uint8)
            y=np.array([y for x,y in samples],dtype=np.uint8)
            if arm=='periodic':targets[i]=y;lengths[i]=n
            else:assert np.array_equal(targets[i],y) and lengths[i]==n
    return {**arrays,'targets':targets,'lengths':lengths}

def evaluation_worlds(excluded,seed=91323,n=128,revision_n=64):
    seen=set(excluded);rng=np.random.Generator(np.random.PCG64(seed));groups={}
    for group,split,rev,count in [('acquire_train_seen','train',False,n),('acquire_dev_seen','dev',False,n),('revision_dev_seen','dev',True,revision_n)]:
        accepted=[]
        while len(accepted)<count:
            ep=distinct_episode(rng.integers(6_000_000_000,7_000_000_000),2,rev,split)
            variants=[ep,swap_assignments(ep)] if group=='acquire_train_seen' else [ep]
            ids={signature(v.events) for v in variants}
            if ids&seen:continue
            seen.update(ids);accepted.append(ep)
        groups[group]=accepted
    groups['acquire_train_swapped']=[swap_assignments(ep) for ep in groups['acquire_train_seen']]
    groups['acquire_dev_heldout']=[retemplate(ep,'heldout') for ep in groups['acquire_dev_seen']]
    groups['revision_dev_heldout']=[retemplate(ep,'heldout') for ep in groups['revision_dev_seen']]
    return {name:[ep.to_dict() for ep in eps] for name,eps in groups.items()}
