"""Closed-codebook compositions and causal replay plans for memory controls."""
from collections import OrderedDict
import hashlib
import json
import numpy as np

NKEYS,NVALUES=64,16


def signature(keys,values):
    return hashlib.sha256(bytes(keys)+b'|'+bytes(values)).hexdigest()


def episode(seed,facts,delay=0,revisions=0,pressure=False):
    if not (2<=facts<=32 and 0<=revisions<=facts and 0<=delay<=64):
        raise ValueError('Unsupported bounded episode geometry')
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),facts,revisions]))
    permutation=rng.permutation(NKEYS).tolist()
    initial=permutation[:facts]
    if facts<=8:
        initial_values=rng.choice(NVALUES,facts,replace=False).tolist()
    else:
        initial_values=rng.integers(NVALUES,size=facts).tolist()
        initial_values[:4]=rng.choice(NVALUES,4,replace=False).tolist()
    keys,values=list(initial),list(initial_values)
    latest=dict(zip(initial,initial_values));old={}
    indices=list(range(2,2+revisions)) if pressure else list(range(revisions))
    if any(i>=facts for i in indices):raise ValueError('Revision indices exceed facts')
    for i in indices:
        key=initial[i];previous=latest[key]
        # Keep strict query values distinct (all K for short, first4 for pressure).
        protected=initial if facts<=8 else initial[:4]
        banned={latest[k] for k in protected}|{previous}
        alternatives=[v for v in range(NVALUES) if v not in banned]
        if not alternatives:raise ValueError('No strict revision value available')
        value=int(rng.choice(alternatives));old[key]=previous;latest[key]=value
        keys.append(key);values.append(value)
    boundary=len(keys)
    drng=np.random.default_rng(np.random.SeedSequence([int(seed),facts,991]))
    table=drng.integers(NVALUES,size=NKEYS).tolist()
    if delay:
        distractors=drng.choice(permutation[facts:],delay,replace=True).tolist()
        for key in distractors:keys.append(key);values.append(table[key])
    positions=[0,1,2,3] if pressure else list(range(facts))
    queries=[dict(key=initial[i],target=latest[initial[i]],old=old.get(initial[i],-1),
                  category='revised' if initial[i] in old else 'stable') for i in positions]
    if pressure and delay:queries.append(dict(key=keys[-1],target=values[-1],old=-1,category='recent'))
    return dict(seed=int(seed),facts=facts,delay=delay,revisions=revisions,pressure=bool(pressure),
        keys=keys,values=values,boundary=boundary,queries=queries,signature=signature(keys,values))


def training_data(steps=1200,batch=32,seed=90302):
    if min(steps,batch)<1:raise ValueError('Positive counts required')
    keys=np.full((steps,batch,14),255,dtype=np.uint8)
    values=keys.copy();queries=np.full((steps,batch,8),255,dtype=np.uint8);targets=queries.copy()
    lengths=np.zeros(steps,dtype=np.int16);counts=lengths.copy();specs=[];seen=set()
    for step in range(steps):
        rng=np.random.default_rng(np.random.SeedSequence([seed,step]))
        facts=int(rng.choice([2,4,8]));rev=max(1,facts//4) if rng.integers(2) else 0
        delay=int(rng.choice([0,4]));seeds=rng.integers(1,1_000_000_000,size=batch)
        for b,s in enumerate(seeds):
            ep=episode(int(s),facts,delay,rev)
            length=len(ep['keys']);keys[step,b,:length]=ep['keys'];values[step,b,:length]=ep['values']
            queries[step,b,:facts]=[q['key'] for q in ep['queries']]
            targets[step,b,:facts]=[q['target'] for q in ep['queries']]
            seen.add(ep['signature'])
        lengths[step]=length;counts[step]=facts
        specs.append(dict(step=step+1,facts=facts,delay=delay,revisions=rev,seeds=seeds.tolist()))
    return dict(keys=keys,values=values,queries=queries,targets=targets,lengths=lengths,counts=counts),specs,seen


def evaluation_data(seed,episodes=128,pressure_episodes=32,seen=(),suite='gate'):
    excluded=set(seen);result={};specs=[('acquire2',2,0,0,False),('acquire4',4,0,0,False),('revise2',2,0,1,False)]
    if suite=='full':
        specs.extend((f'pressure_k{k}_d{d}',k,d,2,True) for k in (8,16,32) for d in (0,16,64))
    for index,(name,k,d,r,pressure) in enumerate(specs):
        # Within each K, delay cells share the identical online prefix.
        rng=np.random.default_rng(np.random.SeedSequence([seed,k,r,500 if pressure else index]))
        count=pressure_episodes if pressure else episodes;items=[];attempts=0
        while len(items)<count:
            attempts+=1
            if attempts>10000:raise RuntimeError('Evaluation rejection cap exceeded')
            candidate=int(rng.integers(2_000_000_000,3_000_000_000))
            ep=episode(candidate,k,d,r,pressure)
            signatures={episode(candidate,k,x,r,True)['signature'] for x in (0,16,64)} if pressure else {ep['signature']}
            if not signatures.intersection(excluded):items.append(ep)
        result[name]=items
    return result


def replay_plan(ep,arm,block=4,capacity=8):
    """Shared online-only prefill; frozen samples from latest-per-key FIFO."""
    if arm not in ('online','repeat','continuous','periodic'):raise ValueError('Unknown arm')
    boundary=ep['boundary'];events=list(zip(ep['keys'],ep['values']));working=OrderedDict()
    plan=[]
    def arrive(i):
        k,v=events[i]
        if k in working:del working[k]
        working[k]=i
        while len(working)>capacity:working.popitem(last=False)
    def item(i,kind,at):
        k,v=events[i];return dict(source=i,key=k,value=v,kind=kind,after_online=at)
    for i in range(boundary):plan.append(item(i,'online',i));arrive(i)
    rng=np.random.default_rng(np.random.SeedSequence([ep['seed'],77]))
    selected=[]
    for start in range(boundary,len(events),block):
        stop=min(start+block,len(events));available=list(working.values())
        if not available:raise ValueError('Pressure replay needs nonempty prefill')
        snapshot=[int(rng.choice(available)) for _ in range(stop-start)];selected.extend(snapshot)
        for offset,i in enumerate(range(start,stop)):
            plan.append(item(i,'online',i));arrive(i)
            if arm=='continuous':plan.append(item(snapshot[offset],'extra',i))
            elif arm=='repeat':plan.append(item(i,'extra',i))
        if arm=='periodic':plan.extend(item(i,'extra',stop-1) for i in snapshot)
    expected=boundary+ep['delay']*(1 if arm=='online' else 2)
    if len(plan)!=expected:raise AssertionError('Replay budget mismatch')
    return dict(events=plan,selected=selected,boundary=boundary,writes=len(plan),
        final_buffer_keys=list(working),buffer_capacity=capacity,snapshot_pairs=block,
        interpretation='Logical pair budget; evaluator trace is not available to read')
