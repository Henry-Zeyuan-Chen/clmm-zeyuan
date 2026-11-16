"""Independent record/query template interventions on frozen K2 histories.

Episode's uniform-family invariants remain intact. Only the visible32-byte
record and31-byte query blocks are independently rendered; labels, key/values,
query identity and current-position tags are preserved across format cells.
"""
from dataclasses import replace
import numpy as np
from .fact_streams import Episode
from .record_selection import (ENCODINGS,VARIANTS,make_codec,make_world,
    variant_records,variant_signatures,render_query_prefix)

TASKS=('select','decode','value')
CELLS={'F0Q0':('seen','seen'),'F0Q1':('seen','heldout'),
       'F1Q0':('heldout','seen'),'F1Q1':('heldout','heldout')}
CONTROLS=('no_facts','cue_erased')


def render(world,codec,task,encoding,cell,variant='original',condition='normal'):
    if task not in TASKS or encoding not in ENCODINGS or cell not in CELLS:
        raise ValueError('Unsupported task, encoding or format cell')
    if condition not in ('normal',*CONTROLS):raise ValueError('Unsupported condition')
    ep=Episode.from_dict(world) if isinstance(world,dict) else world
    records=variant_records(ep,variant)
    fact_family,query_family=CELLS[cell]
    blocks=[]
    for slot,record in enumerate(records):
        record=replace(record,template_family=fact_family)
        raw=record.encoded
        assert len(raw)==32 and raw[:3]==b'   '
        raw=bytes((91,48+slot,93))+raw[3:]
        blocks.append(codec.transform(raw,record.key,encoding))
    context=b''.join(blocks) if condition!='no_facts' else b' '*64
    rows=[]
    for rank,query in enumerate(ep.queries):
        slot=next(i for i,r in enumerate(records) if r.key==query.key)
        query=replace(query,template_family=query_family)
        prefix=context+render_query_prefix(query,codec,task,encoding,slot=slot,
            key_erased=condition=='cue_erased' and task!='decode',
            index_erased=condition=='cue_erased' and task=='decode')
        target=48+slot if task=='select' else ord(records[slot].value)
        assert len(prefix)==95
        rows.append(dict(prefix_hex=prefix.hex(),target=target,rank=rank,key=query.key,slot=slot,
            task=task,encoding=encoding,cell=cell,fact_family=fact_family,query_family=query_family,
            variant=variant,condition=condition,input_bytes=95))
    return rows


def fresh_worlds(excluded,*,seed=91823,n_dev=128,n_train=32,codec=None):
    if any(type(n) is not int or n<1 for n in (n_dev,n_train)):raise ValueError('Positive group sizes required')
    codec=make_codec() if codec is None else codec
    rng=np.random.Generator(np.random.PCG64(seed));seen=set(excluded);groups={}
    for split,n in [('train',n_train),('dev',n_dev)]:
        worlds=[]
        for _ in range(n):
            for attempt in range(100000):
                ep=make_world(int(rng.integers(11000000000,12000000000)),split=split,codec=codec)
                signatures=variant_signatures(ep)
                if signatures.isdisjoint(seen):
                    seen.update(signatures);worlds.append(ep.to_dict());break
            else:raise RuntimeError('Fresh history rejection exhausted')
        groups[split+'_keys']=worlds
    return groups


def samples(worlds,codec):
    rows=[]
    for task in TASKS:
        for encoding in ENCODINGS:
            for group,episodes in worlds.items():
                for episode,world in enumerate(episodes):
                    for cell in CELLS:
                        for variant in VARIANTS:
                            for row in render(world,codec,task,encoding,cell,variant):
                                rows.append(dict(group=group,episode=episode,**row))
                    # Plumbing controls are original-view F0Q0 only.
                    for condition in CONTROLS:
                        for row in render(world,codec,task,encoding,'F0Q0',condition=condition):
                            rows.append(dict(group=group,episode=episode,**row))
    return rows
