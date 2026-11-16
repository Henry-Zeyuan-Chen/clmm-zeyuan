"""Paired familiar/mixed wording data, with a reserved third template family.

Only raw 95-byte observations reach the unchanged model. Episode validation
and the shared template dictionaries are deliberately left intact.
"""
from dataclasses import replace
import numpy as np
from .fact_streams import Episode, RECORD_TEMPLATES, QUERY_TEMPLATES
from .record_selection import make_codec, make_world, variant_records, variant_signatures, VARIANTS

FACT_CONFIRM = ('entry {key} stores {value}.\n', '{key}: stored code is {value}.\n')
QUERY_CONFIRM = ('lookup code belonging to {key}: ', "report {key}'s stored code: ")
FACT_WARMUP_CONFIRM = ('item {key}: its code is {value}.\n', '{key} is paired with code {value}.\n')
QUERY_WARMUP_CONFIRM = ('fetch code tied to key {key}: ', '{key}: what code is stored? ')
CELLS = tuple(f'F{f}Q{q}' for f in range(3) for q in range(3))
EXTENDED_CELLS = tuple(f'F{f}Q{q}' for f in range(4) for q in range(4))
WARMUP_CELLS = tuple(f'F{f}Q{q}' for f in (0,1,3) for q in (0,1,3))
TRAIN_CELLS = ('F0Q0','F0Q1','F1Q0','F1Q1')
PRIMARY_CELLS = ('F2Q0','F0Q2','F2Q2')


def validate_confirmation_lock(lock,packet_sha256,checkpoint_sha256,arms=('familiar','mixed')):
    """Require all eight terminal models before any reserved-wording output."""
    if (lock.get('status')!='all_eight_terminal_locked' or lock.get('packet_metadata_sha256')!=packet_sha256
            or len(lock.get('terminals',[]))!=8 or len(lock.get('initializers',[]))!=4):
        raise ValueError('Incomplete confirmation lock')
    if len(arms)!=2 or len(set(arms))!=2:raise ValueError('Exactly two distinct arms required')
    expected={(p,e,a) for p in (68101,68105) for e in ('digits','atomic') for a in arms}
    actual={(r['parent'],r['encoding'],r['arm']) for r in lock['terminals']}
    if actual!=expected or any(r['status']!='succeeded' or r['completed_steps']!=6000 for r in lock['terminals']):
        raise ValueError('All eight complete terminal conditions are required')
    allowed={r['checkpoint_sha256'] for r in lock['terminals']+lock['initializers']}
    if checkpoint_sha256 not in allowed:raise ValueError('Checkpoint is not locked')
    return True


def templates(include_next=False):
    result=dict(facts=[list(RECORD_TEMPLATES['seen']),list(RECORD_TEMPLATES['heldout']),list(FACT_CONFIRM)],
                queries=[list(QUERY_TEMPLATES['seen']),list(QUERY_TEMPLATES['heldout']),list(QUERY_CONFIRM)])
    if include_next:
        result['facts'].append(list(FACT_WARMUP_CONFIRM));result['queries'].append(list(QUERY_WARMUP_CONFIRM))
    return result


def learning_rate(step,schedule='constant',peak=.0003,warmup_steps=500):
    if type(step) is not int or step<1 or schedule not in ('constant','warmup'):
        raise ValueError('Positive one-indexed step and supported LR schedule required')
    if type(warmup_steps) is not int or warmup_steps<1 or not 0<peak<1:raise ValueError('Invalid LR parameters')
    return peak if schedule=='constant' else peak*min(step/warmup_steps,1.)


def render(world,codec,encoding,cell,variant='original',condition='normal'):
    if cell not in EXTENDED_CELLS or condition not in ('normal','no_facts','cue_erased'):
        raise ValueError('Unknown format cell or observation condition')
    ep=Episode.from_dict(world) if isinstance(world,dict) else world
    records=variant_records(ep,variant);f=int(cell[1]);q=int(cell[3]);blocks=[]
    for slot,r in enumerate(records):
        if f<2:
            raw=replace(r,template_family=('seen','heldout')[f]).encoded
        else:
            raw=(FACT_CONFIRM,FACT_WARMUP_CONFIRM)[f-2][r.template_index].format(key=r.key,value=r.value).encode('ascii').rjust(32,b' ')
        if len(raw)!=32 or raw[:3]!=b'   ':raise ValueError('Fact does not fit tagged record')
        raw=bytes((91,48+slot,93))+raw[3:]
        blocks.append(codec.transform(raw,r.key,encoding))
    context=b' '*64 if condition=='no_facts' else b''.join(blocks)
    rows=[]
    for rank,query in enumerate(ep.queries):
        slot=next(i for i,r in enumerate(records) if r.key==query.key)
        table=(QUERY_TEMPLATES['seen'],QUERY_TEMPLATES['heldout'],QUERY_CONFIRM,QUERY_WARMUP_CONFIRM)[q]
        raw=table[query.template_index].format(key=query.key).encode('ascii').rjust(31,b' ')
        if len(raw)!=31 or raw.count(query.key.encode('ascii'))!=1:raise ValueError('Invalid query')
        start=raw.index(query.key.encode('ascii'));raw=codec.transform(raw,query.key,encoding)
        if condition=='cue_erased':raw=raw[:start]+b'   '+raw[start+3:]
        prefix=context+raw;assert len(prefix)==95
        rows.append(dict(prefix_hex=prefix.hex(),target=ord(records[slot].value),rank=rank,key=query.key,slot=slot,
            cell=cell,variant=variant,condition=condition,encoding=encoding,input_bytes=95))
    return rows


def fresh_training(excluded,steps=6000,contexts=8,seed=92002,codec=None):
    if type(steps) is not int or steps<1 or contexts!=8:raise ValueError('Positive steps and eight contexts required')
    codec=make_codec() if codec is None else codec;seen=set(excluded)
    worlds_rng=np.random.Generator(np.random.PCG64(seed))
    views_rng=np.random.Generator(np.random.PCG64(seed+3))
    cells_rng=np.random.Generator(np.random.PCG64(seed+4));plan=[]
    for step in range(steps):
        episodes=[]
        for j in range(contexts):
            for attempt in range(100000):
                world=make_world(int(worlds_rng.integers(12000000000,13000000000)),codec=codec)
                orbit=variant_signatures(world)
                if not seen&orbit:seen.update(orbit);episodes.append(world.to_dict());break
            else:raise RuntimeError('Training novelty rejection exhausted')
        plan.append(dict(step=step+1,episodes=episodes,variants=[VARIANTS[int(v)] for v in views_rng.integers(0,4,contexts)],
            mixed_cells=[TRAIN_CELLS[int(c)] for c in cells_rng.permutation([0,0,1,1,2,2,3,3])]))
    return plan,seen


def training_arrays(plan,codec):
    n=len(plan);arrays={f'{arm}_{enc}':np.empty((n,16,95),dtype=np.uint8) for arm in ('familiar','mixed') for enc in ('digits','atomic')}
    targets=np.empty((n,16),dtype=np.uint8)
    for i,step in enumerate(plan):
        if sorted(step['mixed_cells'])!=sorted(TRAIN_CELLS*2):raise ValueError('Every mixed step needs two contexts per cell')
        for arm in ('familiar','mixed'):
            for enc in ('digits','atomic'):
                rows=[r for j,w in enumerate(step['episodes']) for r in render(w,codec,enc,'F0Q0' if arm=='familiar' else step['mixed_cells'][j],step['variants'][j])]
                arrays[f'{arm}_{enc}'][i]=np.array([list(bytes.fromhex(r['prefix_hex'])) for r in rows],dtype=np.uint8)
                y=np.array([r['target'] for r in rows],dtype=np.uint8)
                if arm=='familiar' and enc=='digits':targets[i]=y
                else:assert np.array_equal(targets[i],y)
    return dict(**arrays,targets=targets)


def evaluation_samples(worlds,codec,*,confirmation=True,cells=None):
    cells=(CELLS if confirmation else TRAIN_CELLS) if cells is None else tuple(cells)
    if not cells or not set(cells)<=set(EXTENDED_CELLS) or (not confirmation and not set(cells)<=set(TRAIN_CELLS)):
        raise ValueError('Invalid evaluation cells or reserved wording in engineering')
    rows=[]
    for encoding in ('digits','atomic'):
        for group,episodes in worlds.items():
            for i,w in enumerate(episodes):
                for cell in cells:
                    for variant in VARIANTS:
                        rows.extend(dict(group=group,episode=i,**r) for r in render(w,codec,encoding,cell,variant))
                for condition in ('no_facts','cue_erased'):
                    rows.extend(dict(group=group,episode=i,**r) for r in render(w,codec,encoding,'F0Q0',condition=condition))
    for i,row in enumerate(rows):row['packet_row_index']=i
    return rows
