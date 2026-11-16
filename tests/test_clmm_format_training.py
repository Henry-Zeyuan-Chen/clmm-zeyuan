import copy
from collections import Counter
import numpy as np
import pytest
from clmm.pretrain import format_training as t
from clmm.pretrain.format_factorial import render as old_render
from clmm.pretrain.record_selection import make_world,make_codec,variant_signatures,VARIANTS


@pytest.mark.parametrize('encoding',['digits','atomic'])
@pytest.mark.parametrize('cell',t.CELLS)
@pytest.mark.parametrize('variant',VARIANTS)
def test_visible_blocks_labels_and_independent_factors(encoding,cell,variant):
    codec=make_codec();world=make_world(81123,codec=codec)
    rows=t.render(world,codec,encoding,cell,variant)
    original=t.render(world,codec,encoding,'F0Q0',variant)
    for r,b in zip(rows,original):
        raw=bytes.fromhex(r['prefix_hex']);assert len(raw)==95 and raw[:3]==b'[0]' and raw[32:35]==b'[1]'
        assert (r['rank'],r['key'],r['slot'],r['target'])==(b['rank'],b['key'],b['slot'],b['target'])
        if cell[1]=='0':assert raw[:64]==bytes.fromhex(b['prefix_hex'])[:64]
        if cell[3]=='0':assert raw[64:]==bytes.fromhex(b['prefix_hex'])[64:]
    if cell in t.TRAIN_CELLS:
        previous=old_render(world,codec,'value',encoding,cell,variant)
        assert [(r['prefix_hex'],r['target']) for r in rows]==[(r['prefix_hex'],r['target']) for r in previous]


@pytest.mark.parametrize('cell',t.CELLS)
def test_erasure_collision_and_value_swaps(cell):
    c=make_codec();w=make_world(82345,codec=c)
    erased=t.render(w,c,'atomic',cell,condition='cue_erased')
    assert erased[0]['prefix_hex']==erased[1]['prefix_hex'] and erased[0]['target']!=erased[1]['target']
    assert {r['target'] for r in erased}=={r['target'] for r in t.render(w,c,'atomic',cell,'value')}
    for r in t.render(w,c,'digits',cell,condition='no_facts'):assert bytes.fromhex(r['prefix_hex'])[:64]==b' '*64


def test_training_exact_pairing_and_history_closure():
    c=make_codec();excluded=variant_signatures(make_world(82739,codec=c))
    plan,seen=t.fresh_training(excluded,steps=4,codec=c)
    assert len(seen)==4+4*8*4
    orbits=[variant_signatures(w) for step in plan for w in step['episodes']]
    assert len(set.union(*orbits))==128 and not set.union(*orbits)&excluded
    assert all(Counter(s['mixed_cells'])==Counter({cell:2 for cell in t.TRAIN_CELLS}) for s in plan)
    arrays=t.training_arrays(plan,c);assert arrays['targets'].shape==(4,16)
    for enc in ('digits','atomic'):
        assert arrays['familiar_'+enc].shape==arrays['mixed_'+enc].shape==(4,16,95)
        for i,s in enumerate(plan):
            for j,w in enumerate(s['episodes']):
                for arm in ('familiar','mixed'):
                    cell='F0Q0' if arm=='familiar' else s['mixed_cells'][j]
                    rows=t.render(w,c,enc,cell,s['variants'][j])
                    assert [r['target'] for r in rows]==arrays['targets'][i,2*j:2*j+2].tolist()
                    assert arrays[arm+'_'+enc][i,2*j:2*j+2].tobytes()==b''.join(bytes.fromhex(r['prefix_hex']) for r in rows)


def lock():
    return dict(status='all_eight_terminal_locked',packet_metadata_sha256='packet',initializers=[dict(checkpoint_sha256=str(i)) for i in range(4)],
        terminals=[dict(parent=p,encoding=e,arm=a,status='succeeded',completed_steps=6000,checkpoint_sha256=f'{p}{e}{a}') for p in (68101,68105) for e in ('digits','atomic') for a in ('familiar','mixed')])


def test_confirmation_seal_requires_all_eight():
    valid=lock();assert t.validate_confirmation_lock(valid,'packet','0')
    for n in range(8):
        broken=copy.deepcopy(valid);broken['terminals'][n]['completed_steps']=5999
        with pytest.raises(ValueError):t.validate_confirmation_lock(broken,'packet','0')
    for key,value in [('status','running'),('packet_metadata_sha256','wrong'),('terminals',valid['terminals'][:-1]),('initializers',[])]:
        broken=dict(valid);broken[key]=value
        with pytest.raises(ValueError):t.validate_confirmation_lock(broken,'packet','0')
    with pytest.raises(ValueError):t.validate_confirmation_lock(valid,'packet','unlocked')


def test_reserved_wording_absent_from_engineering_packet():
    c=make_codec();worlds={'train_keys':[make_world(83811,codec=c).to_dict()]}
    rows=t.evaluation_samples(worlds,c,confirmation=False)
    assert len(rows)==72 and {r['cell'] for r in rows}==set(t.TRAIN_CELLS)
    final=t.evaluation_samples(worlds,c);assert len(final)==152
    assert [r['packet_row_index'] for r in final]==list(range(152))
