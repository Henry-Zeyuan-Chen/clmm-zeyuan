from dataclasses import replace
import pytest
from clmm.pretrain.format_factorial import CELLS,TASKS,render,fresh_worlds,samples
from clmm.pretrain.record_selection import (ENCODINGS,VARIANTS,make_codec,make_world,
    render_samples,variant_signatures,visible_oracle)
from clmm.pretrain.sequence_freshness_data import retemplate


@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('encoding',ENCODINGS)
@pytest.mark.parametrize('variant',VARIANTS)
def test_independent_visible_blocks_and_counterfactual_labels(task,encoding,variant):
    codec=make_codec();world=make_world(91801,split='dev',codec=codec)
    views={c:render(world,codec,task,encoding,c,variant) for c in CELLS}
    for rank in [0,1]:
        prefix={c:bytes.fromhex(rows[rank]['prefix_hex']) for c,rows in views.items()}
        assert prefix['F0Q0'][:64]==prefix['F0Q1'][:64]
        assert prefix['F1Q0'][:64]==prefix['F1Q1'][:64]
        assert prefix['F0Q0'][64:]==prefix['F1Q0'][64:]
        assert prefix['F0Q1'][64:]==prefix['F1Q1'][64:]
        assert prefix['F0Q0'][:64]!=prefix['F1Q0'][:64]
        assert prefix['F0Q0'][64:]!=prefix['F0Q1'][64:]
        assert len({r[rank]['target'] for r in views.values()})==1
        for cell,x in prefix.items():
            assert len(x)==95
            assert visible_oracle(x,codec,task,encoding)==views[cell][rank]['target']
    for cell,ep in [('F0Q0',world),('F1Q1',retemplate(world,'heldout'))]:
        old=render_samples(ep,codec,task,encoding,variant)
        assert [r['prefix_hex'] for r in views[cell]]==[r['prefix'].hex() for r in old]
        assert [r['target'] for r in views[cell]]==[r['target'] for r in old]


@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('encoding',ENCODINGS)
def test_controls_and_train_namespace_without_weakening_episode(task,encoding):
    codec=make_codec();world=make_world(91802,split='train',codec=codec)
    before=world.to_dict()
    for cell in CELLS:assert all(len(bytes.fromhex(r['prefix_hex']))==95 for r in render(world,codec,task,encoding,cell))
    erased=render(world,codec,task,encoding,'F0Q0',condition='cue_erased')
    assert erased[0]['prefix_hex']==erased[1]['prefix_hex']
    assert erased[0]['target']!=erased[1]['target']
    blank=render(world,codec,task,encoding,'F0Q0',condition='no_facts')
    assert all(bytes.fromhex(r['prefix_hex'])[:64]==b' '*64 for r in blank)
    assert world.to_dict()==before
    with pytest.raises(ValueError):retemplate(world,'heldout')


def test_fresh_full_orbit_exclusions_and_count():
    codec=make_codec();a=fresh_worlds(set(),seed=91812,n_dev=3,n_train=2,codec=codec)
    excluded=set().union(*(variant_signatures(w) for group in a.values() for w in group))
    b=fresh_worlds(excluded,seed=91812,n_dev=3,n_train=2,codec=codec)
    found=set().union(*(variant_signatures(w) for group in b.values() for w in group))
    assert found.isdisjoint(excluded) and len(found)==20
    rows=samples(b,codec)
    assert len(rows)==len(TASKS)*len(ENCODINGS)*5*36
    assert all(r['cell']=='F0Q0' and r['variant']=='original' for r in rows if r['condition']!='normal')


def test_semantic_targets_only_follow_observed_view():
    codec=make_codec();world=make_world(91803,split='dev',codec=codec)
    for cell in CELLS:
        base=render(world,codec,'select','digits',cell)
        order=render(world,codec,'select','digits',cell,'order')
        assert [r['target'] for r in base]==[97-r['target'] for r in order]
        v=render(world,codec,'value','digits',cell)
        swapped=render(world,codec,'value','digits',cell,'value')
        assert v[0]['target']==swapped[1]['target'] and v[1]['target']==swapped[0]['target']
