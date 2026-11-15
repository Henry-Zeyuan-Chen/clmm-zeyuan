from types import SimpleNamespace
import numpy as np
from clmm.pretrain.fact_streams import Episode,KEY_POOLS,render_query
from clmm.pretrain.sequence_freshness_data import (ARMS,distinct_episode,examples,replay_context,
    retemplate,swap_assignments,signature,collect_histories,training_worlds,training_arrays,evaluation_worlds)

def test_label_independent_bridge_and_matched_candidates():
    ep=distinct_episode(91307,2,True)
    candidates=[]
    for arm in ARMS:
        raw,plan=replay_context(ep,arm)
        assert replay_context(SimpleNamespace(events=ep.events,seed=ep.seed),arm)==(raw,plan)
        assert len(raw)==32*(6 if arm=='online' else 10)
        if arm.endswith('_bounded'):assert plan['stale_extras']==0
        if arm.removesuffix('_bounded') in ('continuous','periodic'):
            candidates.append([(e.get('sampled_source',e['source']),e['key'],e.get('sampled_value',e['value'])) for e in plan['events'] if e['kind']=='extra'])
    assert all(v==candidates[0] for v in candidates)

def test_acquisition_prefixes_and_counterfactual_values():
    ep=distinct_episode(91308);swapped=swap_assignments(ep)
    assert ep.queries[0].target==swapped.queries[1].target
    assert signature(ep.events)!=signature(swapped.events)
    original=examples(ep,'online')[0]
    for arm in ARMS:assert examples(ep,arm)[0]==original
    assert all(len(x)==95 for x,y in original)
    assert render_query(ep.queries[0])!=render_query(ep.queries[1])
    assert render_query(ep.queries[0])==render_query(swapped.queries[0])

def test_shared_training_and_template_holdout():
    worlds=training_worlds(40,2);arrays=training_arrays(worlds)
    assert arrays['periodic'].shape==(40,4,351)
    assert set(arrays['lengths'])=={63,95,351}
    for w in worlds:
        for row in w['episodes']:
            ep=Episode.from_dict(row)
            assert all(r.key in KEY_POOLS['train'] for r in ep.events)
            assert ep.template_family=='seen'
    excluded=collect_histories(worlds)
    data=evaluation_worlds(excluded,n=8,revision_n=4)
    for name,rows in data.items():
        for row in rows:assert signature(Episode.from_dict(row).events) not in excluded
    for seen,held in zip(data['acquire_dev_seen'],data['acquire_dev_heldout']):
        a,b=Episode.from_dict(seen),Episode.from_dict(held)
        assert signature(a.events)==signature(b.events)
        assert a.events[0].encoded!=b.events[0].encoded
        assert [q.target for q in a.queries]==[q.target for q in b.queries]

def test_signature_discovery_ignores_nonsemantic_numeric_models():
    ep=distinct_episode(91309)
    assert signature(ep.events) in collect_histories(ep.to_dict())
    assert not collect_histories({'keys':[1,2],'values':[3,4],'records':[{'key':4,'value':5}]})

def test_legacy_seed_only_training_traces_are_excluded():
    from clmm.pretrain.fact_streams import make_episode
    trace=dict(episode_seeds=[123456],facts=2,delay=0,revision_fraction=.5)
    old=make_episode(123456,2,0,'train',revision_fraction=.5)
    assert collect_histories(trace)=={signature(old.events)}
