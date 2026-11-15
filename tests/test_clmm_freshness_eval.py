"""Offline audit-plan integration with the live bounded controller."""
from copy import deepcopy
import pytest
from clmm.pretrain.kv_scheduler import StreamingReplay
from clmm.sleep.multirevision import make_datasets,classify_events
from scripts.freshness_experiment import ARMS,expected_plan


@pytest.mark.parametrize('arm',ARMS)
def test_freshness_plan_matches_live_controller_without_query_access(arm):
    data=make_datasets(91207,2)
    for cell,episodes in data.items():
        if not cell.startswith('pressure'):continue
        for episode in episodes:
            bounded=arm.endswith('_bounded')
            policy=StreamingReplay(arm.removesuffix('_bounded'),episode['seed'],
                                   freshness='bounded' if bounded else 'literal')
            actual=[]
            for index,(key,value) in enumerate(zip(episode['keys'],episode['values'])):
                actual.extend([policy.prime(key,value)] if index<episode['boundary'] else policy.step(key,value))
            actual.extend(policy.finish())
            assert actual==expected_plan(episode,arm)['events']
            changed=deepcopy(episode);changed['queries']=object()
            assert expected_plan(changed,arm)['events']==actual
            classified=classify_events(episode,actual)
            assert len(actual)==len(episode['keys'])+(0 if arm=='online' else episode['delay'])
            if bounded:
                assert not any(e['stale'] for e in classified)
                assert all(e['sampled_source']<=e['after_online'] and e['source']<=e['after_online']
                           for e in actual if e['kind']=='extra')
