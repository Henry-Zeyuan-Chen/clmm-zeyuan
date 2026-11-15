import numpy as np
import pytest
from clmm.pretrain.kv_streams import episode,training_data,evaluation_data,replay_plan,signature


def test_latest_targets_and_strict_revisions():
    for seed in range(30):
        ep=episode(seed,4,4,1)
        observed=dict(zip(ep['keys'],ep['values']))
        assert len({q['target'] for q in ep['queries']})==4
        for q in ep['queries']:
            assert q['target']==observed[q['key']]
            if q['category']=='revised':assert q['old']!=q['target']
        assert len(set(ep['keys'][:4])&set(ep['keys'][ep['boundary']:]))==0


def test_training_padding_and_label_reconstruction():
    data,plan,seen=training_data(24,4,88)
    again,repeat,duplicate=training_data(24,4,88)
    assert plan==repeat and seen==duplicate
    for key in data:np.testing.assert_array_equal(data[key],again[key])
    for step in range(24):
        length,count=int(data['lengths'][step]),int(data['counts'][step])
        assert np.all(data['keys'][step,:,length:]==255)
        assert np.all(data['targets'][step,:,count:]==255)
        for b in range(4):
            keys=data['keys'][step,b,:length].tolist();values=data['values'][step,b,:length].tolist()
            truth=dict(zip(keys,values))
            assert signature(keys,values) in seen
            assert all(truth[int(k)]==int(v) for k,v in zip(data['queries'][step,b,:count],data['targets'][step,b,:count]))


def test_replay_matching_causality_and_prefixes():
    datasets=evaluation_data(90303,2,2,suite='full')
    for k in (8,16,32):
        episodes=[datasets[f'pressure_k{k}_d{d}'][0] for d in (0,16,64)]
        assert episodes[0]['keys']==episodes[2]['keys'][:episodes[0]['boundary']]
        assert episodes[1]['keys']==episodes[2]['keys'][:episodes[1]['boundary']+16]
        for ep in episodes:
            plans={arm:replay_plan(ep,arm) for arm in ('online','repeat','continuous','periodic')}
            c,p=plans['continuous'],plans['periodic']
            assert c['selected']==p['selected']
            assert c['writes']==p['writes']==plans['repeat']['writes']==ep['boundary']+2*ep['delay']
            for plan in plans.values():
                for event in plan['events']:
                    assert event['source']<=event['after_online']
                    assert event['value']==ep['values'][event['source']]
            assert [q['target'] for q in ep['queries'][:4]]==[q['target'] for q in episodes[0]['queries'][:4]]


def test_excluded_history_not_claimed_new():
    original=evaluation_data(81,2)
    excluded={ep['signature'] for eps in original.values() for ep in eps}
    new=evaluation_data(81,2,seen=excluded)
    assert not excluded.intersection(ep['signature'] for eps in new.values() for ep in eps)
