"""Independent pair/epoch contracts for the fixed-set experiment runner."""
import numpy as np
import pytest
from scripts import fixed_set_generalization as runner
from clmm.pretrain.fact_streams import Episode,encode_example
from clmm.pretrain.association_probe import build_bank


def test_fixed_epoch_uses_each_context_once_and_no_label_in_prefix():
    bank=build_bank()
    indices=[]
    for step in range(4):
        x,y,plan=runner.training_batch(bank,step,16,90202,'fixed')
        assert x.shape==(16,95) and y.shape==(16,)
        indices.extend(plan['bank_indices'])
        for i,index in enumerate(plan['bank_indices']):
            ep=Episode.from_dict(bank['episodes'][index])
            for rank,q in enumerate(ep.queries):
                prefix,target,_=encode_example(ep,q,'online')
                assert x[2*i+rank].tobytes()==prefix
                assert y[2*i+rank]==target
            assert y[2*i]!=y[2*i+1]
            np.testing.assert_array_equal(x[2*i,:64],x[2*i+1,:64])
    assert sorted(indices)==list(range(32))
    assert runner.training_batch(bank,0,16,90202,'fixed')[2]!=runner.training_batch(bank,4,16,90202,'fixed')[2]


def test_geometry_independent_sampling_and_fresh_records_are_reconstructable():
    bank=build_bank()
    a=runner.training_batch(bank,17,16,90202,'fresh')
    b=runner.training_batch(bank,17,16,90202,'fresh')
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    assert a[2]==b[2]
    for i,record in enumerate(a[2]['episodes']):
        ep=Episode.from_dict(record)
        assert {r.key for r in ep.events}<=set(bank['key_pool'])
        for rank,q in enumerate(ep.queries):
            prefix,target,_=encode_example(ep,q,'online')
            assert a[0][2*i+rank].tobytes()==prefix and a[1][2*i+rank]==target


@pytest.mark.parametrize('batch',[1,3,14,128])
def test_invalid_rectangular_batches_are_rejected(batch):
    with pytest.raises(ValueError):runner.training_batch(build_bank(),0,batch,90202,'fixed')
