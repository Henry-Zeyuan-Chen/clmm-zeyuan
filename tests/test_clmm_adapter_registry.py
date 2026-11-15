"""Checkpoint identity and explicit slow-only transfer across method variants."""
from pathlib import Path
import pytest
import torch
from clmm.pretrain.adapter_registry import new_associative,load_adapter_checkpoint,save_adapter_checkpoint,fork_associative
from clmm.pretrain.kv_correlated import CorrelatedConfig,CorrelatedNeuralAdapter


def test_checkpoint_roundtrip_and_wrong_architecture_rejection(tmp_path):
    torch.set_num_threads(1)
    model=new_associative(CorrelatedNeuralAdapter(CorrelatedConfig()),update_rule='hebbian',routing='top1')
    path=tmp_path/'checkpoint.pt'
    save_adapter_checkpoint(path,model,source_commit='test-only',source_sha256={},training=dict(step=0))
    restored,payload=load_adapter_checkpoint(path,torch.device('cpu'))
    assert restored.config==model.config
    assert all(torch.equal(t,restored.state_dict()[name]) for name,t in model.state_dict().items())
    state=model.write(torch.tensor([3]),torch.tensor([2]),model.initial_state(1))
    state2=restored.import_state(model.export_state(state))
    assert torch.equal(model.read(torch.tensor([[3]]),state),restored.read(torch.tensor([[3]]),state2))
    payload['architecture']='different'
    torch.save(payload,path)
    with pytest.raises(ValueError,match='Unsupported'):load_adapter_checkpoint(path,torch.device('cpu'))


def test_fork_preserves_slow_bits_and_rng_but_rejects_original_fast_state():
    torch.set_num_threads(1)
    original=new_associative(CorrelatedNeuralAdapter(),update_rule='gradient',routing='dense')
    rng=torch.random.get_rng_state().clone()
    changed=fork_associative(original,update_rule='hebbian',routing='top1')
    assert torch.equal(rng,torch.random.get_rng_state())
    assert changed.method=='hebbian' and changed.config.routing=='top1'
    assert all(torch.equal(t,changed.state_dict()[name]) for name,t in original.state_dict().items())
    assert all(t.data_ptr()!=changed.state_dict()[name].data_ptr() for name,t in original.state_dict().items())
    with pytest.raises(ValueError):changed.import_state(original.export_state(original.initial_state(1)))
    assert changed.initial_state(1).write_count==0


def test_foreign_checkpoint_method_is_rejected(tmp_path):
    model=new_associative(CorrelatedNeuralAdapter(),update_rule='gradient',routing='dense')
    path=tmp_path/'checkpoint.pt'
    save_adapter_checkpoint(path,model,source_commit='test-only',source_sha256={},training={})
    payload=torch.load(path,weights_only=True)
    payload['method']='hebbian'
    torch.save(payload,path)
    with pytest.raises(ValueError,match='method'):load_adapter_checkpoint(path,torch.device('cpu'))
