"""Explicit checkpoint boundaries for the shared component adapter experiments."""
from pathlib import Path
import torch
from clmm.hebbian.adapter import AssociativeAdapter, AssociativeConfig
from clmm.pretrain.kv_correlated_portable import load_portable_checkpoint


def new_associative(shared_model, *, update_rule, routing):
    config=AssociativeConfig(codebook_config=shared_model.config.to_dict(),
                             update_rule=update_rule,routing=routing)
    return AssociativeAdapter(config,shared_model.codebook.keys.detach().cpu(),
                              shared_model.codebook.values.detach().cpu())


def save_adapter_checkpoint(path,model,*,source_commit,source_sha256,training):
    if not isinstance(model,AssociativeAdapter):
        raise ValueError('Reference adapter checkpoints require an associative component')
    model.assert_codebook_valid()
    payload=dict(format='clmm.adapter.reference.v1',architecture='associative_v1',
        method=model.method,config=model.config.to_dict(),model_state=model.state_dict(),
        codebook_hash=model.codebook_hash,source_commit=source_commit,
        source_sha256=source_sha256,training=training)
    torch.save(payload,Path(path))


def load_adapter_checkpoint(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')=='clmm.kv.correlated.v1':
        return load_portable_checkpoint(path,device)
    if payload.get('format')!='clmm.adapter.reference.v1' or payload.get('architecture')!='associative_v1':
        raise ValueError('Unsupported shared adapter checkpoint')
    config=AssociativeConfig(**payload['config'])
    tensors=payload['model_state']
    model=AssociativeAdapter(config,tensors['codebook.keys'],tensors['codebook.values']).to(device)
    if payload['method']!=model.method or payload['codebook_hash']!=model.codebook_hash:
        raise ValueError('Checkpoint method or codebook mismatch')
    model.load_state_dict(tensors,strict=True)
    return model,payload


def fork_associative(model,*,update_rule,routing):
    """Copy only slow tensors to an explicitly changed rule/access configuration.

    The caller starts a new zero-trace session. Fast state is not converted.
    This is a frozen-transfer intervention, not matched adaptation of that rule.
    """
    if not isinstance(model,AssociativeAdapter):
        raise ValueError('Only equal-geometry associative slow weights can be forked')
    config=model.config.to_dict()
    config.update(update_rule=update_rule,routing=routing)
    with torch.random.fork_rng(devices=[]):
        fork=AssociativeAdapter(AssociativeConfig(**config),model.codebook.keys.detach().cpu(),
                               model.codebook.values.detach().cpu()).to(model.device)
    fork.load_state_dict(model.state_dict(),strict=True)
    return fork
