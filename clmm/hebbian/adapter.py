"""Reference DHP and matched gradient trace updates, not a complete Titans model.

The Hebbian EMA follows the form of Miconi et al. (2018), equations 1/2,
with observed-value postsynaptic activity, a fixed router and explicit bounds.
These choices are component defaults, not a teammate's final method.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from clmm.pretrain.kv_control import KVState
from clmm.pretrain.kv_correlated import codebook_tensor_sha256
from clmm.pretrain.kv_correlated_portable import PortableCorrelatedCodebook, PortableCorrelatedConfig


@dataclass(frozen=True)
class AssociativeConfig:
    codebook_config: PortableCorrelatedConfig | dict
    modules: int = 4
    dim: int = 64
    routing: str = "dense"
    update_rule: str = "hebbian"
    eta_init: float = 0.1
    trace_bound: float = 1.0
    alpha_init: float = 1.0
    init_noise: float = 0.01

    def __post_init__(self):
        codebook = self.codebook_config
        if type(codebook) is dict:
            codebook = PortableCorrelatedConfig(**codebook)
        if not isinstance(codebook, PortableCorrelatedConfig):
            raise ValueError("codebook_config must identify a saved literal codebook")
        object.__setattr__(self, "codebook_config", codebook)
        if type(self.modules) is not int or self.modules != 4 or type(self.dim) is not int or self.dim != 64:
            raise ValueError("Reference adapters retain four modules of dimension64")
        if self.routing not in ("dense", "top1") or self.update_rule not in ("gradient", "hebbian"):
            raise ValueError("Unknown routing or update rule")
        for name in ("eta_init", "trace_bound", "alpha_init", "init_noise"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if not 0 < self.eta_init < 1 or self.trace_bound <= 0 or self.init_noise < 0:
            raise ValueError("Invalid learning rate, bound, or initialization noise")

    @property
    def nkeys(self):
        return self.codebook_config.nkeys

    @property
    def nvalues(self):
        return self.codebook_config.nvalues

    def to_dict(self):
        return dict(codebook_config=self.codebook_config.to_dict(), modules=self.modules,
                    dim=self.dim, routing=self.routing, update_rule=self.update_rule,
                    eta_init=self.eta_init, trace_bound=self.trace_bound,
                    alpha_init=self.alpha_init, init_noise=self.init_noise)

    @property
    def fingerprint(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class AssociativeAdapter(nn.Module):
    """Four associative modules sharing the component's literal typed codebook.

    Online writes are analytic tensor operations, including the gradient-rule
    control. Outer training can differentiate through every trace update.
    """
    _payload_fields = {"schema_version", "method", "config", "config_fingerprint",
                       "batch_size", "write_count", "weights", "aux"}
    _prototype_ids = (0, 16, 32, 48)

    def __init__(self, config: AssociativeConfig, keys: Tensor, values: Tensor):
        super().__init__()
        if not isinstance(config, AssociativeConfig):
            raise ValueError("config must be an AssociativeConfig")
        self.config = config
        self.codebook = PortableCorrelatedCodebook(config.codebook_config, keys, values)
        shape = (config.modules, config.dim, config.dim)
        identity = torch.eye(config.dim).expand(config.modules, -1, -1)
        # Every variant consumes exactly the same random draws in the same order.
        self.key_encoder = nn.Parameter(identity.clone() + config.init_noise * torch.randn(shape))
        self.value_encoder = nn.Parameter(identity.clone() + config.init_noise * torch.randn(shape))
        self.decoder = nn.Parameter(identity.clone() + config.init_noise * torch.randn(shape))
        self.W0 = nn.Parameter(torch.zeros(shape))
        self.alpha = nn.Parameter(torch.full(shape, config.alpha_init))
        self.raw_eta = nn.Parameter(torch.tensor(math.log(config.eta_init / (1 - config.eta_init)), dtype=torch.float32))
        self.register_buffer("router_prototypes", self.codebook.keys[list(self._prototype_ids)].clone())

    @property
    def method(self):
        return "hebbian" if self.config.update_rule == "hebbian" else "associative_gradient"

    @property
    def device(self):
        return self.codebook.keys.device

    @property
    def codebook_hash(self):
        return codebook_tensor_sha256(self.codebook.keys)

    @property
    def eta(self):
        return self.raw_eta.sigmoid()

    def parameter_counts(self):
        return dict(registered=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad))

    @property
    def config_deviations(self):
        return dict(component="reference_associative", modules=4, key_value_decoder="shared_literal_codebook",
                    update_rule=self.config.update_rule, routing=self.config.routing,
                    router="fixed cosine key prototypes0,16,32,48; no labels or value access",
                    encoder="module_specific tanh projections", effective_weights="W0+alpha*trace",
                    plasticity="learned scalar sigmoid eta; active-module EMA and explicit clamp",
                    inactive_modules="unchanged, including no decay", online_autograd_calls=False,
                    full_titans_model=False, prototype_status="reference, not final teammate method")

    def assert_codebook_valid(self):
        # The shared portable validator preserves the supplied vector bits.
        from clmm.pretrain.kv_correlated_portable import _validate_literal_codebook
        _validate_literal_codebook(self.config.codebook_config, self.codebook.keys, self.codebook.values)
        expected = self.codebook.keys[list(self._prototype_ids)]
        if (self.router_prototypes.dtype != torch.float32 or self.router_prototypes.shape != (4, 64)
                or not torch.equal(self.router_prototypes, expected)):
            raise ValueError("Router prototypes differ from the fixed key-only construction")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict:
            raise ValueError("Associative checkpoints require strict loading")
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("Checkpoint tensor names do not match the associative adapter")
        from clmm.pretrain.kv_correlated_portable import _validate_literal_codebook
        _validate_literal_codebook(self.config.codebook_config, state_dict["codebook.keys"], state_dict["codebook.values"])
        for name, tensor in state_dict.items():
            if (not isinstance(tensor, Tensor) or tensor.shape != expected[name].shape
                    or tensor.dtype != torch.float32 or tensor.layout != torch.strided
                    or not torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid checkpoint tensor: {name}")
        if not torch.equal(state_dict["router_prototypes"].detach().cpu(),
                           state_dict["codebook.keys"][list(self._prototype_ids)].detach().cpu()):
            raise ValueError("Checkpoint changes the fixed key-only router")
        result = super().load_state_dict(state_dict, strict=True, assign=assign)
        self.assert_codebook_valid()
        return result

    @staticmethod
    def _batch(batch):
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")

    def _state(self, trace, count):
        return KVState({"trace": trace}, None, count, self.method, self.config.fingerprint)

    def _validate_state(self, state, *, device=None, finite=False):
        if (not isinstance(state, KVState) or state.method != self.method
                or state.config_fingerprint != self.config.fingerprint):
            raise ValueError("State method or configuration mismatch")
        if type(state.write_count) is not int or state.write_count < 0 or state.aux is not None:
            raise ValueError("Invalid write counter or auxiliary state")
        if type(state.weights) is not dict or set(state.weights) != {"trace"}:
            raise ValueError("Only bounded associative traces belong in fast state")
        trace = state.weights["trace"]
        if (not isinstance(trace, Tensor) or trace.ndim != 4
                or tuple(trace.shape[1:]) != (4, 64, 64) or trace.shape[0] < 1
                or trace.dtype != torch.float32 or trace.layout != torch.strided
                or device is not None and trace.device != device):
            raise ValueError("Invalid trace shape, dtype or device")
        if finite and (not torch.isfinite(trace).all() or (trace.abs() > self.config.trace_bound).any()):
            raise ValueError("Trace must be finite and bounded")
        return trace.shape[0]

    def initial_state(self, batch):
        self._batch(batch)
        return self._state(self.codebook.keys.new_zeros(batch, 4, 64, 64), 0)

    def route(self, key_ids):
        keys = self.codebook.encode_keys(key_ids)
        if self.config.routing == "dense":
            return torch.ones((*key_ids.shape, 4), device=self.device, dtype=torch.bool)
        similarity = F.normalize(keys, dim=-1) @ F.normalize(self.router_prototypes, dim=-1).T
        return F.one_hot(similarity.argmax(dim=-1), 4).bool()

    def routing_mask(self, key_ids):
        return self.route(key_ids)

    def learned_metrics(self):
        return dict(eta=float(self.eta.detach()), alpha_mean=float(self.alpha.detach().mean()),
                    alpha_min=float(self.alpha.detach().min()), alpha_max=float(self.alpha.detach().max()))

    def write(self, keys, values, state):
        batch = self._validate_state(state, device=self.device)
        if not isinstance(keys, Tensor) or not isinstance(values, Tensor) or keys.shape != (batch,) or values.shape != (batch,):
            raise ValueError("write keys and values must have shape[batch]")
        key = self.codebook.encode_keys(keys)
        value = self.codebook.encode_values(values)
        pre = torch.tanh(torch.einsum("bd,mdh->bmh", key, self.key_encoder))
        post = torch.tanh(torch.einsum("bd,mdh->bmh", value, self.value_encoder))
        trace = state.weights["trace"]
        if self.config.update_rule == "hebbian":
            change = pre[..., :, None] * post[..., None, :]
        else:
            effective = self.W0[None] + self.alpha[None] * trace
            prediction = torch.einsum("bmi,bmij->bmj", pre, effective)
            change = -self.alpha[None] * pre[..., :, None] * (prediction - post)[..., None, :]
        updated = ((1 - self.eta) * trace + self.eta * change).clamp(-self.config.trace_bound, self.config.trace_bound)
        active = self.route(keys)[..., None, None]
        return self._state(torch.where(active, updated, trace), state.write_count + 1)

    def read(self, queries, state):
        batch = self._validate_state(state, device=self.device)
        if not isinstance(queries, Tensor) or queries.ndim != 2 or queries.shape[0] != batch or queries.shape[1] < 1:
            raise ValueError("queries must have shape[batch,queries]")
        key = self.codebook.encode_keys(queries)
        pre = torch.tanh(torch.einsum("bqd,mdh->bqmh", key, self.key_encoder))
        effective = self.W0[None] + self.alpha[None] * state.weights["trace"]
        retrieved = torch.einsum("bqmi,bmij->bqmj", pre, effective)
        decoded = torch.einsum("bqmi,mij->bqmj", retrieved, self.decoder)
        active = self.route(queries).to(decoded.dtype)
        combined = (decoded * active[..., None]).sum(dim=2) / active.sum(dim=2, keepdim=True)
        return self.codebook.decode(combined)

    def detach_state(self, state):
        self._validate_state(state, device=self.device)
        return self._state(state.weights["trace"].detach(), state.write_count)

    def export_state(self, state):
        batch = self._validate_state(state, device=self.device, finite=True)
        self.assert_codebook_valid()
        return dict(schema_version=1, method=self.method, config=self.config.to_dict(),
                    config_fingerprint=self.config.fingerprint, batch_size=batch,
                    write_count=state.write_count, weights={"trace": state.weights["trace"].detach().clone()}, aux=None)

    def import_state(self, payload):
        if (type(payload) is not dict or set(payload) != self._payload_fields
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or type(payload["config"]) is not dict or payload["config"] != self.config.to_dict()
                or payload["method"] != self.method or payload["config_fingerprint"] != self.config.fingerprint):
            raise ValueError("State payload schema, method or configuration mismatch")
        try:
            parsed = AssociativeConfig(**payload["config"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid state configuration") from error
        if parsed.to_dict() != self.config.to_dict():
            raise ValueError("State configuration mismatch")
        self._batch(payload["batch_size"])
        state = KVState(payload["weights"], payload["aux"], payload["write_count"], payload["method"], payload["config_fingerprint"])
        if self._validate_state(state, finite=True) != payload["batch_size"]:
            raise ValueError("Payload batch differs from traces")
        self.assert_codebook_valid()
        return self._state(state.weights["trace"].detach().to(self.device).clone(), state.write_count)
