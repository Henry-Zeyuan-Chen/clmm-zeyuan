"""Reference active-module revision rules with fixed, key-only routing.

The factorial crosses active bulk EMA with rank-one key-local erasure. Inactive
modules are always protected. Erasure acts on the fast trace response x^T H;
it makes no overwrite guarantee for W0 + alpha * H or the decoded output.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor

from clmm.hebbian.adapter import AssociativeAdapter, AssociativeConfig
from clmm.hebbian.routing_decay import RoutingDecayAdapter, RoutingDecayConfig
from clmm.pretrain.kv_control import KVState


REVISION_RULES = ("ema", "additive", "erase_ema", "local_delta")


@dataclass(frozen=True)
class RevisionConfig(RoutingDecayConfig):
    revision_rule: str = "ema"
    projection_eps: float = 1e-8

    def __post_init__(self):
        super().__post_init__()
        if self.update_rule != "hebbian":
            raise ValueError("Revision rules require update_rule=hebbian")
        if self.inactive_decay != "protected":
            raise ValueError("Revision rules always protect inactive modules")
        if self.revision_rule not in REVISION_RULES:
            raise ValueError(f"revision_rule must be one of {REVISION_RULES}")
        if (isinstance(self.projection_eps, bool)
                or not isinstance(self.projection_eps, Real)
                or not math.isfinite(self.projection_eps) or self.projection_eps <= 0):
            raise ValueError("projection_eps must be positive and finite")
        object.__setattr__(self, "projection_eps", float(self.projection_eps))

    @property
    def active_bulk_decay(self):
        return self.revision_rule in ("ema", "erase_ema")

    @property
    def key_local_erasure(self):
        return self.revision_rule in ("erase_ema", "local_delta")

    def to_dict(self):
        return {**super().to_dict(), "revision_rule": self.revision_rule,
                "projection_eps": self.projection_eps}


class RevisionAdapter(RoutingDecayAdapter):
    """Fixed slow weights and bounded fast traces for a 2 x 2 intervention.

    For an active module, x=tanh(key_encoder(k)), y=tanh(value_encoder(v)),
    s=max(sum(x**2), eps), and P_x H=(x/s) outer (x^T H). The update is
    clamp(d * (H - e * P_x H) + eta * x outer y), with d=1-eta for bulk
    EMA and d=1 otherwise; e is one exactly when key-local erasure is on.
    W0 and alpha remain trainable outer parameters and are read-only online.
    """

    def __init__(self, config: RevisionConfig, keys: Tensor, values: Tensor,
                 route_table: Tensor | None = None):
        if not isinstance(config, RevisionConfig):
            raise ValueError("config must be a RevisionConfig")
        super().__init__(config, keys, values, route_table=route_table)

    @classmethod
    def from_parent(cls, parent: AssociativeAdapter, *, revision_rule: str,
                    policy: str, random_seed: int | None = None):
        """Clone slow values and materialize a route table; start with H=0.

        No parent fast state is accepted or copied, and the global CPU RNG and
        parent parameter values, training mode, and gradient flags are untouched.
        """
        if (not isinstance(parent, AssociativeAdapter)
                or not isinstance(parent.config, AssociativeConfig)
                or parent.config.routing != "dense"):
            raise ValueError("A revision fork requires a dense AssociativeAdapter parent")
        parent.assert_codebook_valid()
        fields = parent.config.to_dict()
        fields.pop("routing")
        config = RevisionConfig(**fields, revision_rule=revision_rule, policy=policy,
                                random_seed=random_seed)
        with torch.random.fork_rng(devices=[]):
            child = cls(config, parent.codebook.keys, parent.codebook.values).to(parent.device)
        child.load_state_dict({**parent.state_dict(), "route_table": child.route_table}, strict=True)
        child.train(parent.training)
        source_parameters = dict(parent.named_parameters())
        for name, parameter in child.named_parameters():
            parameter.requires_grad_(source_parameters[name].requires_grad)
        return child

    @property
    def config_deviations(self):
        return {**super().config_deviations,
                "component": "reference_revision_factorial",
                "experiment": "configurable reference rule; run protocol records frozen or trained slow weights",
                "revision_rule": self.config.revision_rule,
                "active_bulk_decay": self.config.active_bulk_decay,
                "key_local_erasure": self.config.key_local_erasure,
                "projection_eps": self.config.projection_eps,
                "plasticity": "active trace rank-one erasure and/or bulk EMA, then Hebbian write and clamp",
                "decay_applications_per_write": int(self.config.active_bulk_decay),
                "projection_applications_per_write": int(self.config.key_local_erasure),
                "erasure_target": "x^T H only; no alpha or W0 correction; no decoded-output overwrite guarantee"}

    def decay_mask(self, key_ids):
        active = self.route(key_ids)
        return active if self.config.active_bulk_decay else torch.zeros_like(active)

    def projection_mask(self, key_ids):
        active = self.route(key_ids)
        return active if self.config.key_local_erasure else torch.zeros_like(active)

    def write(self, keys, values, state):
        if self.config.revision_rule == "ema":
            # Preserve the legacy multiplication/addition/clamp order exactly.
            return super().write(keys, values, state)
        batch = self._validate_state(state, device=self.device)
        if (not isinstance(keys, Tensor) or not isinstance(values, Tensor)
                or keys.shape != (batch,) or values.shape != (batch,)):
            raise ValueError("write keys and values must have shape[batch]")
        key = self.codebook.encode_keys(keys)
        value = self.codebook.encode_values(values)
        pre = torch.tanh(torch.einsum("bd,mdh->bmh", key, self.key_encoder))
        post = torch.tanh(torch.einsum("bd,mdh->bmh", value, self.value_encoder))
        trace = state.weights["trace"]
        retained = trace
        if self.config.key_local_erasure:
            response = torch.einsum("bmi,bmij->bmj", pre, trace)
            # A positive Python epsilon can underflow when cast to float32.
            # Its representable positive floor keeps zero-norm projections safe.
            eps = max(self.config.projection_eps, torch.finfo(pre.dtype).tiny)
            norm = pre.square().sum(dim=-1, keepdim=True).clamp_min(pre.new_tensor(eps))
            projection = (pre / norm)[..., :, None] * response[..., None, :]
            retained = trace - projection
        if self.config.active_bulk_decay:
            retained = (1 - self.eta) * retained
        change = pre[..., :, None] * post[..., None, :]
        updated = (retained + self.eta * change).clamp(-self.config.trace_bound,
                                                     self.config.trace_bound)
        active = self.route(keys)[..., None, None]
        return self._state(torch.where(active, updated, trace), state.write_count + 1)

    def export_state(self, state):
        payload = super().export_state(state)
        payload["schema_version"] = 3
        return payload

    def import_state(self, payload):
        if (type(payload) is not dict or set(payload) != self._payload_fields
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 3
                or type(payload["config"]) is not dict or payload["config"] != self.config.to_dict()
                or payload["method"] != self.method
                or payload["config_fingerprint"] != self.config.fingerprint):
            raise ValueError("Revision state schema, method or configuration mismatch")
        try:
            parsed = RevisionConfig(**payload["config"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid revision state configuration") from error
        if parsed.to_dict() != self.config.to_dict():
            raise ValueError("State configuration mismatch")
        self._batch(payload["batch_size"])
        state = KVState(payload["weights"], payload["aux"], payload["write_count"],
                        payload["method"], payload["config_fingerprint"])
        if self._validate_state(state, finite=True) != payload["batch_size"]:
            raise ValueError("Payload batch differs from traces")
        self.assert_codebook_valid()
        return self._state(state.weights["trace"].detach().to(self.device).clone(), state.write_count)
