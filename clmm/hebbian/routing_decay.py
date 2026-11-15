"""Fixed-slow routing/decay factorial for the reference associative adapter.

The only interventions are a materialized key-only top-one routing table and
whether inactive modules decay during a write. No fast state is transferred
from a parent, and no random choices are made while observing a stream.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from clmm.hebbian.adapter import AssociativeAdapter, AssociativeConfig
from clmm.pretrain.kv_control import KVState
from clmm.pretrain.kv_correlated_portable import PortableCorrelatedConfig


ROUTING_ALGORITHM = "literal_cosine_top1_pcg64_permutation_v1"


def routing_table_sha256(table: Tensor) -> str:
    """Hash shape, dtype, and canonical little-endian int64 table bytes."""
    if (not isinstance(table, Tensor) or table.dtype != torch.int64
            or table.layout != torch.strided or tuple(table.shape) != (64,)
            or ((table < 0) | (table >= 4)).any()):
        raise ValueError("Routing table must contain 64 int64 module IDs in [0, 4)")
    cpu = table.detach().cpu().contiguous()
    header = json.dumps({"shape": list(cpu.shape), "dtype": str(cpu.dtype)},
                        sort_keys=True, separators=(",", ":")).encode("utf8")
    canonical = cpu.numpy().astype("<i8", copy=False).tobytes(order="C")
    return hashlib.sha256(header + b"\0" + canonical).hexdigest()


@dataclass(frozen=True)
class RoutingDecayConfig:
    codebook_config: PortableCorrelatedConfig | dict
    modules: int = 4
    dim: int = 64
    update_rule: str = "hebbian"
    eta_init: float = 0.1
    trace_bound: float = 1.0
    alpha_init: float = 1.0
    init_noise: float = 0.01
    policy: str = "content"
    random_seed: int | None = None
    inactive_decay: str = "protected"
    routing_algorithm: str = ROUTING_ALGORITHM
    routing_table_sha256: str | None = None

    def __post_init__(self):
        # Reuse all frozen geometry and numerical validation without changing
        # the reference configuration or its fingerprint/schema.
        base = self.base_config()
        for name in ("codebook_config", "eta_init", "trace_bound", "alpha_init", "init_noise"):
            object.__setattr__(self, name, getattr(base, name))
        if self.policy not in ("content", "random"):
            raise ValueError("policy must be content or random")
        if self.inactive_decay not in ("protected", "global"):
            raise ValueError("inactive_decay must be protected or global")
        if self.routing_algorithm != ROUTING_ALGORITHM:
            raise ValueError("Unsupported routing algorithm")
        if self.policy == "content" and self.random_seed is not None:
            raise ValueError("Content routing has no random seed")
        if self.policy == "random" and (type(self.random_seed) is not int
                or not 0 <= self.random_seed < 2**64):
            raise ValueError("Random routing requires an unsigned 64-bit seed")
        if (self.routing_table_sha256 is not None
                and (type(self.routing_table_sha256) is not str
                     or re.fullmatch(r"[0-9a-f]{64}", self.routing_table_sha256) is None)):
            raise ValueError("Invalid routing table SHA256")

    def base_config(self):
        return AssociativeConfig(self.codebook_config, modules=self.modules, dim=self.dim,
                                 routing="top1", update_rule=self.update_rule,
                                 eta_init=self.eta_init, trace_bound=self.trace_bound,
                                 alpha_init=self.alpha_init, init_noise=self.init_noise)

    @property
    def nkeys(self):
        return self.codebook_config.nkeys

    @property
    def nvalues(self):
        return self.codebook_config.nvalues

    @property
    def routing(self):
        return "top1"

    def to_dict(self):
        result = self.base_config().to_dict()
        result.pop("routing")
        result.update(policy=self.policy, random_seed=self.random_seed,
                      inactive_decay=self.inactive_decay, routing_algorithm=self.routing_algorithm,
                      routing_table_sha256=self.routing_table_sha256)
        return result

    @property
    def fingerprint(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf8")
        return hashlib.sha256(encoded).hexdigest()


def _materialize_route_table(config: RoutingDecayConfig, keys: Tensor) -> Tensor:
    # Always construct on the CPU from the actual saved literal bits. Moving
    # an adapter to another device never recomputes its routing decisions.
    cpu_keys = keys.detach().cpu()
    prototypes = cpu_keys[list(AssociativeAdapter._prototype_ids)]
    similarities = F.normalize(cpu_keys, dim=-1) @ F.normalize(prototypes, dim=-1).T
    content = similarities.argmax(dim=-1).to(torch.int64)
    if config.policy == "content":
        return content
    # Permute the assignments across key IDs, retaining exact module sizes.
    # A private generator leaves both global NumPy and torch RNGs untouched.
    generator = np.random.Generator(np.random.PCG64(config.random_seed))
    return torch.from_numpy(generator.permutation(content.numpy()).copy())


class RoutingDecayAdapter(AssociativeAdapter):
    """One active association update/read, with independent inactive decay."""

    def __init__(self, config: RoutingDecayConfig, keys: Tensor, values: Tensor,
                 route_table: Tensor | None = None):
        if not isinstance(config, RoutingDecayConfig):
            raise ValueError("config must be a RoutingDecayConfig")
        super().__init__(config.base_config(), keys, values)
        expected = _materialize_route_table(config, self.codebook.keys)
        table = expected if route_table is None else route_table
        digest = routing_table_sha256(table)
        if not torch.equal(table.detach().cpu(), expected):
            raise ValueError("Routing table disagrees with its policy, seed, or literal codebook")
        if config.routing_table_sha256 is not None and config.routing_table_sha256 != digest:
            raise ValueError("Routing table SHA256 differs from configuration")
        self.config = replace(config, routing_table_sha256=digest)
        self.register_buffer("route_table", table.detach().cpu().clone())
        self.assert_codebook_valid()

    @classmethod
    def from_parent(cls, parent: AssociativeAdapter, *, policy: str,
                    inactive_decay: str, random_seed: int | None = None):
        """Clone identical slow values from a dense parent; start new H=0.

        The parent is never modified. Its optimizer or any old fast state is
        deliberately outside this constructor's API.
        """
        if (not isinstance(parent, AssociativeAdapter)
                or not isinstance(parent.config, AssociativeConfig)
                or parent.config.routing != "dense"):
            raise ValueError("A frozen-transfer fork requires a dense AssociativeAdapter parent")
        parent.assert_codebook_valid()
        fields = parent.config.to_dict()
        fields.pop("routing")
        config = RoutingDecayConfig(**fields, policy=policy, inactive_decay=inactive_decay,
                                    random_seed=random_seed)
        with torch.random.fork_rng(devices=[]):
            child = cls(config, parent.codebook.keys, parent.codebook.values).to(parent.device)
        child.load_state_dict({**parent.state_dict(), "route_table": child.route_table}, strict=True)
        child.train(parent.training)
        for name, parameter in child.named_parameters():
            parameter.requires_grad_(dict(parent.named_parameters())[name].requires_grad)
        return child

    @property
    def routing_table_sha256(self):
        return routing_table_sha256(self.route_table)

    @property
    def config_deviations(self):
        return {**super().config_deviations, "component": "reference_routing_decay_factorial",
                "experiment": "fixed identical slow weights; fresh zero traces; no new training",
                "routing": self.config.policy, "routing_algorithm": self.config.routing_algorithm,
                "router": "materialized key-only int64 table; identical write/read assignment",
                "routing_table_sha256": self.config.routing_table_sha256,
                "random_seed": self.config.random_seed, "inactive_decay": self.config.inactive_decay,
                "inactive_modules": ("unchanged, including no decay" if self.config.inactive_decay == "protected"
                                     else "all three inactive modules decay by (1-eta) on each write"),
                "association_updates_per_write": 1,
                "decay_applications_per_write": 1 if self.config.inactive_decay == "protected" else 4,
                "routing_table_bytes": 512, "full_titans_model": False,
                "prototype_status": "reference mechanism, not final teammate method"}

    def assert_codebook_valid(self):
        super().assert_codebook_valid()
        if (routing_table_sha256(self.route_table) != self.config.routing_table_sha256
                or not torch.equal(self.route_table.detach().cpu(),
                                   _materialize_route_table(self.config, self.codebook.keys))):
            raise ValueError("Routing table differs from its saved construction or digest")

    def route(self, key_ids):
        # Reuse exact dtype/device/range validation from the shared codebook.
        self.codebook.encode_keys(key_ids)
        return F.one_hot(self.route_table[key_ids], self.config.modules).bool()

    def decay_mask(self, key_ids):
        active = self.route(key_ids)
        return active if self.config.inactive_decay == "protected" else torch.ones_like(active)

    def write(self, keys, values, state):
        # The base implementation provides the exact pre/post, current-state
        # gradient or Hebbian change, active EMA, clamp, and write counter.
        written = super().write(keys, values, state)
        if self.config.inactive_decay == "protected":
            return written
        active = self.route(keys)[..., None, None]
        decayed = (1 - self.eta) * state.weights["trace"]
        return self._state(torch.where(active, written.weights["trace"], decayed), written.write_count)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict:
            raise ValueError("Routing/decay checkpoints require strict loading")
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("Checkpoint tensor names do not match the routing/decay adapter")
        from clmm.pretrain.kv_correlated_portable import _validate_literal_codebook
        _validate_literal_codebook(self.config.codebook_config,
                                   state_dict["codebook.keys"], state_dict["codebook.values"])
        for name, tensor in state_dict.items():
            dtype = torch.int64 if name == "route_table" else torch.float32
            if (not isinstance(tensor, Tensor) or tensor.shape != expected[name].shape
                    or tensor.dtype != dtype or tensor.layout != torch.strided
                    or not torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid checkpoint tensor: {name}")
        table = state_dict["route_table"]
        if (routing_table_sha256(table) != self.config.routing_table_sha256
                or not torch.equal(table.detach().cpu(),
                                   _materialize_route_table(self.config, state_dict["codebook.keys"]))):
            raise ValueError("Checkpoint routing policy, seed, table, or hash mismatch")
        if not torch.equal(state_dict["router_prototypes"].detach().cpu(),
                           state_dict["codebook.keys"][list(self._prototype_ids)].detach().cpu()):
            raise ValueError("Checkpoint changes the fixed key-only prototypes")
        # Bypass only the reference adapter's all-float32/name restriction.
        result = nn.Module.load_state_dict(self, state_dict, strict=True, assign=assign)
        self.assert_codebook_valid()
        return result

    def export_state(self, state):
        payload = super().export_state(state)
        payload["schema_version"] = 2
        return payload

    def import_state(self, payload):
        if (type(payload) is not dict or set(payload) != self._payload_fields
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 2
                or type(payload["config"]) is not dict or payload["config"] != self.config.to_dict()
                or payload["method"] != self.method or payload["config_fingerprint"] != self.config.fingerprint):
            raise ValueError("Routing/decay state schema, method or configuration mismatch")
        try:
            parsed = RoutingDecayConfig(**payload["config"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid routing/decay state configuration") from error
        if parsed.to_dict() != self.config.to_dict():
            raise ValueError("State configuration mismatch")
        self._batch(payload["batch_size"])
        state = KVState(payload["weights"], payload["aux"], payload["write_count"],
                        payload["method"], payload["config_fingerprint"])
        if self._validate_state(state, finite=True) != payload["batch_size"]:
            raise ValueError("Payload batch differs from traces")
        self.assert_codebook_valid()
        return self._state(state.weights["trace"].detach().to(self.device).clone(), state.write_count)
