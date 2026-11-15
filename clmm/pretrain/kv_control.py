"""Explicit-representation positive control for persistent key/value memory.

This control has its own frozen one-hot interface. Its memory weights are not
compatible with the byte language model. Writes receive facts; reads receive
only keys, and the bounded fast state never contains an event or query log.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn

from titans_pytorch import NeuralMemory
from titans_pytorch.memory_models import MemoryMLP


@dataclass(frozen=True)
class KVConfig:
    nkeys: int = 64
    nvalues: int = 16
    dim: int = 64
    hidden: int = 128
    step_size: float = 0.5

    def __post_init__(self) -> None:
        for name in ("nkeys", "nvalues", "dim", "hidden"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.nkeys > self.dim or self.nvalues > self.dim:
            raise ValueError("nkeys and nvalues must not exceed dim")
        if self.hidden != 2 * self.dim:
            raise ValueError("hidden must equal 2 * dim for the fixed depth-two MLP")
        if (isinstance(self.step_size, bool) or not isinstance(self.step_size, Real)
                or not math.isfinite(self.step_size) or self.step_size <= 0):
            raise ValueError("step_size must be finite and positive")
        object.__setattr__(self, "step_size", float(self.step_size))

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KVState:
    weights: dict[str, Tensor]
    aux: Any
    write_count: int
    method: str
    config_fingerprint: str

    @property
    def batch_size(self) -> int:
        return next(iter(self.weights.values())).shape[0]


class FixedCodebook(nn.Module):
    """Unit orthogonal keys and values, with a fixed coordinate decoder."""

    def __init__(self, config: KVConfig):
        super().__init__()
        self.config = config
        basis = torch.eye(config.dim, dtype=torch.float32)
        self.register_buffer("keys", basis[:config.nkeys].clone())
        self.register_buffer("values", basis[:config.nvalues].clone())

    def _encode(self, ids: Tensor, table: Tensor, name: str) -> Tensor:
        if not isinstance(ids, Tensor) or ids.dtype != torch.long:
            raise ValueError(f"{name} must be an int64 tensor")
        if ids.device != table.device:
            raise ValueError(f"{name} and the adapter must be on the same device")
        if ids.numel() and ((ids < 0).any() or (ids >= table.shape[0]).any()):
            raise ValueError(f"{name} contain an out-of-range code")
        return table[ids]

    def encode_keys(self, ids: Tensor) -> Tensor:
        return self._encode(ids, self.keys, "keys")

    def encode_values(self, ids: Tensor) -> Tensor:
        return self._encode(ids, self.values, "values")

    def decode(self, retrieved: Tensor) -> Tensor:
        if retrieved.ndim < 1 or retrieved.shape[-1] != self.config.dim:
            raise ValueError("retrieved vectors must end in the codebook dimension")
        return retrieved[..., :self.config.nvalues]


def half_sum_squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    return 0.5 * (prediction - target).square().sum(dim=-1)


def _plain_tree(value: Any) -> Any:
    """Remove TensorDict containers without detaching differentiable tensors."""
    if isinstance(value, Tensor) or value is None:
        return value
    if hasattr(value, "items"):
        return {name: _plain_tree(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_tree(item) for item in value)
    raise ValueError("fast state must contain only dicts, tuples, tensors, or None")


def _copy_tree(value: Any, *, device=None, clone=False, memo=None) -> Any:
    """Detach state while preserving repeated tensor references in aux."""
    if memo is None:
        memo = {}
    if isinstance(value, Tensor):
        identity = id(value)
        if identity not in memo:
            copied = value.detach()
            if device is not None:
                copied = copied.to(device=device)
            if clone:
                copied = copied.clone()
            memo[identity] = copied
        return memo[identity]
    if value is None:
        return None
    if type(value) is dict:
        return {name: _copy_tree(item, device=device, clone=clone, memo=memo)
                for name, item in value.items()}
    if type(value) is tuple:
        return tuple(_copy_tree(item, device=device, clone=clone, memo=memo)
                     for item in value)
    raise ValueError("fast state must contain only plain dicts, tuples, tensors, or None")


def physical_state_bytes(state: KVState) -> int:
    """Count unique backing storage bytes, including any auxiliary state views.

    Aliases of fast weights in the vendored update state count once. Storage
    size, rather than numel, also includes unused elements behind tensor views.
    Autograd graphs and the model's separate slow parameters are not fast state.
    """
    storages = {}

    def visit(value):
        if isinstance(value, Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage._cdata)
            storages[key] = storage.nbytes()
        elif type(value) is dict:
            for item in value.values():
                visit(item)
        elif type(value) is tuple:
            for item in value:
                visit(item)
        elif value is not None:
            raise ValueError("invalid fast-state container")

    visit(state.weights)
    visit(state.aux)
    return sum(storages.values())


class _KVAdapter(nn.Module):
    method: str
    _payload_fields = {"schema_version", "method", "config", "config_fingerprint",
                       "batch_size", "write_count", "weights", "aux"}

    def __init__(self, config: KVConfig | None = None):
        super().__init__()
        self.config = config if config is not None else KVConfig()
        if not isinstance(self.config, KVConfig):
            raise ValueError("config must be a KVConfig")
        self.codebook = FixedCodebook(self.config)

    @property
    def device(self) -> torch.device:
        return self.codebook.keys.device

    def parameter_counts(self) -> dict[str, int]:
        parameters = tuple(self.parameters())
        return {"registered": sum(parameter.numel() for parameter in parameters),
                "trainable": sum(parameter.numel() for parameter in parameters
                                 if parameter.requires_grad)}

    def _new_state(self, weights, aux, count):
        return KVState(weights, aux, count, self.method, self.config.fingerprint)

    def _batch(self, batch: int) -> None:
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")

    def _weight_shapes(self, batch: int) -> dict[str, tuple[int, ...]]:
        raise NotImplementedError

    def _validate_weights(self, weights, batch, *, device, finite):
        expected = self._weight_shapes(batch)
        if type(weights) is not dict or set(weights) != set(expected):
            raise ValueError("state weight names do not match the adapter")
        for name, shape in expected.items():
            tensor = weights[name]
            if not isinstance(tensor, Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"state weight {name} has the wrong shape or batch size")
            if tensor.dtype != self.codebook.keys.dtype or tensor.layout != torch.strided:
                raise ValueError(f"state weight {name} has the wrong dtype or layout")
            if device is not None and tensor.device != device:
                raise ValueError("state and the adapter must be on the same device")
            if finite and not torch.isfinite(tensor).all():
                raise ValueError("state tensors must be finite")

    def _validate_state(self, state, *, device=None, finite=False):
        if not isinstance(state, KVState):
            raise ValueError("state must be a KVState")
        if state.method != self.method or state.config_fingerprint != self.config.fingerprint:
            raise ValueError("state method or config does not match the adapter")
        if type(state.write_count) is not int or state.write_count < 0:
            raise ValueError("state write_count must be a nonnegative integer")
        if type(state.weights) is not dict or not state.weights:
            raise ValueError("state weights must be a nonempty plain dict")
        first = next(iter(state.weights.values()))
        if not isinstance(first, Tensor) or first.ndim < 1:
            raise ValueError("state weights must contain batched tensors")
        batch = first.shape[0]
        self._batch(batch)
        self._validate_weights(state.weights, batch, device=device, finite=finite)
        if self.method == "delta":
            if state.aux is not None:
                raise ValueError("delta state must not contain auxiliary data")
        elif state.write_count == 0:
            if state.aux is not None:
                raise ValueError("initial neural state must not contain auxiliary data")
        else:
            if (type(state.aux) is not tuple or len(state.aux) != 2
                    or type(state.aux[1]) is not dict or state.aux[1]):
                raise ValueError("neural auxiliary state must be (fast weights, empty momentum)")
            self._validate_weights(state.aux[0], batch, device=device, finite=finite)
            if finite and any(not torch.equal(state.weights[name], state.aux[0][name])
                              for name in state.weights):
                raise ValueError("neural auxiliary weights must equal the active weights")
        return batch

    def _write_inputs(self, keys, values, state):
        batch = self._validate_state(state, device=self.device)
        if (not isinstance(keys, Tensor) or not isinstance(values, Tensor)
                or keys.shape != (batch,) or values.shape != (batch,)):
            raise ValueError("write keys and values must have shape [batch]")
        return self.codebook.encode_keys(keys), self.codebook.encode_values(values)

    def _read_inputs(self, queries, state):
        batch = self._validate_state(state, device=self.device)
        if (not isinstance(queries, Tensor) or queries.ndim != 2
                or queries.shape[0] != batch or queries.shape[1] < 1):
            raise ValueError("queries must have shape [batch, queries] with queries > 0")
        return self.codebook.encode_keys(queries)

    def detach_state(self, state: KVState) -> KVState:
        self._validate_state(state, device=self.device)
        weights, aux = _copy_tree((state.weights, state.aux))
        return self._new_state(weights, aux, state.write_count)

    def export_state(self, state: KVState) -> dict:
        self._validate_state(state, device=self.device, finite=True)
        weights, aux = _copy_tree((state.weights, state.aux), device="cpu", clone=True)
        return {"schema_version": 1, "method": self.method,
                "config": self.config.to_dict(), "config_fingerprint": self.config.fingerprint,
                "batch_size": state.batch_size, "write_count": state.write_count,
                "weights": weights, "aux": aux}

    def import_state(self, payload: dict) -> KVState:
        if type(payload) is not dict or set(payload) != self._payload_fields:
            raise ValueError("state payload fields do not match the schema")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported state payload schema version")
        if payload["method"] != self.method:
            raise ValueError("state method does not match the adapter")
        if type(payload["config"]) is not dict:
            raise ValueError("state config does not match the adapter")
        try:
            incoming_config = KVConfig(**payload["config"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid state config") from error
        if (incoming_config != self.config
                or payload["config_fingerprint"] != self.config.fingerprint):
            raise ValueError("state config does not match the adapter")
        self._batch(payload["batch_size"])
        state = KVState(payload["weights"], payload["aux"], payload["write_count"],
                        payload["method"], payload["config_fingerprint"])
        batch = self._validate_state(state, finite=True)
        if batch != payload["batch_size"]:
            raise ValueError("payload batch size does not match the state weights")
        weights, aux = _copy_tree((state.weights, state.aux), device=self.device, clone=True)
        return self._new_state(weights, aux, state.write_count)


class DeltaAdapter(_KVAdapter):
    """Exact linear half-sum-SSE gradient write, with step one and no decay."""

    method = "delta"

    @property
    def config_deviations(self) -> dict:
        return {"memory_model": "linear", "inner_step_size": 1.0,
                "weight_decay": 0.0, "fixed_representation": "unit_one_hot"}

    def _weight_shapes(self, batch):
        return {"weight": (batch, self.config.dim, self.config.dim)}

    def initial_state(self, batch: int) -> KVState:
        self._batch(batch)
        weights = self.codebook.keys.new_zeros(batch, self.config.dim, self.config.dim)
        return self._new_state({"weight": weights}, None, 0)

    def write(self, keys: Tensor, values: Tensor, state: KVState) -> KVState:
        key, value = self._write_inputs(keys, values, state)
        weight = state.weights["weight"]
        prediction = torch.bmm(key[:, None], weight)[:, 0]
        updated = weight + key[:, :, None] * (value - prediction)[:, None, :]
        return self._new_state({"weight": updated}, None, state.write_count + 1)

    def read(self, queries: Tensor, state: KVState) -> Tensor:
        key = self._read_inputs(queries, state)
        return self.codebook.decode(torch.bmm(key, state.weights["weight"]))


class NeuralAdapter(_KVAdapter):
    """Vendored nonlinear memory with only its initial fast weights trainable."""

    method = "neural"

    def __init__(self, config: KVConfig | None = None):
        super().__init__(config)
        self.engine = NeuralMemory(
            dim=self.config.dim, model=MemoryMLP(self.config.dim, depth=2, expansion_factor=2),
            heads=1, chunk_size=1, qkv_receives_diff_views=True,
            pre_rmsnorm=False, post_rmsnorm=False, qk_rmsnorm=False,
            momentum=False, mem_model_norm_add_residual=False,
            store_memory_loss_fn=half_sum_squared_error,
            adaptive_step_transform=lambda raw: torch.full_like(raw, self.config.step_size),
            init_decay_bias=-80.0,
        )
        self.engine.to_queries = nn.Identity()
        self.engine.to_keys = nn.Identity()
        self.engine.to_values = nn.Identity()
        self.engine.multihead_rmsnorm = nn.Identity()
        if self.engine.heads != 1 or not isinstance(self.engine.combine_heads, nn.Identity):
            raise RuntimeError("KV control requires an identity single-head output")
        if self.engine.retrieve_gate is not None:
            raise RuntimeError("KV control requires ungated retrieval")
        self.engine.requires_grad_(False)
        self.engine.memory_model_parameters.requires_grad_(True)
        # Both registered copies are zeroed. The first matrix keeps the
        # vendored Xavier initialization, so the zero-output MLP can learn.
        with torch.no_grad():
            self.engine.memory_model.weights[-1].zero_()
            self.engine.memory_model_parameters[-1].zero_()

    @property
    def config_deviations(self) -> dict:
        return {"memory_model": "MemoryMLP", "depth": 2, "expansion_factor": 2,
                "heads": 1, "chunk_size": 1, "qkv_receives_diff_views": True,
                "qkv_projections": "identity", "pre_rmsnorm": False,
                "post_rmsnorm": False, "qk_rmsnorm": False, "momentum": False,
                "mem_model_norm_add_residual": False, "retrieval_gate": None,
                "store_loss": "half_sum_squared_error", "inner_step_size": self.config.step_size,
                "decay_logit": -80.0, "trainable": "initial_memory_weights_only",
                "last_memory_matrix_init": "zero", "fixed_representation": "unit_one_hot"}

    def _weight_shapes(self, batch):
        return {name: (batch, *parameter.shape[1:]) for name, parameter in zip(
            self.engine.memory_model_parameter_names, self.engine.memory_model_parameters)}

    def initial_state(self, batch: int) -> KVState:
        self._batch(batch)
        return self._new_state(_plain_tree(self.engine.init_weights(batch)), None, 0)

    def write(self, keys: Tensor, values: Tensor, state: KVState) -> KVState:
        key, value = self._write_inputs(keys, values, state)
        fact = torch.stack((key[:, None], value[:, None]))
        _, returned = self.engine.store_memories(
            fact, weights=state.weights, past_state=state.aux,
            seq_index=state.write_count, return_surprises=False,
        )
        aux = _plain_tree(returned.states)
        return self._new_state(aux[0], aux, state.write_count + 1)

    def read(self, queries: Tensor, state: KVState) -> Tensor:
        key = self._read_inputs(queries, state)
        active = {name: weight.unsqueeze(1) for name, weight in state.weights.items()}
        # A single weight set is intentionally reused for every key-only query.
        retrieved = torch.cat([self.engine.retrieve_memories(key[:, index:index + 1], active)
                               for index in range(key.shape[1])], dim=1)
        return self.codebook.decode(retrieved)
