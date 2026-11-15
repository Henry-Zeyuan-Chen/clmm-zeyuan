"""Correlated fixed-key controls with the unchanged nonlinear KV memory."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from .kv_control import FixedCodebook, KVConfig, KVState, NeuralAdapter, _copy_tree


def codebook_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash shape, dtype, and canonical little-endian float32 tensor bytes."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
        raise ValueError("Key codebooks must be float32 tensors")
    cpu = tensor.detach().cpu().contiguous()
    header = json.dumps({"shape": list(cpu.shape), "dtype": str(cpu.dtype)},
                        sort_keys=True, separators=(",", ":")).encode("utf8")
    canonical = cpu.numpy().astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(header + b"\0" + canonical).hexdigest()


def _key_codes(nkeys, dim, rank, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(nkeys, dim, generator=generator, dtype=torch.float32, device="cpu")
    keys = torch.zeros(nkeys, dim, dtype=torch.float32, device="cpu")
    keys[:, :rank] = F.normalize(gaussian[:, :rank], dim=-1)
    return keys


@dataclass(frozen=True)
class CorrelatedConfig:
    nkeys: int = 64
    nvalues: int = 16
    dim: int = 64
    hidden: int = 128
    step_size: float = 0.5
    rank: int = 32
    codebook_seed: int = 90501
    key_codebook_sha256: str | None = None

    def __post_init__(self):
        base = KVConfig(self.nkeys, self.nvalues, self.dim, self.hidden, self.step_size)
        if (self.nkeys, self.nvalues, self.dim, self.hidden) != (64, 16, 64, 128):
            raise ValueError("Correlated controls retain the fixed 64/16/64/128 geometry")
        if type(self.rank) is not int or self.rank not in (8, 16, 32):
            raise ValueError("rank must be one of 8, 16, or 32")
        if type(self.codebook_seed) is not int or not 0 <= self.codebook_seed < 2**64:
            raise ValueError("codebook_seed must be an unsigned 64-bit integer")
        digest = codebook_tensor_sha256(_key_codes(self.nkeys, self.dim, self.rank, self.codebook_seed))
        if self.key_codebook_sha256 is not None and self.key_codebook_sha256 != digest:
            raise ValueError("Configured key codebook SHA256 does not match its deterministic construction")
        object.__setattr__(self, "step_size", base.step_size)
        object.__setattr__(self, "key_codebook_sha256", digest)

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf8")
        return hashlib.sha256(encoded).hexdigest()


class CorrelatedCodebook(FixedCodebook):
    """Reuse the fixed encoder/decoder API with nested Gaussian key subspaces."""

    def __init__(self, config: CorrelatedConfig):
        nn.Module.__init__(self)
        self.config = config
        self.register_buffer("keys", _key_codes(config.nkeys, config.dim, config.rank, config.codebook_seed))
        self.register_buffer("values", torch.eye(config.dim, dtype=torch.float32)[:config.nvalues].clone())

    def _validate_tensors(self, keys, values):
        expected_keys = _key_codes(self.config.nkeys, self.config.dim, self.config.rank, self.config.codebook_seed)
        expected_values = torch.eye(self.config.dim, dtype=torch.float32)[:self.config.nvalues]
        for name, tensor, expected in (("keys", keys, expected_keys), ("values", values, expected_values)):
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                    or tensor.layout != torch.strided or tensor.shape != expected.shape
                    or not torch.equal(tensor.detach().cpu(), expected)):
                raise ValueError(f"Loaded {name} codebook does not match the deterministic configuration")
        if codebook_tensor_sha256(keys) != self.config.key_codebook_sha256:
            raise ValueError("Key codebook SHA256 mismatch")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key_name, value_name = prefix + "keys", prefix + "values"
        if key_name not in state_dict or value_name not in state_dict:
            raise ValueError("Correlated model checkpoints must include both fixed codebooks")
        self._validate_tensors(state_dict[key_name], state_dict[value_name])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class CorrelatedNeuralAdapter(NeuralAdapter):
    """Same 64x128x64 engine and writes; only the fixed key representation changes."""
    method = "neural_correlated"

    def __init__(self, config: CorrelatedConfig | None = None):
        config = CorrelatedConfig() if config is None else config
        if not isinstance(config, CorrelatedConfig):
            raise ValueError("config must be a CorrelatedConfig")
        super().__init__(KVConfig(config.nkeys, config.nvalues, config.dim, config.hidden, config.step_size))
        self.config = config
        self.codebook = CorrelatedCodebook(config)
        self.assert_codebook_valid()

    @property
    def codebook_hash(self):
        return codebook_tensor_sha256(self.codebook.keys)

    def assert_codebook_valid(self):
        self.codebook._validate_tensors(self.codebook.keys, self.codebook.values)

    @property
    def config_deviations(self):
        return {**super().config_deviations, "fixed_representation": "unit_gaussian_key_subspace",
                "key_rank": self.config.rank, "codebook_seed": self.config.codebook_seed,
                "key_codebook_sha256": self.config.key_codebook_sha256,
                "value_representation": "unit_one_hot", "decoder": "first_16_raw_coordinates"}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.assert_codebook_valid()
        return result

    def export_state(self, state):
        self.assert_codebook_valid()
        return super().export_state(state)

    def import_state(self, payload):
        if type(payload) is not dict or set(payload) != self._payload_fields:
            raise ValueError("state payload fields do not match the schema")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported state payload schema version")
        if payload["method"] != self.method or type(payload["config"]) is not dict:
            raise ValueError("state method or config does not match the adapter")
        try:
            incoming_config = CorrelatedConfig(**payload["config"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid correlated state config") from error
        if incoming_config != self.config or payload["config_fingerprint"] != self.config.fingerprint:
            raise ValueError("state config or codebook fingerprint does not match the adapter")
        self.assert_codebook_valid()
        self._batch(payload["batch_size"])
        state = KVState(payload["weights"], payload["aux"], payload["write_count"],
                        payload["method"], payload["config_fingerprint"])
        if self._validate_state(state, finite=True) != payload["batch_size"]:
            raise ValueError("payload batch size does not match the state weights")
        weights, aux = _copy_tree((state.weights, state.aux), device=self.device, clone=True)
        return self._new_state(weights, aux, state.write_count)
