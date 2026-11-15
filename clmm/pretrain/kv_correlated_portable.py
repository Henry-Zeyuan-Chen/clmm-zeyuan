"""Load correlated controls with saved literal codebooks as the authority.

The frozen constructor requires normalization bits to reproduce exactly. This
compatibility adapter instead preserves the checkpoint's bits, validates their
canonical digest, and tolerates only small seeded-normalization differences.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re

import torch
from torch import nn

from .kv_control import FixedCodebook, KVConfig, KVState, NeuralAdapter, _copy_tree
from .kv_correlated import _key_codes, codebook_tensor_sha256


NORMALIZATION_ATOL = 2e-7
NORMALIZATION_RTOL = 1e-6


@dataclass(frozen=True)
class PortableCorrelatedConfig:
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
            raise ValueError("Portable correlated controls retain the fixed 64/16/64/128 geometry")
        if type(self.rank) is not int or self.rank not in (8, 16, 32):
            raise ValueError("rank must be one of 8, 16, or 32")
        if type(self.codebook_seed) is not int or not 0 <= self.codebook_seed < 2**64:
            raise ValueError("codebook_seed must be an unsigned 64-bit integer")
        if (type(self.key_codebook_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", self.key_codebook_sha256) is None):
            raise ValueError("A canonical saved key codebook SHA256 is required")
        object.__setattr__(self, "step_size", base.step_size)

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf8")
        return hashlib.sha256(encoded).hexdigest()


def _validate_literal_codebook(config, keys, values):
    for name, tensor, shape in (("keys", keys, (64, 64)), ("values", values, (16, 64))):
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.layout != torch.strided or tuple(tensor.shape) != shape
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"Saved {name} codebook must have finite float32 shape {shape}")
    if codebook_tensor_sha256(keys) != config.key_codebook_sha256:
        raise ValueError("Saved key codebook SHA256 does not match its configuration")
    cpu_keys, cpu_values = keys.detach().cpu(), values.detach().cpu()
    if torch.count_nonzero(cpu_keys[:, config.rank:]):
        raise ValueError("Saved keys violate the configured rank zero padding")
    if not torch.equal(cpu_values, torch.eye(64, dtype=torch.float32)[:16]):
        raise ValueError("Saved values must be the fixed unit one-hot codebook")
    if not torch.allclose(cpu_keys.norm(dim=-1), torch.ones(64),
                          atol=NORMALIZATION_ATOL, rtol=NORMALIZATION_RTOL):
        raise ValueError("Saved key rows must have unit norm")
    expected = _key_codes(config.nkeys, config.dim, config.rank, config.codebook_seed)
    if not torch.allclose(cpu_keys, expected, atol=NORMALIZATION_ATOL, rtol=NORMALIZATION_RTOL):
        raise ValueError("Saved keys do not match the seeded Gaussian construction within normalization tolerance")


class PortableCorrelatedCodebook(FixedCodebook):
    def __init__(self, config, keys, values):
        nn.Module.__init__(self)
        _validate_literal_codebook(config, keys, values)
        self.config = config
        # Clone without normalizing, regenerating, or changing a saved element.
        self.register_buffer("keys", keys.detach().cpu().clone())
        self.register_buffer("values", values.detach().cpu().clone())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key_name, value_name = prefix + "keys", prefix + "values"
        if key_name not in state_dict or value_name not in state_dict:
            raise ValueError("Portable correlated checkpoints must include both literal codebooks")
        _validate_literal_codebook(self.config, state_dict[key_name], state_dict[value_name])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class PortableCorrelatedNeuralAdapter(NeuralAdapter):
    method = "neural_correlated"

    def __init__(self, config: PortableCorrelatedConfig, keys: torch.Tensor, values: torch.Tensor):
        if not isinstance(config, PortableCorrelatedConfig):
            raise ValueError("config must be a PortableCorrelatedConfig")
        codebook = PortableCorrelatedCodebook(config, keys, values)
        super().__init__(KVConfig(config.nkeys, config.nvalues, config.dim, config.hidden, config.step_size))
        # A one-head CPU repeat can alias the frozen model matrices. CUDA .to()
        # separates those registered parameters in the original training run.
        # Separate storage here as well, without changing any initialization
        # value or RNG consumption, so strict loading preserves both copies.
        for index, parameter in enumerate(self.engine.memory_model_parameters):
            self.engine.memory_model_parameters[index] = nn.Parameter(
                parameter.detach().clone(), requires_grad=parameter.requires_grad)
        self.config, self.codebook = config, codebook

    @property
    def codebook_hash(self):
        return codebook_tensor_sha256(self.codebook.keys)

    def assert_codebook_valid(self):
        _validate_literal_codebook(self.config, self.codebook.keys, self.codebook.values)

    @property
    def config_deviations(self):
        return {**super().config_deviations, "fixed_representation": "unit_gaussian_key_subspace",
                "key_rank": self.config.rank, "codebook_seed": self.config.codebook_seed,
                "key_codebook_sha256": self.config.key_codebook_sha256,
                "value_representation": "unit_one_hot", "decoder": "first_16_raw_coordinates",
                "codebook_authority": "saved_literal_bits", "normalization_validation_atol": NORMALIZATION_ATOL,
                "normalization_validation_rtol": NORMALIZATION_RTOL}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        _validate_literal_codebook(self.config, state_dict.get("codebook.keys"), state_dict.get("codebook.values"))
        expected = self.state_dict()
        for name, tensor in state_dict.items():
            if name in expected and (not isinstance(tensor, torch.Tensor)
                    or tensor.dtype != expected[name].dtype or tensor.layout != torch.strided
                    or not torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid checkpoint tensor dtype, layout, or finiteness: {name}")
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
        expected = self.config.to_dict()
        config = payload["config"]
        same_config = (type(config) is dict and set(config) == set(expected)
                       and all(type(config[name]) is type(value) and config[name] == value
                               for name, value in expected.items()))
        if (payload["method"] != self.method or not same_config
                or payload["config_fingerprint"] != self.config.fingerprint):
            raise ValueError("state method, exact config, or codebook fingerprint does not match the adapter")
        self.assert_codebook_valid()
        self._batch(payload["batch_size"])
        state = KVState(payload["weights"], payload["aux"], payload["write_count"],
                        payload["method"], payload["config_fingerprint"])
        if self._validate_state(state, finite=True) != payload["batch_size"]:
            raise ValueError("payload batch size does not match state weights")
        weights, aux = _copy_tree((state.weights, state.aux), device=self.device, clone=True)
        return self._new_state(weights, aux, state.write_count)


def load_portable_checkpoint(path, device):
    """Load frozen format v1 without altering saved codebooks or slow weights."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format") != "clmm.kv.correlated.v1"
            or payload.get("method") != "neural_correlated"):
        raise ValueError("Not a correlated nonlinear checkpoint")
    if type(payload.get("config")) is not dict:
        raise ValueError("Missing portable correlated configuration")
    config = PortableCorrelatedConfig(**payload["config"])
    if payload.get("codebook_hash") != config.key_codebook_sha256:
        raise ValueError("Checkpoint payload and config codebook SHA256 disagree")
    state_dict = payload.get("model_state")
    if not isinstance(state_dict, dict):
        raise ValueError("Missing model state")
    model = PortableCorrelatedNeuralAdapter(config, state_dict.get("codebook.keys"),
                                           state_dict.get("codebook.values")).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model, payload
