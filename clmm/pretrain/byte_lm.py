"""Small complete byte LM built from the unchanged vendored MAC transformer.

This is a reproducible engineering baseline, not a paper-faithful MAC claim.
The retrieval-zero diagnostic keeps the memory update path and is not a
separately trained attention-only baseline.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from titans_pytorch.mac_transformer import MemoryAsContextTransformer
from titans_pytorch.memory_models import MemoryMLP
from titans_pytorch.neural_memory import NeuralMemory


ARCHITECTURE = "clmm.vendored_mac.byte_lm.v1"
FORMAT_VERSION = 1
DEFAULT_SPLIT_BOUNDS = (0, 90_000_000, 95_000_000, 100_000_000)


@dataclass(frozen=True)
class ByteLMConfig:
    num_tokens: int = 256
    dim: int = 128
    depth: int = 2
    dim_head: int = 32
    heads: int = 4
    ff_mult: float = 4.0
    num_residual_streams: int = 1
    segment_len: int = 32
    num_persist_mem_tokens: int = 4
    num_longterm_mem_tokens: int = 0
    memory_layers: tuple[int, ...] = (1, 2)
    memory_segment_len: int = 8
    memory_batch_size: int = 32
    memory_dim_head: int = 32
    memory_heads: int = 4
    memory_depth: int = 2
    memory_expansion_factor: float = 2.0
    momentum: bool = True
    qk_rmsnorm: bool = True
    max_inner_lr: float = 0.01
    memory_norm_residual: bool = True

    def __post_init__(self):
        object.__setattr__(self, "memory_layers", tuple(self.memory_layers))
        positive = ("num_tokens", "dim", "depth", "dim_head", "heads", "segment_len",
                    "memory_segment_len", "memory_batch_size", "memory_dim_head", "memory_heads", "memory_depth")
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_tokens != 256:
            raise ValueError("The byte vocabulary must contain exactly 256 tokens")
        if self.num_residual_streams != 1 or self.num_longterm_mem_tokens != 0:
            raise ValueError("This baseline fixes one residual stream and zero long-term tokens")
        if type(self.num_persist_mem_tokens) is not int or self.num_persist_mem_tokens < 0:
            raise ValueError("Persistent token count must be nonnegative")
        if self.dim_head % 2 or self.dim != self.heads * self.dim_head:
            raise ValueError("Attention needs an even head dimension and heads * dim_head == dim")
        if self.memory_heads == 1 and self.memory_dim_head != self.dim:
            raise ValueError("Vendored single-head memory requires memory_dim_head == dim")
        if self.memory_heads * self.memory_dim_head != self.dim:
            raise ValueError("This baseline requires memory_heads * memory_dim_head == dim")
        if self.memory_batch_size % self.memory_segment_len:
            raise ValueError("Memory batch size must be divisible by its segment length")
        if not self.memory_layers or len(set(self.memory_layers)) != len(self.memory_layers):
            raise ValueError("Memory layers must be nonempty and unique")
        if any(type(layer) is not int or not 1 <= layer <= self.depth for layer in self.memory_layers):
            raise ValueError("Memory layer indices must be one-based and within model depth")
        for name in ("ff_mult", "memory_expansion_factor", "max_inner_lr"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("momentum", "qk_rmsnorm", "memory_norm_residual"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be Boolean")


def build_model(config: ByteLMConfig) -> MemoryAsContextTransformer:
    if not isinstance(config, ByteLMConfig):
        raise TypeError("config must be ByteLMConfig")
    model = MemoryAsContextTransformer(
        num_tokens=config.num_tokens, dim=config.dim, depth=config.depth,
        dim_head=config.dim_head, heads=config.heads, ff_mult=config.ff_mult,
        segment_len=config.segment_len, num_residual_streams=config.num_residual_streams,
        num_persist_mem_tokens=config.num_persist_mem_tokens,
        num_longterm_mem_tokens=config.num_longterm_mem_tokens,
        neural_memory_layers=config.memory_layers,
        neural_memory_segment_len=config.memory_segment_len,
        neural_memory_batch_size=config.memory_batch_size,
        neural_memory_model=MemoryMLP(config.memory_dim_head, config.memory_depth,
                                      expansion_factor=config.memory_expansion_factor),
        neural_memory_kwargs=dict(
            dim_head=config.memory_dim_head, heads=config.memory_heads,
            momentum=config.momentum, qk_rmsnorm=config.qk_rmsnorm,
            default_step_transform_max_lr=config.max_inner_lr,
            mem_model_norm_add_residual=config.memory_norm_residual,
            use_accelerated_scan=False, spectral_norm_surprises=False,
            store_with_lookahead_value=False),
        use_flex_attn=False, sliding_window_attn=False,
        neural_mem_weight_residual=False, neural_memory_add_value_residual=False,
        neural_memory_qkv_receives_diff_views=False, neural_mem_gate_attn_output=False,
    )
    # Upstream per-head initialization uses expanded views with stride zero.
    # Materialize identical values before an optimizer performs in-place updates.
    # This changes storage only; the vendored implementation stays untouched.
    materialized = []
    for name, parameter in model.named_parameters():
        if any(stride == 0 and size > 1 for stride, size in zip(parameter.stride(), parameter.shape)):
            parameter.data = parameter.detach().clone(memory_format=torch.contiguous_format)
            materialized.append(name)
    model.clmm_materialized_parameters = tuple(materialized)
    model.clmm_config = config
    return model


@contextmanager
def zero_memory_output(model):
    """Eval-only retrieval ablation; preserve returned state and restore nested scopes.

    MAC calls ``mem.forward`` directly, bypassing PyTorch forward hooks. Therefore
    this wraps that instance method instead. State construction still executes.
    This changes downstream activations, so later layers can have different state.
    It does not suppress writes or model the result of attention-only training.
    """
    if model.training:
        raise ValueError("zero_memory_output is an evaluation-only forward diagnostic")
    restores = []
    try:
        for memory in model.modules():
            if not isinstance(memory, NeuralMemory):
                continue
            had_override = "forward" in memory.__dict__
            previous_override = memory.__dict__.get("forward")
            original = memory.forward

            def wrapped(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                return (torch.zeros_like(result[0]), *result[1:])

            restores.append((memory, had_override, previous_override))
            memory.forward = wrapped
        if not restores:
            raise ValueError("Model contains no NeuralMemory modules")
        yield model
    finally:
        for memory, had_override, previous_override in reversed(restores):
            if had_override:
                memory.forward = previous_override
            else:
                del memory.forward


@contextmanager
def reset_memory_at_segments(model):
    """Eval-only full-sequence control that removes memory across attention segments.

    Each memory layer still runs its normal within-segment update/read path.
    Returned last-segment state is only for API compatibility; cached decoding
    and supplied memory states are rejected. This is not an attention-only LM.
    """
    if model.training:
        raise ValueError("Segment reset is an evaluation-only diagnostic")
    if getattr(model, "_clmm_reset_segments_active", False):
        raise ValueError("Nested segment-reset diagnostics are unsupported")
    restores = []
    segment_len = model.segment_len
    model._clmm_reset_segments_active = True
    try:
        for memory in model.modules():
            if not isinstance(memory, NeuralMemory):
                continue
            original = memory.forward
            restores.append((memory, "forward" in memory.__dict__, memory.__dict__.get("forward")))

            def reset_forward(seq, *args, _original=original, **kwargs):
                if args or any(kwargs.get(name) is not None for name in ("state", "prev_weights", "store_seq", "store_mask")):
                    raise ValueError("Segment reset rejects incoming state, residual weights or separate storage inputs")
                if kwargs.get("return_surprises", False):
                    raise ValueError("Segment reset does not expose surprise diagnostics")
                pieces, state = [], None
                for piece in seq.split(segment_len, dim=-2):
                    retrieved, state = _original(piece, **kwargs)
                    pieces.append(retrieved)
                return torch.cat(pieces, dim=-2), state

            memory.forward = reset_forward
        if not restores:
            raise ValueError("Model contains no NeuralMemory modules")
        original_model_forward = model.forward
        model_signature = inspect.signature(original_model_forward)
        restores.append((model, "forward" in model.__dict__, model.__dict__.get("forward")))

        def full_sequence_only(*args, **kwargs):
            bound = model_signature.bind(*args, **kwargs)
            if bound.arguments.get("cache") is not None:
                raise ValueError("Segment reset rejects cached decoding")
            return original_model_forward(*args, **kwargs)

        model.forward = full_sequence_only
        yield model
    finally:
        for module, had_override, previous_override in reversed(restores):
            if had_override:
                module.forward = previous_override
            else:
                del module.forward
        del model._clmm_reset_segments_active


def source_info():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "titans_pytorch" / "mac_transformer.py",
             root / "titans_pytorch" / "neural_memory.py", root / "titans_pytorch" / "memory_models.py"]
    return {"architecture": ARCHITECTURE, "torch": str(torch.__version__),
            "compatibility": "Expanded per-head initial parameters are materialized into contiguous storage without changing values.",
            "files_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in paths}}


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def save_checkpoint(path, model, config, *, source_info=None, training_info=None):
    """Atomically save tensor state plus JSON metadata, without pickled modules."""
    if getattr(model, "clmm_config", None) != config:
        raise ValueError("Model and checkpoint configuration disagree")
    provenance = globals()["source_info"]() if source_info is None else source_info
    provenance = _json_copy(provenance)
    if not isinstance(provenance, dict):
        raise ValueError("source_info must be a JSON object")
    provenance["adapter_materialized_parameters"] = list(model.clmm_materialized_parameters)
    payload = {"format_version": FORMAT_VERSION, "architecture": ARCHITECTURE,
               "config": _json_copy(asdict(config)),
               "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
               "source_info": _json_copy(provenance), "training_info": _json_copy(training_info or {})}
    if not all(torch.isfinite(value).all() for value in payload["model_state"].values()):
        raise ValueError("Refusing to save nonfinite model state")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path, *, map_location="cpu", expected_config=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "architecture", "config", "model_state", "source_info", "training_info"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Not a shared byte-LM checkpoint")
    if payload["format_version"] != FORMAT_VERSION or payload["architecture"] != ARCHITECTURE:
        raise ValueError("Unsupported checkpoint format or architecture")
    try:
        config = ByteLMConfig(**payload["config"])
        _json_copy(payload["source_info"])
        _json_copy(payload["training_info"])
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid checkpoint configuration or metadata") from error
    if expected_config is not None and config != expected_config:
        raise ValueError("Checkpoint configuration differs from expected_config")
    if not isinstance(payload["model_state"], dict) or not all(
        isinstance(value, torch.Tensor) and torch.isfinite(value).all()
        for value in payload["model_state"].values()
    ):
        raise ValueError("Invalid or nonfinite checkpoint tensor state")
    with torch.random.fork_rng(devices=[]):
        model = build_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(map_location)
    return model, config, payload


@dataclass(frozen=True)
class ByteCorpus:
    data: bytearray
    sha256: str
    split_bounds: tuple[int, int, int, int]
    path: str

    def bounds(self, split):
        if split not in ("train", "dev", "test"):
            raise ValueError("split must be train, dev or test")
        index = ("train", "dev", "test").index(split)
        return self.split_bounds[index:index + 2]


def load_enwik8(path, *, split_bounds=DEFAULT_SPLIT_BOUNDS, expected_sha256=None):
    """Load raw or gzip bytes; hash decoded data and bound decompression by split size."""
    bounds = tuple(split_bounds)
    if len(bounds) != 4 or bounds[0] != 0 or any(type(value) is not int for value in bounds):
        raise ValueError("Four integer chronological split bounds starting at zero are required")
    if any(left >= right for left, right in zip(bounds, bounds[1:])):
        raise ValueError("Split bounds must be strictly increasing")
    path = Path(path).resolve()
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b" or path.suffix.lower() == ".gz"
    if compressed:
        with gzip.open(path, "rb") as handle:
            data = bytearray(handle.read(bounds[-1] + 1))
    else:
        if path.stat().st_size != bounds[-1]:
            raise ValueError("Corpus size does not match the terminal split bound")
        data = bytearray(path.read_bytes())
    if len(data) != bounds[-1]:
        raise ValueError("Decoded corpus size does not match the terminal split bound")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Corpus SHA256 differs from expected_sha256")
    return ByteCorpus(data, digest, bounds, str(path))


def _window_parameters(corpus, split, seq_len, count):
    if type(seq_len) is not int or seq_len < 1 or type(count) is not int or count < 1:
        raise ValueError("Sequence length and window count must be positive integers")
    start, end = corpus.bounds(split)
    if end - start < seq_len + 1:
        raise ValueError("Split is too short for the requested input and next-token target")
    return start, end


def _gather_windows(corpus, offsets, seq_len):
    data = torch.frombuffer(corpus.data, dtype=torch.uint8)
    indices = offsets[:, None] + torch.arange(seq_len + 1)[None, :]
    return data[indices].long(), offsets


def sample_windows(corpus, split, seq_len, batch_size, generator):
    """Uniform starts with replacement, isolated from global RNG; offsets are global."""
    start, end = _window_parameters(corpus, split, seq_len, batch_size)
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("Provide a local CPU torch.Generator")
    offsets = torch.randint(start, end - seq_len, (batch_size,), generator=generator)
    return _gather_windows(corpus, offsets, seq_len)


def fixed_windows(corpus, split, seq_len, count, seed):
    """Frozen seeded nonoverlapping grid windows; take short prefixes from these."""
    start, end = _window_parameters(corpus, split, seq_len, count)
    available = (end - start) // (seq_len + 1)
    if count > available:
        raise ValueError("Not enough nonoverlapping windows in this split")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    offsets = start + torch.randperm(available, generator=generator)[:count] * (seq_len + 1)
    return _gather_windows(corpus, offsets, seq_len)
