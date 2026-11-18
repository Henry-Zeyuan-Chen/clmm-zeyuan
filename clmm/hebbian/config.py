"""Size presets for the selective-Hebbian-memory experiment.

`tiny` runs on CPU in minutes and is the development / test preset.
`small` targets a single workstation GPU.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # compact MAC host
    dim: int = 64
    depth: int = 2
    heads: int = 4
    segment_len: int = 16
    ff_mult: int = 4
    persist_mode: str = 'hebbian'    # 'hebbian' | 'frozen' | 'none'
    longterm_mode: str = 'neural'    # 'neural' | 'none'
    num_persist_tokens: int = 4      # context tokens for the frozen arm
    memory_source: str = 'contextual'    # 'contextual' (hidden states after the lower blocks) | 'next_token' (embeddings, key t -> value t+1)
    memory_after_block: int = 1      # number of lower blocks run before the memories read and write

    # hebbian persistent slot
    num_modules: int = 8
    dim_assoc: int = 16
    topk: int = 2
    num_read_tokens: int = 4
    hebbian_hidden: int = 64
    eta_init: float = 0.1            # initial trace write rate (slow = mostly persistent)
    alpha_init: float = 0.1          # initial plastic coefficient
    hebbian_read: str = 'both'       # 'token' (per-token recall residual) | 'context' (prepended read tokens) | 'both'

    # adaptive long-term memory (lucidrains NeuralMemory, used unchanged)
    neural_memory_kwargs: dict = field(default_factory=dict)

    def neural_kwargs(self):
        """NeuralMemory kwargs: one chunk and one weight update per segment; `neural_memory_kwargs` overrides"""
        defaults = dict(
            chunk_size = self.segment_len,
            batch_size = self.segment_len,
            store_with_lookahead_value = self.memory_source == 'next_token',
        )

        return {**defaults, **self.neural_memory_kwargs}

    def hebbian_kwargs(self):
        return dict(
            num_modules = self.num_modules,
            dim_assoc = self.dim_assoc,
            topk = self.topk,
            num_read_tokens = self.num_read_tokens,
            dim_hidden = self.hebbian_hidden,
            eta_init = self.eta_init,
            alpha_init = self.alpha_init,
        )


def tiny() -> Config:
    return Config()


def small() -> Config:
    return Config(
        dim = 256,
        depth = 4,
        heads = 8,
        segment_len = 64,
        num_modules = 16,
        dim_assoc = 32,
        topk = 4,
        num_read_tokens = 8,
        hebbian_hidden = 256,
        num_persist_tokens = 8,
    )


PRESETS = dict(tiny = tiny, small = small)
