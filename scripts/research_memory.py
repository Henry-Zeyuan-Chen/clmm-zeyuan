"""Small, trainable wrapper around the vendored Titans NeuralMemory."""

from pathlib import Path
import math
import sys

import torch
from torch import nn


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from titans_pytorch import NeuralMemory


NKEYS = 16
NVALS = 8


def mean_squared_memory_loss(prediction, target):
    return (prediction - target).square().mean(dim=-1)


class Reader(nn.Module):
    """Learn symbolic representations and a nonlinear, per-episode memory.

    A write uses the current fast weights as its gradient anchor. Slow weights
    receive gradients through all writes during outer training. Reads consume
    only the query key and leave the supplied fast state unchanged.
    """

    def __init__(self, dim=64, depth=2, expansion_factor=2.0,
                 init_adaptive_step_bias=1.0, init_decay_bias=-4.0, max_inner_lr=1.0):
        super().__init__()
        if dim < 1 or depth < 1 or expansion_factor <= 0:
            raise ValueError("Memory dimensions and depth must be positive")
        if not math.isfinite(max_inner_lr) or max_inner_lr <= 0:
            raise ValueError("max_inner_lr must be finite and positive")
        self.config = {
            "dim": dim,
            "depth": depth,
            "expansion_factor": expansion_factor,
            "init_adaptive_step_bias": init_adaptive_step_bias,
            "init_decay_bias": init_decay_bias,
            "max_inner_lr": max_inner_lr,
        }
        self.key = nn.Embedding(NKEYS, dim)
        self.value = nn.Embedding(NVALS, dim)
        self.memory = NeuralMemory(
            dim=dim,
            heads=1,
            chunk_size=1,
            qkv_receives_diff_views=True,
            momentum=False,
            mem_model_norm_add_residual=False,
            store_memory_loss_fn=mean_squared_memory_loss,
            default_model_kwargs={"depth": depth, "expansion_factor": expansion_factor},
            init_adaptive_step_bias=init_adaptive_step_bias,
            init_decay_bias=init_decay_bias,
            default_step_transform_max_lr=max_inner_lr,
        )
        self.head = nn.Linear(dim, NVALS)

    def write(self, keys, values):
        if keys.ndim != 2 or keys.shape != values.shape:
            raise ValueError("keys and values must have matching [batch, writes] shapes")
        past = None
        fast = self.memory.init_weights(keys.shape[0])
        for position in range(keys.shape[1]):
            fact = torch.stack((
                self.key(keys[:, position:position + 1]),
                self.value(values[:, position:position + 1]),
            ))
            _, returned = self.memory.store_memories(
                fact,
                weights=fast,
                past_state=past,
                seq_index=position,
                return_surprises=False,
            )
            past = returned.states
            fast = past[0]
        return fast

    def read(self, query, fast):
        active = {name: weights.unsqueeze(1) for name, weights in fast.items()}
        retrieved = self.memory.retrieve_memories(self.key(query[:, None]), active)
        return self.head(retrieved[:, 0])
