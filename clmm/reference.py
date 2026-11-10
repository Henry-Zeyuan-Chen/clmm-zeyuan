"""Loop reference for the Titans associative memory update (equations 11–15)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad


@dataclass
class ReferenceTrace:
    """Entry 0 in each history is the initial state, followed by completed writes."""

    retrieved: Tensor
    weights: list[dict[str, Tensor]]
    momentums: list[dict[str, Tensor]]


def reference_memory(
    model: nn.Module,
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    learning_rate: Tensor,
    momentum: Tensor,
    decay: Tensor,
    gradient_refresh_interval: int,
    read_timing: Literal["before_write", "after_write"],
    update_chunk_size: int = 1,
    initial_weights: dict[str, Tensor] | None = None,
) -> ReferenceTrace:
    """Run one stream through a buffer-free, deterministic memory MLP.

    The caller supplies the projected q/k/v tensors [tokens, features] and
    transformed gates. Learning rates are per token. Momentum and decay each
    have one scalar per complete write chunk. With one-token chunks this gives
    the per-token recurrence. Larger chunks sum the weighted gradients and
    apply momentum and decay once, as in the upstream grouped update.

    Gradients use the weights at the start of each refresh interval. Reads
    use the latest completed write, optionally including the write ending at
    the query token. Queries in a trailing partial chunk read the last
    completed state; its keys and values do not trigger an update.
    The trace preserves autograd history and does not modify model parameters.
    """
    if keys.ndim != 2 or queries.shape != keys.shape or values.shape != keys.shape:
        raise ValueError("queries, keys and values must all have shape [tokens, features]")
    if update_chunk_size < 1 or gradient_refresh_interval < 1:
        raise ValueError("update and gradient refresh intervals must be positive")
    if gradient_refresh_interval % update_chunk_size:
        raise ValueError("gradient_refresh_interval must be a multiple of update_chunk_size")
    if read_timing not in ("before_write", "after_write"):
        raise ValueError("read_timing must be 'before_write' or 'after_write'")

    tokens = keys.shape[0]
    complete_chunks = tokens // update_chunk_size
    if learning_rate.shape != (tokens,):
        raise ValueError("learning_rate must have one entry per token")
    if momentum.shape != (complete_chunks,) or decay.shape != (complete_chunks,):
        raise ValueError("momentum and decay must have one entry per complete write chunk")

    weights = dict(model.named_parameters()) if initial_weights is None else dict(initial_weights)
    velocity = {name: torch.zeros_like(weight) for name, weight in weights.items()}
    weight_history = [weights]
    momentum_history = [velocity]

    def association_loss(parameters, key, value):
        prediction = functional_call(model, parameters, (key,), strict=True)
        return (prediction - value).square().sum()

    loss_gradient = grad(association_loss)
    gradient_weights = weights
    for chunk in range(complete_chunks):
        start = chunk * update_chunk_size
        # Equation 14. Each token contributes its own theta-weighted gradient.
        gradient_sum = {name: torch.zeros_like(weight) for name, weight in weights.items()}
        for token in range(start, start + update_chunk_size):
            gradient = loss_gradient(gradient_weights, keys[token], values[token])
            for name in weights:
                gradient_sum[name] = gradient_sum[name] + learning_rate[token] * gradient[name]

        velocity = {
            name: momentum[chunk] * velocity[name] - gradient_sum[name] for name in weights
        }
        # Equation 13. Decay applies to the previous weights, before adding S.
        weights = {
            name: (1 - decay[chunk]) * weight + velocity[name] for name, weight in weights.items()
        }
        weight_history.append(weights)
        momentum_history.append(velocity)
        if (start + update_chunk_size) % gradient_refresh_interval == 0:
            gradient_weights = weights

    reads = []
    for token, query in enumerate(queries):
        completed = (token + int(read_timing == "after_write")) // update_chunk_size
        reads.append(functional_call(model, weight_history[completed], (query,), strict=True))
    retrieved = torch.stack(reads) if reads else queries.new_empty(queries.shape)
    return ReferenceTrace(retrieved, weight_history, momentum_history)
