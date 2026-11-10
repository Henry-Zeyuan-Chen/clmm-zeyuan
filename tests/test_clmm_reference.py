"""Tiny CPU oracles for the selected Titans memory update and read schedule."""

import pytest
import torch
from torch import nn

from clmm.reference import reference_memory
from titans_pytorch import NeuralMemory


DTYPE = torch.float64
MAX_LR = 0.08


def doubles(data, *, requires_grad=False):
    return torch.tensor(data, dtype=DTYPE, requires_grad=requires_grad)


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(("timing", "expected"), [
    ("before_write", [1.0, 1.0, 1.75, 1.75, 0.625]),
    ("after_write",  [1.0, 1.75, 1.75, 0.625, 0.625]),
])
def test_hand_computed_grouped_updates_and_reads(timing, expected):
    # Both chunks use W0=1 for gradients (b=4); the fifth token cannot write.
    # Feature-summed squared loss: signed updates are +1 and -1/2.
    # S1=1; W1=3/4+1. S2=1/4-1/2; W2=W1/2+S2.
    model = nn.Linear(1, 1, bias=False).double()
    ones = doubles([[1.0]] * 5)
    result = reference_memory(
        model, ones, ones, doubles([[3], [1], [0], [1], [100]]),
        initial_weights={"weight": doubles([[1.0]])},
        learning_rate=doubles([0.25, 0.125, 0.25, 0.125, 1.0]),
        momentum=doubles([0.5, 0.25]), decay=doubles([0.25, 0.5]),
        gradient_refresh_interval=4, update_chunk_size=2, read_timing=timing,
    )
    close(torch.stack([w["weight"].reshape(()) for w in result.weights]),
          doubles([1, 1.75, 0.625]))
    close(torch.stack([s["weight"].reshape(()) for s in result.momentums]),
          doubles([0, 1, -0.25]))
    close(result.retrieved[:, 0], doubles(expected))


def test_grouped_momentum_does_not_apply_once_per_token():
    # Two identical writes. Each supplies +1 at the shared gradient anchor.
    # Per token: S=(1, 1.5), W=(2, 3.5); grouped: S=2 and W=3.
    unit = doubles([[1.0], [1.0]])
    model = nn.Linear(1, 1, bias=False).double()
    for c, terminal in [(1, 3.5), (2, 3.0)]:
        result = reference_memory(
            model, unit, unit, doubles([[3.0], [3.0]]),
            initial_weights={"weight": doubles([[1.0]])},
            learning_rate=doubles([0.25, 0.25]), momentum=doubles([0.5] * (2 // c)),
            decay=doubles([0.0] * (2 // c)), gradient_refresh_interval=2,
            update_chunk_size=c, read_timing="after_write",
        )
        close(result.weights[-1]["weight"], doubles([[terminal]]))


def test_gradient_through_both_writes_and_the_refresh():
    # With theta=1/4, eta2=1/4, no decay, b=c=1, the last read is
    # W2 = W0/8 + 3*v1/8 + v2/2. Gradient checks catch detached refreshes.
    model = nn.Linear(1, 1, bias=False).double()
    initial = doubles([[1.0]], requires_grad=True)
    values = doubles([[3.0], [0.0]], requires_grad=True)
    unit = torch.ones_like(values)
    result = reference_memory(
        model, unit, unit, values, initial_weights={"weight": initial},
        learning_rate=doubles([0.25, 0.25]), momentum=doubles([0, 0.25]),
        decay=doubles([0, 0]), gradient_refresh_interval=1,
        update_chunk_size=1, read_timing="after_write",
    )
    last_read = result.retrieved[-1, 0]
    close(last_read, doubles(1.25))
    dw, dv = torch.autograd.grad(last_read, (initial, values))
    close(dw, doubles([[0.125]]))
    close(dv, doubles([[0.375], [0.5]]))


def feature_sum_loss(pred, target):
    return (pred - target).square().sum(dim=-1)


@pytest.mark.parametrize("tokens", [0, 2])
def test_no_complete_chunk_reads_the_initial_model(tokens):
    model = nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(doubles([[1, 2], [3, 4]]))
    inputs = doubles([[1, 0], [0, 1]])[:tokens]
    result = reference_memory(
        model, inputs, inputs, inputs, learning_rate=doubles([1, 1])[:tokens],
        momentum=doubles([]), decay=doubles([]), gradient_refresh_interval=4,
        update_chunk_size=4, read_timing="after_write",
    )
    assert len(result.weights) == len(result.momentums) == 1
    close(result.retrieved, model(inputs))
    close(result.weights[0]["weight"], doubles([[1, 2], [3, 4]]))


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize(("c", "refresh_chunks"), [
    (1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (4, 2),
])
def test_reference_matches_neural_memory(depth, c, refresh_chunks):
    """Check public outputs and all completed states for matched grouping."""
    b = c * refresh_chunks
    chunks = 3  # includes at least one refresh and a later write in every case
    tail = c - 1 if b > c else 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        engine = NeuralMemory(
            dim=3, heads=1, chunk_size=c, batch_size=b,
            default_model_kwargs=dict(depth=depth, expansion_factor=2),
            default_step_transform_max_lr=MAX_LR,
            store_memory_loss_fn=feature_sum_loss,
            mem_model_norm_add_residual=False, pre_rmsnorm=False,
            post_rmsnorm=False, qk_rmsnorm=False, use_accelerated_scan=False,
        ).double()
        obs = torch.randn(chunks * c + tail, 3, dtype=DTYPE)
        probe = torch.randn_like(obs)

    # Feed the same projected inputs and gates to the independent recurrence.
    # Only complete chunks produce eta/alpha. theta weights individual tokens.
    pooled = obs[:chunks * c].reshape(chunks, c, 3).mean(dim=1)
    lr = MAX_LR * engine.to_adaptive_step[0](obs).squeeze(-1).sigmoid()
    eta = engine.to_momentum[0](pooled).squeeze(-1).sigmoid()
    alpha = engine.to_decay_factor[0](pooled).squeeze(-1).sigmoid()
    initial = {name: weight[0] for name, weight in engine.init_weights(1).items()}
    ref = reference_memory(
        engine.memory_model, engine.to_queries(probe), engine.to_keys(obs),
        engine.to_values(obs), initial_weights=initial, learning_rate=lr,
        momentum=eta, decay=alpha, gradient_refresh_interval=b,
        update_chunk_size=c, read_timing="after_write",
    )

    read, state = engine(probe.unsqueeze(0), store_seq=obs.unsqueeze(0))
    close(read[0], ref.retrieved)
    assert len(ref.weights) == len(ref.momentums) == chunks + 1
    assert state.seq_index == chunks * c
    if tail:
        close(state.cache_store_segment[0], obs[chunks * c:])
    for name in initial:
        close(state.updates[name][0], torch.stack([w[name] for w in ref.weights]))
        close(state.states[1][name][0, 0], ref.momentums[-1][name])

    # forward exposes only the last momentum, so stop at each complete chunk
    # in a second run. This distinguishes intermediate errors that cancel later.
    running = None
    for j in range(1, chunks + 1):
        span = slice((j - 1) * c, j * c)
        _, running = engine(probe[None, span], store_seq=obs[None, span], state=running)
        for name in initial:
            close(running.states[0][name][0], ref.weights[j][name])
            close(running.states[1][name][0, 0], ref.momentums[j][name])
