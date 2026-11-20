import pytest

import torch

from titans_pytorch import NeuralMemory

from clmm.hebbian.hebbian_memory import HebbianMemory
from clmm.host.mac_host import (
    pad_to_multiple,
    segment_attention_mask,
    SegmentAttention,
    FeedForward,
    Block,
    HostState,
    host_state_detach,
)

# helpers

def diff(x, y):
    return (x - y).abs().amax()

def seeded(fn, seed = 0):
    """run fn under a fixed seed, global rng left untouched"""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return fn()

# padding

@pytest.mark.parametrize('n, n_padded', [(15, 16), (16, 16), (17, 32)])
def test_pad_to_multiple(n, n_padded):
    ids = torch.arange(1, 2 * n + 1).view(2, n)                          # (b, n), no pad id inside

    padded, valid = pad_to_multiple(ids, 16, pad_id = 0)                 # (b, n_padded) each

    assert padded.shape == valid.shape == (2, n_padded)
    assert padded.dtype == ids.dtype and valid.dtype == torch.bool

    assert torch.equal(padded[:, :n], ids)
    assert (padded[:, n:] == 0).all()

    assert valid[:, :n].all() and not valid[:, n:].any()

def test_pad_to_multiple_marks_existing_padding():
    ids = torch.tensor([[5, 6, 7, 0], [5, 6, 0, 0]])                     # (b, n), already a multiple of 4

    padded, valid = pad_to_multiple(ids, 4, pad_id = 0)

    assert torch.equal(padded, ids)
    assert torch.equal(valid, torch.tensor([[True, True, True, False], [True, True, False, False]]))

# attention mask

def test_segment_attention_mask():
    # one row of 4 tokens, the last two padded, 2 context tokens

    valid = torch.tensor([[True, True, False, False]])                   # (b, n)

    mask = segment_attention_mask(valid, num_ctx = 2)                    # (b, 1, n, num_ctx + n)

    T, F = True, False

    expected = torch.tensor([
        # ctx   | segment keys
        [T, T,    T, F, F, F],     # query 0 sees itself
        [T, T,    T, T, F, F],     # query 1 sees 0 and itself
        [T, T,    T, T, T, F],     # padded query 2: valid past, plus its own diagonal
        [T, T,    T, T, F, T],     # padded query 3: padded key 2 hidden, own diagonal on
    ])

    assert mask.shape == (1, 1, 4, 6) and mask.dtype == torch.bool
    assert torch.equal(mask[0, 0], expected)

def test_segment_attention_mask_no_context():
    valid = torch.ones(3, 5, dtype = torch.bool)                         # (b, n)

    mask = segment_attention_mask(valid, num_ctx = 0)                    # (b, 1, n, n)

    assert mask.shape == (3, 1, 5, 5)
    assert torch.equal(mask, torch.ones(5, 5, dtype = torch.bool).tril().expand(3, 1, 5, 5))

# attention

def make_attention_inputs(b = 2, n = 6, c = 3, dim = 8):
    x = seeded(lambda: torch.randn(b, n, dim, dtype = torch.float64))       # (b, n, dim)
    ctx = seeded(lambda: torch.randn(b, c, dim, dtype = torch.float64), 1)  # (b, c, dim)
    return x, ctx

def test_segment_attention_shape():
    attn = seeded(lambda: SegmentAttention(dim = 8, heads = 2)).double()
    x, ctx = make_attention_inputs()

    valid = torch.ones(2, 6, dtype = torch.bool)                         # (b, n)

    assert attn(x, ctx, segment_attention_mask(valid, num_ctx = 3)).shape == (2, 6, 8)
    assert attn(x, None, segment_attention_mask(valid, num_ctx = 0)).shape == (2, 6, 8)

@pytest.mark.parametrize('j', [0, 3, 5])
def test_segment_attention_causal(j):
    attn = seeded(lambda: SegmentAttention(dim = 8, heads = 2)).double()
    x, ctx = make_attention_inputs()

    mask = segment_attention_mask(torch.ones(2, 6, dtype = torch.bool), num_ctx = 3)

    perturbed = x.clone()
    perturbed[:, j] += 1.

    out, out_perturbed = attn(x, ctx, mask), attn(perturbed, ctx, mask)  # (b, n, dim) each

    assert out[:, :j].numel() == 0 or diff(out[:, :j], out_perturbed[:, :j]) == 0.

    changed = (out[:, j:] != out_perturbed[:, j:]).flatten(2).any(dim = -1)   # (b, n - j)

    assert changed.all()

def test_segment_attention_reads_context():
    attn = seeded(lambda: SegmentAttention(dim = 8, heads = 2)).double()
    x, ctx = make_attention_inputs()

    mask = segment_attention_mask(torch.ones(2, 6, dtype = torch.bool), num_ctx = 3)

    changed = (attn(x, ctx, mask) != attn(x, ctx + 1., mask)).flatten(2).any(dim = -1)   # (b, n)

    assert changed.all()

def test_segment_attention_all_pad_row_is_finite():
    attn = seeded(lambda: SegmentAttention(dim = 8, heads = 2)).double()
    x, ctx = make_attention_inputs()

    valid = torch.ones(2, 6, dtype = torch.bool)                         # (b, n)
    valid[0] = False

    for context, num_ctx in ((ctx, 3), (None, 0)):
        out = attn(x, context, segment_attention_mask(valid, num_ctx = num_ctx))

        assert out.isfinite().all()

def test_segment_attention_context_has_no_position():
    # context keys are not rotated, so reordering the context tokens changes nothing

    attn = seeded(lambda: SegmentAttention(dim = 8, heads = 2)).double()
    x, ctx = make_attention_inputs()

    mask = segment_attention_mask(torch.ones(2, 6, dtype = torch.bool), num_ctx = 3)

    assert diff(attn(x, ctx, mask), attn(x, ctx.flip(dims = (1,)), mask)) < 1e-12

# feed-forward and block

def test_feedforward_shape():
    ff = seeded(lambda: FeedForward(dim = 8, mult = 4)).double()
    x, _ = make_attention_inputs()

    assert ff.net[1].out_features == 32
    assert ff(x).shape == (2, 6, 8)

def test_block_shape():
    block = seeded(lambda: Block(dim = 8, heads = 2, ff_mult = 4)).double()
    x, ctx = make_attention_inputs()

    valid = torch.ones(2, 6, dtype = torch.bool)                         # (b, n)

    assert block(x, ctx, segment_attention_mask(valid, num_ctx = 3)).shape == (2, 6, 8)
    assert block(x, None, segment_attention_mask(valid, num_ctx = 0)).shape == (2, 6, 8)

def test_block_is_two_residuals():
    block = seeded(lambda: Block(dim = 8, heads = 2)).double()
    x, ctx = make_attention_inputs()

    mask = segment_attention_mask(torch.ones(2, 6, dtype = torch.bool), num_ctx = 3)

    after_attn = x + block.attn(x, ctx, mask)                            # (b, n, dim)
    expected = after_attn + block.ff(after_attn)                         # (b, n, dim)

    assert diff(block(x, ctx, mask), expected) == 0.

@pytest.mark.parametrize('j', [0, 3, 5])
def test_block_causal(j):
    block = seeded(lambda: Block(dim = 8, heads = 2)).double()
    x, ctx = make_attention_inputs()

    mask = segment_attention_mask(torch.ones(2, 6, dtype = torch.bool), num_ctx = 3)

    perturbed = x.clone()
    perturbed[:, j] += 1.

    out, out_perturbed = block(x, ctx, mask), block(perturbed, ctx, mask)   # (b, n, dim) each

    assert out[:, :j].numel() == 0 or diff(out[:, :j], out_perturbed[:, :j]) == 0.
    assert (out[:, j:] != out_perturbed[:, j:]).flatten(2).any(dim = -1).all()

# state

def has_graph(t):
    return t.requires_grad or t.grad_fn is not None

def test_host_state_detach():
    hebbian = seeded(lambda: HebbianMemory(dim = 8, num_modules = 4, dim_assoc = 4, dim_hidden = 8))
    neural = seeded(lambda: NeuralMemory(dim = 8, chunk_size = 4, batch_size = 4))

    x = seeded(lambda: torch.randn(2, 8, 8)).requires_grad_()            # (b, n, dim)

    _, neural_state = neural(x)
    hebbian_state = hebbian.write(x, x, hebbian.init_state(2))

    state = HostState(
        hebbian = hebbian_state,
        neural = neural_state,
        summary_sum = x.sum(dim = 1),                                    # (b, dim)
        summary_count = torch.full((2, 1), 8.),                          # (b, 1)
    )

    assert has_graph(state.hebbian.traces) and has_graph(state.summary_sum)
    assert all(has_graph(t) for t in state.neural.weights.values())
    assert state.neural.updates is not None

    detached = host_state_detach(state)

    assert isinstance(detached, HostState)

    # hebbian and summary: same values, no graph

    assert not has_graph(detached.hebbian.traces)
    assert torch.equal(detached.hebbian.traces, state.hebbian.traces)
    assert detached.hebbian.seq_index == state.hebbian.seq_index

    assert not has_graph(detached.summary_sum) and not has_graph(detached.summary_count)
    assert torch.equal(detached.summary_sum, state.summary_sum)
    assert torch.equal(detached.summary_count, state.summary_count)

    # neural: updates dropped, weights and momentum states keep their values, no graph

    assert detached.neural.updates is None
    assert detached.neural.seq_index == state.neural.seq_index

    last_update, last_momentum = state.neural.states
    detached_update, detached_momentum = detached.neural.states

    for kept, original in ((detached.neural.weights, state.neural.weights), (detached_update, last_update), (detached_momentum, last_momentum)):
        assert set(kept.keys()) == set(original.keys())

        for name in original.keys():
            assert not has_graph(kept[name])
            assert torch.equal(kept[name], original[name])

    # the original state is left as it was

    assert has_graph(state.hebbian.traces) and state.neural.updates is not None

def test_host_state_detach_does_not_change_the_next_segment():
    # dropping `updates` is safe: the next forward gives the same retrieval and the same weights

    neural = seeded(lambda: NeuralMemory(dim = 8, chunk_size = 4, batch_size = 4))

    x = seeded(lambda: torch.randn(2, 8, 8))                             # (b, n, dim)
    x_next = seeded(lambda: torch.randn(2, 8, 8), 1)                     # (b, n, dim)

    _, neural_state = neural(x)

    state = HostState(None, neural_state, torch.zeros(2, 8), torch.zeros(2, 1))
    detached = host_state_detach(state)

    retrieved, next_state = neural(x_next, state = state.neural)
    retrieved_detached, next_state_detached = neural(x_next, state = detached.neural)

    assert diff(retrieved, retrieved_detached) == 0.

    for name in next_state.weights.keys():
        assert torch.equal(next_state.weights[name], next_state_detached.weights[name])

def test_host_state_detach_with_memories_off():
    state = HostState(None, None, torch.zeros(2, 8, requires_grad = True) * 2., torch.zeros(2, 1))

    detached = host_state_detach(state)

    assert detached.hebbian is None and detached.neural is None
    assert not has_graph(detached.summary_sum)
