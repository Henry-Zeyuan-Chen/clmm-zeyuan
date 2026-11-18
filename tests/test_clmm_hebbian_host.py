import pytest

import torch

from clmm.hebbian.mac_host import (
    pad_to_multiple,
    segment_attention_mask,
    SegmentAttention,
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
