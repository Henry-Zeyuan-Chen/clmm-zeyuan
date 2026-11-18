import pytest

import torch

from clmm.hebbian.mac_host import pad_to_multiple, segment_attention_mask

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
