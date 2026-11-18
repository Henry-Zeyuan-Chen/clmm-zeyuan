import pytest

import torch

from clmm.hebbian.mac_host import pad_to_multiple

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
