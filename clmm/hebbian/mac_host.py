"""Compact segment-wise transformer host for the persistent-memory comparison."""

import torch.nn.functional as F


def pad_to_multiple(ids, multiple, pad_id):
    """Right-pad token ids to a multiple of `multiple`.

    ids: (b, n) -> ids (b, n_padded), valid (b, n_padded) bool, False where the id is `pad_id`
    """
    n = ids.shape[-1]
    remainder = -n % multiple

    ids = F.pad(ids, (0, remainder), value = pad_id)                    # (b, n_padded)
    valid = ids != pad_id                                               # (b, n_padded)

    return ids, valid
