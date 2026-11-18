"""Compact segment-wise transformer host for the persistent-memory comparison."""

import torch
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


def segment_attention_mask(valid, num_ctx):
    """Attention mask for one segment with `num_ctx` context tokens prepended to the keys.

    valid: (b, n) bool -> (b, 1, n, num_ctx + n) bool, True = query (row) may attend to key (column)

    Context columns are always visible. Over the segment the mask is causal and hides padded keys,
    except the diagonal, which stays on so that no row is empty (an empty row gives NaN in softmax).
    """
    b, n = valid.shape
    device = valid.device

    causal = torch.ones(n, n, dtype = torch.bool, device = device).tril()   # (n, n)
    diagonal = torch.eye(n, dtype = torch.bool, device = device)            # (n, n)

    segment = (causal & valid[:, None, :]) | diagonal                       # (b, n, n)
    context = torch.ones(b, n, num_ctx, dtype = torch.bool, device = device)   # (b, n, num_ctx)

    mask = torch.cat((context, segment), dim = -1)                          # (b, n, num_ctx + n)

    return mask[:, None]                                                    # (b, 1, n, num_ctx + n)
