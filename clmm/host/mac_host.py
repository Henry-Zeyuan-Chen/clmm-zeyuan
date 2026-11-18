"""Compact segment-wise transformer host for the persistent-memory comparison."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange

from rotary_embedding_torch import RotaryEmbedding


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


class SegmentAttention(Module):
    """Causal self-attention over one segment, with context tokens prepended to the keys and values.

    Pre-norm. Rotary positions go on the queries and the segment keys only, so context tokens carry
    no position and nothing outside attention ever sees one.
    """

    def __init__(
        self,
        dim,
        heads
    ):
        super().__init__()
        assert dim % heads == 0

        dim_head = dim // heads
        assert dim_head % 2 == 0, 'rotary embedding needs an even head dimension'

        self.heads = heads

        self.norm = nn.RMSNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim, bias = False)

        self.rotary = RotaryEmbedding(dim_head)

    def split_heads(self, t):
        return rearrange(t, 'b n (h d) -> b h n d', h = self.heads)

    def forward(self, x, ctx, mask):
        """x: (b, n, dim), ctx: (b, c, dim) or None, mask: (b, 1, n, c + n) bool, True = may attend"""
        n = x.shape[1]

        if ctx is not None:
            x = torch.cat((ctx, x), dim = 1)                                # (b, c + n, dim)

        x = self.norm(x)

        q = self.split_heads(self.to_q(x[:, -n:]))                          # (b, h, n, d)
        k, v = map(self.split_heads, self.to_kv(x).chunk(2, dim = -1))      # (b, h, c + n, d) each

        q = self.rotary.rotate_queries_or_keys(q)

        k_ctx, k_seg = k[..., :-n, :], k[..., -n:, :]                       # (b, h, c, d), (b, h, n, d)
        k = torch.cat((k_ctx, self.rotary.rotate_queries_or_keys(k_seg)), dim = -2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask = mask)     # (b, h, n, d)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)                                             # (b, n, dim)


class FeedForward(Module):
    """Pre-norm two-layer MLP"""

    def __init__(
        self,
        dim,
        mult = 4
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        """x: (b, n, dim) -> (b, n, dim)"""
        return self.net(x)


class Block(Module):
    """Attention residual, then feed-forward residual"""

    def __init__(
        self,
        dim,
        heads,
        ff_mult = 4
    ):
        super().__init__()

        self.attn = SegmentAttention(dim, heads)
        self.ff = FeedForward(dim, ff_mult)

    def forward(self, x, ctx, mask):
        """x: (b, n, dim), ctx: (b, c, dim) or None (the same tensor at every block), mask: (b, 1, n, c + n)"""
        x = x + self.attn(x, ctx, mask)
        x = x + self.ff(x)

        return x                                                            # (b, n, dim)
