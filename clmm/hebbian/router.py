"""Content-based top-k routing over Hebbian associative modules.

Hard top-k selection with softmax gates renormalized over the selected
modules (standard mixture-of-experts style). Gradients flow through the
selected gates; unselected modules receive an exact zero gate, which is
what guarantees their traces stay untouched during writes.
"""

import torch
from torch import nn
from torch.nn import Module


class ContentRouter(Module):
    def __init__(
        self,
        dim,
        num_modules,
        topk
    ):
        super().__init__()
        assert 0 < topk <= num_modules

        self.topk = topk
        self.to_scores = nn.Linear(dim, num_modules, bias = False)

    def forward(self, x):
        """x: (..., dim) -> gates (..., num_modules) sparse, indices (..., topk)"""

        scores = self.to_scores(x)
        gates = scores.softmax(dim = -1)

        _, top_indices = gates.topk(self.topk, dim = -1)

        mask = torch.zeros_like(gates)
        mask.scatter_(-1, top_indices, 1.)

        gates = gates * mask
        gates = gates / gates.sum(dim = -1, keepdim = True).clamp(min = 1e-9)

        return gates, top_indices
