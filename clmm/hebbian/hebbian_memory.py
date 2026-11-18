"""Selective Hebbian memory: a bank of small associative modules with
differentiable Hebbian plasticity (Miconi et al.) and content-based routing.

Each module is encoder -> plastic associative layer -> decoder. The plastic
layer uses effective weights

    W_eff = W + alpha * H_t

where the base weights W, plastic coefficients alpha, write rates eta and the
encoder/decoder/router parameters are learned offline by backpropagating
through the Hebbian update, then frozen. At inference only the traces H_t
change, via local bounded updates:

    H' = H + eta * act_m * (O_m - H)        (a lerp toward the observed
                                             gate-weighted outer products)

with act_m the module's routing activity in the current segment. A module
unselected by every token has act_m = 0 and its trace is exactly unchanged —
this is the "selective access" mechanism. Encoded keys/values are l2-normed,
so each outer product has unit Frobenius norm and the lerp keeps traces
bounded without clipping.

This replaces Titans' *persistent* memory (frozen, input-independent tokens)
with a mostly-persistent, slowly and locally plastic store. It does NOT touch
Titans' adaptive long-term memory (lucidrains NeuralMemory), which the host
uses unchanged.
"""

from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

from einops import rearrange

from clmm.hebbian.router import ContentRouter

# state carried across segments and sessions: only the traces change at inference

HebbianMemState = namedtuple('HebbianMemState', [
    'seq_index',
    'traces',      # (batch, num_modules, dim_assoc, dim_assoc)
])


def hebbian_state_detach(state):
    """cut the graph at a session / truncation boundary, keeping trace values and seq_index"""
    return state._replace(traces = state.traces.detach())


def l2norm(t):
    return F.normalize(t, dim = -1)


def init_per_module_(weight):
    """xavier-init each module's (fan_in, fan_out) slice of a (M, fan_in, fan_out) bank"""
    with torch.no_grad():
        for single in weight:
            nn.init.xavier_uniform_(single)


class HebbianMemory(Module):
    def __init__(
        self,
        dim,
        num_modules = 8,
        dim_assoc = 16,
        topk = 2,
        num_read_tokens = 4,
        dim_hidden = 64,
        eta_init = 0.1,
        alpha_init = 0.1
    ):
        super().__init__()
        M, e, h = num_modules, dim_assoc, dim_hidden

        self.num_modules = M
        self.dim_assoc = e
        self.num_read_tokens = num_read_tokens

        self.router = ContentRouter(dim, M, topk)

        # per-module two-layer encoders and decoder, vectorized over the module axis

        self.enc_k_w1 = Parameter(torch.empty(M, dim, h))
        self.enc_k_w2 = Parameter(torch.empty(M, h, e))
        self.enc_v_w1 = Parameter(torch.empty(M, dim, h))
        self.enc_v_w2 = Parameter(torch.empty(M, h, e))
        self.dec_w1 = Parameter(torch.empty(M, e, h))
        self.dec_w2 = Parameter(torch.empty(M, h, dim))

        for weight in (self.enc_k_w1, self.enc_k_w2, self.enc_v_w1, self.enc_v_w2, self.dec_w1, self.dec_w2):
            init_per_module_(weight)

        # plastic associative layer: base weights, plastic coefficients, write rate

        self.base = Parameter(torch.randn(M, e, e) * e ** -0.5)
        self.alpha = Parameter(torch.full((M, e, e), alpha_init))

        eta_logit = torch.logit(torch.tensor(eta_init))
        self.eta_logit = Parameter(torch.full((M,), eta_logit.item()))

        # reads: learned query tokens, conditioned on a causal summary of the past

        self.read_queries = Parameter(torch.randn(num_read_tokens, dim) * 0.02)
        self.summary_to_query = nn.Linear(dim, dim, bias = False)
        nn.init.zeros_(self.summary_to_query.weight)

    def init_state(self, batch, device = None, dtype = None):
        device = device if device is not None else self.base.device
        dtype = dtype if dtype is not None else self.base.dtype

        traces = torch.zeros(batch, self.num_modules, self.dim_assoc, self.dim_assoc, device = device, dtype = dtype)
        return HebbianMemState(0, traces)

    def _encode(self, x, w1, w2):
        """x: (b, n, dim) -> unit-norm codes (b, n, M, e)"""
        hidden = torch.einsum('b n d, m d h -> b n m h', x, w1)
        hidden = F.gelu(hidden)
        out = torch.einsum('b n m h, m h e -> b n m e', hidden, w2)
        return l2norm(out)

    def effective_weights(self, state):
        """(b, M, e, e) — base plus plastically modulated traces"""
        return self.base + self.alpha * state.traces

    def read(self, summary, state):
        """Retrieve persistent-slot context tokens.

        summary: (b, dim) causal summary of past tokens (zeros at stream start)
        returns: (b, num_read_tokens, dim)
        """
        queries = self.read_queries + self.summary_to_query(summary).unsqueeze(1)   # (b, r, dim)

        return self._associate(queries, state)

    def _associate(self, queries, state):
        """Route queries, pass their keys through the plastic layer, decode.

        queries: (b, r, dim)
        returns: (b, r, dim)
        """
        gates, _ = self.router(queries)                                             # (b, r, M)

        keys = self._encode(queries, self.enc_k_w1, self.enc_k_w2)                  # (b, r, M, e)

        w_eff = self.effective_weights(state)                                       # (b, M, e, e)
        values = torch.einsum('b m f e, b r m e -> b r m f', w_eff, keys)

        hidden = F.gelu(torch.einsum('b r m f, m f h -> b r m h', values, self.dec_w1))
        decoded = torch.einsum('b r m h, m h d -> b r m d', hidden, self.dec_w2)    # (b, r, M, dim)

        return torch.einsum('b r m, b r m d -> b r d', gates, decoded)

    def recall(self, x, state):
        """Per-token content-addressed read: each token queries the store with itself.

        x: (b, n, dim)
        returns: (b, n, dim)
        """
        return self._associate(x, state)

    def write(self, keys_x, values_x, state, mask = None):
        """Fold a segment of (key, value) pairs into the traces of the routed modules.

        keys_x, values_x: (b, n, dim), pair i binds keys_x[:, i] -> values_x[:, i];
        modules are routed by the key. mask: (b, n) bool over pairs, False = ignore (padding)
        """
        assert keys_x.shape == values_x.shape

        b, n, _ = keys_x.shape

        gates, _ = self.router(keys_x)                                              # (b, n, M)

        if mask is not None:
            gates = gates * mask.to(gates.dtype).unsqueeze(-1)
            num_valid = mask.to(gates.dtype).sum(dim = -1).clamp(min = 1.)          # (b,)
        else:
            num_valid = torch.full((b,), float(n), device = keys_x.device, dtype = gates.dtype)

        keys = self._encode(keys_x, self.enc_k_w1, self.enc_k_w2)                   # (b, n, M, e)
        values = self._encode(values_x, self.enc_v_w1, self.enc_v_w2)               # (b, n, M, e)

        outer_sum = torch.einsum('b n m, b n m f, b n m e -> b m f e', gates, values, keys)
        gate_sum = gates.sum(dim = 1)                                               # (b, M)

        target = outer_sum / rearrange(gate_sum.clamp(min = 1e-9), 'b m -> b m 1 1')

        # routing activity in [0, 1]; zero for unselected modules -> trace exactly unchanged

        activity = gate_sum / rearrange(num_valid, 'b -> b 1')

        eta = self.eta_logit.sigmoid()
        rate = rearrange(eta * activity, 'b m -> b m 1 1')

        traces = state.traces.lerp(target, rate)

        return HebbianMemState(state.seq_index + n, traces)

    def forward(self, keys_x, values_x, summary, state, mask = None):
        """Read with the incoming traces, then write the current segment's pairs."""
        read_tokens = self.read(summary, state)
        next_state = self.write(keys_x, values_x, state, mask = mask)
        return read_tokens, next_state
