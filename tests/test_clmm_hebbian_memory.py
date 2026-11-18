import torch
import torch.nn.functional as F

from clmm.hebbian.hebbian_memory import HebbianMemory
from clmm.hebbian.router import ContentRouter

# helpers

def diff(x, y):
    return (x - y).abs().amax()

def make_memory(seed = 0, **kwargs):
    """tiny float64 HebbianMemory with seeded init, global rng left untouched"""
    kwargs = {**dict(dim = 8, num_modules = 4, dim_assoc = 4, dim_hidden = 8), **kwargs}

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return HebbianMemory(**kwargs).double()

# router

def test_router_topk():
    dim, num_modules, topk = 8, 4, 2

    with torch.random.fork_rng():
        torch.manual_seed(0)
        router = ContentRouter(dim, num_modules, topk).double()
        x = torch.randn(3, 5, dim, dtype = torch.float64)                # (b, n, dim)

    gates, top_indices = router(x)                                       # (b, n, M), (b, n, k)

    assert gates.shape == (3, 5, num_modules)
    assert top_indices.shape == (3, 5, topk)

    nonzero = gates != 0.                                                # (b, n, M)

    assert (nonzero.sum(dim = -1) == topk).all()
    assert diff(gates.sum(dim = -1), torch.ones(3, 5, dtype = torch.float64)) < 1e-12

    selected = torch.zeros_like(nonzero).scatter(-1, top_indices, True)  # (b, n, M)

    assert (nonzero == selected).all()

# read

def test_read_shape():
    mem = make_memory(num_read_tokens = 3)
    state = mem.init_state(2)

    summary = torch.zeros(2, 8, dtype = torch.float64)                   # (b, dim)

    assert mem.read(summary, state).shape == (2, 3, 8)

def test_read_ignores_state_when_alpha_zero():
    mem = make_memory()

    with torch.no_grad():
        mem.alpha.zero_()

    with torch.random.fork_rng():
        torch.manual_seed(0)
        summary = torch.randn(2, 8, dtype = torch.float64)               # (b, dim)
        traces = torch.randn(2, 4, 4, 4, dtype = torch.float64)          # (b, M, e, e)

    zero_state = mem.init_state(2)
    random_state = zero_state._replace(traces = traces)

    assert diff(mem.read(summary, zero_state), mem.read(summary, random_state)) == 0.

# recall

def test_recall_shape():
    mem = make_memory()
    state = mem.init_state(2)

    x = torch.zeros(2, 5, 8, dtype = torch.float64)                      # (b, n, dim)

    assert mem.recall(x, state).shape == (2, 5, 8)

def test_recall_zero_traces_is_base_only():
    # with zero traces the plastic term vanishes, so recall equals the base-weights-only path

    mem = make_memory()

    with torch.random.fork_rng():
        torch.manual_seed(0)
        x = torch.randn(2, 5, 8, dtype = torch.float64)                  # (b, n, dim)
        traces = torch.randn(2, 4, 4, 4, dtype = torch.float64)          # (b, M, e, e)

    zero_state = mem.init_state(2)
    random_state = zero_state._replace(traces = traces)

    with_zero_traces = mem.recall(x, zero_state)

    assert diff(with_zero_traces, mem.recall(x, random_state)) > 1e-6    # traces do reach the output

    with torch.no_grad():
        mem.alpha.zero_()

    assert diff(with_zero_traces, mem.recall(x, random_state)) == 0.

# write

def encode_by_hand(x, w1, w2):
    """x: (dim,), one module's (dim, h) and (h, e) weights -> unit-norm code (e,)"""
    return F.normalize(F.gelu(x @ w1) @ w2, dim = -1)

def test_write_formula():
    # one (key, value) pair, so each selected module's target is exactly value_m key_m^T

    mem = make_memory()

    with torch.random.fork_rng():
        torch.manual_seed(0)
        a = torch.randn(8, dtype = torch.float64)                        # (dim,) key token
        b = torch.randn(8, dtype = torch.float64)                        # (dim,) value token
        traces = torch.randn(1, 4, 4, 4, dtype = torch.float64)          # (b, M, e, e)

    state = mem.init_state(1)._replace(traces = traces)
    next_state = mem.write(a.view(1, 1, 8), b.view(1, 1, 8), state)

    gates, _ = mem.router(a)                                             # (M,)
    eta = mem.eta_logit.sigmoid()                                        # (M,)

    assert (gates > 0.).sum() == 2 and (gates < 1.).all()

    for m in range(4):
        key = encode_by_hand(a, mem.enc_k_w1[m], mem.enc_k_w2[m])        # (e,)
        value = encode_by_hand(b, mem.enc_v_w1[m], mem.enc_v_w2[m])      # (e,)

        target = torch.outer(value, key) if gates[m] > 0. else torch.zeros(4, 4, dtype = torch.float64)
        rate = eta[m] * gates[m]                                         # activity = gate, one pair

        expected = traces[0, m] + rate * (target - traces[0, m])         # (e, e)

        assert diff(next_state.traces[0, m], expected) < 1e-12

    assert next_state.seq_index == 1

def test_write_selectivity():
    # 3 keys with topk = 1 touch at most 3 of 8 modules, the rest must be bitwise untouched

    mem = make_memory(num_modules = 8, topk = 1)

    with torch.random.fork_rng():
        torch.manual_seed(0)
        keys_x = torch.randn(2, 3, 8, dtype = torch.float64)             # (b, n, dim)
        values_x = torch.randn(2, 3, 8, dtype = torch.float64)           # (b, n, dim)
        traces = torch.randn(2, 8, 4, 4, dtype = torch.float64)          # (b, M, e, e)

    state = mem.init_state(2)._replace(traces = traces)
    next_state = mem.write(keys_x, values_x, state)

    _, top_indices = mem.router(keys_x)                                  # (b, n, k)

    selected = torch.zeros(2, 8, dtype = torch.bool)                     # (b, M)
    selected.scatter_(-1, top_indices.flatten(1), True)

    changed = (next_state.traces != traces).flatten(2).any(dim = -1)     # (b, M)

    assert selected.any() and not selected.all()
    assert (changed == selected).all()
    assert torch.equal(next_state.traces[~selected], traces[~selected])

def test_write_masked_is_noop():
    mem = make_memory()

    with torch.random.fork_rng():
        torch.manual_seed(0)
        keys_x = torch.randn(2, 6, 8, dtype = torch.float64)             # (b, n, dim)
        values_x = torch.randn(2, 6, 8, dtype = torch.float64)           # (b, n, dim)
        traces = torch.randn(2, 4, 4, 4, dtype = torch.float64)          # (b, M, e, e)

    state = mem.init_state(2)._replace(traces = traces)

    # every pair masked: traces bitwise unchanged

    none_valid = torch.zeros(2, 6, dtype = torch.bool)                   # (b, n)
    next_state = mem.write(keys_x, values_x, state, mask = none_valid)

    assert torch.equal(next_state.traces, traces)

    # last two pairs masked: same traces as writing the first four pairs alone

    mask = torch.ones(2, 6, dtype = torch.bool)                          # (b, n)
    mask[:, 4:] = False

    masked = mem.write(keys_x, values_x, state, mask = mask)
    truncated = mem.write(keys_x[:, :4], values_x[:, :4], state)

    assert diff(masked.traces, truncated.traces) < 1e-12

def test_trace_bounded():
    # each target is a convex combination of unit-norm outer products and the update is a lerp

    mem = make_memory(eta_init = 0.5)
    state = mem.init_state(2)

    with torch.random.fork_rng():
        torch.manual_seed(0)

        with torch.no_grad():
            for _ in range(500):
                keys_x = torch.randn(2, 6, 8, dtype = torch.float64)     # (b, n, dim)
                values_x = torch.randn(2, 6, 8, dtype = torch.float64)   # (b, n, dim)
                state = mem.write(keys_x, values_x, state)

    norms = state.traces.flatten(2).norm(dim = -1)                       # (b, M)

    assert state.traces.isfinite().all()
    assert (norms <= 1. + 1e-9).all()
    assert (norms > 0.).all()

def test_binding():
    # acceptance test for separate key / value inputs: after writing the pair (a, b),
    # querying a selected module's trace with a's key returns b's value, not a's

    mem = make_memory()

    with torch.random.fork_rng():
        torch.manual_seed(0)
        a = torch.randn(8, dtype = torch.float64)                        # (dim,) key token
        b = torch.randn(8, dtype = torch.float64)                        # (dim,) value token

    state = mem.write(a.view(1, 1, 8), b.view(1, 1, 8), mem.init_state(1))

    _, top_indices = mem.router(a)                                       # (k,)

    assert top_indices.numel() == 2

    for m in top_indices.tolist():
        key_a = encode_by_hand(a, mem.enc_k_w1[m], mem.enc_k_w2[m])      # (e,)
        value_a = encode_by_hand(a, mem.enc_v_w1[m], mem.enc_v_w2[m])    # (e,)
        value_b = encode_by_hand(b, mem.enc_v_w1[m], mem.enc_v_w2[m])    # (e,)

        retrieved = state.traces[0, m] @ key_a                           # (e,)

        assert retrieved.norm() > 0.
        assert F.cosine_similarity(retrieved, value_b, dim = -1) > 0.99
        assert F.cosine_similarity(retrieved, value_a, dim = -1) < 0.99
