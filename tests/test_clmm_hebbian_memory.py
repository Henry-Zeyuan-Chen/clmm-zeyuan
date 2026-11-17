import torch

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
