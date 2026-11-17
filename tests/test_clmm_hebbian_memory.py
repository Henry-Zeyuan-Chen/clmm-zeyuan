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
