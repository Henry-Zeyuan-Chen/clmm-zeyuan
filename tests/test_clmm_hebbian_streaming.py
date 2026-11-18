"""Check the plastic state boundary that the MAC host will use."""

from io import BytesIO

import torch

from clmm.hebbian.hebbian_memory import (
    HebbianMemory,
    HebbianMemState,
    hebbian_state_detach,
)


def test_carried_traces_affect_next_segment_and_resume_exactly():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        memory = HebbianMemory(
            dim=8, num_modules=3, dim_assoc=4, topk=2, dim_hidden=12,
            eta_init=0.5, alpha_init=0.7,
        ).eval()
        first_keys, first_values, second_keys, second_values = torch.randn(4, 2, 5, 8)
        summary = torch.randn(2, 8)

    with torch.no_grad():
        initial = memory.init_state(batch=2)
        with_padding = torch.tensor(
            [[True, True, True, True, True], [True, True, False, False, False]]
        )
        _, written = memory(first_keys, first_values, summary, initial, mask=with_padding)
        saved = hebbian_state_detach(written)
        data = BytesIO()
        torch.save({"seq_index": saved.seq_index, "traces": saved.traces}, data)
        data.seek(0)
        restored = HebbianMemState(**torch.load(data, weights_only=True))

        carried_read, carried = memory(second_keys, second_values, summary, written)
        resumed_read, resumed = memory(second_keys, second_values, summary, restored)
        # Match the position at the boundary. Only erase the plastic trace.
        reset = HebbianMemState(written.seq_index, torch.zeros_like(written.traces))
        reset_read, _ = memory(second_keys, second_values, summary, reset)

    assert written.seq_index == restored.seq_index == 5
    assert carried.seq_index == resumed.seq_index == 10
    torch.testing.assert_close(carried_read, resumed_read, rtol=0, atol=0)
    torch.testing.assert_close(carried.traces, resumed.traces, rtol=0, atol=0)
    assert ((carried_read - reset_read).abs().amax(dim=(1, 2)) > 1e-6).all()
    assert torch.count_nonzero(initial.traces).item() == 0
