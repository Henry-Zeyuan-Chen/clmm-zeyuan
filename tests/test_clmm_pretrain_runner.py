"""Scientific metric and bounded-evaluation contracts for the shared runner."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_shared_byte_lm import aggregate, evaluate


class ExtremeFiniteLogits(torch.nn.Module):
    def forward(self, tokens):
        logits = torch.full((*tokens.shape, 256), -3e38)
        logits[..., 1] = 3e38
        return logits


def test_expired_deadline_prevents_model_evaluation():
    class MustNotRun(torch.nn.Module):
        def forward(self, tokens):
            raise AssertionError("Expired evaluation executed inference")
    with pytest.raises(TimeoutError):
        evaluate(MustNotRun(), torch.zeros((2, 5), dtype=torch.long), [0, 5], length=4,
                 stage="final", condition="normal", batch_size=1, device=torch.device("cpu"), deadline=0)


def test_finite_logits_cannot_hide_infinite_cross_entropy():
    with pytest.raises(RuntimeError, match="Nonfinite heldout"):
        evaluate(ExtremeFiniteLogits(), torch.zeros((2, 5), dtype=torch.long), [0, 5], length=4,
                 stage="final", condition="normal", batch_size=1, device=torch.device("cpu"))


def test_aggregate_uses_byte_weighted_nll():
    rows = [dict(stage="final", condition="normal", length=4, bytes=2, nll_sum=2., late_bytes=0, late_nll_sum=0.),
            dict(stage="final", condition="normal", length=4, bytes=4, nll_sum=8., late_bytes=2, late_nll_sum=3.)]
    actual, = aggregate(rows)
    assert actual["windows"] == 2 and actual["bytes"] == 6
    assert actual["nll_per_byte"] == pytest.approx(10 / 6)
    assert actual["bits_per_byte"] == pytest.approx((10 / 6) / math.log(2))
    assert actual["late_nll_per_byte"] == pytest.approx(1.5)
