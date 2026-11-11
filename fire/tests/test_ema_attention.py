# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.nn.modules import C2f, C2fEMA, EMA


def test_ema_validates_constructor_arguments():
    """EMA should reject invalid dimensions instead of entering group selection with invalid values."""
    with pytest.raises(ValueError):
        EMA(0)
    with pytest.raises(ValueError):
        EMA(64, factor=0)


def test_c2f_ema_starts_as_identity_and_backpropagates():
    """C2fEMA should preserve pretrained C2f output initially and learn through its residual gate."""
    torch.manual_seed(0)
    baseline = C2f(64, 64, n=1, shortcut=False)
    attention = C2fEMA(64, 64, n=1, shortcut=False)
    attention.load_state_dict(baseline.state_dict(), strict=False)

    x = torch.randn(2, 64, 20, 20, requires_grad=True)
    expected = baseline(x)
    actual = attention(x)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)

    actual.mean().backward()
    assert attention.ema.gamma.grad is not None
    assert torch.isfinite(attention.ema.gamma.grad).all()