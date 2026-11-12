# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.nn.modules import C2f, C2fEMA, C2fEMACentered, EMA
from ultralytics.utils.loss import v8DetectionLoss


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


def test_centered_c2f_ema_starts_as_identity_and_has_channel_gates():
    """Centered EMA should preserve C2f initially and learn an independent gate for each output channel."""
    torch.manual_seed(0)
    baseline = C2f(128, 128, n=1, shortcut=False)
    attention = C2fEMACentered(128, 128, n=1, shortcut=False)
    attention.load_state_dict(baseline.state_dict(), strict=False)

    x = torch.randn(2, 128, 10, 10, requires_grad=True)
    expected = baseline(x)
    actual = attention(x)

    assert torch.equal(actual, expected)
    assert attention.ema.gamma.shape == (1, 128, 1, 1)

    actual.square().mean().backward()
    assert attention.ema.gamma.grad is not None
    assert torch.isfinite(attention.ema.gamma.grad).all()
    assert torch.count_nonzero(attention.ema.gamma.grad) > 1


def test_detection_loss_reads_model_class_weights():
    """Detection loss should accept positive per-class weights from the model YAML."""
    model = torch.nn.Module()
    model.anchor = torch.nn.Parameter(torch.zeros(1))
    model.args = type("Args", (), {})()
    model.yaml = {"class_loss_weights": [1.25, 2.0, 0.5]}
    head = type("Head", (), {"stride": torch.tensor([8.0, 16.0, 32.0]), "nc": 3, "reg_max": 16})()
    model.model = [head]

    criterion = v8DetectionLoss(model)

    assert criterion.class_loss_weights.shape == (1, 1, 3)
    assert torch.equal(criterion.class_loss_weights.cpu(), torch.tensor([[[1.25, 2.0, 0.5]]]))