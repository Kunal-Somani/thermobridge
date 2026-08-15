"""Unit tests for the Radon-domain consistency loss (Chunk 10, §7 L_rad, ADR-011)."""

from __future__ import annotations

import torch

from src.physics.radon import FastRadonProjector, RadonConsistencyLoss


def _projector() -> FastRadonProjector:
    return FastRadonProjector()


def _loss() -> RadonConsistencyLoss:
    return RadonConsistencyLoss(_projector())


def test_projector_output_shape():
    proj = _projector()
    x = torch.randn(3, 1, 12, 16, 16)
    out = proj(x)
    assert out.shape == (3, 3, 16, 16)


def test_projector_differentiable():
    proj = _projector()
    x = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
    out = proj(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_projector_non_negative():
    proj = _projector()
    x = torch.rand(2, 1, 8, 8, 8)  # non-negative input
    out = proj(x)
    assert torch.all(out >= 0.0)


def test_loss_zero_for_identical():
    loss_fn = _loss()
    x = torch.randn(2, 1, 8, 8, 8)
    loss = loss_fn(x, x, True)
    assert torch.isclose(loss, torch.zeros(()), atol=1e-6)


def test_loss_positive_for_different():
    loss_fn = _loss()
    x = torch.randn(2, 1, 8, 8, 8)
    y = x + torch.randn_like(x) * 2.0
    loss = loss_fn(y, x, True)
    assert loss.item() > 0.0


def test_loss_zero_for_mri_target():
    loss_fn = _loss()
    x = torch.randn(2, 1, 8, 8, 8)
    y = torch.randn(2, 1, 8, 8, 8)
    loss = loss_fn(y, x, False)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() == 0.0


def test_loss_is_tensor():
    loss_fn = _loss()
    x = torch.randn(2, 1, 8, 8, 8)
    y = torch.randn(2, 1, 8, 8, 8)
    for mask in (True, False, torch.tensor([1.0, 0.0])):
        loss = loss_fn(y, x, mask)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


def test_gradients_flow_to_prediction():
    loss_fn = _loss()
    x_hat_0 = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
    x_0 = torch.randn(2, 1, 8, 8, 8)
    loss = loss_fn(x_hat_0, x_0, True)
    loss.backward()
    assert x_hat_0.grad is not None


def test_per_sample_mask_only_averages_active_samples():
    loss_fn = _loss()
    x_0 = torch.randn(2, 1, 8, 8, 8)
    x_hat_0 = x_0.clone()
    x_hat_0[1] = x_hat_0[1] + 5.0  # sample 1 is very wrong

    # Only sample 0 (identical) is CT/CBCT -> loss should be exactly 0,
    # ignoring sample 1's large error entirely.
    mask = torch.tensor([1.0, 0.0])
    loss = loss_fn(x_hat_0, x_0, mask)
    assert torch.isclose(loss, torch.zeros(()), atol=1e-6)
