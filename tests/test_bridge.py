"""Unit tests for I2SBProcess (Chunk 7, Method Spec §3)."""

from __future__ import annotations

import math

import torch

from src.models.bridge import I2SBProcess
from src.models.denoiser import ThermoBridgeDenoiser


def _make_process(s: float = 1.0, num_steps: int = 4, time_weighting: str = "constant") -> I2SBProcess:
    return I2SBProcess(max_variance_s=s, num_steps=num_steps, time_weighting=time_weighting)


def _make_denoiser() -> ThermoBridgeDenoiser:
    torch.manual_seed(0)
    return ThermoBridgeDenoiser(
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        patch_size=(8, 8, 8),
        num_modalities=3,
        modality_embed_dim=8,
    )


def _f_theta(denoiser: ThermoBridgeDenoiser):
    def _call(x_t, t, m_s, m_t, alpha):
        return denoiser(x_t, t, m_s, m_t, alpha)
    return _call


# ---------------------------------------------------------------------------
# forward_marginal
# ---------------------------------------------------------------------------


def test_forward_marginal_mean():
    torch.manual_seed(0)
    process = _make_process()
    x_0 = torch.randn(4, 1, 8, 8, 8)
    x_T = torch.randn(4, 1, 8, 8, 8)

    x_t0, _ = process.forward_marginal(x_0, x_T, torch.zeros(4))
    x_t1, _ = process.forward_marginal(x_0, x_T, torch.ones(4))

    # sigma_t_sq = 0 at t=0 and t=1, so x_t must equal the boundary exactly.
    assert torch.allclose(x_t0, x_0, atol=1e-6)
    assert torch.allclose(x_t1, x_T, atol=1e-6)


def test_forward_marginal_variance():
    torch.manual_seed(0)
    s = 1.0
    process = _make_process(s=s)
    x_0 = torch.zeros(1, 1, 1, 1, 1)
    x_T = torch.zeros(1, 1, 1, 1, 1)
    t_bar = 0.3
    n = 20000  # large N keeps the sample-variance estimator's own noise well under 5%
    t = torch.full((n,), t_bar)
    x_0_rep = x_0.expand(n, 1, 1, 1, 1)
    x_T_rep = x_T.expand(n, 1, 1, 1, 1)

    x_t, _ = process.forward_marginal(x_0_rep, x_T_rep, t)
    empirical_var = x_t.flatten().var(unbiased=True).item()
    expected_var = 2 * s * t_bar * (1 - t_bar)

    assert abs(empirical_var - expected_var) / expected_var < 0.05


def test_marginal_at_midpoint():
    process = _make_process(s=1.0)
    t_bar = torch.linspace(0.0, 1.0, 21)
    _, sigma_t_sq = process.marginal_params(t_bar)
    peak_idx = int(torch.argmax(sigma_t_sq).item())
    assert abs(t_bar[peak_idx].item() - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# bridge_loss
# ---------------------------------------------------------------------------


def test_bridge_loss_scalar():
    torch.manual_seed(0)
    process = _make_process()
    denoiser = _make_denoiser()
    x_0 = torch.randn(2, 1, 16, 16, 16)
    x_T = torch.randn(2, 1, 16, 16, 16)
    t = torch.rand(2)
    m_s = torch.tensor([0, 1])
    m_t = torch.tensor([1, 0])
    alpha = torch.rand(2, 3)

    loss = process.bridge_loss(_f_theta(denoiser), x_0, x_T, t, m_s, m_t, alpha)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_bridge_loss_gradients():
    torch.manual_seed(0)
    process = _make_process()
    denoiser = _make_denoiser()
    x_0 = torch.randn(2, 1, 16, 16, 16)
    x_T = torch.randn(2, 1, 16, 16, 16)
    t = torch.rand(2)
    m_s = torch.tensor([0, 1])
    m_t = torch.tensor([1, 0])
    alpha = torch.rand(2, 3)

    loss = process.bridge_loss(_f_theta(denoiser), x_0, x_T, t, m_s, m_t, alpha)
    loss.backward()

    for name, param in denoiser.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


# ---------------------------------------------------------------------------
# reverse_sample
# ---------------------------------------------------------------------------


def test_reverse_sample_shape():
    torch.manual_seed(0)
    process = _make_process(num_steps=4)
    denoiser = _make_denoiser()
    x_T = torch.randn(1, 1, 16, 16, 16)
    m_s = torch.tensor([0])
    m_t = torch.tensor([1])
    alpha = torch.rand(1, 3)

    with torch.no_grad():
        x_hat_0 = process.reverse_sample(_f_theta(denoiser), x_T, m_s, m_t, alpha, num_steps=4)

    assert x_hat_0.shape == x_T.shape


def test_reverse_sample_deterministic():
    torch.manual_seed(0)
    process = _make_process(num_steps=4)
    denoiser = _make_denoiser()
    x_T = torch.randn(1, 1, 16, 16, 16)
    m_s = torch.tensor([0])
    m_t = torch.tensor([1])
    alpha = torch.rand(1, 3)

    with torch.no_grad():
        out_a = process.reverse_sample(_f_theta(denoiser), x_T, m_s, m_t, alpha, num_steps=4)
        out_b = process.reverse_sample(_f_theta(denoiser), x_T, m_s, m_t, alpha, num_steps=4)

    assert torch.equal(out_a, out_b)


# ---------------------------------------------------------------------------
# Dual-direction batching (ADR-014)
# ---------------------------------------------------------------------------


def test_dual_direction_batching():
    torch.manual_seed(0)
    process = _make_process()
    denoiser = _make_denoiser()
    x_a = torch.randn(1, 1, 16, 16, 16)
    x_b = torch.randn(1, 1, 16, 16, 16)
    t = torch.tensor([0.4])
    alpha = torch.rand(1, 3)

    torch.manual_seed(1)
    loss_fwd = process.bridge_loss(
        _f_theta(denoiser), x_a, x_b, t, torch.tensor([0]), torch.tensor([1]), alpha
    )
    torch.manual_seed(1)
    loss_bwd = process.bridge_loss(
        _f_theta(denoiser), x_b, x_a, t, torch.tensor([1]), torch.tensor([0]), alpha
    )

    assert not torch.allclose(loss_fwd, loss_bwd)


# ---------------------------------------------------------------------------
# time_weighting
# ---------------------------------------------------------------------------


def test_time_weighting_constant():
    process = _make_process(time_weighting="constant")
    t = torch.linspace(0.01, 0.99, 10)
    w = process.time_weighting(t)
    assert torch.allclose(w, torch.ones_like(w))


def test_time_weighting_snr():
    process = _make_process(s=1.0, time_weighting="snr")
    t_low_var = torch.tensor([0.02, 0.98])   # near boundaries -> low variance
    t_high_var = torch.tensor([0.5])          # peak variance

    w_low_var = process.time_weighting(t_low_var)
    w_high_var = process.time_weighting(t_high_var)

    assert (w_low_var > w_high_var).all()
