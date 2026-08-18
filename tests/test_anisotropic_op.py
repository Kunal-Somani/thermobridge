"""Unit tests for the learnable 3D anisotropic-diffusion operator (Chunk 9b, §6)."""

from __future__ import annotations

import torch

from src.models.anisotropic_op import (
    AnisotropicDiffusionOp,
    ConvLocalMixer,
    compute_3d_gradients,
    perona_malik_conductance,
)
from src.models.denoiser import ThermoBridgeDenoiser


def _make_op(
    num_channels: int = 4,
    num_steps: int = 3,
    per_channel_k: bool = True,
    init_conductance_k: float = 1.0,
    init_step_size: float = 0.1,
) -> AnisotropicDiffusionOp:
    torch.manual_seed(0)
    return AnisotropicDiffusionOp(
        num_channels=num_channels,
        num_steps=num_steps,
        per_channel_k=per_channel_k,
        init_conductance_k=init_conductance_k,
        init_step_size=init_step_size,
    )


def test_output_shape():
    op = _make_op()
    x = torch.randn(2, 4, 8, 8, 8)
    y = op(x)
    assert y.shape == x.shape


def test_smooths_flat_region():
    op = _make_op()
    x = torch.full((1, 4, 8, 8, 8), 0.5)
    y = op(x)
    assert torch.allclose(y, x, atol=1e-5)


def test_preserves_step_edge():
    op = _make_op(num_steps=3, init_step_size=0.1)
    x = torch.zeros(1, 4, 8, 8, 8)
    x[:, :, :, :, 4:] = 1.0  # sharp step edge along the W axis

    def edge_grad_mag(vol: torch.Tensor) -> float:
        return (vol[:, :, :, :, 4] - vol[:, :, :, :, 3]).abs().mean().item()

    grad_before = edge_grad_mag(x)
    with torch.no_grad():
        y = op(x)
    grad_after = edge_grad_mag(y)

    # Edge-stopping conductance should keep most of the step's sharpness —
    # far more preserved than a flat/isotropic average would leave.
    assert grad_after > 0.5 * grad_before


def test_K_gradient_flows():
    op = _make_op()
    x = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    y = op(x)
    y.sum().backward()
    assert op.K_raw.grad is not None


def test_dt_gradient_flows():
    op = _make_op()
    x = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    y = op(x)
    y.sum().backward()
    assert op.dt_raw.grad is not None


def test_conductance_at_zero_gradient():
    K = torch.tensor([1.0, 2.0, 5.0])
    s = torch.zeros(3)
    g = perona_malik_conductance(s, K)
    assert torch.allclose(g, torch.ones(3), atol=1e-6)


def test_conductance_decreases_with_gradient():
    K = torch.tensor([1.0])
    s_small = torch.tensor([0.1])
    s_large = torch.tensor([10.0])
    g_small = perona_malik_conductance(s_small, K)
    g_large = perona_malik_conductance(s_large, K)
    assert (g_large < g_small).all()


def test_denoiser_set_local_mixer():
    torch.manual_seed(0)
    denoiser = ThermoBridgeDenoiser(
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        patch_size=(8, 8, 8),
        num_modalities=3,
        modality_embed_dim=8,
    )
    op = AnisotropicDiffusionOp(
        num_channels=16,
        num_steps=2,
        per_channel_k=True,
        init_conductance_k=1.0,
        init_step_size=0.1,
    )
    denoiser.set_local_mixer(op)

    # Each block gets its own instance, not a shared one.
    assert len(denoiser.local_mixers) == 2
    assert denoiser.local_mixers[0] is not denoiser.local_mixers[1]
    assert denoiser.local_mixers[0] is not op

    x_T = torch.randn(2, 1, 16, 16, 16)
    t = torch.rand(2)
    m_s = torch.tensor([0, 1])
    m_t = torch.tensor([1, 0])
    alpha = torch.rand(2, 3)

    out = denoiser(x_T, t, m_s, m_t, alpha)
    assert out.shape == x_T.shape


def test_conv_local_mixer_shape():
    mixer = ConvLocalMixer(num_channels=8)
    x = torch.randn(2, 8, 10, 10, 10)
    y = mixer(x)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# compute_3d_gradients / spatial_gradients
# ---------------------------------------------------------------------------


def test_compute_3d_gradients_shape():
    """Output channel count must be 3*C and spatial dims must be preserved."""
    x = torch.randn(2, 1, 16, 16, 16)
    g = compute_3d_gradients(x)
    assert g.shape == (2, 3, 16, 16, 16), f"expected (2,3,16,16,16), got {tuple(g.shape)}"


def test_compute_3d_gradients_differentiable():
    """Gradients must flow back to the input."""
    x = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
    g = compute_3d_gradients(x)
    g.sum().backward()
    assert x.grad is not None, "x.grad is None — compute_3d_gradients is not differentiable"


def test_spatial_gradients_matches_module_fn():
    """AnisotropicDiffusionOp.spatial_gradients must equal compute_3d_gradients."""
    op = _make_op(num_channels=1)
    x = torch.randn(2, 1, 8, 8, 8)
    assert torch.allclose(op.spatial_gradients(x), compute_3d_gradients(x))


def test_compute_3d_gradients_boundary_zero():
    """The last slice along each spatial dim must be zero (no-flux BC)."""
    x = torch.randn(1, 1, 8, 8, 8)
    g = compute_3d_gradients(x)  # (1, 3, 8, 8, 8) — channels: dz, dy, dx
    assert g[:, 0, -1, :, :].abs().max().item() == 0.0, "dz boundary not zeroed"
    assert g[:, 1, :, -1, :].abs().max().item() == 0.0, "dy boundary not zeroed"
    assert g[:, 2, :, :, -1].abs().max().item() == 0.0, "dx boundary not zeroed"
