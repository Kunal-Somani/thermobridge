"""Unit tests for ThermoBridgeDenoiser (Chunk 8, Method Spec §4/§4.1)."""

from __future__ import annotations

import torch

from src.models.denoiser import ThermoBridgeDenoiser


def _make_model(hidden_dim: int = 16, num_layers: int = 2, num_heads: int = 4) -> ThermoBridgeDenoiser:
    torch.manual_seed(0)
    return ThermoBridgeDenoiser(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        patch_size=(8, 8, 8),
        num_modalities=3,
        modality_embed_dim=8,
    )


def _perturb_gates(model: ThermoBridgeDenoiser) -> None:
    """Break the (correct, load-bearing) zero-init so conditioning-dependence
    can be observed. adaLN-Zero makes the network an exact identity map of
    x_T at initialization by design (§4.1) — output is mathematically
    independent of t/m_s/m_t until the gates move away from zero via
    training. This mirrors that first training update for testing purposes.
    """
    torch.manual_seed(1)
    for block in model.blocks:
        linear = block.adaLN_modulation[-1]
        linear.weight.data.normal_(0, 0.02)
        linear.bias.data.normal_(0, 0.02)


def test_output_shape():
    model = _make_model()
    x_T = torch.randn(1, 1, 48, 48, 48)
    t = torch.tensor([0.3])
    m_s = torch.tensor([0])
    m_t = torch.tensor([1])
    alpha = torch.rand(1, 3)
    out = model(x_T, t, m_s, m_t, alpha)
    assert out.shape == (1, 1, 48, 48, 48)


def test_timestep_conditioning_changes_output():
    model = _make_model()
    _perturb_gates(model)
    model.eval()
    x_T = torch.randn(1, 1, 16, 16, 16)
    m_s = torch.tensor([0])
    m_t = torch.tensor([1])
    alpha = torch.rand(1, 3)

    with torch.no_grad():
        out_t0 = model(x_T, torch.tensor([0.0]), m_s, m_t, alpha)
        out_t1 = model(x_T, torch.tensor([0.5]), m_s, m_t, alpha)

    assert not torch.allclose(out_t0, out_t1)


def test_modality_conditioning_changes_output():
    model = _make_model()
    _perturb_gates(model)
    model.eval()
    x_T = torch.randn(1, 1, 16, 16, 16)
    t = torch.tensor([0.2])
    alpha = torch.rand(1, 3)

    with torch.no_grad():
        out_a = model(x_T, t, torch.tensor([0]), torch.tensor([1]), alpha)
        out_b = model(x_T, t, torch.tensor([0]), torch.tensor([2]), alpha)

    assert not torch.allclose(out_a, out_b)


def test_gradient_flows_to_all_params():
    model = _make_model()
    x_T = torch.randn(1, 1, 16, 16, 16)
    t = torch.tensor([0.4])
    m_s = torch.tensor([0])
    m_t = torch.tensor([1])
    alpha = torch.rand(1, 3)

    out = model(x_T, t, m_s, m_t, alpha)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_adaLN_zero_init():
    model = _make_model()
    for block in model.blocks:
        linear = block.adaLN_modulation[-1]
        # chunks are (shift1, scale1, gate1, shift2, scale2, gate2); gate
        # chunks are indices 2 and 5 of the 6 equal-sized slices.
        hidden_dim = linear.out_features // 6
        gate1_w = linear.weight[2 * hidden_dim : 3 * hidden_dim]
        gate2_w = linear.weight[5 * hidden_dim : 6 * hidden_dim]
        gate1_b = linear.bias[2 * hidden_dim : 3 * hidden_dim]
        gate2_b = linear.bias[5 * hidden_dim : 6 * hidden_dim]
        assert torch.all(gate1_w == 0)
        assert torch.all(gate2_w == 0)
        assert torch.all(gate1_b == 0)
        assert torch.all(gate2_b == 0)


def test_alpha_passthrough():
    model = _make_model()
    x_T = torch.randn(1, 1, 16, 16, 16)
    t = torch.tensor([0.1])
    m_s = torch.tensor([0])
    m_t = torch.tensor([1])
    alpha = torch.rand(1, 5)  # arbitrary A; identity placeholder must accept any shape
    out = model(x_T, t, m_s, m_t, alpha)
    assert out.shape == (1, 1, 16, 16, 16)
