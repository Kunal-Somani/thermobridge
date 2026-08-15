"""Unit tests for the N-way sparse anatomy routing gate (Chunk 9, Method Spec §5)."""

from __future__ import annotations

import torch

from src.models.denoiser import ThermoBridgeDenoiser
from src.models.routing import (
    AnatomyAdapter,
    AnatomyRouter,
    RoutedAdapterBlock,
    RoutingLoss,
)


def _make_router(
    in_channels: int = 1,
    hidden_dim: int = 8,
    num_anatomies: int = 3,
    top_k: int = 2,
    adapter_rank: int = 4,
    tau_max: float = 2.0,
    tau_min: float = 0.5,
    total_epochs: int = 10,
) -> AnatomyRouter:
    torch.manual_seed(0)
    return AnatomyRouter(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_anatomies=num_anatomies,
        top_k=top_k,
        adapter_rank=adapter_rank,
        tau_max=tau_max,
        tau_min=tau_min,
        total_epochs=total_epochs,
    )


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_alpha_sums_to_one():
    router = _make_router()
    x_T = torch.randn(4, 1, 16, 16, 16)
    alpha_soft = router.compute_alpha_soft(x_T)
    sums = alpha_soft.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)


def test_top_k_sparsity():
    router = _make_router(num_anatomies=5, top_k=2)
    x_T = torch.randn(6, 1, 16, 16, 16)
    alpha, S = router(x_T)
    nonzero_counts = (alpha > 0).sum(dim=-1)
    assert torch.all(nonzero_counts == 2)
    assert S.shape == (6, 2)


def test_sparse_alpha_sums_to_one():
    router = _make_router(num_anatomies=5, top_k=2)
    x_T = torch.randn(6, 1, 16, 16, 16)
    alpha, _S = router(x_T)
    sums = alpha.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(6), atol=1e-5)


def test_temperature_annealing():
    router = _make_router(tau_max=2.0, tau_min=0.5, total_epochs=10)
    taus = [router.tau_schedule(e) for e in range(11)]
    assert abs(taus[0] - 2.0) < 1e-6
    assert abs(taus[-1] - 0.5) < 1e-6
    assert all(taus[i] > taus[i + 1] for i in range(len(taus) - 1))


# ---------------------------------------------------------------------------
# Routing losses
# ---------------------------------------------------------------------------


def test_entropy_loss_shape():
    router = _make_router(num_anatomies=4, top_k=2)
    x_T = torch.randn(5, 1, 16, 16, 16)
    alpha, _S = router(x_T)
    loss = RoutingLoss.entropy_loss(alpha)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_balance_loss_shape():
    router = _make_router(num_anatomies=4, top_k=2)
    x_T = torch.randn(5, 1, 16, 16, 16)
    alpha, _S = router(x_T)
    loss = RoutingLoss.balance_loss(alpha)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_cls_loss_none_labels():
    router = _make_router(num_anatomies=4)
    x_T = torch.randn(3, 1, 16, 16, 16)
    alpha_soft = router.compute_alpha_soft(x_T)
    loss = RoutingLoss.cls_loss(alpha_soft, None)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() == 0.0


def test_cls_loss_with_labels():
    router = _make_router(num_anatomies=4)
    x_T = torch.randn(3, 1, 16, 16, 16)
    alpha_soft = router.compute_alpha_soft(x_T)
    labels = torch.tensor([0, 2, 1])
    loss = RoutingLoss.cls_loss(alpha_soft, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def test_adapter_output_shape():
    adapter = AnatomyAdapter(dim=16, rank=4)
    x = torch.randn(2, 10, 16)
    y = adapter(x)
    assert y.shape == x.shape


def test_routed_adapter_block_output_shape():
    block = RoutedAdapterBlock(dim=16, num_anatomies=3, adapter_rank=4)
    x = torch.randn(2, 10, 16)
    alpha = torch.tensor([[0.6, 0.4, 0.0], [0.0, 0.3, 0.7]])
    y = block(x, alpha)
    assert y.shape == x.shape


def test_gradients_flow_to_adapters():
    block = RoutedAdapterBlock(dim=16, num_anatomies=3, adapter_rank=4)
    x = torch.randn(2, 10, 16)
    alpha = torch.tensor([[0.6, 0.4, 0.0], [0.0, 0.3, 0.7]])
    y = block(x, alpha)
    y.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_gradients_flow_to_gate():
    router = _make_router(num_anatomies=4, top_k=2)
    x_T = torch.randn(3, 1, 16, 16, 16)
    alpha, _S = router(x_T)
    alpha.sum().backward()
    for name, param in router.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


# ---------------------------------------------------------------------------
# Denoiser integration
# ---------------------------------------------------------------------------


def test_denoiser_set_adapters():
    torch.manual_seed(0)
    denoiser = ThermoBridgeDenoiser(
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        patch_size=(8, 8, 8),
        num_modalities=3,
        modality_embed_dim=8,
    )
    router = _make_router(hidden_dim=8, num_anatomies=3, top_k=2, adapter_rank=4)
    adapter_blocks = torch.nn.ModuleList(
        [RoutedAdapterBlock(dim=16, num_anatomies=3, adapter_rank=4) for _ in range(2)]
    )
    denoiser.set_adapters(router, adapter_blocks)

    x_T = torch.randn(2, 1, 16, 16, 16)
    t = torch.rand(2)
    m_s = torch.tensor([0, 1])
    m_t = torch.tensor([1, 0])
    alpha_unused = torch.rand(2, 3)  # ignored once router is installed

    out = denoiser(x_T, t, m_s, m_t, alpha_unused)
    assert out.shape == x_T.shape
