"""Final integration test — Chunk 10: denoiser + routing gate + anisotropic
operator + I2SB bridge + Radon loss, all wired through LitBridge.training_step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.anisotropic_op import AnisotropicDiffusionOp
from src.models.routing import AnatomyRouter, RoutedAdapterBlock
from src.training.lit_bridge import LitBridge


def _make_cfg():
    return OmegaConf.create({
        "model": {
            "bridge": {
                "max_variance_s": 1.0,
                "num_steps": 4,
                "time_weighting": "constant",
                "prediction_target": "x0",
            },
            "denoiser": {
                "patch_size": [8, 8, 8],
                "hidden_dim": 16,
                "num_layers": 2,
                "num_heads": 4,
                "num_modalities": 3,
                "modality_embed_dim": 8,
            },
            "routing": {
                "num_anatomies": 3,
                "top_k": 2,
                "adapter_rank": 4,
                "tau_max": 2.0,
                "tau_min": 0.5,
            },
            "anisotropic_op": {
                "use_anisotropic_op": True,
                "num_steps": 2,
                "per_channel_k": True,
                "init_conductance_k": 1.0,
                "init_step_size": 0.1,
            },
        },
        "loss": {
            "weights": {"rec": 1.0, "bnd": 0.1, "rad": 0.1, "ent": 0.01, "bal": 0.01, "cls": 0.0},
        },
        "data": {"all_anatomies": ["brain", "pelvis", "HN"]},
        "training": {
            "optimizer": {"lr": 1.0e-4, "weight_decay": 1.0e-2, "betas": [0.9, 0.999]},
            "scheduler": {"warmup_epochs": 1, "min_lr": 1.0e-6},
            "max_epochs": 10,
        },
        "patch": {"size": [16, 16, 16], "inference_overlap": 0.5},
    })


def _build_full_lit_bridge(cfg) -> LitBridge:
    """Build LitBridge with the routing gate (§5) and anisotropic operator (§6) installed."""
    torch.manual_seed(0)
    mod = LitBridge(cfg)

    d = cfg.model.denoiser
    r = cfg.model.routing
    router = AnatomyRouter(
        in_channels=1,
        hidden_dim=8,
        num_anatomies=r.num_anatomies,
        top_k=r.top_k,
        adapter_rank=r.adapter_rank,
        tau_max=r.tau_max,
        tau_min=r.tau_min,
        total_epochs=int(cfg.training.max_epochs),
    )
    adapter_blocks = nn.ModuleList([
        RoutedAdapterBlock(dim=d.hidden_dim, num_anatomies=r.num_anatomies, adapter_rank=r.adapter_rank)
        for _ in range(d.num_layers)
    ])
    mod.denoiser.set_adapters(router, adapter_blocks)

    a = cfg.model.anisotropic_op
    op = AnisotropicDiffusionOp(
        num_channels=d.hidden_dim,
        num_steps=a.num_steps,
        per_channel_k=a.per_channel_k,
        init_conductance_k=a.init_conductance_k,
        init_step_size=a.init_step_size,
    )
    mod.denoiser.set_local_mixer(op)

    return mod


def _random_batch(batch_size: int, direction_ids: list[int]) -> dict:
    return {
        "source": torch.randn(batch_size, 1, 16, 16, 16),
        "target": torch.randn(batch_size, 1, 16, 16, 16),
        "direction_id": torch.tensor(direction_ids, dtype=torch.long),
    }


def test_full_forward_pass():
    cfg = _make_cfg()
    mod = _build_full_lit_bridge(cfg)

    batch = _random_batch(batch_size=2, direction_ids=[0, 1])
    loss = mod.training_step(batch, 0)

    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    missing = [name for name, p in mod.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"


def test_dual_direction_full():
    cfg = _make_cfg()
    mod = _build_full_lit_bridge(cfg)

    torch.manual_seed(42)
    x_a = torch.randn(1, 1, 16, 16, 16)
    x_b = torch.randn(1, 1, 16, 16, 16)

    batch_fwd = {"source": x_a, "target": x_b, "direction_id": torch.tensor([0])}
    batch_bwd = {"source": x_b, "target": x_a, "direction_id": torch.tensor([1])}

    torch.manual_seed(7)
    loss_fwd = mod.training_step(batch_fwd, 0)
    torch.manual_seed(7)
    loss_bwd = mod.training_step(batch_bwd, 0)

    assert not torch.allclose(loss_fwd, loss_bwd)


def test_all_loss_terms_logged():
    cfg = _make_cfg()
    mod = _build_full_lit_bridge(cfg)

    logged: dict[str, torch.Tensor] = {}

    def fake_log(name, value, *args, **kwargs):
        logged[name] = value

    mod.log = fake_log  # type: ignore[assignment]

    batch = _random_batch(batch_size=2, direction_ids=[0, 1])
    mod.training_step(batch, 0)

    for term in ("rec", "bnd", "rad", "ent", "bal", "cls"):
        key = f"train/loss_{term}"
        assert key in logged, f"{key} was not logged"
