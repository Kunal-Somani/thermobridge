"""Factory functions for ThermoBridge models."""

from __future__ import annotations

from .denoiser import ThermoBridgeDenoiser
from .unet3d import UNet3D, build_unet3d as _build_unet3d


def build_denoiser(cfg) -> ThermoBridgeDenoiser:
    """Instantiate ThermoBridgeDenoiser from cfg.model.denoiser (§4, §4.1)."""
    d = cfg.model.denoiser
    return ThermoBridgeDenoiser(
        hidden_dim=d.hidden_dim,
        num_layers=d.num_layers,
        num_heads=d.num_heads,
        patch_size=tuple(d.patch_size),
        num_modalities=d.num_modalities,
        modality_embed_dim=d.modality_embed_dim,
    )


def build_unet3d(cfg) -> UNet3D:
    """Instantiate UNet3D from cfg.training (existing baseline builder)."""
    return _build_unet3d(cfg.training)
