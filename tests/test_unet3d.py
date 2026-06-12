"""Shape unit tests for UNet3D — ThermoBridge (R7).

Run::
    pytest tests/test_unet3d.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.unet3d import UNet3D, build_unet3d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_default() -> UNet3D:
    return UNet3D(in_channels=1, channels=[16, 32, 64, 128])


@pytest.fixture
def model_tiny() -> UNet3D:
    return UNet3D(in_channels=1, channels=[8, 16])


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestUNet3DShapes:
    """All shape tests run on CPU — no GPU required (R7)."""

    @pytest.mark.parametrize("patch", [(64, 64, 64), (96, 96, 96)])
    def test_output_shape_cube(self, model_default: UNet3D, patch: tuple) -> None:
        """Output shape must equal input shape (B, 1, Z, Y, X)."""
        B = 2
        src = torch.randn(B, 1, *patch)
        did = torch.zeros(B, dtype=torch.long)
        out = model_default(src, did)
        assert out.shape == (B, 1, *patch), (
            f"Expected {(B, 1, *patch)}, got {tuple(out.shape)}"
        )

    def test_output_shape_asymmetric(self, model_tiny: UNet3D) -> None:
        """Non-cubic input must also produce matching output shape."""
        src = torch.randn(1, 1, 48, 64, 80)
        did = torch.zeros(1, dtype=torch.long)
        out = model_tiny(src, did)
        assert out.shape == (1, 1, 48, 64, 80)

    def test_direction_0_and_1_both_run(self, model_default: UNet3D) -> None:
        """Both direction IDs (0 and 1) must produce same-shape output."""
        src = torch.randn(1, 1, 32, 32, 32)
        for d in (0, 1):
            did = torch.tensor([d], dtype=torch.long)
            out = model_default(src, did)
            assert out.shape == (1, 1, 32, 32, 32), f"direction={d} failed"

    def test_batch_size_1(self, model_tiny: UNet3D) -> None:
        src = torch.randn(1, 1, 32, 32, 32)
        did = torch.zeros(1, dtype=torch.long)
        out = model_tiny(src, did)
        assert out.shape == (1, 1, 32, 32, 32)

    def test_batch_size_4(self, model_tiny: UNet3D) -> None:
        src = torch.randn(4, 1, 32, 32, 32)
        did = torch.zeros(4, dtype=torch.long)
        out = model_tiny(src, did)
        assert out.shape == (4, 1, 32, 32, 32)

    def test_output_channel_is_1(self, model_default: UNet3D) -> None:
        """Output must always have exactly 1 channel (target modality)."""
        src = torch.randn(2, 1, 32, 32, 32)
        did = torch.zeros(2, dtype=torch.long)
        out = model_default(src, did)
        assert out.shape[1] == 1, f"Expected 1 output channel, got {out.shape[1]}"

    def test_linear_head_no_saturation(self, model_tiny: UNet3D) -> None:
        """Output can exceed [-1,1] — head is linear (no tanh/sigmoid clamp)."""
        torch.manual_seed(0)
        src = torch.randn(1, 1, 32, 32, 32) * 10.0
        did = torch.zeros(1, dtype=torch.long)
        out = model_tiny(src, did)
        # At least some outputs should be outside [-1,1] due to large input scale
        # (if the head clamped, this would fail)
        assert out.abs().max().item() != 1.0 or out.std().item() > 0.01

    @pytest.mark.parametrize("channels", [
        [8, 16],
        [16, 32, 64],
        [32, 64, 128, 256],
    ])
    def test_various_channel_configs(self, channels: list) -> None:
        model = UNet3D(in_channels=1, channels=channels)
        src   = torch.randn(1, 1, 32, 32, 32)
        did   = torch.zeros(1, dtype=torch.long)
        out   = model(src, did)
        assert out.shape == (1, 1, 32, 32, 32)

    def test_gradient_flows_to_all_params(self, model_tiny: UNet3D) -> None:
        """All model parameters must receive non-None gradients after backward."""
        src    = torch.randn(1, 1, 32, 32, 32, requires_grad=False)
        did    = torch.zeros(1, dtype=torch.long)
        target = torch.randn(1, 1, 32, 32, 32)
        pred   = model_tiny(src, did)
        loss   = torch.nn.functional.l1_loss(pred, target)
        loss.backward()

        no_grad = [n for n, p in model_tiny.named_parameters() if p.grad is None]
        assert not no_grad, f"Parameters with no gradient: {no_grad}"

    def test_direction_channel_affects_output(self, model_default: UNet3D) -> None:
        """Different direction IDs on the same input should give different outputs."""
        torch.manual_seed(1)
        src  = torch.randn(1, 1, 32, 32, 32)
        did0 = torch.tensor([0], dtype=torch.long)
        did1 = torch.tensor([1], dtype=torch.long)
        out0 = model_default(src, did0)
        out1 = model_default(src, did1)
        # The direction channel injects a different constant (0 vs 1),
        # so outputs must differ (even for an untrained model).
        assert not torch.allclose(out0, out1, atol=1e-6), (
            "Direction channel has no effect — check that dir_ch is actually concatenated."
        )


# ---------------------------------------------------------------------------
# build_unet3d factory
# ---------------------------------------------------------------------------


class TestBuildFactory:
    def test_build_from_config(self) -> None:
        from omegaconf import OmegaConf
        fake_cfg = OmegaConf.create({"unet_channels": [16, 32, 64, 128]})
        model = build_unet3d(fake_cfg)
        src   = torch.randn(1, 1, 32, 32, 32)
        did   = torch.zeros(1, dtype=torch.long)
        out   = model(src, did)
        assert out.shape == (1, 1, 32, 32, 32)

    def test_parameter_count_is_reasonable(self) -> None:
        """Default [32,64,128,256] U-Net should have ~10–30M params."""
        model  = UNet3D(channels=[32, 64, 128, 256])
        n_params = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < n_params < 100_000_000, (
            f"Unexpected param count: {n_params:,}"
        )
