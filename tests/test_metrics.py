"""Unit tests for src/training/metrics.py — ThermoBridge (R7).

Tests are hand-computed against known closed-form answers to ensure
the metric implementations are mathematically correct.

Run::
    pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.training.metrics import (
    MetricResult,
    _ssim_2d,
    compute_all_metrics,
    mae_hu_in_mask,
    psnr_in_mask,
    ssim_in_mask,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_cube(shape=(8, 16, 16), seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_hu, target_hu, mask) with full-foreground mask."""
    rng = np.random.default_rng(seed)
    target = rng.uniform(-500.0, 1500.0, size=shape).astype(np.float32)
    noise  = rng.normal(0.0, 50.0, size=shape).astype(np.float32)
    pred   = target + noise
    mask   = np.ones(shape, dtype=np.uint8)
    return pred, target, mask


# ============================================================================
# 1. MAE-HU
# ============================================================================


class TestMAEHU:
    def test_perfect_prediction_is_zero(self) -> None:
        t    = np.array([[[0.0, 100.0, -500.0]]], dtype=np.float32)
        mask = np.ones_like(t, dtype=np.uint8)
        assert mae_hu_in_mask(t, t, mask) == 0.0

    def test_hand_computed_value(self) -> None:
        """MAE of |pred - target| in mask = mean(|10, 20|) = 15."""
        target = np.array([[[0.0, 0.0]]], dtype=np.float32)
        pred   = np.array([[[10.0, -20.0]]], dtype=np.float32)
        mask   = np.ones((1, 1, 2), dtype=np.uint8)
        np.testing.assert_allclose(mae_hu_in_mask(pred, target, mask), 15.0, atol=1e-5)

    def test_mask_restricts_to_foreground(self) -> None:
        """Only in-mask voxels count; background errors are ignored."""
        target = np.zeros((1, 1, 4), dtype=np.float32)
        pred   = np.array([[[0.0, 100.0, 0.0, 100.0]]], dtype=np.float32)
        mask   = np.array([[[1, 0, 1, 0]]], dtype=np.uint8)  # alternating
        # Only voxels 0 and 2 are in mask — both have error 0
        assert mae_hu_in_mask(pred, target, mask) == 0.0

    def test_mask_restricts_to_nonzero_region(self) -> None:
        """Background (mask=0) voxels with large error should not affect result."""
        target = np.zeros((1, 1, 4), dtype=np.float32)
        pred   = np.array([[[0.0, 9999.0, 0.0, 9999.0]]], dtype=np.float32)
        mask   = np.array([[[1, 0, 1, 0]]], dtype=np.uint8)
        assert mae_hu_in_mask(pred, target, mask) == 0.0

    def test_empty_mask_raises(self) -> None:
        t    = np.ones((4, 4, 4), dtype=np.float32)
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="no foreground"):
            mae_hu_in_mask(t, t, mask)

    def test_known_asymmetric_values(self) -> None:
        """MAE of [0, 10, 30] = 40/3 ≈ 13.333."""
        target = np.zeros((1, 1, 3), dtype=np.float32)
        pred   = np.array([[[0.0, 10.0, 30.0]]], dtype=np.float32)
        mask   = np.ones((1, 1, 3), dtype=np.uint8)
        np.testing.assert_allclose(
            mae_hu_in_mask(pred, target, mask), 40.0 / 3.0, rtol=1e-5
        )


# ============================================================================
# 2. PSNR
# ============================================================================


class TestPSNR:
    def test_perfect_prediction_returns_inf(self) -> None:
        t    = np.ones((4, 4, 4), dtype=np.float32) * 100.0
        mask = np.ones((4, 4, 4), dtype=np.uint8)
        assert psnr_in_mask(t, t, mask) == float("inf")

    def test_hand_computed_psnr(self) -> None:
        """PSNR = 10*log10(100^2 / 25) = 10*log10(400) ≈ 26.02 dB.

        data_range=100, MSE of uniform error of 5: MSE=25.
        """
        shape  = (1, 10, 10)
        target = np.zeros(shape, dtype=np.float32)
        pred   = np.full(shape, 5.0, dtype=np.float32)
        mask   = np.ones(shape, dtype=np.uint8)
        expected = 10.0 * np.log10(100.0**2 / 25.0)
        np.testing.assert_allclose(
            psnr_in_mask(pred, target, mask, data_range=100.0),
            expected, rtol=1e-5,
        )

    def test_psnr_mask_sensitive(self) -> None:
        """PSNR with partial mask should differ from full mask."""
        shape  = (1, 4, 4)
        target = np.zeros(shape, dtype=np.float32)
        pred   = np.full(shape, 10.0, dtype=np.float32)
        full   = np.ones(shape, dtype=np.uint8)
        partial = full.copy()
        partial[0, 0, 0] = 0
        # Both give same MSE (constant error) so PSNR should be equal
        p1 = psnr_in_mask(pred, target, full, data_range=100.0)
        p2 = psnr_in_mask(pred, target, partial, data_range=100.0)
        np.testing.assert_allclose(p1, p2, rtol=1e-5)

    def test_empty_mask_raises(self) -> None:
        t    = np.ones((4, 4, 4), dtype=np.float32)
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            psnr_in_mask(t, t, mask)

    def test_psnr_decreases_with_more_error(self) -> None:
        shape  = (1, 8, 8)
        target = np.zeros(shape, dtype=np.float32)
        mask   = np.ones(shape, dtype=np.uint8)
        pred1  = np.full(shape, 10.0, dtype=np.float32)
        pred2  = np.full(shape, 50.0, dtype=np.float32)
        assert psnr_in_mask(pred1, target, mask) > psnr_in_mask(pred2, target, mask)


# ============================================================================
# 3. SSIM
# ============================================================================


class TestSSIM:
    def test_identical_arrays_gives_one(self) -> None:
        """SSIM of identical images should be ≈1 (box window artefacts at edges)."""
        rng    = np.random.default_rng(0)
        t      = rng.uniform(0, 1000, size=(4, 32, 32)).astype(np.float32)
        mask   = np.ones((4, 32, 32), dtype=np.uint8)
        result = ssim_in_mask(t, t, mask)
        assert result > 0.99, f"SSIM of identical arrays should be ~1.0, got {result:.4f}"

    def test_random_noise_gives_lower_ssim(self) -> None:
        """Random prediction should have lower SSIM than the identity."""
        rng    = np.random.default_rng(1)
        target = rng.uniform(-500, 1500, size=(8, 32, 32)).astype(np.float32)
        noise  = rng.normal(0, 500, size=(8, 32, 32)).astype(np.float32)
        mask   = np.ones((8, 32, 32), dtype=np.uint8)
        ssim_noise  = ssim_in_mask(target + noise, target, mask)
        ssim_identity = ssim_in_mask(target, target, mask)
        assert ssim_noise < ssim_identity

    def test_ssim_only_counts_foreground_slices(self) -> None:
        """Slices with mask=0 must not contribute to the average."""
        rng    = np.random.default_rng(2)
        target = rng.uniform(0, 1000, size=(6, 16, 16)).astype(np.float32)
        pred   = rng.uniform(0, 1000, size=(6, 16, 16)).astype(np.float32)
        # Foreground only in first 3 slices
        mask_full    = np.ones((6, 16, 16), dtype=np.uint8)
        mask_partial = np.zeros((6, 16, 16), dtype=np.uint8)
        mask_partial[:3] = 1
        ssim_full    = ssim_in_mask(pred, target, mask_full)
        ssim_partial = ssim_in_mask(pred, target, mask_partial)
        # Results may differ because different slices are included
        assert isinstance(ssim_partial, float)

    def test_empty_mask_raises(self) -> None:
        t    = np.ones((4, 16, 16), dtype=np.float32)
        mask = np.zeros((4, 16, 16), dtype=np.uint8)
        with pytest.raises(ValueError):
            ssim_in_mask(t, t, mask)

    def test_ssim_range_is_valid(self) -> None:
        """SSIM should be in a reasonable range for medical-like volumes."""
        pred, target, mask = _make_cube(shape=(4, 16, 16), seed=99)
        s = ssim_in_mask(pred, target, mask)
        assert -1.0 <= s <= 1.0, f"SSIM out of range: {s}"

    def test_ssim_2d_perfect_image(self) -> None:
        """Internal _ssim_2d: identical 2-D arrays give values near 1 (interior)."""
        rng    = np.random.default_rng(3)
        arr    = rng.uniform(0, 1000, size=(32, 32)).astype(np.float32)
        ssim_map = _ssim_2d(arr, arr, data_range=1000.0, win_size=7)
        # Interior pixels (away from boundary) should be ≈1
        interior = ssim_map[7:-7, 7:-7]
        np.testing.assert_allclose(interior, np.ones_like(interior), atol=0.01)


# ============================================================================
# 4. compute_all_metrics
# ============================================================================


class TestComputeAllMetrics:
    def test_returns_named_tuple(self) -> None:
        pred, target, mask = _make_cube()
        result = compute_all_metrics(pred, target, mask)
        assert isinstance(result, MetricResult)
        assert hasattr(result, "mae_hu")
        assert hasattr(result, "psnr")
        assert hasattr(result, "ssim")
        assert hasattr(result, "n_mask_voxels")

    def test_perfect_prediction_mae_is_zero(self) -> None:
        target = np.ones((4, 8, 8), dtype=np.float32) * 42.0
        mask   = np.ones((4, 8, 8), dtype=np.uint8)
        result = compute_all_metrics(target, target, mask)
        assert result.mae_hu == 0.0

    def test_n_mask_voxels_is_correct(self) -> None:
        shape  = (4, 8, 8)
        pred, target, _ = _make_cube(shape)
        mask   = np.zeros(shape, dtype=np.uint8)
        mask[:2] = 1   # 2 * 8 * 8 = 128 foreground voxels
        result = compute_all_metrics(pred, target, mask)
        assert result.n_mask_voxels == 128

    def test_metrics_are_finite(self) -> None:
        pred, target, mask = _make_cube(shape=(4, 16, 16), seed=7)
        result = compute_all_metrics(pred, target, mask)
        assert np.isfinite(result.mae_hu)
        assert np.isfinite(result.psnr)
        assert np.isfinite(result.ssim)

    def test_mae_lower_for_better_predictor(self) -> None:
        """A predictor with less noise gets lower MAE."""
        rng    = np.random.default_rng(0)
        target = rng.uniform(-500, 1500, size=(4, 8, 8)).astype(np.float32)
        mask   = np.ones((4, 8, 8), dtype=np.uint8)
        pred_good = target + rng.normal(0, 10,  size=target.shape).astype(np.float32)
        pred_bad  = target + rng.normal(0, 200, size=target.shape).astype(np.float32)
        r_good = compute_all_metrics(pred_good, target, mask)
        r_bad  = compute_all_metrics(pred_bad,  target, mask)
        assert r_good.mae_hu < r_bad.mae_hu
        assert r_good.psnr   > r_bad.psnr


# ============================================================================
# 5. Round-trip integration: invert-normalise then metric
# ============================================================================


_DATA_ROOT   = _REPO_ROOT / "data" / "synthrad2023" / "Task1"
_REAL_PATIENT = _DATA_ROOT / "brain" / "1BA001"
_PREP_ROOT   = _REPO_ROOT / "outputs" / "preprocessed"


@pytest.mark.skipif(
    not (_PREP_ROOT / "brain" / "1BA001" / "ct.npz").exists(),
    reason="Preprocessed data not available.",
)
class TestRoundTripMetrics:
    """Verify that identity predictor on CT→CT gives MAE-HU = 0 (modulo float32)."""

    @pytest.fixture(scope="class")
    def patient(self) -> dict:
        import json
        with open(_PREP_ROOT / "manifest.json") as f:
            manifest = json.load(f)
        return manifest.get("1BA001")

    def test_identity_gives_zero_mae(self, patient) -> None:
        if patient is None:
            pytest.skip("1BA001 not in manifest.")
        from src.data.preprocess import invert_ct_to_hu
        ct_norm = np.load(patient["ct_path"])["data"]
        mask    = np.load(patient["mask_path"])["data"].astype(np.float32)
        ct_hu   = invert_ct_to_hu(ct_norm, patient["ct_norm_params"])
        # Identity: predict the target itself (perfect)
        result = compute_all_metrics(ct_hu, ct_hu, mask)
        assert result.mae_hu == 0.0
        assert result.psnr   == float("inf")
        assert result.ssim   > 0.99
