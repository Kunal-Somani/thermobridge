"""Mask-aware evaluation metrics for ThermoBridge sCT synthesis.

All metrics computed INSIDE the patient body mask only (R2).
CT predictions MUST be inverse-normalised to HU before MAE is computed (R2).

Metrics implemented:
    mae_hu_in_mask   — mean absolute error in HU, in-mask only (headline metric §9).
    psnr_in_mask     — peak signal-to-noise ratio, in-mask only.
    ssim_in_mask     — slice-averaged 2-D SSIM across the axial axis, in-mask.

SSIM implementation note:
    True 3-D SSIM is prohibitively expensive for full-volume evaluation.  We compute
    2-D SSIM on every axial slice and average across slices that contain ≥1 foreground
    voxel (mask-weighted averaging).  This is stated explicitly so paper==code (R1).

Math-heavy functions are unit-tested in tests/test_metrics.py (R7).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.ndimage import uniform_filter


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------


class MetricResult(NamedTuple):
    """Per-patient metric bundle returned by compute_all_metrics."""

    mae_hu: float       # in-mask MAE in HU (headline, R2)
    psnr: float         # in-mask PSNR  (dB)
    ssim: float         # slice-averaged in-mask SSIM  ∈ [0, 1]
    n_mask_voxels: int  # number of foreground voxels (for sanity checks)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def mae_hu_in_mask(
    pred_hu: np.ndarray,
    target_hu: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute mean absolute error in Hounsfield Units, masked.

    Args:
        pred_hu:   Prediction in HU, shape (Z, Y, X).  Must already be
                   inverse-normalised (NOT the normalised [-1,1] array).
        target_hu: Ground-truth CT in HU, same shape.
        mask:      Binary patient mask, same shape.  Non-zero = foreground.

    Returns:
        Scalar MAE in HU.

    Raises:
        ValueError: If no foreground voxels exist in the mask.
    """
    mask_bool = mask.astype(bool)
    n = mask_bool.sum()
    if n == 0:
        raise ValueError("mask contains no foreground voxels — cannot compute MAE.")
    return float(np.abs(pred_hu[mask_bool] - target_hu[mask_bool]).mean())


def psnr_in_mask(
    pred_hu: np.ndarray,
    target_hu: np.ndarray,
    mask: np.ndarray,
    data_range: float | None = None,
) -> float:
    """Peak signal-to-noise ratio computed on in-mask voxels only.

    PSNR = 10 · log₁₀(data_range² / MSE_in_mask)

    Args:
        pred_hu:    Prediction in HU, shape (Z, Y, X).
        target_hu:  Ground-truth in HU, same shape.
        mask:       Binary patient mask.
        data_range: Value range for PSNR denominator.  Defaults to
                    target_hu[mask].max() - target_hu[mask].min().

    Returns:
        PSNR in dB.  Returns +inf if MSE == 0 (perfect prediction).
    """
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        raise ValueError("mask contains no foreground voxels.")

    if data_range is None:
        t = target_hu[mask_bool]
        data_range = float(t.max() - t.min())
        if data_range == 0.0:
            data_range = 1.0  # degenerate: avoid log(0)

    mse = float(np.mean((pred_hu[mask_bool] - target_hu[mask_bool]) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def _ssim_2d(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> np.ndarray:
    """Compute 2-D SSIM map for one axial slice.

    Uses a uniform (box) window for speed.  Returns a 2-D map of local SSIM
    values; the caller averages over the mask region.

    Args:
        pred, target: 2-D arrays of the same shape (Y, X).
        data_range:   Max − min of target values.
        win_size:     Window size for local statistics.
        k1, k2:       Stability constants (SSIM paper defaults).

    Returns:
        2-D SSIM map of same shape as inputs.
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    size = (win_size, win_size)

    mu1  = uniform_filter(pred.astype(np.float64),   size=size)
    mu2  = uniform_filter(target.astype(np.float64), size=size)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sig1_sq = uniform_filter(pred.astype(np.float64) ** 2,           size=size) - mu1_sq
    sig2_sq = uniform_filter(target.astype(np.float64) ** 2,         size=size) - mu2_sq
    sig12   = uniform_filter(pred.astype(np.float64) * target.astype(np.float64), size=size) - mu12

    num = (2 * mu12 + c1) * (2 * sig12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sig1_sq + sig2_sq + c2)
    return (num / (den + 1e-10)).astype(np.float32)


def ssim_in_mask(
    pred_hu: np.ndarray,
    target_hu: np.ndarray,
    mask: np.ndarray,
    win_size: int = 7,
) -> float:
    """Slice-averaged 2-D SSIM, restricted to foreground slices.

    SSIM is computed slice-by-slice along the axial (Z) axis using a 2-D
    uniform-window approximation, then averaged across slices that contain
    at least one foreground voxel.  This is the stated methodology (R1).

    Args:
        pred_hu:   Prediction in HU, shape (Z, Y, X).
        target_hu: Ground-truth in HU, same shape.
        mask:      Binary patient mask, same shape.
        win_size:  Sliding window size for local SSIM statistics.

    Returns:
        Scalar SSIM ∈ [-1, 1] (typically ∈ [0, 1] for medical images).
    """
    mask_bool = mask.astype(bool)
    target_vals = target_hu[mask_bool]
    if len(target_vals) == 0:
        raise ValueError("mask contains no foreground voxels.")

    data_range = float(target_vals.max() - target_vals.min())
    if data_range == 0.0:
        data_range = 1.0

    Z = pred_hu.shape[0]
    slice_ssims: list[float] = []

    for z in range(Z):
        mask_slice = mask[z].astype(bool)
        if mask_slice.sum() == 0:
            continue  # skip background-only slices
        ssim_map = _ssim_2d(pred_hu[z], target_hu[z], data_range, win_size)
        # Average SSIM over the foreground pixels of this slice
        slice_ssims.append(float(ssim_map[mask_slice].mean()))

    if not slice_ssims:
        raise ValueError("No foreground slices found.")

    return float(np.mean(slice_ssims))


# ---------------------------------------------------------------------------
# Convenience: compute all three metrics at once
# ---------------------------------------------------------------------------


def compute_all_metrics(
    pred_hu: np.ndarray,
    target_hu: np.ndarray,
    mask: np.ndarray,
    psnr_data_range: float | None = None,
) -> MetricResult:
    """Compute MAE-HU, PSNR, and SSIM for one patient.

    All arrays must be in HU and same shape (Z, Y, X).
    mask is binary (non-zero = foreground).
    """
    mask_bool = mask.astype(bool)
    return MetricResult(
        mae_hu        = mae_hu_in_mask(pred_hu, target_hu, mask),
        psnr          = psnr_in_mask(pred_hu, target_hu, mask, psnr_data_range),
        ssim          = ssim_in_mask(pred_hu, target_hu, mask),
        n_mask_voxels = int(mask_bool.sum()),
    )
