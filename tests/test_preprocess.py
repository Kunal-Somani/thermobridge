"""Unit tests for src/data/preprocess.py — ThermoBridge (R7).

Covers:
    1. CT normalize / invert round-trip on synthetic arrays.
    2. MR normalize / invert round-trip on synthetic arrays.
    3. CT clipping is lossy (expected, not a bug).
    4. Real-volume round-trip: invert(normalize(ct_inmask)) == ct_inmask
       within float32 tolerance (the load-bearing R2 requirement).
    5. Manifest entry has all required inverse-normalization keys.

Run with:
    pytest tests/test_preprocess.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure src/ is importable regardless of install state
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.preprocess import (
    invert_ct_to_hu,
    invert_mr,
    normalize_ct,
    normalize_mr_percentile99,
)

# ---------------------------------------------------------------------------
# Constants matching configs/default.yaml (R1 — must agree with config)
# ---------------------------------------------------------------------------
_CLIP_HU_MIN: float = -1024.0
_CLIP_HU_MAX: float = 2000.0
_HU_RANGE: float = _CLIP_HU_MAX - _CLIP_HU_MIN  # 3024.0

_DATA_ROOT = _REPO_ROOT / "data" / "synthrad2023" / "Task1"
_REAL_PATIENT_DIR = _DATA_ROOT / "brain" / "1BA001"  # deterministic choice


# ============================================================================
# Helpers
# ============================================================================


def _make_ct_array(seed: int = 0) -> np.ndarray:
    """Synthetic CT-like array with values spanning the typical HU range."""
    rng = np.random.default_rng(seed)
    # Mix of tissue-like values within clip range
    return rng.uniform(_CLIP_HU_MIN, _CLIP_HU_MAX, size=(32, 32, 32)).astype(np.float32)


def _make_mr_array(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic MR-like array + matching mask."""
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0, 1500, size=(32, 32, 32)).astype(np.float32)
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[4:28, 4:28, 4:28] = 1  # central foreground cube
    return arr, mask


# ============================================================================
# 1. CT round-trip — synthetic
# ============================================================================


class TestCTNormalize:
    """CT normalization and inversion correctness (synthetic data)."""

    def test_output_range_in_minus_one_to_one(self) -> None:
        """Normalized CT must be in [-1, 1] for all values within clip range."""
        ct = _make_ct_array(seed=1)
        norm, _ = normalize_ct(ct, _CLIP_HU_MIN, _CLIP_HU_MAX)
        assert norm.min() >= -1.0 - 1e-6, f"Norm min below -1: {norm.min()}"
        assert norm.max() <=  1.0 + 1e-6, f"Norm max above +1: {norm.max()}"

    def test_boundary_values(self) -> None:
        """clip_hu_min → -1.0; clip_hu_max → +1.0 exactly."""
        ct = np.array([_CLIP_HU_MIN, _CLIP_HU_MAX], dtype=np.float32)
        norm, _ = normalize_ct(ct, _CLIP_HU_MIN, _CLIP_HU_MAX)
        np.testing.assert_allclose(norm[0], -1.0, atol=1e-6)
        np.testing.assert_allclose(norm[1],  1.0, atol=1e-6)

    def test_midpoint_maps_to_zero(self) -> None:
        """Midpoint of HU range → 0.0."""
        mid_hu = (_CLIP_HU_MIN + _CLIP_HU_MAX) / 2.0
        ct = np.array([mid_hu], dtype=np.float32)
        norm, _ = normalize_ct(ct, _CLIP_HU_MIN, _CLIP_HU_MAX)
        np.testing.assert_allclose(norm[0], 0.0, atol=1e-6)

    def test_roundtrip_within_clip_range(self) -> None:
        """R2: invert(normalize(x)) == x for all x in [clip_hu_min, clip_hu_max].

        This is the load-bearing test for the MAE-HU headline metric.
        Tolerance: 0.1 HU — well within clinical relevance thresholds.
        """
        ct = _make_ct_array(seed=42)
        norm, params = normalize_ct(ct, _CLIP_HU_MIN, _CLIP_HU_MAX)
        ct_recovered = invert_ct_to_hu(norm, params)
        # All synthetic values are within clip range (uniform draw)
        np.testing.assert_allclose(
            ct_recovered, ct, atol=0.1,
            err_msg="CT HU round-trip exceeds 0.1 HU tolerance — R2 violation."
        )

    def test_roundtrip_float32_precision(self) -> None:
        """Round-trip precision is limited by float32; check it stays tiny."""
        ct = _make_ct_array(seed=7)
        norm, params = normalize_ct(ct, _CLIP_HU_MIN, _CLIP_HU_MAX)
        ct_recovered = invert_ct_to_hu(norm, params)
        max_err = float(np.abs(ct_recovered - ct).max())
        # float32 with HU range 3024 → max representable error ~3024 * 1.2e-7 ≈ 0.00036 HU
        assert max_err < 0.01, f"Float32 round-trip error too large: {max_err:.6f} HU"

    def test_params_contains_required_keys(self) -> None:
        """Params dict must contain all keys needed for inversion (R2)."""
        _, params = normalize_ct(np.zeros(1, dtype=np.float32), _CLIP_HU_MIN, _CLIP_HU_MAX)
        required = {"method", "clip_hu_min", "clip_hu_max", "hu_range", "norm_min", "norm_max"}
        assert required.issubset(params.keys()), f"Missing keys: {required - params.keys()}"


# ============================================================================
# 2. CT clipping is lossy (correct behaviour, not a bug)
# ============================================================================


class TestCTClipping:
    """Values outside clip range are clipped — round-trip is intentionally lossy."""

    def test_values_above_max_are_clamped(self) -> None:
        """CT > clip_hu_max → normalized to +1.0 (clamped)."""
        ct = np.array([3000.0, 5000.0], dtype=np.float32)
        norm, _ = normalize_ct(ct, _CLIP_HU_MIN, _CLIP_HU_MAX)
        np.testing.assert_allclose(norm, [1.0, 1.0], atol=1e-6)

    def test_values_below_min_are_clamped(self) -> None:
        """CT < clip_hu_min → normalized to -1.0 (clamped)."""
        ct = np.array([-2000.0, -4096.0], dtype=np.float32)
        norm, _ = normalize_ct(ct, _CLIP_HU_MIN, _CLIP_HU_MAX)
        np.testing.assert_allclose(norm, [-1.0, -1.0], atol=1e-6)

    def test_out_of_range_roundtrip_is_lossy(self) -> None:
        """Voxels outside clip range do NOT round-trip to original HU (expected)."""
        ct_high = np.array([3000.0], dtype=np.float32)
        norm, params = normalize_ct(ct_high, _CLIP_HU_MIN, _CLIP_HU_MAX)
        ct_recovered = invert_ct_to_hu(norm, params)
        # Recovered value is clip_hu_max (2000), NOT 3000
        np.testing.assert_allclose(ct_recovered, [_CLIP_HU_MAX], atol=0.1)
        assert float(ct_recovered[0]) != float(ct_high[0])


# ============================================================================
# 3. MR round-trip — synthetic
# ============================================================================


class TestMRNormalize:
    """MR percentile-99 normalization and inversion correctness."""

    def test_output_range_in_zero_to_one(self) -> None:
        """Normalized MR must be in [0, 1]."""
        mr, mask = _make_mr_array(seed=1)
        norm, _ = normalize_mr_percentile99(mr, mask)
        assert norm.min() >= -1e-6, f"Norm below 0: {norm.min()}"
        assert norm.max() <=  1.0 + 1e-6, f"Norm above 1: {norm.max()}"

    def test_p1_maps_to_zero_p99_maps_to_one(self) -> None:
        """In-mask p1 → ~0.0; in-mask p99 → ~1.0."""
        mr, mask = _make_mr_array(seed=2)
        norm, params = normalize_mr_percentile99(mr, mask)
        mask_bool = mask.astype(bool)
        in_mask_norm = norm[mask_bool]
        # p1 and p99 of the normalized in-mask values should be near 0 and 1
        assert float(np.percentile(in_mask_norm, 1))  < 0.02
        assert float(np.percentile(in_mask_norm, 99)) > 0.98

    def test_roundtrip_in_mask_within_p1_p99(self) -> None:
        """invert(normalize(mr))[mask] ≈ mr[mask] for values in [p1, p99]."""
        mr, mask = _make_mr_array(seed=42)
        norm, params = normalize_mr_percentile99(mr, mask)
        mr_recovered = invert_mr(norm, params)

        p1, p99 = params["p1"], params["p99"]
        in_range = (mr >= p1) & (mr <= p99) & mask.astype(bool)
        np.testing.assert_allclose(
            mr_recovered[in_range], mr[in_range],
            rtol=1e-5,
            err_msg="MR round-trip fails for in-range in-mask voxels."
        )

    def test_params_contains_required_keys(self) -> None:
        """Params dict must have all keys needed for inversion."""
        mr, mask = _make_mr_array()
        _, params = normalize_mr_percentile99(mr, mask)
        required = {"method", "p1", "p99", "mr_range", "norm_min", "norm_max"}
        assert required.issubset(params.keys()), f"Missing keys: {required - params.keys()}"

    def test_empty_mask_does_not_crash(self) -> None:
        """All-zero mask should return zeros (no division by zero)."""
        mr = np.ones((8, 8, 8), dtype=np.float32) * 500.0
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        norm, _ = normalize_mr_percentile99(mr, mask)
        assert norm.shape == mr.shape
        assert np.isfinite(norm).all()

    def test_degenerate_constant_volume(self) -> None:
        """Constant in-mask MR (p1 == p99) returns zeros without crashing."""
        mr = np.ones((8, 8, 8), dtype=np.float32) * 100.0
        mask = np.ones((8, 8, 8), dtype=np.uint8)
        norm, _ = normalize_mr_percentile99(mr, mask)
        assert np.isfinite(norm).all()


# ============================================================================
# 4. Real-volume round-trip (R2 load-bearing test)
# ============================================================================


@pytest.mark.skipif(
    not _REAL_PATIENT_DIR.exists(),
    reason="Real data not available at data/synthrad2023/Task1/brain/1BA001",
)
class TestRealVolumeRoundtrip:
    """Verify the HU round-trip on an actual patient volume (R2)."""

    @pytest.fixture(scope="class")
    def patient_data(self) -> dict:
        """Load one real patient's CT, MR, and mask."""
        import SimpleITK as sitk

        def to_f32(p: Path) -> np.ndarray:
            return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)

        ct   = to_f32(_REAL_PATIENT_DIR / "ct.nii.gz")
        mr   = to_f32(_REAL_PATIENT_DIR / "mr.nii.gz")
        mask = to_f32(_REAL_PATIENT_DIR / "mask.nii.gz").astype(np.uint8)
        return {"ct": ct, "mr": mr, "mask": mask}

    def test_ct_hu_roundtrip_real_volume(self, patient_data: dict) -> None:
        """R2: invert(normalize(ct)) == ct (within 0.1 HU) for in-clip-range voxels.

        This is the exact guarantee the MAE-HU metric depends on.
        Tolerance 0.1 HU is negligible in any clinical sCT application.
        """
        ct = patient_data["ct"]
        norm, params = normalize_ct(ct, _CLIP_HU_MIN, _CLIP_HU_MAX)
        ct_recovered = invert_ct_to_hu(norm, params)

        in_range = (ct >= _CLIP_HU_MIN) & (ct <= _CLIP_HU_MAX)
        max_err = float(np.abs(ct_recovered[in_range] - ct[in_range]).max())

        assert max_err < 0.1, (
            f"CT HU round-trip on real volume exceeds 0.1 HU: {max_err:.4f} HU — R2 violation."
        )

    def test_ct_inmask_roundtrip_real_volume(self, patient_data: dict) -> None:
        """Specifically test in-mask voxels — these are what MAE-HU is computed on."""
        ct   = patient_data["ct"]
        mask = patient_data["mask"].astype(bool)

        norm, params = normalize_ct(ct, _CLIP_HU_MIN, _CLIP_HU_MAX)
        ct_recovered = invert_ct_to_hu(norm, params)

        in_clip = (ct >= _CLIP_HU_MIN) & (ct <= _CLIP_HU_MAX)
        test_mask = mask & in_clip

        np.testing.assert_allclose(
            ct_recovered[test_mask], ct[test_mask],
            atol=0.1,
            err_msg="In-mask CT HU round-trip exceeds 0.1 HU — R2 violation."
        )

    def test_mr_roundtrip_real_volume(self, patient_data: dict) -> None:
        """invert(normalize(mr))[in_p1p99] ≈ mr[in_p1p99] on a real MR volume."""
        mr   = patient_data["mr"]
        mask = patient_data["mask"]

        norm, params = normalize_mr_percentile99(mr, mask)
        mr_recovered = invert_mr(norm, params)

        p1, p99 = params["p1"], params["p99"]
        in_range = (mr >= p1) & (mr <= p99)
        np.testing.assert_allclose(
            mr_recovered[in_range], mr[in_range],
            rtol=1e-5,
            err_msg="Real-volume MR round-trip failed."
        )

    def test_normalized_ct_dtype_is_float32(self, patient_data: dict) -> None:
        """Ensure normalized arrays are float32 (training pipeline assumption)."""
        norm, _ = normalize_ct(patient_data["ct"], _CLIP_HU_MIN, _CLIP_HU_MAX)
        assert norm.dtype == np.float32, f"Expected float32, got {norm.dtype}"

    def test_normalized_mr_dtype_is_float32(self, patient_data: dict) -> None:
        norm, _ = normalize_mr_percentile99(patient_data["mr"], patient_data["mask"])
        assert norm.dtype == np.float32, f"Expected float32, got {norm.dtype}"


# ============================================================================
# 5. Manifest structure
# ============================================================================


_MANIFEST_PATH = _REPO_ROOT / "outputs" / "preprocessed" / "manifest.json"
_REQUIRED_MANIFEST_KEYS = {
    "patient_id", "anatomy", "center",
    "patient_dir", "out_dir", "mr_path", "ct_path", "mask_path",
    "orientation",
    "original_spacing_mm", "original_size_xyz",
    "resampled_spacing_mm", "resampled_size_xyz", "resampled_shape_zyx",
    "ct_norm_params", "mr_norm_params",
    "processed_at",
}
_REQUIRED_CT_PARAMS = {"method", "clip_hu_min", "clip_hu_max", "hu_range", "norm_min", "norm_max"}
_REQUIRED_MR_PARAMS = {"method", "p1", "p99", "mr_range", "norm_min", "norm_max"}


@pytest.mark.skipif(
    not _MANIFEST_PATH.exists(),
    reason="manifest.json not yet generated — run preprocess.py first.",
)
class TestManifest:
    """Validate the generated manifest.json has all inverse-norm params (R2)."""

    @pytest.fixture(scope="class")
    def manifest(self) -> dict:
        with open(_MANIFEST_PATH) as f:
            return json.load(f)

    def test_manifest_has_360_entries(self, manifest: dict) -> None:
        assert len(manifest) == 360, (
            f"Expected 360 manifest entries, got {len(manifest)}. "
            "Some patients may have failed preprocessing."
        )

    def test_all_entries_have_required_top_level_keys(self, manifest: dict) -> None:
        missing: list[str] = []
        for pid, entry in manifest.items():
            diff = _REQUIRED_MANIFEST_KEYS - entry.keys()
            if diff:
                missing.append(f"{pid}: {diff}")
        assert not missing, f"Entries missing required keys:\n" + "\n".join(missing[:10])

    def test_ct_norm_params_present_in_all_entries(self, manifest: dict) -> None:
        """Every entry must have full CT norm params for MAE-HU inversion (R2)."""
        missing: list[str] = []
        for pid, entry in manifest.items():
            ct_p = entry.get("ct_norm_params", {})
            diff = _REQUIRED_CT_PARAMS - ct_p.keys()
            if diff:
                missing.append(f"{pid}: {diff}")
        assert not missing, "CT norm params incomplete:\n" + "\n".join(missing[:10])

    def test_mr_norm_params_present_in_all_entries(self, manifest: dict) -> None:
        missing: list[str] = []
        for pid, entry in manifest.items():
            mr_p = entry.get("mr_norm_params", {})
            diff = _REQUIRED_MR_PARAMS - mr_p.keys()
            if diff:
                missing.append(f"{pid}: {diff}")
        assert not missing, "MR norm params incomplete:\n" + "\n".join(missing[:10])

    def test_ct_clip_values_consistent_with_config(self, manifest: dict) -> None:
        """All CT norm params must match the pinned config values (R1)."""
        for pid, entry in manifest.items():
            p = entry["ct_norm_params"]
            assert p["clip_hu_min"] == _CLIP_HU_MIN, (
                f"{pid}: clip_hu_min={p['clip_hu_min']}, expected {_CLIP_HU_MIN}"
            )
            assert p["clip_hu_max"] == _CLIP_HU_MAX, (
                f"{pid}: clip_hu_max={p['clip_hu_max']}, expected {_CLIP_HU_MAX}"
            )

    def test_npz_files_exist_for_all_entries(self, manifest: dict) -> None:
        """All three .npz files must exist on disk."""
        missing_files: list[str] = []
        for pid, entry in manifest.items():
            for key in ("mr_path", "ct_path", "mask_path"):
                p = Path(entry.get(key, ""))
                if not p.exists():
                    missing_files.append(f"{pid}: {key} → {p}")
        assert not missing_files, (
            f"{len(missing_files)} missing files:\n" + "\n".join(missing_files[:10])
        )

    def test_anatomy_counts(self, manifest: dict) -> None:
        """180 brain + 180 pelvis (matches SynthRAD2023 Task1 counts)."""
        from collections import Counter
        counts = Counter(e["anatomy"] for e in manifest.values())
        assert counts["brain"]  == 180, f"Expected 180 brain, got {counts['brain']}"
        assert counts["pelvis"] == 180, f"Expected 180 pelvis, got {counts['pelvis']}"
