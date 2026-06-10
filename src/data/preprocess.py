"""Preprocessing pipeline for SynthRAD2023 Task1 — ThermoBridge Phase 0.

Reorient → Resample → Normalize → Save (.npz) → manifest.json

CLI::
    python src/data/preprocess.py --config configs/default.yaml [--force] [--data-root ...] [--out-dir ...]

Outputs (R5 — never writes under data/):
    outputs/preprocessed/<anat>/<pid>/mr.npz
    outputs/preprocessed/<anat>/<pid>/ct.npz
    outputs/preprocessed/<anat>/<pid>/mask.npz
    outputs/preprocessed/manifest.json

Rules observed:
    R1 — every normalisation constant is derived from config (no hidden terms).
    R2 — CT inverse-normalisation is load-bearing; stored exactly in manifest.
    R5 — all writes go to outputs/, data/ is read-only.
    R6 — idempotent (skips completed patients unless --force).
    R7 — typed, docstrings, round-trip tested in tests/test_preprocess.py.
    R8 — no constants invented; all derived from config or the patient volume.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import SimpleITK as sitk

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Public path so tests can import without installing the package
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.utils.config import load_config  # noqa: E402

# ============================================================================
# CT normalisation / inversion  (R2 — load-bearing for MAE-HU metric)
# ============================================================================


def normalize_ct(
    arr: np.ndarray,
    clip_hu_min: float,
    clip_hu_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip to [clip_hu_min, clip_hu_max] then min-max scale to [-1, 1].

    Args:
        arr:         Raw CT array in Hounsfield Units (HU), float32.
        clip_hu_min: Lower HU clip bound (from config; e.g. -1024).
        clip_hu_max: Upper HU clip bound (from config; e.g. 2000).

    Returns:
        normalized: float32 array in [-1, 1].
        params:     Dict containing every value needed to invert exactly (R2).

    Notes:
        Forward: norm = (clip(x) - clip_hu_min) / (clip_hu_max - clip_hu_min) * 2 - 1
        Values outside [clip_hu_min, clip_hu_max] are clipped before scaling —
        the round-trip is exact only for voxels within the clip range.
    """
    hu_range = float(clip_hu_max) - float(clip_hu_min)
    clipped = np.clip(arr, clip_hu_min, clip_hu_max)
    normalized = ((clipped - clip_hu_min) / hu_range * 2.0 - 1.0).astype(np.float32)
    params: dict[str, Any] = {
        "method": "min_max",
        "clip_hu_min": float(clip_hu_min),
        "clip_hu_max": float(clip_hu_max),
        "hu_range": hu_range,
        "norm_min": -1.0,
        "norm_max": 1.0,
    }
    return normalized, params


def invert_ct_to_hu(arr: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Invert a normalized CT array back to Hounsfield Units.

    This is the load-bearing inverse used by the MAE-HU metric (R2).
    Inverse: hu = (norm + 1) / 2 * hu_range + clip_hu_min

    Args:
        arr:    Normalized CT array (values in [-1, 1]).
        params: The params dict returned by :func:`normalize_ct`.

    Returns:
        float32 array in approximately [clip_hu_min, clip_hu_max] HU.
    """
    clip_hu_min: float = params["clip_hu_min"]
    hu_range: float = params["hu_range"]
    hu = ((arr.astype(np.float64) + 1.0) / 2.0 * hu_range + clip_hu_min).astype(
        np.float32
    )
    return hu


# ============================================================================
# MR normalisation / inversion
# ============================================================================


def normalize_mr_percentile99(
    arr: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-volume percentile normalization (p1–p99 of in-mask voxels) to [0, 1].

    Using in-mask percentiles avoids background bias, is robust to outliers,
    and scales across the 25× cross-patient MR intensity spread observed in EDA.

    Args:
        arr:  Raw MR volume, any non-negative float scale.
        mask: Binary patient mask (same shape as arr), used to compute percentiles.

    Returns:
        normalized: float32 array clipped+scaled to [0, 1].
        params:     Dict with p1, p99 for exact inversion.

    Notes:
        Forward: norm = clip(x, p1, p99) - p1) / (p99 - p1)
        Values outside [p1, p99] are clipped; round-trip is exact only for
        voxels whose raw value lies within [p1, p99].
    """
    mask_bool = mask.astype(bool)
    in_mask = arr[mask_bool]
    if in_mask.size == 0:
        p1, p99 = 0.0, 1.0
    else:
        p1 = float(np.percentile(in_mask, 1))
        p99 = float(np.percentile(in_mask, 99))

    mr_range = p99 - p1
    if mr_range < 1e-8:
        # Degenerate volume — return zeros rather than divide-by-zero
        normalized = np.zeros_like(arr, dtype=np.float32)
    else:
        clipped = np.clip(arr, p1, p99)
        normalized = ((clipped - p1) / mr_range).astype(np.float32)

    params: dict[str, Any] = {
        "method": "percentile_99",
        "p1": p1,
        "p99": p99,
        "mr_range": mr_range,
        "norm_min": 0.0,
        "norm_max": 1.0,
    }
    return normalized, params


def invert_mr(arr: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Invert a normalized MR array back to the original intensity scale.

    Inverse: x_orig ≈ norm * (p99 - p1) + p1

    Args:
        arr:    Normalized MR array (values in [0, 1]).
        params: The params dict returned by :func:`normalize_mr_percentile99`.

    Returns:
        float32 array in approximately [p1, p99] of the original scale.
    """
    p1: float = params["p1"]
    mr_range: float = params["mr_range"]
    return (arr.astype(np.float64) * mr_range + p1).astype(np.float32)


# ============================================================================
# SimpleITK helpers
# ============================================================================

_ORIENTATION = "RAS"


def reorient_to_ras(img: sitk.Image) -> sitk.Image:
    """Reorient a SimpleITK image to RAS canonical orientation."""
    return sitk.DICOMOrient(img, _ORIENTATION)


def resample_sitk(
    img: sitk.Image,
    new_spacing_xyz: list[float],
    interpolator: int = sitk.sitkLinear,
    default_pixel_value: float = 0.0,
) -> sitk.Image:
    """Resample ``img`` to ``new_spacing_xyz`` (x, y, z order in mm).

    Preserves the physical extent of the volume by computing the new size
    from the original size × spacing.

    Args:
        img:               Input SimpleITK image.
        new_spacing_xyz:   Target spacing in mm, (x, y, z) order.
        interpolator:      SimpleITK interpolator constant.
        default_pixel_value: Fill value for areas outside original FOV.

    Returns:
        Resampled SimpleITK image.
    """
    orig_spacing = img.GetSpacing()  # (x, y, z)
    orig_size = img.GetSize()        # (x, y, z)
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / new_spacing_xyz[i]))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing_xyz)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(img)


def sitk_to_f32(img: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image to float32 numpy (Z, Y, X) array."""
    return sitk.GetArrayFromImage(img).astype(np.float32)


# ============================================================================
# Manifest helpers
# ============================================================================

_MANIFEST_FILENAME = "manifest.json"


def load_manifest(path: Path) -> dict[str, Any]:
    """Load existing manifest or return empty dict if file not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Atomically write manifest to disk (indent=2 for readability)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(path)


# ============================================================================
# Per-patient processing
# ============================================================================


def process_patient(
    patient_dir: Path,
    anatomy: str,
    preprocessed_root: Path,
    cfg_preproc: Any,
    force: bool = False,
) -> dict[str, Any] | None:
    """Reorient, resample, normalize, and save one patient.

    Args:
        patient_dir:       Path to raw patient folder (contains mr/ct/mask.nii.gz).
        anatomy:           'brain' or 'pelvis'.
        preprocessed_root: Root of outputs/preprocessed/.
        cfg_preproc:       OmegaConf node: config.preprocessing.
        force:             If True, reprocess even if outputs exist.

    Returns:
        Manifest entry dict on success, None on skip (already done, not forced).
        Raises on errors (caller wraps in try/except).
    """
    pid = patient_dir.name
    out_dir = preprocessed_root / anatomy / pid

    mr_out  = out_dir / "mr.npz"
    ct_out  = out_dir / "ct.npz"
    msk_out = out_dir / "mask.npz"

    if not force and mr_out.exists() and ct_out.exists() and msk_out.exists():
        return None  # already done

    # ── 1. Read ─────────────────────────────────────────────────────────────
    mr_img  = sitk.ReadImage(str(patient_dir / "mr.nii.gz"))
    ct_img  = sitk.ReadImage(str(patient_dir / "ct.nii.gz"))
    msk_img = sitk.ReadImage(str(patient_dir / "mask.nii.gz"))

    orig_spacing = list(mr_img.GetSpacing())   # (x, y, z)
    orig_size    = list(mr_img.GetSize())       # (x, y, z)

    # ── 2. Reorient to RAS ─────────────────────────────────────────────────
    mr_img  = reorient_to_ras(mr_img)
    ct_img  = reorient_to_ras(ct_img)
    msk_img = reorient_to_ras(msk_img)

    # ── 3. Resample ────────────────────────────────────────────────────────
    new_spacing: list[float] = list(cfg_preproc.target_spacing_mm)
    mr_img  = resample_sitk(mr_img,  new_spacing, sitk.sitkLinear,  default_pixel_value=0.0)
    ct_img  = resample_sitk(ct_img,  new_spacing, sitk.sitkLinear,  default_pixel_value=float(cfg_preproc.ct.clip_hu_min))
    msk_img = resample_sitk(msk_img, new_spacing, sitk.sitkNearestNeighbor, default_pixel_value=0.0)

    new_size = list(mr_img.GetSize())

    # ── 4. To numpy ─────────────────────────────────────────────────────────
    mr  = sitk_to_f32(mr_img)           # (Z, Y, X)
    ct  = sitk_to_f32(ct_img)
    msk = sitk_to_f32(msk_img)
    msk = (msk > 0.5).astype(np.uint8)  # binarise after nearest-neighbour resample

    # ── 5. Normalise CT (global clip+scale, invertible) ─────────────────────
    ct_norm, ct_params = normalize_ct(
        ct,
        clip_hu_min=float(cfg_preproc.ct.clip_hu_min),
        clip_hu_max=float(cfg_preproc.ct.clip_hu_max),
    )

    # ── 6. Normalise MR (per-volume, invertible) ─────────────────────────────
    mr_norm, mr_params = normalize_mr_percentile99(mr, msk)

    # ── 7. Save ──────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(mr_out),  data=mr_norm)
    np.savez_compressed(str(ct_out),  data=ct_norm)
    np.savez_compressed(str(msk_out), data=msk)

    # ── 8. Build manifest entry ──────────────────────────────────────────────
    center = pid[2] if len(pid) >= 3 else "?"
    entry: dict[str, Any] = {
        "patient_id":         pid,
        "anatomy":            anatomy,
        "center":             center,
        "patient_dir":        str(patient_dir),
        "out_dir":            str(out_dir),
        "mr_path":            str(mr_out),
        "ct_path":            str(ct_out),
        "mask_path":          str(msk_out),
        "orientation":        _ORIENTATION,
        "original_spacing_mm":  orig_spacing,
        "original_size_xyz":    orig_size,
        "resampled_spacing_mm": new_spacing,
        "resampled_size_xyz":   new_size,
        "resampled_shape_zyx":  list(mr_norm.shape),
        "ct_norm_params":     ct_params,
        "mr_norm_params":     mr_params,
        "processed_at":       datetime.datetime.utcnow().isoformat() + "Z",
    }
    return entry


# ============================================================================
# Visualisation — one brain + one pelvis preprocessed pair
# ============================================================================


def render_preprocessed_sample(
    manifest: dict[str, Any],
    out_path: Path,
    seed: int = 42,
    n_per_anatomy: int = 2,
) -> None:
    """Render mid-axial slices of preprocessed MR + CT for a mixed sample.

    Columns: [MR (norm, [0,1]), CT (norm, [-1,1] → displayed), mask contour on MR]
    Saves at 300 DPI (spec: ≥300 DPI for all figures).
    """
    rng = np.random.default_rng(seed)

    brain_entries  = [e for e in manifest.values() if e["anatomy"] == "brain"]
    pelvis_entries = [e for e in manifest.values() if e["anatomy"] == "pelvis"]

    sample: list[dict[str, Any]] = []
    for pool in (brain_entries, pelvis_entries):
        if pool:
            chosen = rng.choice(pool, size=min(n_per_anatomy, len(pool)), replace=False)
            sample.extend(chosen.tolist())

    n = len(sample)
    if n == 0:
        print("  No entries in manifest — skipping visualisation.")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), dpi=300)
    if n == 1:
        axes = axes[np.newaxis, :]

    for j, title in enumerate(["MR (normalised [0,1])", "CT (normalised [−1,1])", "Mask contour over MR"]):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold", pad=5)

    for i, entry in enumerate(sample):
        try:
            mr_arr  = np.load(entry["mr_path"])["data"]
            ct_arr  = np.load(entry["ct_path"])["data"]
            msk_arr = np.load(entry["mask_path"])["data"]
        except Exception as exc:
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f"Load error:\n{exc}",
                                ha="center", va="center",
                                transform=axes[i, j].transAxes, fontsize=7)
            continue

        mid_z = mr_arr.shape[0] // 2
        mr_sl  = mr_arr[mid_z]
        ct_sl  = ct_arr[mid_z]
        msk_sl = msk_arr[mid_z]

        label = f"{entry['patient_id']}\n({entry['anatomy']}, ctr {entry['center']})\nshape ZYX: {tuple(mr_arr.shape)}"

        axes[i, 0].imshow(mr_sl, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(label, fontsize=7)
        axes[i, 1].imshow(ct_sl, cmap="gray", origin="lower", vmin=-1, vmax=1)
        axes[i, 2].imshow(mr_sl, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[i, 2].contour(msk_sl, levels=[0.5], colors=["red"], linewidths=0.8)

        for j in range(3):
            axes[i, j].axis("off")

    fig.suptitle(
        f"Preprocessed sample — resampled to {sample[0].get('resampled_spacing_mm')} mm, "
        f"RAS orientation (seed={seed})",
        fontsize=11, fontweight="bold", y=1.002,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ThermoBridge Phase 0 — preprocessing pipeline (R1/R2/R5/R6/R7)."
    )
    p.add_argument("--config", required=True, type=Path,
                   help="Path to YAML config (e.g. configs/default.yaml).")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Override config data.root_dir.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override config output.root_dir.")
    p.add_argument("--force", action="store_true",
                   help="Re-process patients even if outputs already exist.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the visualisation sample (R6).")
    return p.parse_args()


def collect_patients(data_root: Path) -> list[tuple[Path, str]]:
    """Return sorted list of (patient_dir, anatomy) pairs."""
    patients: list[tuple[Path, str]] = []
    for anatomy in ("brain", "pelvis"):
        anat_dir = data_root / anatomy
        if not anat_dir.exists():
            print(f"  WARNING: {anat_dir} not found — skipping.", file=sys.stderr)
            continue
        for entry in sorted(anat_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("1") and len(entry.name) >= 4:
                patients.append((entry, anatomy))
    return patients


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    # Allow CLI overrides of paths (keep everything else from config)
    data_root = (args.data_root or Path(cfg.data.root_dir)).resolve()
    out_root  = (args.out_dir   or Path(cfg.output.root_dir)).resolve()

    preprocessed_root = out_root / "preprocessed"
    manifest_path     = preprocessed_root / _MANIFEST_FILENAME

    # Validate preconditions (R8 — stop if values still null)
    cfg_pp = cfg.preprocessing
    for attr, label in [
        (cfg_pp.target_spacing_mm, "preprocessing.target_spacing_mm"),
        (cfg_pp.ct.clip_hu_min,    "preprocessing.ct.clip_hu_min"),
        (cfg_pp.ct.clip_hu_max,    "preprocessing.ct.clip_hu_max"),
        (cfg_pp.mr.normalization,  "preprocessing.mr.normalization"),
    ]:
        if attr is None:
            sys.exit(
                f"STOP: '{label}' is still null in config. "
                "Run EDA (inventory.py) first and pin the value before preprocessing."
            )

    if not data_root.exists():
        sys.exit(f"ERROR: data root '{data_root}' does not exist.")

    patients = collect_patients(data_root)
    print(f"\nFound {len(patients)} patients. Loading manifest…")

    manifest = load_manifest(manifest_path)
    n_already = sum(
        1 for pid, e in manifest.items()
        if Path(e.get("mr_path", "")).exists()
        and Path(e.get("ct_path", "")).exists()
        and Path(e.get("mask_path", "")).exists()
    )
    print(f"Manifest has {len(manifest)} entries ({n_already} with files on disk).")
    if not args.force:
        print("  (use --force to reprocess existing patients)")

    n_processed = n_skipped = n_failed = 0

    for idx, (pdir, anatomy) in enumerate(patients, 1):
        pid = pdir.name

        # Idempotency: skip if manifest entry is complete and files exist
        if not args.force and pid in manifest:
            e = manifest[pid]
            if (
                Path(e.get("mr_path", "")).exists()
                and Path(e.get("ct_path", "")).exists()
                and Path(e.get("mask_path", "")).exists()
            ):
                n_skipped += 1
                if idx % 50 == 0 or idx == len(patients):
                    print(f"  [{idx:4d}/{len(patients)}] (skipped {n_skipped} so far)")
                continue

        try:
            entry = process_patient(
                pdir, anatomy, preprocessed_root, cfg_pp, force=args.force
            )
            if entry is None:
                # process_patient returned None → already done, not forced
                n_skipped += 1
            else:
                manifest[pid] = entry
                n_processed += 1
                save_manifest(manifest, manifest_path)  # crash-safe: write after each
        except Exception:
            err = traceback.format_exc().splitlines()[-1]
            print(f"  [{idx:4d}/{len(patients)}] ✗ {pid} — {err}", file=sys.stderr)
            n_failed += 1
            continue

        if idx % 50 == 0 or idx == len(patients):
            print(f"  [{idx:4d}/{len(patients)}]  processed={n_processed}  skipped={n_skipped}  failed={n_failed}")

    # Final manifest write
    save_manifest(manifest, manifest_path)

    print(f"\nManifest written → {manifest_path}")
    print(f"  Entries: {len(manifest)}  |  processed: {n_processed}  "
          f"|  skipped: {n_skipped}  |  failed: {n_failed}")

    if n_failed > 0:
        print(f"\nWARNING: {n_failed} patients failed. Check stderr above.")

    # Visualise a preprocessed sample
    fig_path = out_root / "figures" / "preprocessed_sample.png"
    print("\nRendering preprocessed sample…")
    render_preprocessed_sample(manifest, fig_path, seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
