"""Preprocessing pipeline for SynthRAD2025 Task1 (MRI-CT) and Task2 (CBCT-CT)
— ThermoBridge Phase 1 (ADR-012, Chunk N1).

Reorient -> Resample -> Normalize -> Save (.npz) -> manifest_2025.json

Reuses normalize_ct(), normalize_mr_percentile99(), reorient_to_ras(),
resample_sitk(), sitk_to_f32(), load_manifest(), save_manifest() from
src/data/preprocess.py verbatim — no duplicated normalization logic. CBCT is
normalized with the *same* normalize_ct() clip+affine transform as CT (ADR-013:
all modalities must share the [-1,1] symmetric range for bridge symmetry),
just with CBCT-specific clip bounds. The stored ct_norm_params/src_norm_params
are invert_ct_to_hu()/invert_mr()-compatible (same schema preprocess.py
already produces), so downstream eval code reuses those inverse functions
unchanged — this script only ever normalizes forward, never inverts.

Directory layout (source, on the cloud GPU):
    <task1-root>/{HN,TH,AB}/<patientID>/{ct.mha, mr.mha, mask.mha}
    <task2-root>/{HN,TH,AB}/<patientID>/{ct.mha, cbct.mha, mask.mha}
Patient ID format: 1HNA001 (Task1, HN, Center A, #001); 2THB023 (Task2, TH, Center B, #023).

Outputs (R5 — never writes under data/):
    outputs/preprocessed_2025/task1/<anat>/<pid>/{mr.npz, ct.npz, mask.npz}
    outputs/preprocessed_2025/task2/<anat>/<pid>/{cbct.npz, ct.npz, mask.npz}
    outputs/preprocessed_2025/manifest_2025.json

manifest_2025.json entry schema (one dict per patient_id):
    patient_id, anatomy ("HN"/"TH"/"AB"), task ("task1"/"task2"), center,
    modality_src ("mr"/"cbct"), src_path, ct_path, mask_path,
    src_norm_params, ct_norm_params, orientation,
    original_spacing_mm, original_size_xyz,
    resampled_spacing_mm, resampled_size_xyz, resampled_shape_zyx,
    processed_at, out_dir, patient_dir

CLI::
    python scripts/run_preprocess_2025.py --config configs/default.yaml \\
        --task1-root data/synthrad2025/Task1_data \\
        --task2-root data/synthrad2025/Task2_data \\
        --out-dir outputs [--force]

DO NOT run on the local dev machine — SynthRAD2025 data lives on the cloud
GPU only (ADR-008). This script is written to be ready to run there.

Rules observed:
    R1 — every normalisation constant is derived from config (no hidden terms).
    R2 — CT/CBCT inverse-normalisation params stored exactly in the manifest.
    R5 — all writes go to outputs/, data/ is read-only.
    R6 — idempotent (skips completed patients unless --force).
    R8 — CT/CBCT clip ranges not yet pinned by EDA fall back to documented
         placeholders, with an explicit WARNING printed (never silently
         treated as final — see get_ct_clip_range / get_cbct_clip_range).
"""

from __future__ import annotations

import argparse
import datetime
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.preprocess import (  # noqa: E402
    load_manifest,
    normalize_ct,
    normalize_mr_percentile99,
    reorient_to_ras,
    resample_sitk,
    save_manifest,
    sitk_to_f32,
)
from src.utils.config import load_config  # noqa: E402

_ANATOMIES = ("HN", "TH", "AB")

# Fallback clip ranges (§2, Rule 8) — used ONLY when configs/default.yaml
# still has null for the corresponding anatomy/modality. These are NOT
# EDA-pinned values; every use prints a WARNING so the manifest/logs make
# clear which patients were processed with a provisional range.
_CT_CLIP_FALLBACK = (-1024.0, 2000.0)     # same as SynthRAD2023 (ADR-006)
_CBCT_CLIP_FALLBACK = (-1000.0, 3071.0)   # wider — CBCT scatter artifacts


# ============================================================================
# Clip-range resolution (config-first, fallback + warning otherwise; Rule 8)
# ============================================================================


def get_ct_clip_range(cfg_synthrad2025: Any, anatomy: str) -> tuple[float, float]:
    """CT clip [c_lo, c_hi] for one SynthRAD2025 anatomy (HN/TH/AB may differ
    from SynthRAD2023's brain/pelvis — TODO: pin from EDA, Chunk N1)."""
    key = f"{anatomy.lower()}_ct"
    node = getattr(cfg_synthrad2025, key, None)
    if node is None or node.clip_hu_min is None or node.clip_hu_max is None:
        lo, hi = _CT_CLIP_FALLBACK
        print(
            f"  WARNING: data.synthrad2025.{key}.clip_hu_{{min,max}} is null — "
            f"using UNPINNED fallback [{lo}, {hi}] HU. TODO: pin from EDA (Chunk N1).",
            file=sys.stderr,
        )
        return lo, hi
    return float(node.clip_hu_min), float(node.clip_hu_max)


def get_cbct_clip_range(cfg_synthrad2025: Any) -> tuple[float, float]:
    """CBCT clip [b_lo, b_hi] — distinct from CT due to scatter artifacts (§2).
    TODO: pin from EDA (Chunk N1); falls back to a documented placeholder."""
    node = cfg_synthrad2025.cbct
    if node.clip_hu_min is None or node.clip_hu_max is None:
        lo, hi = _CBCT_CLIP_FALLBACK
        print(
            f"  WARNING: data.synthrad2025.cbct.clip_hu_{{min,max}} is null — "
            f"using UNPINNED fallback [{lo}, {hi}] HU. TODO: pin from EDA (Chunk N1).",
            file=sys.stderr,
        )
        return lo, hi
    return float(node.clip_hu_min), float(node.clip_hu_max)


# ============================================================================
# Patient ID parsing (format: <task_digit><anatomy_code><center><number>)
# ============================================================================


def parse_patient_id_2025(pid: str) -> dict[str, str]:
    """Parse '1HNA001' -> task_digit='1', anatomy_code='HN', center='A', number='001'."""
    return {
        "task_digit": pid[0] if len(pid) >= 1 else "?",
        "anatomy_code": pid[1:3] if len(pid) >= 3 else "??",
        "center": pid[3] if len(pid) >= 4 else "?",
        "number": pid[4:] if len(pid) >= 5 else "?",
    }


def collect_patients_2025(task_root: Path) -> list[tuple[Path, str]]:
    """Return sorted list of (patient_dir, anatomy) pairs under task_root/{HN,TH,AB}/."""
    patients: list[tuple[Path, str]] = []
    for anatomy in _ANATOMIES:
        anat_dir = task_root / anatomy
        if not anat_dir.exists():
            print(f"  WARNING: {anat_dir} not found — skipping.", file=sys.stderr)
            continue
        for entry in sorted(anat_dir.iterdir()):
            if entry.is_dir():
                patients.append((entry, anatomy))
    return patients


# ============================================================================
# Per-patient processing
# ============================================================================


def _process_common(
    patient_dir: Path,
    src_filename: str,
    modality_src: str,
    anatomy: str,
    task: str,
    out_dir: Path,
    target_spacing_mm: list[float],
    src_clip: tuple[float, float] | None,
    ct_clip: tuple[float, float],
) -> dict[str, Any]:
    """Shared reorient/resample/normalize/save logic for one Task1 or Task2 patient.

    Args:
        src_filename:  'mr.mha' or 'cbct.mha'.
        modality_src:  'mr' or 'cbct'.
        src_clip:      (lo, hi) HU clip for the source modality if it is
                        normalized like CT (CBCT); None if it uses the
                        MR per-volume percentile method instead.
    """
    pid = patient_dir.name

    ct_img  = sitk.ReadImage(str(patient_dir / "ct.mha"))
    src_img = sitk.ReadImage(str(patient_dir / src_filename))
    msk_img = sitk.ReadImage(str(patient_dir / "mask.mha"))

    orig_spacing = list(ct_img.GetSpacing())  # (x, y, z)
    orig_size    = list(ct_img.GetSize())

    # Reorient to RAS
    ct_img  = reorient_to_ras(ct_img)
    src_img = reorient_to_ras(src_img)
    msk_img = reorient_to_ras(msk_img)

    # Resample — CT/CBCT fill outside FOV with their own clip minimum
    # (matches SynthRAD2023 convention: avoid injecting artificial edges).
    ct_img  = resample_sitk(ct_img,  target_spacing_mm, sitk.sitkLinear, default_pixel_value=ct_clip[0])
    src_fill = src_clip[0] if src_clip is not None else 0.0
    src_img = resample_sitk(src_img, target_spacing_mm, sitk.sitkLinear, default_pixel_value=src_fill)
    msk_img = resample_sitk(msk_img, target_spacing_mm, sitk.sitkNearestNeighbor, default_pixel_value=0.0)

    new_size = list(ct_img.GetSize())

    ct  = sitk_to_f32(ct_img)
    src = sitk_to_f32(src_img)
    msk = sitk_to_f32(msk_img)
    msk = (msk > 0.5).astype(msk.dtype)

    # Normalize CT (always clip+affine to [-1,1])
    ct_norm, ct_params = normalize_ct(ct, clip_hu_min=ct_clip[0], clip_hu_max=ct_clip[1])

    # Normalize source modality
    if src_clip is not None:
        # CBCT: same clip+affine transform as CT (ADR-013 symmetric-range requirement).
        src_norm, src_params = normalize_ct(src, clip_hu_min=src_clip[0], clip_hu_max=src_clip[1])
    else:
        # MR: per-volume in-mask percentile normalization (§2).
        src_norm, src_params = normalize_mr_percentile99(src, msk)

    out_dir.mkdir(parents=True, exist_ok=True)
    src_out = out_dir / f"{modality_src}.npz"
    ct_out  = out_dir / "ct.npz"
    msk_out = out_dir / "mask.npz"
    np.savez_compressed(str(src_out), data=src_norm)
    np.savez_compressed(str(ct_out),  data=ct_norm)
    np.savez_compressed(str(msk_out), data=msk.astype("uint8"))

    parsed = parse_patient_id_2025(pid)
    entry: dict[str, Any] = {
        "patient_id":            pid,
        "anatomy":               anatomy,
        "task":                  task,
        "center":                parsed["center"],
        "modality_src":          modality_src,
        "patient_dir":           str(patient_dir),
        "out_dir":               str(out_dir),
        "src_path":              str(src_out),
        "ct_path":               str(ct_out),
        "mask_path":             str(msk_out),
        "orientation":           "RAS",
        "original_spacing_mm":   orig_spacing,
        "original_size_xyz":     orig_size,
        "resampled_spacing_mm":  target_spacing_mm,
        "resampled_size_xyz":    new_size,
        "resampled_shape_zyx":   list(ct_norm.shape),
        "src_norm_params":       src_params,
        "ct_norm_params":        ct_params,
        "processed_at":          datetime.datetime.utcnow().isoformat() + "Z",
    }
    return entry


def process_patient_task1(
    patient_dir: Path,
    anatomy: str,
    preprocessed_root: Path,
    target_spacing_mm: list[float],
    ct_clip: tuple[float, float],
) -> dict[str, Any] | None:
    """Task1 (MRI-CT): normalize MR (per-volume p1/p99) and CT (clip per anatomy)."""
    out_dir = preprocessed_root / "task1" / anatomy / patient_dir.name
    if (out_dir / "mr.npz").exists() and (out_dir / "ct.npz").exists() and (out_dir / "mask.npz").exists():
        return None
    return _process_common(
        patient_dir, "mr.mha", "mr", anatomy, "task1", out_dir,
        target_spacing_mm, src_clip=None, ct_clip=ct_clip,
    )


def process_patient_task2(
    patient_dir: Path,
    anatomy: str,
    preprocessed_root: Path,
    target_spacing_mm: list[float],
    ct_clip: tuple[float, float],
    cbct_clip: tuple[float, float],
) -> dict[str, Any] | None:
    """Task2 (CBCT-CT): normalize CBCT (own clip range) and CT (same as Task1)."""
    out_dir = preprocessed_root / "task2" / anatomy / patient_dir.name
    if (out_dir / "cbct.npz").exists() and (out_dir / "ct.npz").exists() and (out_dir / "mask.npz").exists():
        return None
    return _process_common(
        patient_dir, "cbct.mha", "cbct", anatomy, "task2", out_dir,
        target_spacing_mm, src_clip=cbct_clip, ct_clip=ct_clip,
    )


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ThermoBridge Phase 1 — SynthRAD2025 preprocessing (R1/R2/R5/R6/R8)."
    )
    p.add_argument("--config", required=True, type=Path,
                   help="Path to YAML config (e.g. configs/default.yaml).")
    p.add_argument("--task1-root", type=Path, default=None,
                   help="Override config data.synthrad2025.task1_dir.")
    p.add_argument("--task2-root", type=Path, default=None,
                   help="Override config data.synthrad2025.task2_dir.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override config output.root_dir.")
    p.add_argument("--force", action="store_true",
                   help="Re-process patients even if outputs already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg_sr2025 = cfg.data.synthrad2025

    task1_root = (args.task1_root or Path(cfg_sr2025.task1_dir)).resolve()
    task2_root = (args.task2_root or Path(cfg_sr2025.task2_dir)).resolve()
    out_root   = (args.out_dir or Path(cfg.output.root_dir)).resolve()

    target_spacing_mm = list(cfg.preprocessing.target_spacing_mm)
    if any(v is None for v in target_spacing_mm):
        sys.exit("STOP: preprocessing.target_spacing_mm is still null in config.")

    preprocessed_root = out_root / "preprocessed_2025"
    manifest_path      = preprocessed_root / "manifest_2025.json"

    # Resolve clip ranges once up front (prints WARNINGs for any unpinned values).
    ct_clip_by_anat = {anat: get_ct_clip_range(cfg_sr2025, anat) for anat in _ANATOMIES}
    cbct_clip = get_cbct_clip_range(cfg_sr2025)

    task1_ok = task1_root.exists()
    task2_ok = task2_root.exists()
    if not task1_ok:
        print(f"  WARNING: task1 root '{task1_root}' does not exist — skipping Task1.", file=sys.stderr)
    if not task2_ok:
        print(f"  WARNING: task2 root '{task2_root}' does not exist — skipping Task2.", file=sys.stderr)
    if not task1_ok and not task2_ok:
        sys.exit("ERROR: neither task1-root nor task2-root exists. Nothing to do.")

    jobs: list[tuple[str, Path, str]] = []  # (task, patient_dir, anatomy)
    if task1_ok:
        jobs.extend(("task1", pdir, anat) for pdir, anat in collect_patients_2025(task1_root))
    if task2_ok:
        jobs.extend(("task2", pdir, anat) for pdir, anat in collect_patients_2025(task2_root))

    print(f"\nFound {len(jobs)} patients across Task1+Task2. Loading manifest…")
    manifest = load_manifest(manifest_path)
    print(f"Manifest has {len(manifest)} entries.")
    if not args.force:
        print("  (use --force to reprocess existing patients)")

    n_processed = n_skipped = n_failed = 0
    total = len(jobs)

    for idx, (task, pdir, anatomy) in enumerate(jobs, 1):
        pid = pdir.name

        if not args.force and pid in manifest:
            e = manifest[pid]
            if (
                Path(e.get("src_path", "")).exists()
                and Path(e.get("ct_path", "")).exists()
                and Path(e.get("mask_path", "")).exists()
            ):
                n_skipped += 1
                if idx % 50 == 0 or idx == total:
                    print(f"  [{idx:4d}/{total}] processed={n_processed} skipped={n_skipped} failed={n_failed}")
                continue

        try:
            if task == "task1":
                entry = process_patient_task1(
                    pdir, anatomy, preprocessed_root, target_spacing_mm,
                    ct_clip=ct_clip_by_anat[anatomy],
                )
            else:
                entry = process_patient_task2(
                    pdir, anatomy, preprocessed_root, target_spacing_mm,
                    ct_clip=ct_clip_by_anat[anatomy], cbct_clip=cbct_clip,
                )

            if entry is None:
                n_skipped += 1
            else:
                manifest[pid] = entry
                n_processed += 1
                save_manifest(manifest, manifest_path)  # crash-safe: write after each
        except Exception:
            err = traceback.format_exc().splitlines()[-1]
            print(f"  [{idx:4d}/{total}] ✗ {pid} — {err}", file=sys.stderr)
            n_failed += 1
            continue

        if idx % 50 == 0 or idx == total:
            print(f"  [{idx:4d}/{total}] processed={n_processed} skipped={n_skipped} failed={n_failed}")

    save_manifest(manifest, manifest_path)

    print(f"\nManifest written -> {manifest_path}")
    print(f"  Entries: {len(manifest)}  |  processed: {n_processed}  "
          f"|  skipped: {n_skipped}  |  failed: {n_failed}")
    if n_failed > 0:
        print(f"\nWARNING: {n_failed} patients failed. Check stderr above.")

    print("\nDone.")


if __name__ == "__main__":
    main()
