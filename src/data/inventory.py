"""EDA inventory for SynthRAD2023 Task1.

CLI:
    python src/data/inventory.py \\
        --data-root data/synthrad2023/Task1 \\
        --out-dir outputs \\
        [--n-samples N]  # subset for figures only (None = auto-pick 6) \\
        [--seed 42]

Outputs (Rule 5 — never writes under data/):
    outputs/reports/inventory.csv
    outputs/figures/sample_slices.png
    outputs/figures/intensity_hist.png

Rules observed:
    R2  — CT stats reported in HU (no normalisation applied here).
    R5  — read-only access to data/.
    R6  — seed-controlled figure sample.
    R7  — typed, docstrings, try/except per patient.
    R8  — no thresholds invented; data speaks.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CENTERS = {"A", "B", "C"}


def _parse_patient_id(pid: str, anatomy: str) -> dict[str, str]:
    """Extract center letter from patient ID (format: 1[B|P][A|B|C][NNN])."""
    center = pid[2] if len(pid) >= 3 else "?"
    if center not in _VALID_CENTERS:
        center = "?"
    return {"patient_id": pid, "anatomy": anatomy, "center": center}


def _sitk_to_np(img: sitk.Image) -> np.ndarray:
    """Return numpy array in (Z, Y, X) order (standard radiological)."""
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _intensity_stats(arr: np.ndarray) -> dict[str, float]:
    """Compute min/max/mean/p1/p50/p99 of a flat array."""
    flat = arr.ravel()
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "p1": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
    }


# ---------------------------------------------------------------------------
# Per-patient scan
# ---------------------------------------------------------------------------

def scan_patient(patient_dir: Path, anatomy: str) -> dict[str, Any]:
    """Scan one patient directory and return a flat stats dict.

    Returns a dict with either full stats or an 'error' key on failure.
    Never modifies data/ (R5).
    """
    pid = patient_dir.name
    row: dict[str, Any] = _parse_patient_id(pid, anatomy)
    row["patient_dir"] = str(patient_dir)

    try:
        mr_img = sitk.ReadImage(str(patient_dir / "mr.nii.gz"))
        ct_img = sitk.ReadImage(str(patient_dir / "ct.nii.gz"))
        mask_img = sitk.ReadImage(str(patient_dir / "mask.nii.gz"))

        mr = _sitk_to_np(mr_img)
        ct = _sitk_to_np(ct_img)
        mask = _sitk_to_np(mask_img).astype(bool)

        # Shapes
        row["mr_shape"] = str(mr.shape)
        row["ct_shape"] = str(ct.shape)
        row["mask_shape"] = str(mask.shape)
        row["shapes_match"] = (mr.shape == ct.shape == mask.shape)

        # Spacing (SimpleITK gives X,Y,Z order)
        sp = mr_img.GetSpacing()
        row["spacing_x_mm"] = round(sp[0], 4)
        row["spacing_y_mm"] = round(sp[1], 4)
        row["spacing_z_mm"] = round(sp[2], 4)

        # Mask volume fraction
        row["mask_vox_fraction"] = round(float(mask.sum()) / mask.size, 4)

        # MR stats — whole volume
        mr_s = _intensity_stats(mr)
        for k, v in mr_s.items():
            row[f"mr_{k}"] = round(v, 4)

        # CT stats — whole volume
        ct_s = _intensity_stats(ct)
        for k, v in ct_s.items():
            row[f"ct_{k}"] = round(v, 4)

        # CT stats — INSIDE mask (the HU values that matter for sCT, R2)
        if mask.sum() > 0:
            ct_mask_s = _intensity_stats(ct[mask])
            for k, v in ct_mask_s.items():
                row[f"ct_inmask_{k}"] = round(v, 4)
        else:
            for k in ("min", "max", "mean", "p1", "p50", "p99"):
                row[f"ct_inmask_{k}"] = float("nan")

        # MR stats — inside mask
        if mask.sum() > 0:
            mr_mask_s = _intensity_stats(mr[mask])
            for k, v in mr_mask_s.items():
                row[f"mr_inmask_{k}"] = round(v, 4)
        else:
            for k in ("min", "max", "mean", "p1", "p50", "p99"):
                row[f"mr_inmask_{k}"] = float("nan")

        # Flags
        row["flag_neg_mr"] = bool((mr < 0).any())
        row["flag_ct_metal"] = bool(
            mask.sum() > 0 and float(ct[mask].max()) > 3000
        )
        row["error"] = ""

    except Exception:
        row["error"] = traceback.format_exc().splitlines()[-1]

    return row


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "patient_id", "anatomy", "center", "patient_dir",
    "mr_shape", "ct_shape", "mask_shape", "shapes_match",
    "spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
    "mask_vox_fraction",
    "mr_min", "mr_max", "mr_mean", "mr_p1", "mr_p50", "mr_p99",
    "mr_inmask_min", "mr_inmask_max", "mr_inmask_mean",
    "mr_inmask_p1", "mr_inmask_p50", "mr_inmask_p99",
    "ct_min", "ct_max", "ct_mean", "ct_p1", "ct_p50", "ct_p99",
    "ct_inmask_min", "ct_inmask_max", "ct_inmask_mean",
    "ct_inmask_p1", "ct_inmask_p50", "ct_inmask_p99",
    "flag_neg_mr", "flag_ct_metal", "error",
]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write inventory rows to CSV; missing keys → empty string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _CSV_FIELDS})


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print aggregate summary to stdout."""
    ok = [r for r in rows if not r.get("error")]
    bad = [r for r in rows if r.get("error")]

    print("\n" + "=" * 70)
    print("  THERMOBRIDGE EDA SUMMARY")
    print("=" * 70)

    # ── Patient counts ───────────────────────────────────────────────────
    print(f"\nTotal patients scanned : {len(rows)}")
    print(f"  Successful           : {len(ok)}")
    print(f"  Failed (errors)      : {len(bad)}")
    if bad:
        for r in bad:
            print(f"    ✗ {r['patient_id']} — {r['error']}")

    print("\n── Counts by anatomy × center ──")
    from collections import Counter
    counts: Counter = Counter()
    for r in ok:
        counts[(r["anatomy"], r["center"])] += 1
    for (anat, ctr), n in sorted(counts.items()):
        print(f"  {anat:8s} center {ctr}: {n:4d}")

    # ── Shapes ───────────────────────────────────────────────────────────
    print("\n── CT volume shapes ──")
    shape_ctr: Counter = Counter(r["ct_shape"] for r in ok)
    n_distinct = len(shape_ctr)
    most_common, mc_n = shape_ctr.most_common(1)[0]
    print(f"  Distinct shapes      : {n_distinct}")
    print(f"  Most common          : {most_common}  (n={mc_n})")

    mismatch = [r for r in ok if not r.get("shapes_match")]
    print(f"  Shape mismatches     : {len(mismatch)}")
    if mismatch:
        for r in mismatch:
            print(f"    ✗ {r['patient_id']}: MR={r['mr_shape']} CT={r['ct_shape']}")

    # ── Voxel spacing ────────────────────────────────────────────────────
    print("\n── Voxel spacing (mm) — min / median / max ──")
    for ax in ("x", "y", "z"):
        vals = [r[f"spacing_{ax}_mm"] for r in ok if f"spacing_{ax}_mm" in r]
        if vals:
            print(
                f"  {ax}-axis : {min(vals):.4f} / "
                f"{float(np.median(vals)):.4f} / {max(vals):.4f}"
            )

    # ── CT in-mask HU ────────────────────────────────────────────────────
    print("\n── In-mask CT (HU) statistics ──")
    for key, label in [
        ("ct_inmask_min", "global min"),
        ("ct_inmask_p1",  "p1  range"),
        ("ct_inmask_p99", "p99 range"),
        ("ct_inmask_max", "global max"),
    ]:
        vals = [r[key] for r in ok if isinstance(r.get(key), float) and not np.isnan(r[key])]
        if vals:
            print(f"  {label:12s}: [{min(vals):.1f}, {max(vals):.1f}]")

    # ── MR in-mask ───────────────────────────────────────────────────────
    print("\n── In-mask MR intensity statistics ──")
    for key, label in [
        ("mr_inmask_p99", "p99 range"),
        ("mr_inmask_max", "global max"),
    ]:
        vals = [r[key] for r in ok if isinstance(r.get(key), float) and not np.isnan(r[key])]
        if vals:
            print(f"  {label:12s}: [{min(vals):.1f}, {max(vals):.1f}]")

    # ── Flags ────────────────────────────────────────────────────────────
    print("\n── Flags ──")
    neg_mr = [r["patient_id"] for r in ok if r.get("flag_neg_mr")]
    metal  = [r["patient_id"] for r in ok if r.get("flag_ct_metal")]
    print(f"  Negative MR values   : {len(neg_mr)}"
          + (f"  → {neg_mr[:5]}" if neg_mr else ""))
    print(f"  CT > 3000 HU (metal) : {len(metal)}"
          + (f"  → {metal[:5]}" if metal else ""))

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _mid_axial(arr: np.ndarray) -> np.ndarray:
    """Return the mid-axial slice (axis 0 = Z)."""
    return arr[arr.shape[0] // 2]


def figure_sample_slices(
    rows: list[dict[str, Any]],
    data_root: Path,
    out_path: Path,
    n_samples: int,
    seed: int,
) -> None:
    """3-column grid: MR | CT windowed | mask contour over MR (R6 seed)."""
    ok = [r for r in rows if not r.get("error")]
    rng = random.Random(seed)

    # Anatomy-mixed sample: half brain, half pelvis (or as close as possible)
    brain_rows = [r for r in ok if r["anatomy"] == "brain"]
    pelvis_rows = [r for r in ok if r["anatomy"] == "pelvis"]
    half = n_samples // 2
    sample = (
        rng.sample(brain_rows, min(half, len(brain_rows)))
        + rng.sample(pelvis_rows, min(n_samples - half, len(pelvis_rows)))
    )
    rng.shuffle(sample)
    n = len(sample)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), dpi=300)
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["MR", "CT (HU −1000→+1000)", "Mask contour over MR"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold", pad=6)

    for i, row in enumerate(sample):
        pdir = Path(row["patient_dir"])
        try:
            mr = _sitk_to_np(sitk.ReadImage(str(pdir / "mr.nii.gz")))
            ct = _sitk_to_np(sitk.ReadImage(str(pdir / "ct.nii.gz")))
            msk = _sitk_to_np(sitk.ReadImage(str(pdir / "mask.nii.gz")))
        except Exception as exc:
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f"Load error:\n{exc}",
                                ha="center", va="center", transform=axes[i, j].transAxes)
            continue

        mr_sl  = _mid_axial(mr)
        ct_sl  = _mid_axial(ct)
        msk_sl = _mid_axial(msk)

        label = f"{row['patient_id']}\n({row['anatomy']}, ctr {row['center']})"

        # col 0: MR
        axes[i, 0].imshow(mr_sl, cmap="gray", origin="lower",
                          vmin=np.percentile(mr_sl, 1),
                          vmax=np.percentile(mr_sl, 99))
        axes[i, 0].set_ylabel(label, fontsize=8)

        # col 1: CT windowed
        axes[i, 1].imshow(ct_sl, cmap="gray", origin="lower",
                          vmin=-1000, vmax=1000)

        # col 2: mask contour over MR
        axes[i, 2].imshow(mr_sl, cmap="gray", origin="lower",
                          vmin=np.percentile(mr_sl, 1),
                          vmax=np.percentile(mr_sl, 99))
        axes[i, 2].contour(msk_sl, levels=[0.5], colors=["red"], linewidths=0.8)

        for j in range(3):
            axes[i, j].axis("off")

    fig.suptitle(
        f"SynthRAD2023 Task1 — mid-axial sample (n={n}, seed={seed})",
        fontsize=13, fontweight="bold", y=1.002,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def figure_intensity_hist(
    rows: list[dict[str, Any]],
    data_root: Path,
    out_path: Path,
    n_samples: int,
    seed: int,
) -> None:
    """Log-y histograms: in-mask CT (HU) and in-mask MR, anatomy-coloured."""
    ok = [r for r in rows if not r.get("error")]
    rng = random.Random(seed + 1)

    brain_rows  = [r for r in ok if r["anatomy"] == "brain"]
    pelvis_rows = [r for r in ok if r["anatomy"] == "pelvis"]
    half = n_samples // 2
    sample = (
        rng.sample(brain_rows, min(half, len(brain_rows)))
        + rng.sample(pelvis_rows, min(n_samples - half, len(pelvis_rows)))
    )

    ct_brain, ct_pelvis = [], []
    mr_brain, mr_pelvis = [], []

    for row in sample:
        pdir = Path(row["patient_dir"])
        try:
            ct  = _sitk_to_np(sitk.ReadImage(str(pdir / "ct.nii.gz")))
            mr  = _sitk_to_np(sitk.ReadImage(str(pdir / "mr.nii.gz")))
            msk = _sitk_to_np(sitk.ReadImage(str(pdir / "mask.nii.gz"))).astype(bool)
            if msk.sum() > 0:
                if row["anatomy"] == "brain":
                    ct_brain.append(ct[msk])
                    mr_brain.append(mr[msk])
                else:
                    ct_pelvis.append(ct[msk])
                    mr_pelvis.append(mr[msk])
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # CT in-mask
    ax = axes[0]
    bins_ct = np.linspace(-1200, 3200, 200)
    if ct_brain:
        ax.hist(np.concatenate(ct_brain), bins=bins_ct, log=True,
                alpha=0.65, color="#4C72B0", label="Brain")
    if ct_pelvis:
        ax.hist(np.concatenate(ct_pelvis), bins=bins_ct, log=True,
                alpha=0.65, color="#DD8452", label="Pelvis")
    ax.set_xlabel("CT intensity (HU)", fontsize=11)
    ax.set_ylabel("Voxel count (log)", fontsize=11)
    ax.set_title("In-mask CT histogram (HU)", fontsize=12, fontweight="bold")
    ax.axvline(-1000, color="gray", lw=0.8, ls="--", label="HU −1000")
    ax.axvline(3000,  color="red",  lw=0.8, ls="--", label="HU +3000")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # MR in-mask
    ax = axes[1]
    all_mr = (np.concatenate(mr_brain) if mr_brain else np.array([])) 
    all_mr = np.concatenate(
        [np.concatenate(mr_brain) if mr_brain else np.array([]),
         np.concatenate(mr_pelvis) if mr_pelvis else np.array([])]
    )
    if all_mr.size:
        bins_mr = np.linspace(np.percentile(all_mr, 0.1), np.percentile(all_mr, 99.9), 200)
    else:
        bins_mr = 100
    if mr_brain:
        axes[1].hist(np.concatenate(mr_brain), bins=bins_mr, log=True,
                     alpha=0.65, color="#4C72B0", label="Brain")
    if mr_pelvis:
        axes[1].hist(np.concatenate(mr_pelvis), bins=bins_mr, log=True,
                     alpha=0.65, color="#DD8452", label="Pelvis")
    axes[1].set_xlabel("MR intensity (a.u.)", fontsize=11)
    axes[1].set_ylabel("Voxel count (log)", fontsize=11)
    axes[1].set_title("In-mask MR histogram", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"SynthRAD2023 Task1 — in-mask intensity distributions (n={len(sample)} patients, seed={seed})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SynthRAD2023 Task1 EDA inventory (ThermoBridge, R5/R6/R7/R8)."
    )
    p.add_argument("--data-root", required=True, type=Path,
                   help="Path to data/synthrad2023/Task1")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Root output dir (outputs/)")
    p.add_argument("--n-samples", type=int, default=6,
                   help="Patients to sample for figures (default: 6, must be even)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (R6)")
    return p.parse_args()


def collect_patient_dirs(data_root: Path) -> list[tuple[Path, str]]:
    """Enumerate all patient directories, skipping non-patient folders."""
    patients: list[tuple[Path, str]] = []
    for anatomy in ("brain", "pelvis"):
        anat_dir = data_root / anatomy
        if not anat_dir.exists():
            print(f"  WARNING: {anat_dir} not found — skipping.", file=sys.stderr)
            continue
        for entry in sorted(anat_dir.iterdir()):
            if not entry.is_dir():
                continue
            # Patient IDs follow 1[B|P][A|B|C][NNN] — skip 'overview' etc.
            if not (entry.name.startswith("1") and len(entry.name) >= 4):
                continue
            patients.append((entry, anatomy))
    return patients


def main() -> None:
    args = parse_args()

    # Validate data root (R5 — read-only, must already exist)
    data_root = args.data_root.resolve()
    if not data_root.exists():
        sys.exit(f"ERROR: --data-root '{data_root}' does not exist.")

    out_dir = args.out_dir.resolve()
    csv_path = out_dir / "reports" / "inventory.csv"
    fig_slices = out_dir / "figures" / "sample_slices.png"
    fig_hist   = out_dir / "figures" / "intensity_hist.png"

    # Collect patient dirs
    patients = collect_patient_dirs(data_root)
    print(f"\nFound {len(patients)} patient directories. Scanning…\n")

    rows: list[dict[str, Any]] = []
    for idx, (pdir, anatomy) in enumerate(patients, 1):
        row = scan_patient(pdir, anatomy)
        rows.append(row)
        status = "✓" if not row.get("error") else "✗"
        if idx % 50 == 0 or idx == len(patients):
            print(f"  [{idx:4d}/{len(patients)}] {status} {row['patient_id']}")
        elif row.get("error"):
            print(f"  [{idx:4d}/{len(patients)}] {status} {row['patient_id']} — {row['error']}")

    # Write CSV
    write_csv(rows, csv_path)
    print(f"\nInventory written → {csv_path}")

    # Print summary
    print_summary(rows)

    # Figures — n_samples must be even; cap at available
    ok_rows = [r for r in rows if not r.get("error")]
    n_fig = min(args.n_samples, len(ok_rows))
    if n_fig % 2 != 0:
        n_fig -= 1  # keep even for balanced anatomy mix
    n_fig = max(n_fig, 2)

    print("Rendering figures…")
    figure_sample_slices(rows, data_root, fig_slices, n_samples=n_fig, seed=args.seed)
    figure_intensity_hist(rows, data_root, fig_hist,  n_samples=n_fig, seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
