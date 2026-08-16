"""Full-volume evaluation harness for ThermoBridge sCT synthesis.

Given any ``predictor`` callable with signature::

    predictor(source_norm: np.ndarray, direction_id: int) -> np.ndarray

(where source_norm is the preprocessed normalised array (Z,Y,X) and the return
is also in normalised space), this module:

1. Runs full-volume inference using MONAI sliding-window with overlap blending.
2. Inverse-normalises the CT prediction back to true HU (R2, load-bearing).
3. Computes per-patient MAE-HU, PSNR, SSIM (all in-mask).
4. Aggregates mean ± std per (anatomy × direction) and overall.
5. Reports a paired Wilcoxon signed-rank significance test for comparing two
   predictors on the same patient set.
6. Writes outputs/reports/eval_<name>.csv (one row per patient).

Two trivial baselines are provided:
    IdentityPredictor   — returns the source volume unchanged (copy-source).
    MeanCTPredictor     — returns the per-anatomy mean CT image (train-set mean).

These numbers are the floor every real model must beat (R4, §9 spec).

Rules:
    R1  — SSIM method documented here and in metrics.py.
    R2  — CT inversion happens inside the harness, never exposed to caller.
    R4  — mean ± std, significance test; NO peak single-case numbers.
    R5  — writes under outputs/, never under data/.
    R6  — deterministic eval ordering (val/test datasets are ordered).
    R7  — typed, docstrings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from scipy.stats import wilcoxon

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.preprocess import invert_ct_to_hu, invert_mr
from src.training.metrics import MetricResult, compute_all_metrics
from src.utils.config import load_config

# ---------------------------------------------------------------------------
# Predictor protocol
# ---------------------------------------------------------------------------


class Predictor(Protocol):
    """Interface every predictor must satisfy."""

    @property
    def name(self) -> str: ...

    def predict(
        self,
        source_norm: np.ndarray,
        direction_id: int,
    ) -> np.ndarray:
        """Return a predicted normalised volume of the same shape as source_norm.

        Args:
            source_norm:  Input volume in normalised space, shape (Z, Y, X).
            direction_id: 0 = MR→CT  (source=MR, predict CT)
                          1 = CT→MR  (source=CT, predict MR).

        Returns:
            Predicted volume in the SAME normalised space as the target modality.
        """
        ...


# ---------------------------------------------------------------------------
# Trivial baseline (a): Identity / copy-source
# ---------------------------------------------------------------------------


class IdentityPredictor:
    """Baseline: output the source volume unchanged.

    For MR→CT direction this is a terrible prediction (MR ≠ CT HU scale),
    but it is a useful sanity check that the harness correctly measures
    large errors for bad predictors.
    """

    name = "identity"

    def predict(self, source_norm: np.ndarray, direction_id: int) -> np.ndarray:
        return source_norm.copy()


# ---------------------------------------------------------------------------
# Trivial baseline (b): Per-anatomy mean CT
# ---------------------------------------------------------------------------


class MeanCTPredictor:
    """Baseline: predict the per-anatomy mean CT image (computed from train).

    For direction 0 (MR→CT): predicts the anatomy-mean CT.
    For direction 1 (CT→MR): predicts the anatomy-mean MR (less meaningful).

    The mean is computed lazily on first call; train patient list is taken
    from splits.json so it cannot overlap with eval patients (R3).
    """

    name = "mean_ct"

    def __init__(
        self,
        manifest: dict,
        train_ids: list[str],
        target_shape_zyx: tuple[int, int, int] | None = None,
    ) -> None:
        self._manifest = manifest
        self._train_ids = train_ids
        self._target_shape = target_shape_zyx
        # Lazy cache: (anatomy, direction_id) -> mean normalised volume
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        # We need anatomy context at predict time — caller injects it
        self._current_anatomy: str = "brain"

    def set_anatomy(self, anatomy: str) -> None:
        """Must be called before predict() for each patient."""
        self._current_anatomy = anatomy

    def _build_mean(self, anatomy: str, direction_id: int) -> np.ndarray:
        """Compute mean normalised target volume over the train set."""
        pids = [
            p for p in self._train_ids
            if self._manifest[p]["anatomy"] == anatomy
        ]
        if not pids:
            raise RuntimeError(f"No train patients for anatomy '{anatomy}'.")

        # Load a few patients to get a shape reference then accumulate sum
        sum_vol: np.ndarray | None = None
        count = 0
        for pid in pids:
            e = self._manifest[pid]
            if direction_id == 0:       # MR→CT: target is CT
                vol = np.load(e["ct_path"])["data"]
            else:                        # CT→MR: target is MR
                vol = np.load(e["mr_path"])["data"]

            if sum_vol is None:
                sum_vol = vol.astype(np.float64)
            else:
                # Resize to match sum shape via simple crop/pad to median size
                # (simplification: use the first patient's shape as reference)
                if vol.shape != sum_vol.shape:
                    vol = _resize_to(vol, sum_vol.shape)
                sum_vol += vol.astype(np.float64)
            count += 1

        return (sum_vol / count).astype(np.float32)  # type: ignore[return-value]

    def predict(self, source_norm: np.ndarray, direction_id: int) -> np.ndarray:
        key = (self._current_anatomy, direction_id)
        if key not in self._cache:
            print(f"  [MeanCTPredictor] building mean for {key} …", flush=True)
            mean_vol = self._build_mean(self._current_anatomy, direction_id)
            self._cache[key] = mean_vol

        mean = self._cache[key]
        # Resize mean to match current patient's volume shape
        if mean.shape != source_norm.shape:
            mean = _resize_to(mean, source_norm.shape)
        return mean


def _resize_to(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Crop or zero-pad arr to target_shape (Z, Y, X)."""
    out = np.zeros(target_shape, dtype=arr.dtype)
    slices_src = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
    slices_dst = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
    out[slices_dst] = arr[slices_src]
    return out


# ---------------------------------------------------------------------------
# Full-volume inference with sliding-window (MONAI)
# ---------------------------------------------------------------------------


def sliding_window_predict(
    predictor: Predictor,
    source_norm: np.ndarray,
    direction_id: int,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
    device: "torch.device | str | None" = None,
) -> np.ndarray:
    """Run predictor over a full volume with MONAI sliding-window overlap blending.

    For trivial (non-neural) predictors the sliding window is unnecessary, but
    we keep it here so the harness API is identical for real models.

    Args:
        predictor:    Any object satisfying the Predictor protocol.
        source_norm:  Full volume in normalised space, shape (Z, Y, X).
        direction_id: Translation direction (0 or 1).
        patch_size:   Inference patch size (Z, Y, X).
        overlap:      Fraction overlap between adjacent windows.
        device:       Device to run inference on. If None, auto-detects from
            `predictor._mod.denoiser` when available (LitBridge's
            _BridgePredictor), else falls back to CPU.

    Returns:
        Predicted volume in normalised target space, shape (Z, Y, X).
    """
    import torch
    from monai.inferers import sliding_window_inference

    Z, Y, X = source_norm.shape
    pZ, pY, pX = patch_size

    if device is None:
        mod = getattr(predictor, "_mod", None)
        denoiser = getattr(mod, "denoiser", None)
        device = next(denoiser.parameters()).device if denoiser is not None else torch.device("cpu")
    device = torch.device(device)

    # Never run a whole (possibly large) volume through the net at once — that
    # path can OOM the 8 GB GPU and crash the shared display session. Clamp the
    # ROI to the volume size and always sliding-window it instead.
    roi_size = [min(p, s) for p, s in zip(patch_size, (Z, Y, X))]

    # MONAI sliding_window_inference expects (B, C, Z, Y, X). Move to the
    # target device up front so window extraction/blending stays on-GPU;
    # only the final result is brought back to CPU/numpy for metrics.
    src_t = torch.from_numpy(source_norm[np.newaxis, np.newaxis]).float().to(device)

    def _predictor_fn(patch: "torch.Tensor") -> "torch.Tensor":
        patch_np = patch[0, 0].cpu().numpy()  # (Z, Y, X)
        out_np = predictor.predict(patch_np, direction_id)
        return torch.from_numpy(out_np[np.newaxis, np.newaxis]).float().to(device)

    with torch.no_grad():
        result = sliding_window_inference(
            inputs       = src_t,
            roi_size     = roi_size,
            sw_batch_size= 1,
            predictor    = _predictor_fn,
            overlap      = overlap,
            mode         = "gaussian",
        )

    return result[0, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# Per-patient evaluation
# ---------------------------------------------------------------------------


def evaluate_patient(
    entry: dict,
    predictor: Predictor,
    direction_id: int,
    patch_size: tuple[int, int, int],
    overlap: float,
) -> dict:
    """Run inference + metrics for one (patient, direction) pair.

    Returns a flat dict suitable for CSV writing.
    """
    mr_norm  = np.load(entry["mr_path"])["data"]    # (Z, Y, X) normalised [0,1]
    ct_norm  = np.load(entry["ct_path"])["data"]    # (Z, Y, X) normalised [-1,1]
    mask     = np.load(entry["mask_path"])["data"].astype(np.float32)

    ct_params = entry["ct_norm_params"]
    mr_params = entry["mr_norm_params"]

    if direction_id == 0:          # MR → CT
        source_norm  = mr_norm
        target_norm  = ct_norm
        target_hu    = invert_ct_to_hu(ct_norm, ct_params)
    else:                          # CT → MR
        source_norm  = ct_norm
        target_norm  = mr_norm
        target_hu    = None        # MR has no HU; see below

    # Inject anatomy context for MeanCTPredictor (no-op for others)
    if hasattr(predictor, "set_anatomy"):
        predictor.set_anatomy(entry["anatomy"])

    pred_norm = sliding_window_predict(
        predictor, source_norm, direction_id, patch_size, overlap
    )

    # ── Metrics ─────────────────────────────────────────────────────────────
    if direction_id == 0:          # MR→CT: MAE-HU is the headline metric
        pred_hu   = invert_ct_to_hu(pred_norm, ct_params)
        result    = compute_all_metrics(pred_hu, target_hu, mask)
        mae_label = "mae_hu"
    else:                          # CT→MR: MAE on normalised scale (no HU unit)
        # For MR, MAE is on the per-volume-normalised [0,1] scale.
        # We report it as "mae_mr_norm" to be explicit (R1).
        mask_bool = mask.astype(bool)
        mae_mr    = float(np.abs(pred_norm[mask_bool] - mr_norm[mask_bool]).mean())
        # Also compute PSNR/SSIM on normalised MR (invertible to original scale)
        pred_mr_orig   = invert_mr(pred_norm,   mr_params)
        target_mr_orig = invert_mr(mr_norm,     mr_params)
        result = compute_all_metrics(pred_mr_orig, target_mr_orig, mask)
        # Override mae_hu with MAE on restored MR scale
        result = result._replace(mae_hu=mae_mr)
        mae_label = "mae_mr_norm"

    return {
        "patient_id":   entry["patient_id"],
        "anatomy":      entry["anatomy"],
        "center":       entry["center"],
        "direction_id": direction_id,
        "direction":    "MR->CT" if direction_id == 0 else "CT->MR",
        mae_label:      round(result.mae_hu, 4),
        "psnr_db":      round(result.psnr, 4),
        "ssim":         round(result.ssim, 6),
        "n_mask_vox":   result.n_mask_voxels,
    }


# ---------------------------------------------------------------------------
# Aggregation + significance test
# ---------------------------------------------------------------------------


def aggregate_results(rows: list[dict]) -> None:
    """Print mean ± std per (anatomy × direction) and overall.  No peak numbers (R4)."""
    print("\n" + "=" * 76)
    print("  EVALUATION SUMMARY  — mean ± std across patients  (R4)")
    print("=" * 76)

    def _stats(grp: list[dict], mae_key: str) -> str:
        if not grp:
            return "  (no data)"
        maes  = [r[mae_key]  for r in grp if mae_key in r]
        psnrs = [r["psnr_db"] for r in grp]
        ssims = [r["ssim"]    for r in grp]
        n = len(grp)
        mae_s  = f"{np.mean(maes):>8.2f}±{np.std(maes):<6.2f}" if maes else f"{'N/A':>15}"
        psnr_s = f"{np.mean(psnrs):>6.2f}±{np.std(psnrs):.2f}"
        ssim_s = f"{np.mean(ssims):>7.4f}±{np.std(ssims):.4f}"
        return f"{mae_s}  {psnr_s}  {ssim_s}  {n:>4}"

    header = (f"  {'group':<24}  {'MAE(HU|norm)':>15}  "
              f"{'psnr_db':>11}  {'ssim':>12}  {'n':>4}")
    print(header)
    print("  " + "-" * (len(header) - 1))

    for dirn, mae_key in [("MR->CT", "mae_hu"), ("CT->MR", "mae_mr_norm")]:
        dirn_rows = [r for r in rows if r["direction"] == dirn]
        anat_grps: dict[str, list[dict]] = defaultdict(list)
        for r in dirn_rows:
            anat_grps[r["anatomy"]].append(r)
        for anat in sorted(anat_grps):
            label = f"{anat}/{dirn}"
            print(f"  {label:<24}  {_stats(anat_grps[anat], mae_key)}")
        label = f"ALL/{dirn}"
        print(f"  {label:<24}  {_stats(dirn_rows, mae_key)}")
        print()

    print("=" * 76 + "\n")


def wilcoxon_compare(
    rows_a: list[dict],
    rows_b: list[dict],
    metric: str = "mae_hu",
) -> dict:
    """Paired Wilcoxon signed-rank test comparing two predictor result sets.

    Rows must correspond to the same patients in the same order.

    Args:
        rows_a, rows_b: Per-patient result rows from evaluate_split().
        metric:         Column to compare (default 'mae_hu').

    Returns:
        Dict with keys 'statistic', 'pvalue', 'n_pairs', 'direction'
        (lower is better for MAE, so we test A < B).
    """
    vals_a = np.array([r.get(metric, float("nan")) for r in rows_a])
    vals_b = np.array([r.get(metric, float("nan")) for r in rows_b])
    valid  = ~(np.isnan(vals_a) | np.isnan(vals_b))
    if valid.sum() < 2:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n_pairs": 0}

    stat, pval = wilcoxon(vals_a[valid], vals_b[valid], alternative="two-sided")
    return {
        "statistic": float(stat),
        "pvalue":    float(pval),
        "n_pairs":   int(valid.sum()),
        "metric":    metric,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_split(
    predictor: Predictor,
    split_ids: list[str],
    manifest: dict,
    patch_size: tuple[int, int, int],
    overlap: float,
    out_csv: Path,
    verbose: bool = True,
) -> list[dict]:
    """Evaluate predictor on all patients in split_ids for BOTH directions.

    Args:
        predictor:  Any Predictor-protocol object.
        split_ids:  Patient IDs from the target split (e.g., val).
        manifest:   Preprocessed manifest dict.
        patch_size: Inference window size.
        overlap:    Sliding-window overlap fraction.
        out_csv:    Path to write per-patient results CSV.
        verbose:    Print progress.

    Returns:
        List of per-patient result dicts (2 rows per patient: one per direction).
    """
    rows: list[dict] = []
    n = len(split_ids)

    for idx, pid in enumerate(sorted(split_ids), 1):
        entry = manifest[pid]
        if verbose:
            print(f"  [{idx:3d}/{n}] {pid} ({entry['anatomy']}, ctr {entry['center']})")

        for direction_id in (0, 1):
            try:
                row = evaluate_patient(
                    entry, predictor, direction_id, patch_size, overlap
                )
                rows.append(row)
            except Exception as exc:
                import traceback
                print(f"    ✗ direction {direction_id} failed: {exc}", flush=True)
                traceback.print_exc()

    # Write CSV
    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"\nResults written → {out_csv}")

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ThermoBridge evaluation harness — trivial baselines (R4)."
    )
    p.add_argument("--config",   required=True, type=Path)
    p.add_argument("--split",    default="val", choices=["val", "test"],
                   help="Which split to evaluate on.")
    p.add_argument("--manifest", type=Path,
                   default=_REPO_ROOT / "outputs" / "preprocessed" / "manifest.json")
    p.add_argument("--splits",   type=Path,
                   default=_REPO_ROOT / "outputs" / "splits.json")
    p.add_argument("--out-dir",  type=Path,
                   default=_REPO_ROOT / "outputs" / "reports")
    p.add_argument("--n-patients", type=int, default=None,
                   help="Limit to first N patients (for quick sanity checks).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.splits) as f:
        splits = json.load(f)

    split_ids: list[str] = splits[args.split]
    if args.n_patients:
        split_ids = split_ids[: args.n_patients]

    patch_size = tuple(int(x) for x in cfg.patch.size)
    overlap    = float(cfg.patch.inference_overlap)

    print(f"\nEvaluating on '{args.split}' split — {len(split_ids)} patients")
    print(f"  patch_size={patch_size}  overlap={overlap}")

    # ── Baseline (a): Identity ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("BASELINE (a): Identity / copy-source")
    print(f"{'─'*60}")
    id_pred  = IdentityPredictor()
    id_rows  = evaluate_split(
        id_pred, split_ids, manifest, patch_size, overlap,
        out_csv=args.out_dir / f"eval_identity_{args.split}.csv",
    )
    aggregate_results(id_rows)

    # ── Baseline (b): Per-anatomy mean CT ────────────────────────────────────
    print(f"{'─'*60}")
    print("BASELINE (b): Per-anatomy mean CT (train-set mean)")
    print(f"{'─'*60}")
    train_ids   = splits["train"]
    mean_pred   = MeanCTPredictor(manifest, train_ids)
    mean_rows   = evaluate_split(
        mean_pred, split_ids, manifest, patch_size, overlap,
        out_csv=args.out_dir / f"eval_mean_ct_{args.split}.csv",
    )
    aggregate_results(mean_rows)

    # ── Wilcoxon: mean_ct vs identity for MR→CT MAE ─────────────────────────
    print("── Wilcoxon test: mean_ct vs identity  (MR→CT, MAE-HU) ──")
    id_mr2ct   = [r for r in id_rows   if r["direction_id"] == 0]
    mean_mr2ct = [r for r in mean_rows if r["direction_id"] == 0]
    if id_mr2ct and mean_mr2ct:
        wres = wilcoxon_compare(id_mr2ct, mean_mr2ct, metric="mae_hu")
        print(f"  n={wres['n_pairs']}  W={wres['statistic']:.1f}  p={wres['pvalue']:.4g}")

    print("\nDone.")


if __name__ == "__main__":
    main()
