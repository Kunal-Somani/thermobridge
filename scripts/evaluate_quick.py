"""Quick proxy evaluation — single center-crop forward pass, no sliding window.

Loads a trained checkpoint and, for a handful of test patients per anatomy,
runs ONE bridge reverse-sample call on the center [96,96,96] (cfg.patch.size)
crop of each volume — instead of MONAI sliding-window inference over the
whole volume. This trades full-volume coverage for speed: seconds per
patient instead of minutes, giving a directional MAE-HU proxy metric.

This is NOT a substitute for LitBridge.evaluate_full() (the full-volume,
sliding-window, R2/R4-compliant harness) — it's a fast sanity check to run
between full evaluations, e.g. right after training completes or when
comparing checkpoints quickly.

Usage::
    python scripts/evaluate_quick.py --config configs/default.yaml \\
        --checkpoint outputs/runs/thermobridge_v2/checkpoints/best_epoch=079_val/loss_patch=0.0590.ckpt \\
        --manifest-2023 outputs/preprocessed/manifest.json \\
        --manifest-2025 outputs/preprocessed_2025/manifest_2025.json \\
        --splits-2023 outputs/splits.json \\
        --splits-2025 outputs/splits_synthrad2025.json \\
        --patients-per-anatomy 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.preprocess import invert_ct_to_hu, invert_mr
from src.training.lit_bridge import LitBridge, _CBCT_IDX, _CT_IDX, _MR_IDX
from src.training.metrics import compute_all_metrics
from src.utils.config import load_config

_MOD_NAME = {_MR_IDX: "mr", _CT_IDX: "ct", _CBCT_IDX: "cbct"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick center-crop proxy evaluation (no sliding window)."
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest-2023", type=Path,
                   default=_REPO_ROOT / "outputs" / "preprocessed" / "manifest.json")
    p.add_argument("--manifest-2025", type=Path,
                   default=_REPO_ROOT / "outputs" / "preprocessed_2025" / "manifest_2025.json")
    p.add_argument("--splits-2023", type=Path,
                   default=_REPO_ROOT / "outputs" / "splits.json")
    p.add_argument("--splits-2025", type=Path,
                   default=_REPO_ROOT / "outputs" / "splits_synthrad2025.json")
    p.add_argument("--split", default="test", choices=["val", "test"],
                   help="Which split to sample patients from.")
    p.add_argument("--patients-per-anatomy", type=int, default=5,
                   help="Max patients evaluated per anatomy (for speed).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Center crop
# ---------------------------------------------------------------------------


def center_crop(arr: np.ndarray, patch_size: tuple[int, int, int]) -> np.ndarray:
    """Crop the center patch_size window out of a (Z, Y, X) volume."""
    Z, Y, X = arr.shape
    pZ, pY, pX = (min(p, s) for p, s in zip(patch_size, (Z, Y, X)))
    z0, y0, x0 = (Z - pZ) // 2, (Y - pY) // 2, (X - pX) // 2
    return arr[z0:z0 + pZ, y0:y0 + pY, x0:x0 + pX]


# ---------------------------------------------------------------------------
# Single forward pass (one reverse_sample call, no sliding window)
# ---------------------------------------------------------------------------


def predict_patch(
    model: LitBridge,
    source_norm: np.ndarray,
    m_s_val: int,
    m_t_val: int,
    device: torch.device,
) -> np.ndarray:
    """Run ONE bridge reverse-sample call on a single patch."""
    x_T = torch.from_numpy(source_norm[np.newaxis, np.newaxis]).float().to(device)
    m_s = torch.tensor([m_s_val], dtype=torch.long, device=device)
    m_t = torch.tensor([m_t_val], dtype=torch.long, device=device)
    alpha = model._make_alpha(1, device)
    num_steps = int(model.cfg.model.bridge.num_steps)
    with torch.no_grad():
        x_hat_0 = model.bridge.reverse_sample(
            model.denoiser, x_T, m_s, m_t, alpha, num_steps=num_steps
        )
    return x_hat_0[0, 0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# Per-patient, per-direction evaluation
# ---------------------------------------------------------------------------


def evaluate_patient_direction(
    model: LitBridge,
    entry: dict[str, Any],
    pid: str,
    is_2025: bool,
    direction_id: int,
    patch_size: tuple[int, int, int],
    device: torch.device,
) -> dict[str, Any]:
    """Center-crop, single forward pass, HU-inverted metrics for one (patient, direction)."""
    if is_2025:
        modality_src = entry["modality_src"]  # "mr" or "cbct"
        m_src = _MR_IDX if modality_src == "mr" else _CBCT_IDX
        src_full  = np.load(entry["src_path"])["data"]
        ct_full   = np.load(entry["ct_path"])["data"]
        mask_full = np.load(entry["mask_path"])["data"].astype(np.float32)
        src_params = entry.get("src_norm_params", {})
        ct_params  = entry["ct_norm_params"]
    else:
        m_src = _MR_IDX
        src_full  = np.load(entry["mr_path"])["data"]
        ct_full   = np.load(entry["ct_path"])["data"]
        mask_full = np.load(entry["mask_path"])["data"].astype(np.float32)
        src_params = entry["mr_norm_params"]
        ct_params  = entry["ct_norm_params"]

    if direction_id == 0:   # source -> CT
        m_s, m_t = m_src, _CT_IDX
        source_full, target_full, target_params = src_full, ct_full, ct_params
    else:                    # CT -> source
        m_s, m_t = _CT_IDX, m_src
        source_full, target_full, target_params = ct_full, src_full, src_params

    source_crop = center_crop(source_full, patch_size)
    target_crop = center_crop(target_full, patch_size)
    mask_crop   = center_crop(mask_full,   patch_size)

    pred_norm = predict_patch(model, source_crop, m_s, m_t, device)

    if m_t == _CT_IDX:       # target is CT: headline HU metric
        pred_hu   = invert_ct_to_hu(pred_norm,   target_params)
        target_hu = invert_ct_to_hu(target_crop, target_params)
        result    = compute_all_metrics(pred_hu, target_hu, mask_crop)
        mae_label = "mae_hu"
    elif target_params:      # target is MR/CBCT with known norm params
        pred_orig   = invert_mr(pred_norm,   target_params)
        target_orig = invert_mr(target_crop, target_params)
        result      = compute_all_metrics(pred_orig, target_orig, mask_crop)
        mae_label   = "mae_hu"
    else:                     # no norm params to invert — normalized MAE only
        result    = compute_all_metrics(pred_norm, target_crop, mask_crop)
        mae_label = "mae_norm"

    return {
        "patient_id":  pid,
        "anatomy":     entry["anatomy"],
        "direction":   f"{_MOD_NAME[m_s]}->{_MOD_NAME[m_t]}",
        mae_label:     round(result.mae_hu, 4),
        "psnr_db":     round(result.psnr, 4),
        "ssim":        round(result.ssim, 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    with open(args.manifest_2023) as f:
        manifest_2023 = json.load(f)
    with open(args.manifest_2025) as f:
        manifest_2025 = json.load(f)
    with open(args.splits_2023) as f:
        splits_2023 = json.load(f)
    with open(args.splits_2025) as f:
        splits_2025 = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint} onto {device} …")
    model = LitBridge.load_from_checkpoint(
        str(args.checkpoint),
        cfg=cfg,
        manifest_path=args.manifest_2023,
        splits_path=args.splits_2023,
        manifest_2025_path=args.manifest_2025,
        map_location=device,
    )
    model.to(device)
    model.eval()

    patch_size = tuple(int(x) for x in cfg.patch.size)

    # ── Gather + limit patients per anatomy ──────────────────────────────
    anatomy_to_pids: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for pid in sorted(splits_2023[args.split]):
        anatomy_to_pids[manifest_2023[pid]["anatomy"]].append((pid, False))
    for pid in sorted(splits_2025[args.split]):
        anatomy_to_pids[manifest_2025[pid]["anatomy"]].append((pid, True))

    selected: list[tuple[str, bool]] = []
    for anatomy in sorted(anatomy_to_pids):
        selected.extend(anatomy_to_pids[anatomy][: args.patients_per_anatomy])

    print(f"Evaluating {len(selected)} patients "
          f"(<= {args.patients_per_anatomy} per anatomy, split='{args.split}') "
          f"— center-crop {patch_size}, no sliding window.\n")

    # ── Run ───────────────────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    for pid, is_2025 in selected:
        entry = manifest_2025[pid] if is_2025 else manifest_2023[pid]
        for direction_id in (0, 1):
            try:
                row = evaluate_patient_direction(
                    model, entry, pid, is_2025, direction_id, patch_size, device
                )
                rows.append(row)
                mae_key = "mae_hu" if "mae_hu" in row else "mae_norm"
                print(f"  {pid:<12} {entry['anatomy']:<8} {row['direction']:<8} "
                      f"{mae_key}={row[mae_key]:>8.2f}  psnr={row['psnr_db']:>6.2f}dB")
            except Exception as exc:
                print(f"  {pid:<12} direction {direction_id} FAILED: {exc}")

    # ── Aggregate: per anatomy x direction ──────────────────────────────
    print("\n" + "=" * 72)
    print("  QUICK EVAL SUMMARY — center-crop proxy MAE (mean over patients)")
    print("=" * 72)

    anatomies = sorted({r["anatomy"] for r in rows})
    for anatomy in anatomies:
        anat_rows = [r for r in rows if r["anatomy"] == anatomy]
        directions = sorted({r["direction"] for r in anat_rows})
        for direction in directions:
            grp = [r for r in anat_rows if r["direction"] == direction]
            mae_key = "mae_hu" if "mae_hu" in grp[0] else "mae_norm"
            maes = [r[mae_key] for r in grp]
            psnrs = [r["psnr_db"] for r in grp]
            print(f"  {anatomy:<8} {direction:<8} "
                  f"{mae_key}={np.mean(maes):>8.2f}±{np.std(maes):<6.2f}  "
                  f"psnr={np.mean(psnrs):>6.2f}dB  n={len(grp)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
