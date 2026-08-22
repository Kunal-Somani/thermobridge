"""Fast center-crop evaluation — full model (routing + anisotropic op + bridge).

Loads a trained checkpoint into the SAME architecture train_thermobridge.py
builds (denoiser + AnatomyRouter/adapters + AnisotropicDiffusionOp + I2SB
bridge — see build_model() there), then for a handful of test patients per
anatomy runs bridge.reverse_sample() with num_steps=10 on the center
[96,96,96] crop of each volume — no MONAI sliding window, no DataLoader
(npz files loaded directly with numpy). This trades full-volume coverage
and reverse-sampling fidelity for speed: this is a fast proxy metric, NOT
a substitute for LitBridge.evaluate_full() (the full-volume, sliding-window,
R2/R4-compliant harness).

Primary direction (source->CT, selected via --direction) is HU-inverted and
compared against fixed baselines (U-Net: 88.29 HU, mean-CT floor: 655 HU —
see outputs/reports/eval_mean_ct_*.csv / prior U-Net run logs for where
these numbers come from). The reverse direction (CT->source) is also
evaluated for the same patients, reported as normalized MAE/PSNR/SSIM (HU
inversion doesn't apply to MR/CBCT).

Usage::
    python scripts/evaluate_quick.py --config configs/default.yaml \\
        --checkpoint outputs/runs/thermobridge_v2/checkpoints/best_epoch=079_val/loss_patch=0.0590.ckpt \\
        --manifest-2023 outputs/preprocessed/manifest.json \\
        --manifest-2025 outputs/preprocessed_2025/manifest_2025.json \\
        --splits-2023 outputs/splits.json \\
        --splits-2025 outputs/splits_synthrad2025.json \\
        --patients-per-anatomy 5 --direction mr_to_ct
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

from scripts.train_thermobridge import build_model
from src.data.preprocess import invert_ct_to_hu, invert_mr
from src.training.lit_baseline import LitBaseline
from src.training.lit_bridge import LitBridge, _CBCT_IDX, _CT_IDX, _MR_IDX
from src.training.metrics import compute_all_metrics
from src.utils.config import load_config

_MOD_NAME = {_MR_IDX: "mr", _CT_IDX: "ct", _CBCT_IDX: "cbct"}

# Fixed comparison points (from prior baseline/U-Net evaluation runs — see
# outputs/reports/eval_mean_ct_*.csv and the U-Net baseline run logs).
BASELINE_UNET_MAE_HU = 88.29
BASELINE_MEAN_CT_MAE_HU = 655.0

PATCH_SIZE = (96, 96, 96)   # center crop — fixed, not read from cfg (R6: reproducible)
NUM_STEPS = 10              # fast inference — not cfg.model.bridge.num_steps (25)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast center-crop proxy evaluation (no sliding window, no DataLoader)."
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model-type", default="thermobridge", choices=["thermobridge", "unet"],
                   help="thermobridge (default): LitBridge + reverse_sample diffusion. "
                        "unet: LitBaseline, direct UNet3D forward pass (no bridge/diffusion).")
    p.add_argument("--manifest-2023", type=Path,
                   default=_REPO_ROOT / "outputs" / "preprocessed" / "manifest.json")
    p.add_argument("--manifest-2025", type=Path,
                   default=_REPO_ROOT / "outputs" / "preprocessed_2025" / "manifest_2025.json")
    p.add_argument("--splits-2023", type=Path,
                   default=_REPO_ROOT / "outputs" / "splits.json")
    p.add_argument("--splits-2025", type=Path,
                   default=_REPO_ROOT / "outputs" / "splits_synthrad2025.json")
    p.add_argument("--patients-per-anatomy", type=int, default=5,
                   help="Max patients evaluated per anatomy (for speed).")
    p.add_argument("--direction", default="mr_to_ct", choices=["mr_to_ct", "cbct_to_ct"],
                   help="Primary source->CT direction to evaluate + compare vs baselines. "
                        "The reverse (CT->source) direction is always evaluated too.")
    p.add_argument("--num-steps", type=int, default=10, help="Reverse-sampling steps (overrides NUM_STEPS).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Center crop
# ---------------------------------------------------------------------------


def center_crop(arr: np.ndarray, patch_size: tuple[int, int, int]) -> np.ndarray:
    """Crop the center patch_size window out of a (Z, Y, X) volume. Deterministic (R6)."""
    Z, Y, X = arr.shape
    pZ, pY, pX = (min(p, s) for p, s in zip(patch_size, (Z, Y, X)))
    z0, y0, x0 = (Z - pZ) // 2, (Y - pY) // 2, (X - pX) // 2
    return arr[z0:z0 + pZ, y0:y0 + pY, x0:x0 + pX]


# ---------------------------------------------------------------------------
# Single reverse_sample call (num_steps=10, no sliding window)
# ---------------------------------------------------------------------------


def predict_patch(
    model: LitBridge,
    source_norm: np.ndarray,
    m_s_val: int,
    m_t_val: int,
    device: torch.device,
) -> np.ndarray:
    x_T = torch.from_numpy(source_norm[np.newaxis, np.newaxis]).float().to(device)
    m_s = torch.tensor([m_s_val], dtype=torch.long, device=device)
    m_t = torch.tensor([m_t_val], dtype=torch.long, device=device)
    alpha = model._make_alpha(1, device)
    with torch.no_grad():
        x_hat_0 = model.bridge.reverse_sample(
            model.denoiser, x_T, m_s, m_t, alpha, num_steps=NUM_STEPS
        )
    return x_hat_0[0, 0].float().cpu().numpy()


class _UNetPredictor:
    """Direct UNet3D forward pass — no bridge reverse_sample, no diffusion.

    Mirrors _BridgePredictor (src/training/lit_bridge.py)'s role as a thin
    model wrapper, but calls model.model(...) directly instead of
    bridge.reverse_sample.
    """

    def __init__(self, model: LitBaseline) -> None:
        self.model = model

    def predict(
        self,
        source_norm: np.ndarray,
        m_s_val: int,
        m_t_val: int,
        device: torch.device,
    ) -> np.ndarray:
        src_t = torch.from_numpy(source_norm[np.newaxis, np.newaxis]).float().to(device)
        direction_id_val = 0 if m_t_val == _CT_IDX else 1  # 0 = ->CT, 1 = CT->source
        dir_t = torch.tensor([direction_id_val], dtype=torch.long, device=device)
        with torch.no_grad():
            out = self.model.model(src_t, dir_t)
        return out[0, 0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# Patient loading (no DataLoader — direct npz reads, R6)
# ---------------------------------------------------------------------------


def load_patient_arrays(entry: dict[str, Any], is_2025: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (source, ct, mask) full volumes for the patient's source modality."""
    if is_2025:
        source = np.load(entry["src_path"])["data"]
    else:
        source = np.load(entry["mr_path"])["data"]
    ct   = np.load(entry["ct_path"])["data"]
    mask = np.load(entry["mask_path"])["data"].astype(np.float32)
    return source, ct, mask


def gather_patients(
    manifest_2023: dict, splits_2023: dict,
    manifest_2025: dict, splits_2025: dict,
    direction: str, split: str = "test",
) -> list[tuple[str, bool]]:
    """(pid, is_2025) pairs whose source modality matches `direction`."""
    candidates: list[tuple[str, bool]] = []
    if direction == "mr_to_ct":
        for pid in sorted(splits_2023[split]):
            candidates.append((pid, False))
        for pid in sorted(splits_2025[split]):
            if manifest_2025[pid]["modality_src"] == "mr":
                candidates.append((pid, True))
    else:  # cbct_to_ct — SynthRAD2025 Task2 only
        for pid in sorted(splits_2025[split]):
            if manifest_2025[pid]["modality_src"] == "cbct":
                candidates.append((pid, True))
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global NUM_STEPS
    args = parse_args()
    NUM_STEPS = args.num_steps
    cfg = load_config(args.config)

    with open(args.manifest_2023) as f:
        manifest_2023 = json.load(f)
    with open(args.manifest_2025) as f:
        manifest_2025 = json.load(f)
    with open(args.splits_2023) as f:
        splits_2023 = json.load(f)
    with open(args.splits_2025) as f:
        splits_2025 = json.load(f)

    if args.model_type == "unet":
        # ── Build the LitBaseline UNet3D model (no bridge, no router/adapters).
        model = LitBaseline(
            cfg,
            manifest_path=args.manifest_2023,
            splits_path=args.splits_2023,
            manifest_2025_path=args.manifest_2025,
        )
    else:
        # ── Build the full model (denoiser + router/adapters + anisotropic op +
        # bridge), same assembly as train_thermobridge.py's build_model(). This
        # reads hidden_dim from cfg.model.denoiser.hidden_dim for the router —
        # NOT a hardcoded value — since build_model() already does so.
        model = build_model(cfg, args)

    print(f"Loading checkpoint (map_location='cpu'): {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  warning: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  warning: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    model = model.cuda()
    model.eval()
    if args.model_type == "unet":
        device = next(model.model.parameters()).device
    else:
        device = next(model.denoiser.parameters()).device

    if args.model_type == "unet":
        predictor = _UNetPredictor(model)
        predict_fn = lambda src, m_s, m_t: predictor.predict(src, m_s, m_t, device)
    else:
        predict_fn = lambda src, m_s, m_t: predict_patch(model, src, m_s, m_t, device)

    # ── Gather + limit patients per anatomy ──────────────────────────────
    candidates = gather_patients(manifest_2023, splits_2023, manifest_2025, splits_2025, args.direction)

    anatomy_to_pids: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for pid, is_2025 in candidates:
        entry = manifest_2025[pid] if is_2025 else manifest_2023[pid]
        anatomy_to_pids[entry["anatomy"]].append((pid, is_2025))

    selected: list[tuple[str, bool]] = []
    for anatomy in sorted(anatomy_to_pids):
        selected.extend(anatomy_to_pids[anatomy][: args.patients_per_anatomy])

    total = len(selected)
    print(f"\nEvaluating {total} patients (<= {args.patients_per_anatomy} per anatomy), "
          f"direction={args.direction}, center-crop {PATCH_SIZE}, num_steps={NUM_STEPS}.\n")

    # ── Run: primary (source->CT, HU) + reverse (CT->source, normalized) ──
    primary_rows: list[dict[str, Any]] = []
    reverse_rows: list[dict[str, Any]] = []

    for i, (pid, is_2025) in enumerate(selected, 1):
        entry = manifest_2025[pid] if is_2025 else manifest_2023[pid]
        anatomy = entry["anatomy"]
        try:
            if is_2025:
                m_src = _MR_IDX if entry["modality_src"] == "mr" else _CBCT_IDX
                src_params = entry.get("src_norm_params", {})
            else:
                m_src = _MR_IDX
                src_params = entry["mr_norm_params"]
            ct_params = entry["ct_norm_params"]

            source_full, ct_full, mask_full = load_patient_arrays(entry, is_2025)
            source_crop = center_crop(source_full, PATCH_SIZE)
            target_crop = center_crop(ct_full,     PATCH_SIZE)
            mask_crop   = center_crop(mask_full,   PATCH_SIZE)

            # Primary: source -> CT, HU-inverted, in-mask MAE (R2)
            pred_norm = predict_fn(source_crop, m_src, _CT_IDX)
            pred_hu   = invert_ct_to_hu(pred_norm,  ct_params)
            target_hu = invert_ct_to_hu(target_crop, ct_params)
            result    = compute_all_metrics(pred_hu, target_hu, mask_crop)

            print(f"[{i}/{total}] {pid} {anatomy} ({_MOD_NAME[m_src]}->ct) MAE={result.mae_hu:.2f} HU")
            primary_rows.append({
                "pid": pid, "anatomy": anatomy,
                "mae_hu": result.mae_hu, "psnr": result.psnr, "ssim": result.ssim,
            })

            # Also evaluate CT -> source direction (normalized MAE/PSNR/SSIM)
            pred_norm_rev = predict_fn(target_crop, _CT_IDX, m_src)
            if src_params:
                pred_orig   = invert_mr(pred_norm_rev, src_params)
                target_orig = invert_mr(source_crop,   src_params)
                result_rev  = compute_all_metrics(pred_orig, target_orig, mask_crop)
            else:
                result_rev = compute_all_metrics(pred_norm_rev, source_crop, mask_crop)
            reverse_rows.append({
                "pid": pid, "anatomy": anatomy,
                "mae": result_rev.mae_hu, "psnr": result_rev.psnr, "ssim": result_rev.ssim,
            })
        except Exception as exc:
            print(f"[{i}/{total}] {pid} {anatomy} FAILED: {exc}")

    # ── Summary: primary direction, per anatomy + overall, vs baselines ──
    print("\n" + "=" * 72)
    print(f"  QUICK EVAL SUMMARY — model_type={args.model_type} — {args.direction} — center-crop MAE-HU (mean ± std)")
    print("=" * 72)

    for anatomy in sorted({r["anatomy"] for r in primary_rows}):
        group = [r["mae_hu"] for r in primary_rows if r["anatomy"] == anatomy]
        print(f"  {anatomy:<8}  MAE-HU = {np.mean(group):>8.2f} ± {np.std(group):<6.2f}  n={len(group)}")

    if primary_rows:
        all_mae = [r["mae_hu"] for r in primary_rows]
        print("  " + "-" * 68)
        print(f"  {'ALL':<8}  MAE-HU = {np.mean(all_mae):>8.2f} ± {np.std(all_mae):<6.2f}  n={len(all_mae)}")

        print("\n  vs. baselines:")
        if args.model_type != "unet":
            print(f"    U-Net baseline    : {BASELINE_UNET_MAE_HU:>8.2f} HU")
        print(f"    mean-CT floor     : {BASELINE_MEAN_CT_MAE_HU:>8.2f} HU")
        model_label = "ThermoBridge" if args.model_type == "thermobridge" else "U-Net"
        print(f"    {model_label} (here): {np.mean(all_mae):>8.2f} HU")
    print("=" * 72)

    # ── Summary: reverse direction (CT -> source), normalized ────────────
    print(f"\n  CT->source — normalized MAE / PSNR / SSIM (mean ± std)")
    print("  " + "-" * 68)
    for anatomy in sorted({r["anatomy"] for r in reverse_rows}):
        group = [r for r in reverse_rows if r["anatomy"] == anatomy]
        maes  = [r["mae"]  for r in group]
        psnrs = [r["psnr"] for r in group]
        ssims = [r["ssim"] for r in group]
        print(f"  {anatomy:<8}  MAE={np.mean(maes):>8.4f}±{np.std(maes):<7.4f}  "
              f"PSNR={np.mean(psnrs):>6.2f}dB  SSIM={np.mean(ssims):>6.4f}  n={len(group)}")
    print()


if __name__ == "__main__":
    main()
