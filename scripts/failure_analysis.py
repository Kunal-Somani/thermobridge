"""Per-slice failure analysis — find worst slices, plot error maps."""
import argparse, json, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.utils.config import load_config
from src.data.preprocess import invert_ct_to_hu
from scripts.train_thermobridge import build_model
from scripts.visualize_predictions import center_crop, run_inference

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--manifest-2023", type=Path, default=_REPO/"outputs/preprocessed/manifest.json")
    p.add_argument("--manifest-2025", type=Path, default=_REPO/"outputs/preprocessed_2025/manifest_2025.json")
    p.add_argument("--splits", type=Path, default=_REPO/"outputs/splits.json")
    p.add_argument("--n-patients", type=int, default=10)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=_REPO/"outputs/figures/failure_analysis")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    import argparse as ap
    fake_args = ap.Namespace(manifest_2023=args.manifest_2023,
                              manifest_2025=args.manifest_2025,
                              splits_2023=args.splits)
    model = build_model(cfg, fake_args)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    device = next(model.denoiser.parameters()).device

    with open(args.manifest_2023) as f: m23 = json.load(f)
    with open(args.manifest_2025) as f: m25 = json.load(f)
    with open(args.splits) as f: splits = json.load(f)
    manifest = {**m23, **m25}
    patch = tuple(int(x) for x in cfg.patch.size)

    val_ids = splits.get("val", [])[:args.n_patients]
    summary = []

    for pid in val_ids:
        if pid not in manifest: continue
        entry = manifest[pid]
        mr = np.load(entry.get("mr_path") or entry.get("src_path"))["data"]
        ct = np.load(entry["ct_path"])["data"]
        mask = np.load(entry["mask_path"])["data"]
        ct_params = entry["ct_norm_params"]

        mr_crop = center_crop(mr, patch)
        ct_crop = center_crop(ct, patch)
        mask_crop = center_crop(mask, patch)

        pred_norm = run_inference(model, mr_crop, device, args.num_steps)
        pred_hu = invert_ct_to_hu(pred_norm, ct_params)
        gt_hu = invert_ct_to_hu(ct_crop, ct_params)
        error = np.abs(pred_hu - gt_hu)

        # Per-slice MAE
        slice_mae = []
        for z in range(error.shape[0]):
            m = mask_crop[z] > 0.5
            if m.sum() > 100:
                slice_mae.append((error[z][m].mean(), z))

        if not slice_mae: continue
        slice_mae.sort(reverse=True)
        overall_mae = error[mask_crop>0.5].mean()
        summary.append((pid, entry.get("anatomy","?"), overall_mae))

        # Plot worst 3 slices
        worst = slice_mae[:3]
        fig, axes = plt.subplots(len(worst), 3, figsize=(12, 4*len(worst)), dpi=300)
        if len(worst) == 1: axes = axes[np.newaxis,:]

        for row, (mae_val, z) in enumerate(worst):
            axes[row,0].imshow(gt_hu[z], cmap='gray', vmin=-200, vmax=1500)
            axes[row,0].set_title(f"GT CT (slice {z})", fontweight='bold')
            axes[row,0].axis('off')

            axes[row,1].imshow(pred_hu[z], cmap='gray', vmin=-200, vmax=1500)
            axes[row,1].set_title(f"Predicted CT", fontweight='bold')
            axes[row,1].axis('off')

            # Error with threshold overlay
            err_display = error[z].copy()
            im = axes[row,2].imshow(gt_hu[z], cmap='gray', vmin=-200, vmax=1500)
            overlay = np.zeros((*err_display.shape, 4))
            high_err = err_display > 150  # highlight >150 HU error in red
            overlay[high_err] = [1, 0, 0, 0.6]
            axes[row,2].imshow(overlay)
            axes[row,2].set_title(f"Error overlay (red>150HU) | MAE={mae_val:.1f}HU",
                                   fontweight='bold')
            axes[row,2].axis('off')

        fig.suptitle(f"{pid} ({entry.get('anatomy','?')}) — Overall MAE: {overall_mae:.1f} HU",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = args.out_dir / f"failure_{pid}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{pid}: overall MAE={overall_mae:.1f} HU, worst slice={worst[0][0]:.1f} HU")

    # Print summary table
    print("\n=== FAILURE ANALYSIS SUMMARY ===")
    print(f"{'Patient':<12} {'Anatomy':<8} {'MAE-HU':>8}")
    print("-" * 32)
    for pid, anat, mae in sorted(summary, key=lambda x: -x[2]):
        print(f"{pid:<12} {anat:<8} {mae:>8.1f}")

if __name__ == "__main__":
    main()
