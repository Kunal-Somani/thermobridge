"""Test MAE-HU/PSNR/SSIM vs input noise level for ThermoBridge vs UNet."""
import argparse, json, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.utils.config import load_config
from src.data.preprocess import invert_ct_to_hu
from src.training.metrics import compute_all_metrics
from scripts.train_thermobridge import build_model
from scripts.visualize_predictions import center_crop, run_inference

def run_unet(model, mr_crop, device):
    src_t = torch.from_numpy(mr_crop[None,None]).float().to(device)
    dir_t = torch.tensor([0], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model.model(src_t, dir_t)
    return pred[0,0].float().cpu().numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tb-checkpoint", required=True, type=Path)
    p.add_argument("--unet-checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--manifest-2023", type=Path, default=_REPO/"outputs/preprocessed/manifest.json")
    p.add_argument("--splits", type=Path, default=_REPO/"outputs/splits.json")
    p.add_argument("--n-patients", type=int, default=5)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=_REPO/"outputs/figures")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]

    # Load ThermoBridge
    import argparse as ap
    fake = ap.Namespace(manifest_2023=args.manifest_2023,
                         manifest_2025=_REPO/"outputs/preprocessed_2025/manifest_2025.json",
                         splits_2023=args.splits)
    tb = build_model(cfg, fake)
    ckpt = torch.load(str(args.tb_checkpoint), map_location="cpu")
    tb.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    tb = tb.cuda().eval()
    tb_dev = next(tb.denoiser.parameters()).device

    # Load UNet
    from src.training.lit_baseline import LitBaseline
    unet = LitBaseline(cfg, manifest_path=args.manifest_2023, splits_path=args.splits)
    ckpt2 = torch.load(str(args.unet_checkpoint), map_location="cpu")
    unet.load_state_dict(ckpt2.get("state_dict", ckpt2), strict=False)
    unet = unet.cuda().eval()
    unet_dev = next(unet.model.parameters()).device

    with open(args.manifest_2023) as f: manifest = json.load(f)
    with open(args.splits) as f: splits = json.load(f)
    patch = tuple(int(x) for x in cfg.patch.size)
    val_ids = [p for p in splits["val"] if p in manifest][:args.n_patients]

    results = {m: {"tb_mae":[], "unet_mae":[], "tb_psnr":[], "unet_psnr":[]}
               for m in noise_levels}

    for pid in val_ids:
        entry = manifest[pid]
        mr = np.load(entry["mr_path"])["data"]
        ct = np.load(entry["ct_path"])["data"]
        mask = np.load(entry["mask_path"])["data"]
        ct_params = entry["ct_norm_params"]
        mr_crop = center_crop(mr, patch)
        ct_crop = center_crop(ct, patch)
        mask_crop = center_crop(mask, patch)
        gt_hu = invert_ct_to_hu(ct_crop, ct_params)

        for sigma in noise_levels:
            if sigma > 0:
                noisy_mr = mr_crop + np.random.normal(0, sigma, mr_crop.shape).astype(np.float32)
                noisy_mr = np.clip(noisy_mr, -1, 1)
            else:
                noisy_mr = mr_crop

            # ThermoBridge
            pred_tb = invert_ct_to_hu(run_inference(tb, noisy_mr, tb_dev, args.num_steps), ct_params)
            r_tb = compute_all_metrics(pred_tb, gt_hu, mask_crop)
            results[sigma]["tb_mae"].append(r_tb.mae_hu)
            results[sigma]["tb_psnr"].append(r_tb.psnr)

            # UNet
            pred_unet = invert_ct_to_hu(run_unet(unet, noisy_mr, unet_dev), ct_params)
            r_unet = compute_all_metrics(pred_unet, gt_hu, mask_crop)
            results[sigma]["unet_mae"].append(r_unet.mae_hu)
            results[sigma]["unet_psnr"].append(r_unet.psnr)

        print(f"Done: {pid}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    tb_mae = [np.mean(results[s]["tb_mae"]) for s in noise_levels]
    unet_mae = [np.mean(results[s]["unet_mae"]) for s in noise_levels]
    tb_psnr = [np.mean(results[s]["tb_psnr"]) for s in noise_levels]
    unet_psnr = [np.mean(results[s]["unet_psnr"]) for s in noise_levels]

    axes[0].plot(noise_levels, tb_mae, 'r-o', label='ThermoBridge', linewidth=2)
    axes[0].plot(noise_levels, unet_mae, 'b-o', label='U-Net', linewidth=2)
    axes[0].set_xlabel("Input Noise Level (σ)", fontsize=12)
    axes[0].set_ylabel("MAE-HU", fontsize=12)
    axes[0].set_title("MAE-HU vs Input Noise", fontsize=13, fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(noise_levels, tb_psnr, 'r-o', label='ThermoBridge', linewidth=2)
    axes[1].plot(noise_levels, unet_psnr, 'b-o', label='U-Net', linewidth=2)
    axes[1].set_xlabel("Input Noise Level (σ)", fontsize=12)
    axes[1].set_ylabel("PSNR (dB)", fontsize=12)
    axes[1].set_title("PSNR vs Input Noise", fontsize=13, fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = args.out_dir / "noise_robustness.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out}")

    # Print table
    print(f"\n{'Noise':>8} {'TB MAE':>10} {'UNet MAE':>10} {'TB PSNR':>10} {'UNet PSNR':>10}")
    for s in noise_levels:
        print(f"{s:>8.2f} {np.mean(results[s]['tb_mae']):>10.2f} "
              f"{np.mean(results[s]['unet_mae']):>10.2f} "
              f"{np.mean(results[s]['tb_psnr']):>10.2f} "
              f"{np.mean(results[s]['unet_psnr']):>10.2f}")

if __name__ == "__main__":
    main()
