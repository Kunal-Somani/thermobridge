"""Visualize MRI input, GT CT, predicted CT, and error map for selected patients."""
import argparse, json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.utils.config import load_config
from src.data.preprocess import invert_ct_to_hu
from scripts.train_thermobridge import build_model

def center_crop(vol, patch_size=(96,96,96)):
    pZ,pY,pX = patch_size
    Z,Y,X = vol.shape
    z0=max(0,(Z-pZ)//2); y0=max(0,(Y-pY)//2); x0=max(0,(X-pX)//2)
    return vol[z0:z0+pZ, y0:y0+pY, x0:x0+pX]

def run_inference(model, mr_crop, device, num_steps=10):
    from src.training.lit_bridge import _DIRECTION_TO_MODALITIES, _MR_IDX, _CT_IDX
    src_t = torch.from_numpy(mr_crop[None,None]).float().to(device)
    m_s = torch.tensor([_MR_IDX], dtype=torch.long, device=device)
    m_t = torch.tensor([_CT_IDX], dtype=torch.long, device=device)
    A = model.router.num_anatomies if model.router is not None else 5
    alpha = torch.full((1,A), 1.0/A, device=device)
    with torch.no_grad():
        pred = model.bridge.reverse_sample(
            model.denoiser, src_t, m_s, m_t, alpha, num_steps=num_steps
        )
    return pred[0,0].float().cpu().numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--manifest-2023", type=Path, default=_REPO/"outputs/preprocessed/manifest.json")
    p.add_argument("--manifest-2025", type=Path, default=_REPO/"outputs/preprocessed_2025/manifest_2025.json")
    p.add_argument("--patients", nargs="+", default=["1ABA044","1BA022","1PA026"])
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=_REPO/"outputs/figures")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    # Load model
    import argparse as ap
    fake_args = ap.Namespace(manifest_2023=args.manifest_2023,
                              manifest_2025=args.manifest_2025,
                              splits_2023=_REPO/"outputs/splits.json")
    model = build_model(cfg, fake_args)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    device = next(model.denoiser.parameters()).device

    # Load manifests
    with open(args.manifest_2023) as f: m23 = json.load(f)
    with open(args.manifest_2025) as f: m25 = json.load(f)
    manifest = {**m23, **m25}

    patch = tuple(int(x) for x in cfg.patch.size)

    for pid in args.patients:
        if pid not in manifest:
            print(f"Patient {pid} not found"); continue
        entry = manifest[pid]
        mr = np.load(entry.get("mr_path") or entry.get("src_path"))["data"]
        ct = np.load(entry["ct_path"])["data"]
        mask = np.load(entry["mask_path"])["data"]
        ct_params = entry["ct_norm_params"]

        mr_crop = center_crop(mr, patch)
        ct_crop = center_crop(ct, patch)
        mask_crop = center_crop(mask, patch)

        pred_norm = run_inference(model, mr_crop, device, args.num_steps)
        pred_hu   = invert_ct_to_hu(np.clip(pred_norm, -1, 1), ct_params)
        gt_hu     = invert_ct_to_hu(ct_crop, ct_params)
        error = np.abs(pred_hu - gt_hu)
        error[mask_crop < 0.5] = 0

        # Plot middle axial slice
        z_mid = mr_crop.shape[0] // 2
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)

        axes[0].imshow(mr_crop[z_mid], cmap='gray')
        axes[0].set_title("MRI Input", fontweight='bold')
        axes[0].axis('off')

        vmin, vmax = -200, 1500
        axes[1].imshow(gt_hu[z_mid], cmap='gray', vmin=vmin, vmax=vmax)
        axes[1].set_title("Ground Truth CT", fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(np.clip(pred_hu[z_mid], vmin, vmax), cmap='gray', vmin=vmin, vmax=vmax)
        axes[2].set_title("ThermoBridge Prediction", fontweight='bold')
        axes[2].axis('off')

        err_img = axes[3].imshow(error[z_mid], cmap='hot', vmin=0, vmax=300)
        axes[3].set_title("Error Map |pred-GT| (HU)", fontweight='bold')
        axes[3].axis('off')
        plt.colorbar(err_img, ax=axes[3], fraction=0.046, pad=0.04, label='HU')

        anatomy = entry.get("anatomy", "unknown")
        fig.suptitle(f"Patient {pid} ({anatomy}) — MAE: {error[mask_crop>0.5].mean():.1f} HU",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = args.out_dir / f"prediction_{pid}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out} | MAE: {error[mask_crop>0.5].mean():.1f} HU")

if __name__ == "__main__":
    main()
