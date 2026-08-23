"""Plot train/val loss curves for ThermoBridge v4 and UNet from CSV logs."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

runs = {
    "ThermoBridge v4": "outputs/runs/thermobridge_v4/logs/version_0/metrics.csv",
    "UNet (combined)": "outputs/runs/unet_baseline_combined/logs/version_0/metrics.csv",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

colors = {"ThermoBridge v4": "#e74c3c", "UNet (combined)": "#2980b9"}

for name, csv_path in runs.items():
    p = Path(csv_path)
    if not p.exists():
        print(f"MISSING: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    color = colors[name]

    # Training loss
    train = df[df["train/loss_total_epoch"].notna()][["epoch","train/loss_total_epoch"]].dropna()
    if not train.empty:
        axes[0].plot(train["epoch"], train["train/loss_total_epoch"],
                     label=name, color=color, linewidth=1.5)

    # Val loss
    if "val/mae_hu" in df.columns:
        val = df[df["val/mae_hu"].notna()][["epoch","val/mae_hu"]].dropna()
        if not val.empty:
            axes[1].plot(val["epoch"], val["val/mae_hu"],
                         label=name, color=color, linewidth=1.5, marker='o', markersize=3)
    elif "val/loss_patch" in df.columns:
        val = df[df["val/loss_patch"].notna()][["epoch","val/loss_patch"]].dropna()
        if not val.empty:
            axes[1].plot(val["epoch"], val["val/loss_patch"],
                         label=name, color=color, linewidth=1.5, marker='o', markersize=3)

axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Training Loss", fontsize=12)
axes[0].set_title("Training Loss vs Epoch", fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Validation Metric", fontsize=12)
axes[1].set_title("Validation Loss vs Epoch", fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = out_dir / "training_curves.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved: {out}")
