"""Dedicated noise chart showing PSNR and MAE separately — cleaner than combined."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
noise_labels = ['0\n(Clean)', '0.01', '0.05', '0.10', '0.20']

tb_mae = [183.58, 183.82, 180.19, 180.95, 169.67]
unet_mae = [84.70, 84.59, 84.35, 96.22, 133.14]
tb_psnr = [19.26, 19.25, 19.32, 19.33, 19.71]
unet_psnr = [24.46, 24.48, 24.63, 23.97, 21.97]

# Percentage degradation
tb_pct = [(v - tb_mae[0])/tb_mae[0]*100 for v in tb_mae]
unet_pct = [(v - unet_mae[0])/unet_mae[0]*100 for v in unet_mae]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

# Plot 1: MAE-HU
axes[0].plot(noise_levels, tb_mae, 'r-o', linewidth=2.5,
             markersize=8, label='ThermoBridge', zorder=5)
axes[0].plot(noise_levels, unet_mae, 'b-s', linewidth=2.5,
             markersize=8, label='U-Net')
axes[0].fill_between(noise_levels, tb_mae, alpha=0.1, color='red')
axes[0].fill_between(noise_levels, unet_mae, alpha=0.1, color='blue')
axes[0].set_xlabel('Input Noise Level (σ)', fontsize=12)
axes[0].set_ylabel('MAE-HU', fontsize=12)
axes[0].set_title('MAE-HU vs Input Noise', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(noise_levels)
axes[0].set_xticklabels(noise_labels)

# Plot 2: PSNR
axes[1].plot(noise_levels, tb_psnr, 'r-o', linewidth=2.5,
             markersize=8, label='ThermoBridge', zorder=5)
axes[1].plot(noise_levels, unet_psnr, 'b-s', linewidth=2.5,
             markersize=8, label='U-Net')
axes[1].set_xlabel('Input Noise Level (σ)', fontsize=12)
axes[1].set_ylabel('PSNR (dB)', fontsize=12)
axes[1].set_title('PSNR vs Input Noise', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(noise_levels)
axes[1].set_xticklabels(noise_labels)

# Plot 3: % degradation
axes[2].plot(noise_levels, tb_pct, 'r-o', linewidth=2.5,
             markersize=8, label=f'ThermoBridge: {tb_pct[-1]:.1f}%', zorder=5)
axes[2].plot(noise_levels, unet_pct, 'b-s', linewidth=2.5,
             markersize=8, label=f'U-Net: {unet_pct[-1]:.1f}%')
axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=1)
axes[2].fill_between(noise_levels, tb_pct, 0, alpha=0.1, color='red')
axes[2].fill_between(noise_levels, unet_pct, 0, alpha=0.1, color='blue')
axes[2].set_xlabel('Input Noise Level (σ)', fontsize=12)
axes[2].set_ylabel('MAE-HU Change (%)', fontsize=12)
axes[2].set_title('% MAE Degradation vs Input Noise\n(ThermoBridge: 8% vs U-Net: 57%)',
                  fontsize=12, fontweight='bold')
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(noise_levels)
axes[2].set_xticklabels(noise_labels)

plt.suptitle('Noise Robustness Analysis: ThermoBridge vs U-Net',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
out = out_dir / "noise_chart.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved: {out}")
