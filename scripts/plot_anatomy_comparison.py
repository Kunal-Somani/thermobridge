"""Bar chart comparing ThermoBridge vs UNet per anatomy — like IC3 anatomy_split figure."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

anatomies = ["AB", "HN", "TH", "Brain", "Pelvis", "ALL"]

# ThermoBridge v4 MR->CT (10 patients)
tb_mr = [90.2, 168.8, 174.7, 167.1, 200.9, 158.6]
tb_mr_std = [22.3, 34.6, 39.1, 21.5, 67.2, 53.7]

# UNet MR->CT (10 patients)
unet_mr = [60.9, 129.2, 144.2, 96.3, 81.2, 103.3]
unet_mr_std = [13.9, 14.7, 39.9, 10.0, 16.7, 37.9]

# ThermoBridge CBCT->CT
tb_cbct = [126.0, 138.3, 104.4, None, None, 122.9]
tb_cbct_std = [51.0, 11.1, 6.4, None, None, 33.4]

# Mean-CT floor
mean_ct = [655, 655, 655, 655, 655, 655]

x = np.arange(len(anatomies))
width = 0.22

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

bars1 = ax.bar(x - 1.5*width, unet_mr, width,
               label='U-Net MR→CT', color='#2980b9',
               yerr=unet_mr_std, capsize=4, error_kw={'linewidth':1.2})
bars2 = ax.bar(x - 0.5*width, tb_mr, width,
               label='ThermoBridge MR→CT', color='#e74c3c',
               yerr=tb_mr_std, capsize=4, error_kw={'linewidth':1.2})

# CBCT bars (skip None)
cbct_vals = [v if v is not None else 0 for v in tb_cbct]
cbct_stds = [v if v is not None else 0 for v in tb_cbct_std]
cbct_mask = [v is not None for v in tb_cbct]
bars3 = ax.bar(x + 0.5*width, cbct_vals, width,
               label='ThermoBridge CBCT→CT', color='#27ae60',
               yerr=cbct_stds, capsize=4, error_kw={'linewidth':1.2},
               alpha=0.9)
# Zero out bars where no data
for i, (bar, has_data) in enumerate(zip(bars3, cbct_mask)):
    if not has_data:
        bar.set_height(0)
        bar.set_alpha(0)

# Mean-CT floor as horizontal dashed line
ax.axhline(y=655, color='gray', linestyle='--', linewidth=1.5,
           label='Mean-CT floor (655 HU)', alpha=0.7)

ax.set_xlabel('Anatomy', fontsize=13)
ax.set_ylabel('MAE-HU (lower is better)', fontsize=13)
ax.set_title('Per-Anatomy MAE-HU: ThermoBridge vs U-Net\n'
             '(10 patients/anatomy, center-crop 96³, 10 reverse steps)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(anatomies, fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim(0, 280)
ax.grid(axis='y', alpha=0.3)
ax.yaxis.set_minor_locator(plt.MultipleLocator(25))

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x()+bar.get_width()/2, h+3, f'{h:.0f}',
                ha='center', va='bottom', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x()+bar.get_width()/2, h+3, f'{h:.0f}',
                ha='center', va='bottom', fontsize=8)
for bar, has_data in zip(bars3, cbct_mask):
    h = bar.get_height()
    if h > 0 and has_data:
        ax.text(bar.get_x()+bar.get_width()/2, h+3, f'{h:.0f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
out = out_dir / "anatomy_comparison.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved: {out}")
