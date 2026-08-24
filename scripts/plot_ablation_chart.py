"""Bar chart for ablation study results."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

models = ['Mean-CT\nFloor', 'TB w/o\nAnisotropic Op\n(B1)', 
          'ThermoBridge\nv4 (Full)', 'U-Net\n(Fair Baseline)']
mae_hu = [655.0, 143.7, 158.6, 103.3]
std = [38.2, 36.7, 53.7, 37.9]
colors = ['#95a5a6', '#e67e22', '#e74c3c', '#2980b9']

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

bars = ax.bar(models, mae_hu, color=colors, width=0.5,
              yerr=std, capsize=6, error_kw={'linewidth':1.5},
              edgecolor='black', linewidth=0.8)

# Value labels
for bar, val, s in zip(bars, mae_hu, std):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+8,
            f'{val:.1f} HU', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

ax.set_ylabel('MAE-HU (lower is better)', fontsize=13)
ax.set_title('Ablation Study — Effect of Architecture Components\n'
             'ALL Anatomies, Center-Crop 96³, 10 Reverse Steps',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 750)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=655, color='gray', linestyle='--',
           linewidth=1.2, alpha=0.7, label='Mean-CT floor')

# Annotate improvement arrows
ax.annotate('', xy=(2, 158.6), xytext=(1, 143.7),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(1.5, 155, '+15 HU\n(op adds\ncomplexity)', ha='center',
        fontsize=9, color='black')

ax.legend(fontsize=11)
plt.tight_layout()
out = out_dir / "ablation_chart.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved: {out}")
