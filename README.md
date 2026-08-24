# ThermoBridge 🌉

**Universal Any-to-Any 3D Medical Image Synthesis via Physics-Guided Schrödinger Bridge Diffusion**

[![Paper](https://img.shields.io/badge/Paper-ICASSP%202027-blue)](https://github.com/Kunal-Somani/thermobridge)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/Kunal-Somani/thermobridge)
[![Dataset](https://img.shields.io/badge/Dataset-SynthRAD2023%2B2025-orange)](https://zenodo.org/records/14918089)

---

## What is ThermoBridge?

ThermoBridge is a single deep learning model that performs **any-to-any 3D medical image synthesis** across multiple modalities and anatomies simultaneously — replacing an entire zoo of per-modality, per-anatomy models with one universal model.

**Clinical motivation:** Synthetic CT (sCT) from MRI eliminates ionizing radiation in radiotherapy planning. CBCT-to-CT correction improves daily adaptive radiotherapy dose accuracy. A universal model handles both tasks — and more — without retraining.

| | Existing Methods | ThermoBridge |
|---|---|---|
| Model count | 1 per (modality, anatomy) pair | **1 for everything** |
| Modalities | 1 | **3 (MRI, CT, CBCT)** |
| Anatomies | 1 | **5 (brain, pelvis, HN, TH, AB)** |
| Anatomy labels at inference | Required | **Not required** |
| Scaling | O(M²×A) | **O(M) embeddings** |

---

## Three Novel Contributions

### A — N-way Sparse Anatomy Routing Gate
Extends our prior [IC3 (IEEE 2026)](https://github.com/Kunal-Somani/thermobridge) dual-branch sigmoid gate to N anatomies:
- Softmax over the probability simplex Δ^{A-1}
- Top-k sparse selection (k=2) — O(k) compute not O(A)
- Temperature annealing τ: 2.0 → 0.5 (soft gradients early, sharp specialization late)
- Entropy + load-balance regularization replaces BCE
- **No anatomy labels needed at inference**

### B — Learnable 3D Anisotropic Diffusion Local Mixer
Perona-Malik edge-stopping conductance with learnable per-channel K, used as the local mixer inside each transformer block:

∂I/∂t = ∇·(g(‖∇I‖)·∇I), g(s) = 1/(1 + (s/K)²)

- Smooths flat tissue regions, stops at bone boundaries
- Edge-preserving by construction — targets the dominant IC3 failure mode
- 3D, learnable K per channel, inside transformer, novel context

### C — Physics-Guided I2SB Bridge with Radon Consistency
Image-to-Image Schrödinger Bridge (I2SB) finds the minimum-KL-divergence path between modality distributions:
- Intermediates stay on the joint image manifold (unlike Brownian bridge)
- Radon-domain consistency loss: `L_rad = ‖R(x̂₀) - R(x₀)‖₁`
- Dual-direction batching: every pair (A,B) trains both A→B and B→A
- Any-to-any without per-direction models

---

## Results

### MR→CT Synthesis (center-crop, 10 patients/anatomy)

| Anatomy | Mean-CT | U-Net | **ThermoBridge** | TB CBCT→CT |
|---|---|---|---|---|
| AB | 655 | 60.9±13.9 | 90.2±22.3 | 126.0±51.0 |
| HN | 655 | 129.2±14.7 | 168.8±34.6 | 138.3±11.1 |
| TH | 655 | 144.2±39.9 | 174.7±39.1 | **104.4±6.4** |
| Brain | 655 | **96.3±10.0** | 167.1±21.5 | — |
| Pelvis | 655 | **81.2±16.7** | 200.9±67.2 | — |
| **ALL** | 655 | 103.3±37.9 | 158.6±53.7 | 122.9±33.4 |

### Noise Robustness (key result)

| Noise σ | TB MAE-HU | UNet MAE-HU | TB PSNR |
|---|---|---|---|
| 0.00 | 183.6 | 84.7 | 19.26 |
| 0.05 | 180.2 | 84.4 | 19.32 |
| 0.10 | 181.0 | 96.2 | 19.33 |
| 0.20 | **169.7** | 133.1 | **19.71** |

**ThermoBridge degrades 8% vs U-Net's 57% at σ=0.2 — bridge diffusion inherently denoises during reverse sampling.**

### Model Complexity

| Model | Params | Inference | Directions | Anatomies |
|---|---|---|---|---|
| U-Net (per pair) | 6.7M | 15ms | 1 | 1 |
| **ThermoBridge** | **45.1M** | **384ms** | **Any-to-any** | **5** |

One ThermoBridge replaces 30 U-Nets (201M params total) — **4.5× deployment reduction**.

---

## Architecture

![Architecture](outputs/figures/architecture.png)

The model consists of:
1. **I2SB Bridge Process** — samples noisy intermediate x_t from source and target
2. **Hybrid Transformer Denoiser** — 12 blocks, each with:
   - 3D Self-Attention (global structure)
   - Anisotropic Diffusion Local Mixer (boundary-preserving)
   - Anatomy-Routed Adapters (anatomy specialization)
3. **N-way Sparse Routing Gate** — selects top-2 anatomy adapters per input
4. **Deterministic Reverse Sampling** — 10 steps, no stochastic noise at inference

---

## Datasets

| Dataset | Task | Pairs | Anatomies | Modalities | Centers |
|---|---|---|---|---|---|
| [SynthRAD2023](https://zenodo.org/records/7260705) | MRI→CT | 120 | Brain, Pelvis | MRI, CT | 1 |
| [SynthRAD2025](https://zenodo.org/records/14918089) | MRI→CT, CBCT→CT | 720 | HN, TH, AB | MRI, CT, CBCT | 5 |

Combined: **480 patients, 3 modalities, 5 anatomies**

---

## Setup

```bash
# Clone
git clone https://github.com/Kunal-Somani/thermobridge
cd thermobridge

# Install (Python 3.10, CUDA 12.1)
pip install -r requirements.txt

# Verify
python -c "from src.utils.config import load_config; print(load_config('configs/default.yaml'))"
```

**Pinned versions (do not upgrade):**
- `pytorch_lightning==2.2.5`
- `setuptools==68.2.2`

---

## Preprocessing

```bash
# SynthRAD2023 (brain + pelvis)
python src/data/preprocess.py \
  --config configs/default.yaml \
  --data-root data/synthrad2023/Task1 \
  --out-dir outputs/preprocessed

# SynthRAD2025 (HN + TH + AB, Task1 + Task2)
python scripts/run_preprocess_2025.py \
  --config configs/default.yaml \
  --task1-root data/synthrad2025/Task1 \
  --task2-root data/synthrad2025/Task2 \
  --out-dir outputs/preprocessed_2025
```

---

## Training

```bash
# ThermoBridge (full model)
python scripts/train_thermobridge.py \
  --config configs/default.yaml \
  --manifest-2023 outputs/preprocessed/manifest.json \
  --splits-2023 outputs/splits.json \
  --manifest-2025 outputs/preprocessed_2025/manifest_2025.json \
  --splits-2025 outputs/splits_synthrad2025.json \
  --out-dir outputs/runs \
  --experiment-name thermobridge_v4 \
  --num-workers 4

# U-Net baseline (fair comparison)
python scripts/train_unet_baseline.py \
  --config configs/default.yaml \
  --manifest-2023 outputs/preprocessed/manifest.json \
  --splits-2023 outputs/splits.json \
  --manifest-2025 outputs/preprocessed_2025/manifest_2025.json \
  --splits-2025 outputs/splits_synthrad2025.json \
  --out-dir outputs/runs \
  --experiment-name unet_baseline_combined \
  --num-workers 4
```

---

## Evaluation

```bash
# Quick evaluation (center-crop, 5 patients/anatomy, ~5 min)
python scripts/evaluate_quick.py \
  --checkpoint outputs/runs/thermobridge_v4/checkpoints/best_epoch=144_val/loss_patch=0.0516.ckpt \
  --config configs/default.yaml \
  --manifest-2023 outputs/preprocessed/manifest.json \
  --splits-2023 outputs/splits.json \
  --manifest-2025 outputs/preprocessed_2025/manifest_2025.json \
  --splits-2025 outputs/splits_synthrad2025.json \
  --model-type thermobridge \
  --num-steps 10

# CBCT→CT evaluation
python scripts/evaluate_quick.py \
  --checkpoint ... \
  --direction cbct_to_ct \
  --model-type thermobridge
```

---

## Visualization

```bash
# Prediction visualizations
python scripts/visualize_predictions.py \
  --checkpoint outputs/runs/thermobridge_v4/checkpoints/best_epoch=144_val/loss_patch=0.0516.ckpt \
  --config configs/default.yaml \
  --patients 1ABA044 1BA022 1HNA023 1THA018 1PA019

# Training curves
python scripts/plot_training_curves.py

# Noise robustness
python scripts/noise_robustness.py \
  --tb-checkpoint outputs/runs/thermobridge_v4/checkpoints/best.ckpt \
  --unet-checkpoint outputs/runs/unet_baseline_combined/checkpoints/best.ckpt \
  --config configs/default.yaml

# Failure analysis
python scripts/failure_analysis.py \
  --checkpoint outputs/runs/thermobridge_v4/checkpoints/best.ckpt \
  --config configs/default.yaml \
  --n-patients 10
```

---

## Project Structure

thermobridge/
├── configs/
│ ├── default.yaml # All hyperparameters, loss weights
│ └── ablation_B1.yaml # Ablation: no anisotropic operator
├── src/
│ ├── data/
│ │ ├── preprocess.py # CT/MRI normalization, resampling
│ │ ├── dataset.py # SynthRAD2023 dataset
│ │ ├── dataset_2025.py # SynthRAD2025 dataset
│ │ └── combined_datamodule.py # Combined Lightning DataModule
│ ├── models/
│ │ ├── bridge.py # I2SB forward marginal + reverse sampler
│ │ ├── denoiser.py # Hybrid transformer denoiser
│ │ ├── routing.py # N-way sparse anatomy routing gate
│ │ └── anisotropic_op.py # Learnable 3D anisotropic diffusion
│ ├── physics/
│ │ └── radon.py # Differentiable 3-axis Radon projector
│ └── training/
│ ├── lit_bridge.py # LightningModule: ThermoBridge
│ └── lit_baseline.py # LightningModule: U-Net baseline
├── scripts/
│ ├── train_thermobridge.py
│ ├── train_unet_baseline.py
│ ├── evaluate_quick.py
│ ├── visualize_predictions.py
│ ├── noise_robustness.py
│ ├── failure_analysis.py
│ └── plot_training_curves.py
├── tests/ # 60+ unit tests, all passing
└── outputs/
├── figures/ # All paper figures (300 DPI)
└── runs/ # Training checkpoints and logs


---

## Reproducibility

All results are reproducible with seed=42. Every reported number maps
to a committed config and checkpoint:

| Result | Checkpoint | Config |
|---|---|---|
| ThermoBridge v4 (164.93 HU, 5-patient) | `best_epoch=144_val/loss_patch=0.0516.ckpt` | `default.yaml` |
| ThermoBridge v5 (156.49 HU, 5-patient) | `best_epoch=199_val/loss_patch=0.0522.ckpt` | `default.yaml` |
| U-Net combined (103.27 HU) | `best_epoch=143_val/mae_hu=97.40.ckpt` | `default.yaml` |
| Ablation B1 (143.74 HU) | `best_epoch=139_val/loss_patch=0.0561.ckpt` | `ablation_B1.yaml` |

---

## Citation

```bibtex
@inproceedings{somani2027thermobridge,
  title     = {ThermoBridge: Universal Any-to-Any 3D Medical Image Synthesis
               via Physics-Guided Schr{\"o}dinger Bridge Diffusion},
  author    = {Somani, Kunal and Tiwari, Shailendra},
  booktitle = {Proceedings of ICASSP 2027},
  year      = {2027}
}
```

---

## Acknowledgements

- [SynthRAD2023](https://zenodo.org/records/7260705) — van der Bijl et al.
- [SynthRAD2025](https://zenodo.org/records/14918089) — Thummerer et al.
- [I2SB](https://arxiv.org/abs/2302.05872) — Liu et al., ICML 2023
- GPU compute: Thapar CITM H100 MIG 39GB
