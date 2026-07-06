# ThermoBridge

**Bidirectional 3D MRI↔CT synthesis for radiotherapy synthetic CT generation.** A thermodynamically grounded Brownian-bridge diffusion process that transports
directly between paired MRI and CT volumes in both directions, using a hybrid
transformer denoiser whose local mixer is a novel learnable 3D anisotropic-diffusion
operator (edge-preserving by construction). Evaluated cross-organ on brain and pelvis
with deep bone-boundary failure analysis. Target venue: MICCAI.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (Phase-0)
pip install -r requirements.txt

# Verify installation
python -c "from src.utils.config import load_config; print(load_config('configs/default.yaml'))"
```

## Project Structure

```
configs/          # Experiment configs (default.yaml)
src/              # Source code
  data/           # Dataset, preprocessing, dataloaders
  models/         # Bridge, denoiser, anisotropic operator
  training/       # Training loops, loss, schedulers
  utils/          # Config, seeding, logging utilities
tests/            # Unit and integration tests
scripts/          # Thin CLI entrypoints
outputs/          # Runtime outputs (gitignored)
  figures/        # Publication-quality plots (≥300 DPI)
  reports/        # EDA reports, metric tables
  checkpoints/    # Model checkpoints
data/             # SynthRAD2023 dataset (read-only, gitignored)
```

## Usage

```bash
# Lint
ruff check .

# Format
ruff format .

# Run tests
pytest
```
