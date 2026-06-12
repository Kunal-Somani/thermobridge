"""ThermoBridge training entry point — baseline U-Net.

Usage::
    # Overfit smoke test (checks gradient wiring on 4 patients, ~50 steps)
    python src/training/train.py --config configs/overfit_smoke.yaml

    # Full training run
    python src/training/train.py --config configs/default.yaml

    # Resume from checkpoint
    python src/training/train.py --config configs/default.yaml --ckpt outputs/checkpoints/last.ckpt

Writes:
    outputs/runs/<run_id>/resolved_config.yaml  — frozen config dump (R6)
    outputs/checkpoints/best.ckpt               — best by val/mae_hu (min)
    outputs/checkpoints/last.ckpt               — always written
    outputs/runs/<run_id>/                       — CSV logs
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.datamodule import ThermoBridgeDataModule
from src.training.lit_baseline import LitBaseline
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ThermoBridge — train baseline U-Net.")
    p.add_argument("--config", required=True, type=Path,
                   help="Path to YAML config (e.g. configs/default.yaml).")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Resume from checkpoint path.")
    p.add_argument("--run-id", type=str, default=None,
                   help="Run identifier (auto-generated from timestamp if not given).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    pl.seed_everything(int(cfg.seed), workers=True)

    # ── Run directory + config dump (R6) ──────────────────────────────────
    run_id  = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _REPO_ROOT / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Dump resolved config for reproducibility
    from omegaconf import OmegaConf
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    print(f"Run dir: {run_dir}")

    # ── DataModule ───────────────────────────────────────────────────────
    overfit_n = 4 if ("smoke" in str(args.config) or "overfit" in str(args.config)) else None
    dm = ThermoBridgeDataModule(cfg, overfit_n=overfit_n)
    if overfit_n:
        print(f"⚡ OVERFIT SMOKE MODE: restricting to {overfit_n} patients")

    # ── Model ────────────────────────────────────────────────────────────
    model = LitBaseline(cfg)

    # ── Callbacks ────────────────────────────────────────────────────────
    ckpt_dir = _REPO_ROOT / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath   = str(ckpt_dir),
        filename  = "best",
        monitor   = str(cfg.training.ckpt_monitor),
        mode      = str(cfg.training.ckpt_mode),
        save_last = True,
        verbose   = True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_cb, lr_monitor]
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    # ── Logger ───────────────────────────────────────────────────────────
    logger = CSVLogger(save_dir=str(run_dir), name="logs", version=0)

    # ── Precision ────────────────────────────────────────────────────────
    precision = "16-mixed" if bool(cfg.training.mixed_precision) else "32"

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs          = int(cfg.training.max_epochs),
        precision           = precision,
        gradient_clip_val   = float(cfg.training.gradient_clip_norm),
        callbacks           = callbacks,
        logger              = logger,
        log_every_n_steps   = 10,
        num_sanity_val_steps= 0,      # val is expensive; skip sanity
        accelerator         = "auto",
        devices             = 1,
        deterministic       = False,  # True would block CuDNN; False is standard
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm, ckpt_path=str(args.ckpt) if args.ckpt else None)

    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val/mae_hu: {checkpoint_cb.best_model_score:.2f} HU")


if __name__ == "__main__":
    main()
