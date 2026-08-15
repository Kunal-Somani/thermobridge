"""ThermoBridge training launch script — 3-D U-Net baseline (SynthRAD2023 only).

Usage::
    python scripts/train_unet_baseline.py --config configs/default.yaml \\
        --data-root-2023 data/synthrad2023/Task1 \\
        --manifest-2023 outputs/preprocessed/manifest.json \\
        --splits-2023 outputs/splits.json \\
        --out-dir outputs/runs --experiment-name unet_baseline_v1

    # Resume
    python scripts/train_unet_baseline.py --config configs/default.yaml --resume outputs/runs/unet_baseline_v1/checkpoints/last.ckpt

Writes:
    <out-dir>/<experiment-name>/resolved_config.yaml
    <out-dir>/<experiment-name>/checkpoints/{best_*.ckpt, last.ckpt}
    <out-dir>/<experiment-name>/logs/                 — CSV logs (no wandb)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.datamodule import ThermoBridgeDataModule
from src.training.lit_baseline import LitBaseline
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ThermoBridge — train U-Net baseline (SynthRAD2023 only).")
    p.add_argument("--config", required=True, type=Path, help="Path to YAML config (e.g. configs/default.yaml).")
    p.add_argument("--data-root-2023", type=Path, default=None, help="SynthRAD2023 data root (informational; paths come from manifest).")
    p.add_argument("--data-root-2025", type=Path, default=None, help="Unused by this baseline (SynthRAD2023 only); accepted for CLI symmetry.")
    p.add_argument("--manifest-2023", type=Path, default=None, help="Path to SynthRAD2023 preprocessed manifest.json.")
    p.add_argument("--manifest-2025", type=Path, default=None, help="Unused by this baseline; accepted for CLI symmetry.")
    p.add_argument("--splits-2023", type=Path, default=None, help="Path to SynthRAD2023 outputs/splits.json.")
    p.add_argument("--splits-2025", type=Path, default=None, help="Unused by this baseline; accepted for CLI symmetry.")
    p.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "outputs" / "runs", help="Root directory for run outputs.")
    p.add_argument("--experiment-name", type=str, default="unet_baseline", help="Experiment/run name (subdirectory of --out-dir).")
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from.")
    return p.parse_args()


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    pl.seed_everything(int(cfg.seed), workers=True)

    print(f"Git commit: {_git_commit_hash()}")
    print("=" * 80)
    print("Resolved config:")
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    run_dir = args.out_dir / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    print(f"Run dir: {run_dir}")

    # ── DataModule (SynthRAD2023 only) ──────────────────────────────────
    dm = ThermoBridgeDataModule(
        cfg,
        splits_path=args.splits_2023,
        manifest_path=args.manifest_2023,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = LitBaseline(cfg)

    # ── Callbacks ────────────────────────────────────────────────────────
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_{epoch:03d}_{val/mae_hu:.2f}",
        monitor="val/mae_hu",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_cb, lr_monitor]
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    # ── Logger (CSV only, no wandb) ─────────────────────────────────────
    logger = CSVLogger(save_dir=str(run_dir), name="logs", version=0)

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=1.0,
        callbacks=callbacks,
        logger=logger,
    )

    # ── Train ────────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm, ckpt_path=str(args.resume) if args.resume else None)

    print("\n" + "=" * 80)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    best_score = checkpoint_cb.best_model_score
    if best_score is not None:
        print(f"Best val/mae_hu: {float(best_score):.4f} HU")
    else:
        print("Best val/mae_hu: N/A (no validation metric recorded)")


if __name__ == "__main__":
    main()
