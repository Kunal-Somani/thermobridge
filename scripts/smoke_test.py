"""ThermoBridge smoke test — 1-epoch sanity check for the full training pipeline.

Patches the loaded config to use the smallest possible settings (num_workers=0,
max_epochs=1, batch_size=1, patch_size=[32,32,32]) so the test completes quickly
on any GPU without touching configs/default.yaml.

Usage::
    python scripts/smoke_test.py \\
        --config configs/default.yaml \\
        --manifest-2023 outputs/preprocessed/manifest.json \\
        --splits-2023 outputs/splits.json \\
        --manifest-2025 outputs/preprocessed_2025/manifest_2025.json \\
        --splits-2025 outputs/splits_synthrad2025.json

Exits 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

from src.data.combined_datamodule import CombinedDataModule
from src.models.anisotropic_op import AnisotropicDiffusionOp, ConvLocalMixer
from src.models.routing import RoutedAdapterBlock
from src.training.lit_bridge import LitBridge
from src.utils.config import load_config

# Expected loss keys that must appear in trainer.callback_metrics after one epoch
_EXPECTED_LOSS_KEYS = [
    "train/loss_rec",
    "train/loss_ssim",
    "train/loss_bnd",
    "train/loss_rad",
    "train/loss_ent",
    "train/loss_bal",
    "train/loss_cls",
    "train/loss_total",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ThermoBridge smoke test — 1-epoch sanity check (exits 0=PASS, 1=FAIL)."
    )
    p.add_argument("--config", required=True, type=Path,
                   help="Path to YAML config (e.g. configs/default.yaml).")
    p.add_argument("--manifest-2023", type=Path, default=None)
    p.add_argument("--splits-2023", type=Path, default=None)
    p.add_argument("--manifest-2025", type=Path, default=None)
    p.add_argument("--splits-2025", type=Path, default=None)
    p.add_argument("--accelerator", type=str, default="auto",
                   help="Lightning accelerator (auto|gpu|cpu). Default: auto.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Callback: captures the loss dict after each training step
# ---------------------------------------------------------------------------


class _LossCapture(Callback):
    """Stores a copy of the logged metrics after each training step."""

    def __init__(self) -> None:
        self.step_metrics: list[dict] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.step_metrics.append(dict(trainer.callback_metrics))


# ---------------------------------------------------------------------------
# Build model (mirrors build_model() in train_thermobridge.py)
# ---------------------------------------------------------------------------


def _build_model(cfg, args: argparse.Namespace) -> LitBridge:
    lit_kwargs: dict = {}
    if args.manifest_2023 is not None:
        lit_kwargs["manifest_path"] = args.manifest_2023
    if args.splits_2023 is not None:
        lit_kwargs["splits_path"] = args.splits_2023
    if args.manifest_2025 is not None:
        lit_kwargs["manifest_2025_path"] = args.manifest_2025
    lit = LitBridge(cfg, **lit_kwargs)
    denoiser = lit.denoiser

    # Anatomy routing
    routing_cfg = cfg.model.routing
    router = lit.router
    from torch import nn
    adapter_blocks = [
        RoutedAdapterBlock(
            dim=int(cfg.model.denoiser.hidden_dim),
            num_anatomies=int(routing_cfg.num_anatomies),
            adapter_rank=int(routing_cfg.adapter_rank),
        )
        for _ in range(int(cfg.model.denoiser.num_layers))
    ]
    denoiser.set_adapters(router, nn.ModuleList(adapter_blocks))

    # Local mixer
    aniso_cfg = cfg.model.anisotropic_op
    if bool(aniso_cfg.use_anisotropic_op):
        local_mixer = AnisotropicDiffusionOp(
            num_channels=int(cfg.model.denoiser.hidden_dim),
            num_steps=int(aniso_cfg.num_steps),
            per_channel_k=bool(aniso_cfg.per_channel_k),
            init_conductance_k=float(aniso_cfg.init_conductance_k),
            init_step_size=float(aniso_cfg.init_step_size),
        )
    else:
        local_mixer = ConvLocalMixer(num_channels=int(cfg.model.denoiser.hidden_dim))
    denoiser.set_local_mixer(local_mixer)

    return lit


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_loss_finite_and_positive(metrics: list[dict]) -> list[str]:
    """Return a list of failure messages (empty = all OK)."""
    failures: list[str] = []
    if not metrics:
        failures.append("No training steps were logged (DataLoader may be empty).")
        return failures
    last = metrics[-1]
    for key in _EXPECTED_LOSS_KEYS:
        if key not in last:
            failures.append(f"Missing metric key: {key!r}")
            continue
        val = float(last[key])
        if not math.isfinite(val):
            failures.append(f"{key} = {val} (not finite)")
    total_key = "train/loss_total"
    if total_key in last:
        total = float(last[total_key])
        if math.isfinite(total) and total <= 0.0:
            failures.append(f"{total_key} = {total:.6f} (must be > 0)")
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # ── Patch config for fast smoke run (no YAML edit) ───────────────────
    OmegaConf.update(cfg, "training.num_workers",       0,          merge=True)
    OmegaConf.update(cfg, "training.max_epochs",        1,          merge=True)
    OmegaConf.update(cfg, "training.batch_size",        1,          merge=True)
    OmegaConf.update(cfg, "training.samples_per_volume", 1,         merge=True)
    OmegaConf.update(cfg, "patch.size",                 [32, 32, 32], merge=True)

    print("=" * 70)
    print("  ThermoBridge SMOKE TEST")
    print(f"  config:      {args.config}")
    print(f"  patch_size:  {list(cfg.patch.size)}")
    print(f"  batch_size:  {cfg.training.batch_size}")
    print(f"  max_epochs:  {cfg.training.max_epochs}")
    print(f"  num_workers: {cfg.training.num_workers}")
    print("=" * 70)

    pl.seed_everything(0, workers=True)

    # ── DataModule ────────────────────────────────────────────────────────
    dm = CombinedDataModule(
        cfg,
        splits_path=args.splits_2023,
        manifest_path=args.manifest_2023,
        synthrad2025_splits_path=args.splits_2025,
        manifest_2025_path=args.manifest_2025,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = _build_model(cfg, args)

    # ── Trainer (minimal: 1 epoch, 1 val step, no checkpointing) ─────────
    loss_capture = _LossCapture()

    with tempfile.TemporaryDirectory() as tmpdir:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(save_dir=tmpdir, name="smoke", version=0)

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator=args.accelerator,
            devices=1,
            precision="16-mixed",
            gradient_clip_val=1.0,
            log_every_n_steps=1,       # log every step so we capture metrics
            check_val_every_n_epoch=1, # run validation at end of the single epoch
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[loss_capture],
            logger=logger,
            limit_train_batches=1,     # exactly 1 training step
            limit_val_batches=1,       # exactly 1 validation step
        )

        trainer.fit(model, datamodule=dm)

    # ── Verify ────────────────────────────────────────────────────────────
    failures = _check_loss_finite_and_positive(loss_capture.step_metrics)

    # Also check val/loss_patch was logged
    val_key = "val/loss_patch"
    if val_key in trainer.callback_metrics:
        val_loss = float(trainer.callback_metrics[val_key])
        if not math.isfinite(val_loss):
            failures.append(f"{val_key} = {val_loss} (not finite)")
        else:
            print(f"  val/loss_patch = {val_loss:.6f}  ✓")
    else:
        failures.append(f"Missing validation metric: {val_key!r}")

    if loss_capture.step_metrics:
        last = loss_capture.step_metrics[-1]
        for key in _EXPECTED_LOSS_KEYS:
            if key in last:
                print(f"  {key:<30} = {float(last[key]):.6f}  ✓")

    print()
    if failures:
        print("SMOKE TEST FAILED")
        for msg in failures:
            print(f"  ✗ {msg}")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
