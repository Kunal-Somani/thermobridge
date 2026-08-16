"""ThermoBridge training launch script — full model (denoiser + routing + anisotropic op + bridge).

Usage::
    python scripts/train_thermobridge.py --config configs/default.yaml \\
        --data-root-2023 data/synthrad2023/Task1 \\
        --data-root-2025 data/synthrad2025 \\
        --manifest-2023 outputs/preprocessed/manifest.json \\
        --manifest-2025 outputs/preprocessed_2025/manifest_2025.json \\
        --splits-2023 outputs/splits.json \\
        --splits-2025 outputs/splits_synthrad2025.json \\
        --out-dir outputs/runs --experiment-name thermobridge_v1

    # Resume
    python scripts/train_thermobridge.py --config configs/default.yaml --resume outputs/runs/thermobridge_v1/checkpoints/last.ckpt

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
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.combined_datamodule import CombinedDataModule
from src.models.anisotropic_op import AnisotropicDiffusionOp, ConvLocalMixer
from src.models.build import build_denoiser
from src.models.routing import AnatomyAdapter, AnatomyRouter, RoutedAdapterBlock
from src.training.lit_bridge import LitBridge
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ThermoBridge — train full model (routing + anisotropic op + I2SB bridge).")
    p.add_argument("--config", required=True, type=Path, help="Path to YAML config (e.g. configs/default.yaml).")
    p.add_argument("--data-root-2023", type=Path, default=None, help="SynthRAD2023 data root (informational; paths come from manifest).")
    p.add_argument("--data-root-2025", type=Path, default=None, help="SynthRAD2025 data root (informational; paths come from manifest).")
    p.add_argument("--manifest-2023", type=Path, default=None, help="Path to SynthRAD2023 preprocessed manifest.json.")
    p.add_argument("--manifest-2025", type=Path, default=None, help="Path to SynthRAD2025 preprocessed manifest_2025.json.")
    p.add_argument("--splits-2023", type=Path, default=None, help="Path to SynthRAD2023 outputs/splits.json.")
    p.add_argument("--splits-2025", type=Path, default=None, help="Path to SynthRAD2025 challenge splits JSON.")
    p.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "outputs" / "runs", help="Root directory for run outputs.")
    p.add_argument("--experiment-name", type=str, default="thermobridge", help="Experiment/run name (subdirectory of --out-dir).")
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from.")
    return p.parse_args()


class RouterTauScheduleCallback(Callback):
    """Anneals AnatomyRouter.tau each epoch per its own tau_schedule (§5)."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: LitBridge) -> None:
        router = pl_module.denoiser.router
        if router is not None:
            router.tau = router.tau_schedule(trainer.current_epoch)


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_model(cfg, args: argparse.Namespace) -> LitBridge:
    """Assemble ThermoBridgeDenoiser + AnatomyRouter/adapters + AnisotropicDiffusionOp + I2SBProcess."""
    lit_kwargs: dict = {}
    if args.manifest_2023 is not None:
        lit_kwargs["manifest_path"] = args.manifest_2023
    if args.splits_2023 is not None:
        lit_kwargs["splits_path"] = args.splits_2023
    if args.manifest_2025 is not None:
        lit_kwargs["manifest_2025_path"] = args.manifest_2025
    lit = LitBridge(cfg, **lit_kwargs)
    denoiser = lit.denoiser

    # --- Anatomy routing (§5) ---
    routing_cfg = cfg.model.routing
    router = AnatomyRouter(
        in_channels=1,
        hidden_dim=int(cfg.model.denoiser.hidden_dim),
        num_anatomies=int(routing_cfg.num_anatomies),
        top_k=int(routing_cfg.top_k),
        adapter_rank=int(routing_cfg.adapter_rank),
        tau_max=float(routing_cfg.tau_max),
        tau_min=float(routing_cfg.tau_min),
        total_epochs=int(cfg.training.max_epochs),
    )
    adapter_blocks = [
        RoutedAdapterBlock(
            dim=int(cfg.model.denoiser.hidden_dim),
            num_anatomies=int(routing_cfg.num_anatomies),
            adapter_rank=int(routing_cfg.adapter_rank),
        )
        for _ in range(int(cfg.model.denoiser.num_layers))
    ]
    from torch import nn
    denoiser.set_adapters(router, nn.ModuleList(adapter_blocks))

    # --- Local mixer: anisotropic diffusion op (§6) or plain conv ablation ---
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    pl.seed_everything(int(cfg.seed), workers=True)

    print(f"Git commit: {_git_commit_hash()}")
    from omegaconf import OmegaConf
    print(f"Experiment: {args.experiment_name} | patch={list(cfg.patch.size)} | batch={cfg.training.batch_size} | epochs={cfg.training.max_epochs}")

    run_dir = args.out_dir / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    print(f"Run dir: {run_dir}")

    # ── DataModule ───────────────────────────────────────────────────────
    dm = CombinedDataModule(
        cfg,
        splits_path=args.splits_2023,
        manifest_path=args.manifest_2023,
        synthrad2025_splits_path=args.splits_2025,
        manifest_2025_path=args.manifest_2025,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(cfg, args)

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

    callbacks = [checkpoint_cb, lr_monitor, RouterTauScheduleCallback()]
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
        print(f"Best val/mae_hu_mean: {float(best_score):.4f} HU")
    else:
        print("Best val/mae_hu_mean: N/A (no validation metric recorded)")


if __name__ == "__main__":
    main()
