"""Lightning training module for the 3-D U-Net baseline — ThermoBridge.

Loss terms
----------
Every term is logged separately (Rule 1: no hidden loss terms).
    train/loss_l1    — L1 on normalised prediction vs target patch
    train/loss_ssim  — 1 − SSIM on normalised prediction (optional, weight from config)
    train/loss_total — weighted sum

Validation
----------
Runs FULL-VOLUME sliding-window inference (chunk-5 harness) on each val patient,
inverts CT→HU, computes MAE-HU / PSNR / SSIM per (anatomy × direction), logs:
    val/mae_hu       — primary checkpoint monitor (lower=better)
    val/psnr_db
    val/ssim
    val/mae_hu_brain_mr2ct, val/mae_hu_pelvis_mr2ct, …

Rules:
    R1 — all loss weights visible in config; each term logged separately.
    R2 — CT inversion done inside the val loop via chunk-5 harness, never skipped.
    R4 — epoch-level mean reported; no single-case peak.
    R6 — seed workers, deterministic val ordering.
    R7 — typed, docstrings.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.unet3d import build_unet3d
from src.training.evaluate import evaluate_patient, sliding_window_predict
from src.training.metrics import compute_all_metrics
from src.data.preprocess import invert_ct_to_hu, invert_mr


# ---------------------------------------------------------------------------
# SSIM loss (3-D patch-level, uniform window — same formula as metrics.py)
# ---------------------------------------------------------------------------


def _patch_ssim_loss(pred: torch.Tensor, target: torch.Tensor, win: int = 7) -> torch.Tensor:
    """1 − mean SSIM over a batch of 3-D patches; used as an auxiliary loss.

    Args:
        pred, target: (B, 1, Z, Y, X) tensors in normalised space.

    Returns:
        Scalar tensor (1 − SSIM), higher = worse.
    """
    B = pred.shape[0]
    losses = []
    for b in range(B):
        p = pred[b, 0]   # (Z, Y, X)
        t = target[b, 0]
        dr = float((t.max() - t.min()).item())
        if dr < 1e-8:
            dr = 1.0
        c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
        # Slice-average along Z
        ssim_slices: list[torch.Tensor] = []
        for z in range(p.shape[0]):
            pz, tz = p[z].unsqueeze(0).unsqueeze(0), t[z].unsqueeze(0).unsqueeze(0)
            mu1 = F.avg_pool2d(pz, win, stride=1, padding=win // 2)
            mu2 = F.avg_pool2d(tz, win, stride=1, padding=win // 2)
            mu1sq, mu2sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
            sig1sq = F.avg_pool2d(pz ** 2, win, stride=1, padding=win // 2) - mu1sq
            sig2sq = F.avg_pool2d(tz ** 2, win, stride=1, padding=win // 2) - mu2sq
            sig12  = F.avg_pool2d(pz * tz,  win, stride=1, padding=win // 2) - mu12
            num = (2 * mu12 + c1) * (2 * sig12 + c2)
            den = (mu1sq + mu2sq + c1) * (sig1sq + sig2sq + c2)
            ssim_slices.append((num / (den + 1e-8)).mean())
        losses.append(1.0 - torch.stack(ssim_slices).mean())
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Model predictor wrapper (for chunk-5 harness)
# ---------------------------------------------------------------------------


class _UNetPredictor:
    """Wraps the Lightning module as a Predictor for the chunk-5 harness."""

    name = "unet3d"

    def __init__(self, lit_module: "LitBaseline") -> None:
        self._mod = lit_module

    def predict(self, source_norm: np.ndarray, direction_id: int) -> np.ndarray:
        dev = next(self._mod.model.parameters()).device
        src_t = torch.from_numpy(source_norm[np.newaxis, np.newaxis]).float().to(dev)
        dir_t = torch.tensor([direction_id], dtype=torch.long, device=dev)
        use_amp = (dev.type == "cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = self._mod.model(src_t, dir_t)
        return out[0, 0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------


class LitBaseline(pl.LightningModule):
    """LightningModule wrapping UNet3D with L1 (+ optional SSIM) loss.

    Args:
        cfg:  Fully resolved OmegaConf config (from load_config).
        manifest_path: Path to outputs/preprocessed/manifest.json (SynthRAD2023).
        splits_path:   Path to outputs/splits.json (SynthRAD2023).
        manifest_2025_path: Path to outputs/preprocessed_2025/manifest_2025.json
            (SynthRAD2025). Optional — validation_step falls back to this
            manifest for patient IDs not found in manifest_path (needed when
            trained via CombinedDataModule, see train_unet_baseline.py).
    """

    def __init__(
        self,
        cfg: Any,
        manifest_path: Path = _REPO_ROOT / "outputs" / "preprocessed" / "manifest.json",
        splits_path:   Path = _REPO_ROOT / "outputs" / "splits.json",
        manifest_2025_path: Path | None = _REPO_ROOT / "outputs" / "preprocessed_2025" / "manifest_2025.json",
    ) -> None:
        super().__init__()
        self.cfg                = cfg
        self.manifest_path      = manifest_path
        self.splits_path        = splits_path
        self.manifest_2025_path = manifest_2025_path

        # Build model
        self.model = build_unet3d(cfg.training)

        # Loss weights (R1 — from config, explicit)
        losses = dict(cfg.training.losses)
        self.w_l1   = float(losses.get("l1",   1.0))
        self.w_ssim = float(losses.get("ssim",  0.0))

        # Combined anatomy index map (Chunk N1) — anatomy arrives as int when
        # trained via CombinedDataModule, string when via ThermoBridgeDataModule.
        combined_cfg = getattr(cfg.data, "combined", None)
        self.anatomy_to_idx = dict(combined_cfg.anatomy_to_idx) if combined_cfg is not None else {"brain": 0, "pelvis": 1}

        # Lazy-loaded manifest / splits for val loop
        self._manifest:      dict | None = None  # SynthRAD2023
        self._manifest_2025: dict | None = None  # SynthRAD2025
        self._val_ids:  list[str] | None = None

        self.save_hyperparameters(ignore=["cfg"])

    # ------------------------------------------------------------------
    # Lazy data loading (avoids pickling OmegaConf in worker processes)
    # ------------------------------------------------------------------

    def _load_val_data(self) -> tuple[dict, list[str]]:
        if self._manifest is None:
            with open(self.manifest_path) as f:
                self._manifest = json.load(f)
        if self._manifest_2025 is None and self.manifest_2025_path is not None:
            manifest_2025_path = Path(self.manifest_2025_path)
            if manifest_2025_path.exists():
                with open(manifest_2025_path) as f:
                    self._manifest_2025 = json.load(f)
        if self._val_ids is None:
            with open(self.splits_path) as f:
                splits = json.load(f)
            self._val_ids = splits["val"]
        return self._manifest, self._val_ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, source: torch.Tensor, direction_id: torch.Tensor) -> torch.Tensor:
        return self.model(source, direction_id)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        source = batch["source"]     # (B, 1, Z, Y, X)
        target = batch["target"]     # (B, 1, Z, Y, X)
        dir_id = batch["direction_id"]  # (B,)

        pred = self(source, dir_id)

        # ── L1 loss (always on) ─────────────────────────────────────────
        loss_l1 = F.l1_loss(pred, target)

        # ── SSIM loss (optional) ────────────────────────────────────────
        if self.w_ssim > 0.0:
            loss_ssim = _patch_ssim_loss(pred, target)
        else:
            loss_ssim = torch.zeros(1, device=self.device)

        # ── Total ───────────────────────────────────────────────────────
        loss_total = self.w_l1 * loss_l1 + self.w_ssim * loss_ssim

        # ── Log EACH term separately (R1 — no hidden terms) ─────────────
        self.log("train/loss_l1",    loss_l1,    on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss_ssim",  loss_ssim,  on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss_total", loss_total, on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)

        return loss_total

    # ------------------------------------------------------------------
    # Validation — full-volume, sliding-window, HU-inverted (R2)
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Collect per-patient results during the val epoch.

        We accumulate into self._val_results and aggregate in
        on_validation_epoch_end().
        """
        source   = batch["source"][0, 0].cpu().numpy()   # (Z, Y, X)
        target_t = batch["target"][0, 0].cpu().numpy()
        mask_np  = batch["mask"][0, 0].cpu().numpy()
        dir_id   = int(batch["direction_id"][0].item())
        pid      = batch["patient_id"][0]
        anatomy  = batch["anatomy"][0]

        self._load_val_data()
        if pid in self._manifest:
            entry     = self._manifest[pid]
            ct_params = entry["ct_norm_params"]
            mr_params = entry["mr_norm_params"]
        else:
            # SynthRAD2025 patient (ADR-012) — different manifest, different
            # param keys (src_norm_params covers both CBCT and MRI sources).
            if self._manifest_2025 is None or pid not in self._manifest_2025:
                raise KeyError(
                    f"Patient id {pid!r} not found in either manifest_2023 "
                    f"({self.manifest_path}) or manifest_2025 ({self.manifest_2025_path})."
                )
            entry     = self._manifest_2025[pid]
            ct_params = entry["ct_norm_params"]
            mr_params = entry.get("src_norm_params", {})

        # Sliding-window inference on CPU (GPU device handled by predictor)
        predictor = _UNetPredictor(self)
        patch_size = tuple(int(x) for x in self.cfg.patch.size)
        overlap    = float(self.cfg.patch.inference_overlap)

        pred_norm = sliding_window_predict(predictor, source, dir_id, patch_size, overlap)

        # Invert to HU / original scale (R2)
        if dir_id == 0:   # MR→CT: headline metric
            pred_hu   = invert_ct_to_hu(pred_norm, ct_params)
            target_hu = invert_ct_to_hu(target_t,  ct_params)
            result    = compute_all_metrics(pred_hu, target_hu, mask_np)
        elif mr_params and "p1" in mr_params:   # CT→MR: report on restored MR/CBCT scale
            pred_mr   = invert_mr(pred_norm, mr_params)
            target_mr = invert_mr(target_t,  mr_params)
            result    = compute_all_metrics(pred_mr, target_mr, mask_np)
        else:
            # SynthRAD2025 CT->MR with no src_norm_params available to invert —
            # skip HU inversion for the MRI target and report normalized MAE only.
            result = compute_all_metrics(pred_norm, target_t, mask_np)

        self._val_results.append({
            "mae_hu":   result.mae_hu,
            "psnr":     result.psnr,
            "ssim":     result.ssim,
            "dir_id":   dir_id,
            "anatomy":  anatomy,
            "pid":      pid,
        })

    def on_validation_epoch_start(self) -> None:
        self._val_results: list[dict] = []

    def on_validation_epoch_end(self) -> None:
        if not self._val_results:
            return

        # ── Overall ──────────────────────────────────────────────────────
        mr2ct = [r for r in self._val_results if r["dir_id"] == 0]
        ct2mr = [r for r in self._val_results if r["dir_id"] == 1]

        def _mean(lst: list[dict], key: str) -> float:
            vals = [r[key] for r in lst if np.isfinite(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        if mr2ct:
            self.log("val/mae_hu",  _mean(mr2ct, "mae_hu"), prog_bar=True,  sync_dist=True)
            self.log("val/psnr_db", _mean(mr2ct, "psnr"),   prog_bar=False, sync_dist=True)
            self.log("val/ssim",    _mean(mr2ct, "ssim"),   prog_bar=False, sync_dist=True)

        if ct2mr:
            self.log("val/mae_mr",      _mean(ct2mr, "mae_hu"), sync_dist=True)
            self.log("val/psnr_ct2mr",  _mean(ct2mr, "psnr"),   sync_dist=True)

        # ── Per anatomy × direction ───────────────────────────────────────
        for anat_name, anat_idx in self.anatomy_to_idx.items():
            for dir_id, tag in [(0, "mr2ct"), (1, "ct2mr")]:
                sub = [r for r in self._val_results
                       if r["anatomy"] == anat_idx and r["dir_id"] == dir_id]
                if sub:
                    self.log(f"val/mae_hu_{anat_name}_{tag}", _mean(sub, "mae_hu"), sync_dist=True)
                    self.log(f"val/psnr_{anat_name}_{tag}",   _mean(sub, "psnr"),   sync_dist=True)

        # ── Print summary ─────────────────────────────────────────────────
        ep = self.current_epoch
        mae = _mean(mr2ct, "mae_hu") if mr2ct else float("nan")
        print(f"\n[Epoch {ep}] val/mae_hu (MR→CT) = {mae:.2f} HU", flush=True)

    # ------------------------------------------------------------------
    # Optimiser + scheduler  (warmup → cosine)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_cfg  = self.cfg.training.optimizer
        sch_cfg  = self.cfg.training.scheduler
        max_ep   = int(self.cfg.training.max_epochs)
        warmup   = int(sch_cfg.warmup_epochs)

        optimizer = AdamW(
            self.parameters(),
            lr           = float(opt_cfg.lr),
            weight_decay = float(opt_cfg.weight_decay),
            betas        = tuple(float(b) for b in opt_cfg.betas),
        )

        # 5-epoch linear warmup → cosine to min_lr
        warmup_sched = LinearLR(
            optimizer,
            start_factor = 1e-4,
            end_factor   = 1.0,
            total_iters  = warmup,
        )
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max  = max(max_ep - warmup, 1),
            eta_min = float(sch_cfg.min_lr),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers = [warmup_sched, cosine_sched],
            milestones = [warmup],
        )
        return {
            "optimizer":  optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
                "name":      "lr/cosine_warmup",
            },
        }
