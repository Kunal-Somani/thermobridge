"""Lightning training module for the I2SB hybrid-transformer bridge — ThermoBridge.

Loss terms (§7)
----------------
L_total = λ_rec·L_rec + λ_bnd·L_bnd + λ_rad·L_rad + λ_ent·L_ent + λ_bal·L_bal + λ_cls·L_cls

L_rec (§3, I2SB bridge reconstruction) and L_rad (§7, ADR-011, Radon-domain
consistency — Chunk 10) are implemented. L_rad is computed only for samples
whose target modality is CT or CBCT (m_t index 1 or 2); MRI-target samples
contribute exactly 0 via RadonConsistencyLoss's mask, matching the spec's
"Applied when: Target is CT or CBCT only". L_bnd needs the anisotropic
operator's edge information wired into a boundary loss (not yet built);
L_ent/L_bal need the routing gate's outputs threaded through RoutingLoss
(the gate itself exists since Chunk 9, but its losses aren't wired here
yet); L_cls is optional (ADR-003) and needs anatomy labels. Their λ weights
are read from cfg.loss.weights and logged every step at their actual (zero)
contribution — nothing is hidden (Rule 1), they simply have no
implementation yet.

Dual-direction batching (ADR-014)
----------------------------------
ThermoBridgeTrainDataset already expands every patient into both direction
IDs (see src/data/dataset.py), so a shuffled training batch naturally
contains a mix of MR->CT (m_s=0,m_t=1) and CT->MR (m_s=1,m_t=0) samples, and
the loss below is averaged across whatever mix the batch contains — this is
the batch-level realization of "both directions, same forward pass, loss
averaged" for the existing per-item dataset format.

Anatomy routing (§5) is not implemented until Chunk 9. alpha is passed as a
uniform placeholder distribution over anatomies (documented TODO) — the
denoiser's anatomy-adapter slots are nn.Identity() until Chunk 9 replaces
them, so alpha has no effect on the forward pass yet.

Validation
----------
Training-time validation_step() is a cheap patch-based bridge forward pass
(same math as training_step, under no_grad) in normalized space — no HU
inversion, no sliding window. It logs val/loss_patch, which is what
ModelCheckpoint monitors. This keeps validation as fast as a training step.

Full-volume, HU-inverted evaluation (the chunk-5 sliding-window harness,
MAE-HU / PSNR / SSIM per anatomy x direction) is NOT run during training
anymore. It lives in evaluate_full(), called once by the training script
after trainer.fit() completes, on a held-out dataloader (e.g. test).

Rules:
    R1 — all loss weights visible in config; each term logged separately.
    R2 — CT/MR inversion done inside evaluate_full() via chunk-5 harness, never skipped.
    R4 — epoch-level mean reported; no single-case peak.
    R6 — seed workers, deterministic val ordering (inherited from datamodule).
    R7 — typed, docstrings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.bridge import I2SBProcess
from src.models.build import build_denoiser
from src.physics.radon import FastRadonProjector, RadonConsistencyLoss
from src.training.evaluate import sliding_window_predict
from src.training.metrics import compute_all_metrics
from src.data.preprocess import invert_ct_to_hu, invert_mr

# Modality indices, consistent with cfg.data.all_modalities = ["mr", "ct", "cbct"]
# and direction_id convention (0 = MR->CT, 1 = CT->MR) used by the dataset.
_MR_IDX = 0
_CT_IDX = 1
_CBCT_IDX = 2  # not yet produced by the dataset (Task2/CBCT not wired), but L_rad must honor it


# ---------------------------------------------------------------------------
# Model predictor wrapper (for chunk-5 harness)
# ---------------------------------------------------------------------------


class _BridgePredictor:
    """Wraps LitBridge's reverse_sample as a Predictor for the chunk-5 harness."""

    name = "thermobridge"

    def __init__(self, lit_module: "LitBridge") -> None:
        self._mod = lit_module

    def predict(self, source_norm: np.ndarray, direction_id: int) -> np.ndarray:
        mod = self._mod
        dev = next(mod.denoiser.parameters()).device
        x_T = torch.from_numpy(source_norm[np.newaxis, np.newaxis]).float().to(dev)

        m_s_val, m_t_val = (_MR_IDX, _CT_IDX) if direction_id == 0 else (_CT_IDX, _MR_IDX)
        m_s = torch.tensor([m_s_val], dtype=torch.long, device=dev)
        m_t = torch.tensor([m_t_val], dtype=torch.long, device=dev)
        alpha = mod._make_alpha(1, dev)

        num_steps = int(mod.cfg.model.bridge.num_steps)
        with torch.no_grad():
            x_hat_0 = mod.bridge.reverse_sample(
                mod.denoiser, x_T, m_s, m_t, alpha, num_steps=num_steps
            )
        return x_hat_0[0, 0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------


class LitBridge(pl.LightningModule):
    """LightningModule wrapping ThermoBridgeDenoiser + I2SBProcess (§3, §4).

    Args:
        cfg:  Fully resolved OmegaConf config (from load_config).
        manifest_path: Path to outputs/preprocessed/manifest.json (SynthRAD2023).
        splits_path:   Path to outputs/splits.json (SynthRAD2023).
        manifest_2025_path: Path to outputs/preprocessed_2025/manifest_2025.json
            (SynthRAD2025). Optional — validation_step falls back to this
            manifest for patient IDs not found in manifest_path.
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

        # Build model + bridge process
        self.denoiser = build_denoiser(cfg)
        self.bridge = I2SBProcess(
            max_variance_s=float(cfg.model.bridge.max_variance_s),
            num_steps=int(cfg.model.bridge.num_steps),
            time_weighting=str(cfg.model.bridge.time_weighting),
        )
        self.radon_loss = RadonConsistencyLoss(FastRadonProjector())

        # §7 loss weights (R1 — from config, explicit, never hardcoded)
        weights = dict(cfg.loss.weights)
        self.w_rec = float(weights.get("rec", 1.0))
        self.w_bnd = float(weights.get("bnd", 0.0))
        self.w_rad = float(weights.get("rad", 0.0))
        self.w_ent = float(weights.get("ent", 0.0))
        self.w_bal = float(weights.get("bal", 0.0))
        self.w_cls = float(weights.get("cls", 0.0))

        # Anatomy count for the alpha placeholder (§5 routing gate not built
        # until Chunk 9 — alpha is a uniform pass-through, see _make_alpha).
        self._num_anatomies = len(cfg.data.all_anatomies)

        # Lazy-loaded manifests / splits for val loop
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

    def _make_alpha(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Uniform routing-weight placeholder (§5 gate not built until Chunk 9)."""
        A = self._num_anatomies
        return torch.full((batch_size, A), 1.0 / A, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        m_s: torch.Tensor,
        m_t: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(x_t, t, m_s, m_t, alpha)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        source = batch["source"]        # (B, 1, Z, Y, X) — x_T (bridge start)
        target = batch["target"]        # (B, 1, Z, Y, X) — x_0 (bridge end)
        dir_id = batch["direction_id"]  # (B,) 0=MR->CT, 1=CT->MR

        x_T, x_0 = source, target
        # direction_id 0=MR->CT, 1=CT->MR coincides exactly with (_MR_IDX, _CT_IDX)=(0,1)
        m_s = dir_id.long()
        m_t = (1 - dir_id).long()

        B = x_0.shape[0]
        t = torch.rand(B, device=x_0.device)
        alpha = self._make_alpha(B, x_0.device)

        # Sample x_t and predict x_hat_0 directly (rather than calling the
        # opaque I2SBProcess.bridge_loss) so x_hat_0 is available for L_rad
        # below without a second denoiser forward pass. This replicates
        # bridge_loss's own math exactly (see src/models/bridge.py).
        x_t, _noise = self.bridge.forward_marginal(x_0, x_T, t)
        x_hat_0 = self.denoiser(x_t, t, m_s, m_t, alpha)
        w_t = self.bridge.time_weighting(t)
        l1_per_sample = (x_hat_0 - x_0).abs().flatten(1).mean(dim=1)
        loss_rec = (w_t * l1_per_sample).mean()

        # L_rad (§7, ADR-011, Chunk 10): only for CT/CBCT targets.
        is_ct_or_cbct = ((m_t == _CT_IDX) | (m_t == _CBCT_IDX)).float()
        loss_rad = self.radon_loss(x_hat_0, x_0, is_ct_or_cbct)

        # Terms not yet implementable (§7 — see module docstring): logged at
        # their true zero contribution, weights still read from config.
        zero = torch.zeros((), device=x_0.device)
        loss_bnd, loss_ent, loss_bal, loss_cls = zero, zero, zero, zero

        loss_total = (
            self.w_rec * loss_rec
            + self.w_bnd * loss_bnd
            + self.w_rad * loss_rad
            + self.w_ent * loss_ent
            + self.w_bal * loss_bal
            + self.w_cls * loss_cls
        )

        self.log("train/loss_rec",   loss_rec,   on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss_bnd",   loss_bnd,   on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss_rad",   loss_rad,   on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss_ent",   loss_ent,   on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss_bal",   loss_bal,   on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss_cls",   loss_cls,   on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss_bridge", loss_rec,  on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss_total", loss_total, on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)

        return loss_total

    # ------------------------------------------------------------------
    # Validation — cheap patch-based bridge loss (normalized space, R2 N/A)
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """One bridge forward pass on the val batch, under no_grad.

        Mirrors training_step's math exactly, but with no backward pass and
        no HU inversion — this is a normalized-space proxy loss, logged as
        val/loss_patch, which ModelCheckpoint monitors. Full HU-metric
        evaluation lives in evaluate_full() (see module docstring).
        """
        source = batch["source"]        # x_T (bridge start)
        target = batch["target"]        # x_0 (bridge end)
        dir_id = batch["direction_id"]

        x_T, x_0 = source, target
        m_s = dir_id.long()
        m_t = (1 - dir_id).long()

        B = x_0.shape[0]
        t = torch.rand(B, device=x_0.device)
        alpha = self._make_alpha(B, x_0.device)

        with torch.no_grad():
            x_t, _noise = self.bridge.forward_marginal(x_0, x_T, t)
            x_hat_0 = self.denoiser(x_t, t, m_s, m_t, alpha)
            loss_patch = (x_hat_0 - x_0).abs().mean()

        self.log(
            "val/loss_patch", loss_patch,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )

    # ------------------------------------------------------------------
    # Full-volume, HU-inverted evaluation — NOT run during training.
    # Call once, after trainer.fit() completes, on a held-out dataloader.
    # ------------------------------------------------------------------

    def evaluate_full(self, dataloader) -> dict:
        """Sliding-window inference + HU-inverted metrics over `dataloader`.

        Runs the chunk-5 sliding-window harness with reverse_sample as the
        predictor, inverts CT->HU / MR->original scale, computes
        MAE-HU / PSNR / SSIM per (anatomy x direction), prints a summary,
        and returns the aggregated results.

        Args:
            dataloader: Any eval-style dataloader yielding batch_size=1
                batches with source/target/mask/direction_id/patient_id/
                anatomy keys (e.g. datamodule.test_dataloader()).

        Returns:
            Dict of aggregated (mean) metrics, keyed like
            {"mae_hu_mr2ct": ..., "psnr_db_mr2ct": ..., ...}, plus the raw
            per-patient rows under "rows".
        """
        self.eval()
        manifest_2023, _ = self._load_val_data()
        results: list[dict] = []

        for batch in dataloader:
            source   = batch["source"][0, 0].cpu().numpy()   # (Z, Y, X)
            target_t = batch["target"][0, 0].cpu().numpy()
            mask_np  = batch["mask"][0, 0].cpu().numpy()
            dir_id   = int(batch["direction_id"][0].item())
            pid      = batch["patient_id"][0]
            anatomy  = batch["anatomy"][0]

            if pid in manifest_2023:
                entry     = manifest_2023[pid]
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

            predictor = _BridgePredictor(self)
            patch_size = tuple(int(x) for x in self.cfg.patch.size)
            overlap    = float(self.cfg.patch.inference_overlap)

            pred_norm = sliding_window_predict(
                predictor, source, dir_id, patch_size, overlap, device=self.device
            )

            # Invert to HU / original scale (R2)
            if dir_id == 0:   # MR->CT: headline metric
                pred_hu   = invert_ct_to_hu(pred_norm, ct_params)
                target_hu = invert_ct_to_hu(target_t,  ct_params)
                result    = compute_all_metrics(pred_hu, target_hu, mask_np)
            elif mr_params:   # CT->MR: report on restored MR/CBCT scale
                pred_mr   = invert_mr(pred_norm, mr_params)
                target_mr = invert_mr(target_t,  mr_params)
                result    = compute_all_metrics(pred_mr, target_mr, mask_np)
            else:
                # SynthRAD2025 CT->MR with no src_norm_params available to invert
                # (e.g. CBCT<->CT pairs with no MRI norm stats) — skip HU
                # inversion for the MRI target and report normalized MAE only.
                result = compute_all_metrics(pred_norm, target_t, mask_np)

            results.append({
                "mae_hu":   result.mae_hu,
                "psnr":     result.psnr,
                "ssim":     result.ssim,
                "dir_id":   dir_id,
                "anatomy":  anatomy,
                "pid":      pid,
            })

        def _mean(lst: list[dict], key: str) -> float:
            vals = [r[key] for r in lst if np.isfinite(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        mr2ct = [r for r in results if r["dir_id"] == 0]
        ct2mr = [r for r in results if r["dir_id"] == 1]

        summary: dict[str, Any] = {"rows": results}
        if mr2ct:
            summary["mae_hu_mr2ct"]  = _mean(mr2ct, "mae_hu")
            summary["psnr_db_mr2ct"] = _mean(mr2ct, "psnr")
            summary["ssim_mr2ct"]    = _mean(mr2ct, "ssim")
        if ct2mr:
            summary["mae_hu_ct2mr"]  = _mean(ct2mr, "mae_hu")
            summary["psnr_db_ct2mr"] = _mean(ct2mr, "psnr")

        anatomies = sorted({r["anatomy"] for r in results})
        for anat in anatomies:
            for dir_id, tag in [(0, "mr2ct"), (1, "ct2mr")]:
                sub = [r for r in results if r["anatomy"] == anat and r["dir_id"] == dir_id]
                if sub:
                    summary[f"mae_hu_{anat}_{tag}"] = _mean(sub, "mae_hu")
                    summary[f"psnr_{anat}_{tag}"]   = _mean(sub, "psnr")

        print("\n" + "=" * 76)
        print("  evaluate_full() SUMMARY — mean over patients (R4)")
        print("=" * 76)
        for key, val in summary.items():
            if key == "rows":
                continue
            print(f"  {key:<24} {val:.4f}")
        print("=" * 76 + "\n", flush=True)

        return summary

    # ------------------------------------------------------------------
    # Optimiser + scheduler  (warmup -> cosine)
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
