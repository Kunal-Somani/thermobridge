"""PyTorch Lightning DataModule for ThermoBridge.

Wraps train / val / test loaders with:
- Reproducible seeded worker_init_fn (fixes the multi-worker validation
  non-reproducibility bug from a prior project).
- Train loader: shuffled, foreground-biased patches, bidirectional.
- Val/Test loaders: deterministic full-volume ordering, batch_size=1
  (variable volume shapes cannot be stacked without padding).

Usage::
    from src.data.datamodule import ThermoBridgeDataModule
    from src.utils.config import load_config
    cfg = load_config("configs/default.yaml")
    dm = ThermoBridgeDataModule(cfg)
    dm.setup("fit")
    loader = dm.train_dataloader()
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ThermoBridgeEvalDataset, ThermoBridgeTrainDataset
from src.data.splits import load_splits


# ---------------------------------------------------------------------------
# Reproducible worker seeding  (R6)
# ---------------------------------------------------------------------------


def seed_worker(worker_id: int) -> None:
    """Per-worker seed initialiser — prevents non-reproducible eval sampling.

    Called by DataLoader for each worker process.  Uses the worker's initial
    seed (which PyTorch sets deterministically from the main-process seed +
    epoch) so that each worker's numpy/random state is unique but reproducible.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class ThermoBridgeDataModule(pl.LightningDataModule):
    """Lightning DataModule for bidirectional 3-D MR↔CT synthesis.

    Args:
        cfg:           Fully-resolved OmegaConf config.
        splits_path:   Path to outputs/splits.json  (default: inferred).
        manifest_path: Path to outputs/preprocessed/manifest.json (default: inferred).
    """

    def __init__(
        self,
        cfg: Any,
        splits_path:   Path | None = None,
        manifest_path: Path | None = None,
        overfit_n:     int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.splits_path   = splits_path   or (_REPO_ROOT / "outputs" / "splits.json")
        self.manifest_path = manifest_path or (_REPO_ROOT / "outputs" / "preprocessed" / "manifest.json")
        self.overfit_n     = overfit_n     # if set, restrict train+val to N patients

        self.train_ds: ThermoBridgeTrainDataset | None = None
        self.val_ds:   ThermoBridgeEvalDataset  | None = None
        self.test_ds:  ThermoBridgeEvalDataset  | None = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets.  Called once per process by Lightning."""
        splits   = load_splits(self.splits_path)
        with open(self.manifest_path) as f:
            manifest = json.load(f)

        patch_size         = list(self.cfg.patch.size)          # [96,96,96]
        samples_per_volume = int(self.cfg.patch.samples_per_volume)
        fg_fraction        = float(self.cfg.patch.foreground_fraction)
        base_seed          = int(self.cfg.seed)

        if stage in ("fit", None):
            train_ids = splits["train"]
            val_ids   = splits["val"]
            if self.overfit_n is not None:
                # Restrict to N patients (2 brain + 2 pelvis where possible)
                brain_ids  = [p for p in train_ids if manifest[p]["anatomy"] == "brain"]
                pelvis_ids = [p for p in train_ids if manifest[p]["anatomy"] == "pelvis"]
                n_each     = max(1, self.overfit_n // 2)
                train_ids  = brain_ids[:n_each] + pelvis_ids[:n_each]
                val_ids    = train_ids  # deliberately overfit: val == train

            self.train_ds = ThermoBridgeTrainDataset(
                patient_ids        = train_ids,
                manifest           = manifest,
                patch_size         = patch_size,
                samples_per_volume = samples_per_volume,
                fg_fraction        = fg_fraction,
                base_seed          = base_seed,
            )
            self.val_ds = ThermoBridgeEvalDataset(
                patient_ids = val_ids,
                manifest    = manifest,
            )

        if stage in ("test", None):
            self.test_ds = ThermoBridgeEvalDataset(
                patient_ids = splits["test"],
                manifest    = manifest,
            )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "Call setup('fit') first."
        nw = int(self.cfg.training.num_workers)
        return DataLoader(
            self.train_ds,
            batch_size  = int(self.cfg.training.batch_size),
            shuffle     = True,
            num_workers = nw,
            pin_memory  = (nw > 0),
            drop_last   = True,
            worker_init_fn = seed_worker if nw > 0 else None,
            generator   = _make_generator(int(self.cfg.seed)),
            persistent_workers = (nw > 0),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None, "Call setup('fit') first."
        nw = int(self.cfg.training.num_workers)
        return DataLoader(
            self.val_ds,
            batch_size  = 1,
            shuffle     = False,
            num_workers = nw,
            pin_memory  = (nw > 0),
            worker_init_fn = seed_worker if nw > 0 else None,
            persistent_workers = (nw > 0),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, "Call setup('test') first."
        return DataLoader(
            self.test_ds,
            batch_size  = 1,
            shuffle     = False,
            num_workers = int(self.cfg.training.num_workers),
            pin_memory  = True,
            worker_init_fn = seed_worker,
            persistent_workers = (int(self.cfg.training.num_workers) > 0),
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def train_size(self) -> int:
        return len(self.train_ds) if self.train_ds else 0

    @property
    def val_size(self) -> int:
        return len(self.val_ds) if self.val_ds else 0

    @property
    def test_size(self) -> int:
        return len(self.test_ds) if self.test_ds else 0
