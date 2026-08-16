"""CombinedDataModule — SynthRAD2023 (brain+pelvis, MRI<->CT) + SynthRAD2025
(HN+TH+AB, MRI<->CT + CBCT<->CT), ThermoBridge Phase 1 (ADR-012, Chunk N1).

Combined index mapping (authoritative — routing gate's num_anatomies becomes
5 once this is wired into training):
    anatomy_to_idx:  brain=0, pelvis=1, HN=2, TH=3, AB=4
    modality_to_idx: MRI=0, CT=1, CBCT=2

SynthRAD2023 split: existing outputs/splits.json (ADR-007).
SynthRAD2025 split: challenge pre-split (ADR-012, use_challenge_split=true) —
loaded from a splits JSON with the same {"train":[...], "val":[...], "test":[...]}
format as outputs/splits.json. ADR-012 explicitly forbids re-splitting
SynthRAD2025 (G3) — this module never generates one; the file must be
derived from the official challenge split and supplied via
`synthrad2025_splits_path`.

src/data/dataset.py's ThermoBridgeTrainDataset/ThermoBridgeEvalDataset return
`anatomy` as a string and have no m_s/m_t keys (Phase 0 was MRI<->CT only, so
those were implicit). Rather than modifying that file (out of scope for this
chunk), `_Legacy2023Wrapper` below adapts its batches to the combined schema:
`anatomy` remapped string->combined-int, `m_s`/`m_t` added from
direction_id (0=MRI->CT, 1=CT->MRI, matching that dataset's own convention).
This keeps both datasets' `__getitem__` outputs collate-compatible before
ConcatDataset chains them, without touching src/data/dataset.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ThermoBridgeEvalDataset, ThermoBridgeTrainDataset
from src.data.dataset_2025 import SynthRAD2025Dataset, SynthRAD2025EvalDataset
from src.data.datamodule import seed_worker, _make_generator
from src.data.splits import load_splits

# Authoritative combined index mapping (Chunk N1) — must match
# configs/default.yaml's data.combined section exactly.
DEFAULT_ANATOMY_TO_IDX = {"brain": 0, "pelvis": 1, "HN": 2, "TH": 3, "AB": 4}
DEFAULT_MODALITY_TO_IDX = {"mr": 0, "ct": 1, "cbct": 2}

# configs/default.yaml's data.combined.modality_to_idx uses the human-facing
# keys MRI/CT/CBCT; the dataset classes use the lowercase mr/ct/cbct
# vocabulary already established by manifest.json's "modality_src" field
# (Chunk N1/3). This maps one to the other explicitly (not a blind
# .lower(), since "MRI".lower() != "mr").
_CONFIG_TO_DATASET_MODALITY_KEY = {"MRI": "mr", "CT": "ct", "CBCT": "cbct"}


def _config_modality_to_idx(cfg_modality_to_idx: Any) -> dict[str, int]:
    return {
        _CONFIG_TO_DATASET_MODALITY_KEY[k]: v
        for k, v in dict(cfg_modality_to_idx).items()
    }


class _Legacy2023Wrapper(Dataset):
    """Adapts a src/data/dataset.py dataset's batches to the combined schema
    (anatomy string -> combined int index; adds m_s/m_t) without modifying
    that file. See module docstring for why this adapter exists.
    """

    def __init__(
        self,
        inner: Dataset,
        anatomy_to_idx: dict[str, int],
        modality_to_idx: dict[str, int],
    ) -> None:
        self.inner = inner
        self.anatomy_to_idx = anatomy_to_idx
        self.modality_to_idx = modality_to_idx

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.inner[idx])
        dir_id = int(item["direction_id"])
        m_mr, m_ct = self.modality_to_idx["mr"], self.modality_to_idx["ct"]
        m_s, m_t = (m_mr, m_ct) if dir_id == 0 else (m_ct, m_mr)
        item["m_s"] = torch.tensor(m_s, dtype=torch.long)
        item["m_t"] = torch.tensor(m_t, dtype=torch.long)
        item["anatomy"] = self.anatomy_to_idx[item["anatomy"]]
        return item


class CombinedDataModule(pl.LightningDataModule):
    """Lightning DataModule combining SynthRAD2023 + SynthRAD2025 (ADR-012).

    Args:
        cfg:                     Fully-resolved OmegaConf config.
        splits_path:              outputs/splits.json for SynthRAD2023 (ADR-007).
        manifest_path:             outputs/preprocessed/manifest.json for SynthRAD2023.
        synthrad2025_splits_path:  Challenge pre-split JSON for SynthRAD2025 (ADR-012).
        manifest_2025_path:        outputs/preprocessed_2025/manifest_2025.json.
    """

    def __init__(
        self,
        cfg: Any,
        splits_path: Path | None = None,
        manifest_path: Path | None = None,
        synthrad2025_splits_path: Path | None = None,
        manifest_2025_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.splits_path = splits_path or (_REPO_ROOT / "outputs" / "splits.json")
        self.manifest_path = manifest_path or (_REPO_ROOT / "outputs" / "preprocessed" / "manifest.json")
        self.synthrad2025_splits_path = synthrad2025_splits_path or (
            _REPO_ROOT / "outputs" / "splits_synthrad2025.json"
        )
        self.manifest_2025_path = manifest_2025_path or (
            _REPO_ROOT / "outputs" / "preprocessed_2025" / "manifest_2025.json"
        )

        combined_cfg = getattr(cfg.data, "combined", None)
        self.anatomy_to_idx = dict(combined_cfg.anatomy_to_idx) if combined_cfg is not None else dict(DEFAULT_ANATOMY_TO_IDX)
        self.modality_to_idx = (
            _config_modality_to_idx(combined_cfg.modality_to_idx)
            if combined_cfg is not None else dict(DEFAULT_MODALITY_TO_IDX)
        )

        self.train_ds: ConcatDataset | None = None
        self.val_ds: ConcatDataset | None = None
        self.test_ds: ConcatDataset | None = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        splits_2023 = load_splits(self.splits_path)
        with open(self.manifest_path) as f:
            manifest_2023 = json.load(f)

        splits_2025 = load_splits(self.synthrad2025_splits_path)
        with open(self.manifest_2025_path) as f:
            manifest_2025 = json.load(f)

        patch_size = list(self.cfg.patch.size)
        samples_per_volume = int(self.cfg.patch.samples_per_volume)
        fg_fraction = float(self.cfg.patch.foreground_fraction)
        base_seed = int(self.cfg.seed)

        if stage in ("fit", None):
            ds_2023_train = ThermoBridgeTrainDataset(
                patient_ids=splits_2023["train"], manifest=manifest_2023,
                patch_size=patch_size, samples_per_volume=samples_per_volume,
                fg_fraction=fg_fraction, base_seed=base_seed,
            )
            ds_2025_train = SynthRAD2025Dataset(
                patient_ids=splits_2025["train"], manifest=manifest_2025,
                patch_size=patch_size, samples_per_volume=samples_per_volume,
                fg_fraction=fg_fraction, modality_to_idx=self.modality_to_idx,
                anatomy_to_idx=self.anatomy_to_idx, base_seed=base_seed,
            )
            self.train_ds = ConcatDataset([
                _Legacy2023Wrapper(ds_2023_train, self.anatomy_to_idx, self.modality_to_idx),
                ds_2025_train,
            ])

            ds_2023_val = ThermoBridgeEvalDataset(patient_ids=splits_2023["val"], manifest=manifest_2023)
            ds_2025_val = SynthRAD2025EvalDataset(
                patient_ids=splits_2025["val"], manifest=manifest_2025,
                modality_to_idx=self.modality_to_idx, anatomy_to_idx=self.anatomy_to_idx,
            )
            self.val_ds = ConcatDataset([
                _Legacy2023Wrapper(ds_2023_val, self.anatomy_to_idx, self.modality_to_idx),
                ds_2025_val,
            ])

        if stage in ("test", None):
            ds_2023_test = ThermoBridgeEvalDataset(patient_ids=splits_2023["test"], manifest=manifest_2023)
            ds_2025_test = SynthRAD2025EvalDataset(
                patient_ids=splits_2025["test"], manifest=manifest_2025,
                modality_to_idx=self.modality_to_idx, anatomy_to_idx=self.anatomy_to_idx,
            )
            self.test_ds = ConcatDataset([
                _Legacy2023Wrapper(ds_2023_test, self.anatomy_to_idx, self.modality_to_idx),
                ds_2025_test,
            ])

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "Call setup('fit') first."
        nw = int(self.cfg.training.num_workers)
        return DataLoader(
            self.train_ds,
            batch_size=int(self.cfg.training.batch_size),
            shuffle=True,
            num_workers=nw,
            pin_memory=(nw > 0),
            drop_last=True,
            worker_init_fn=seed_worker if nw > 0 else None,
            generator=_make_generator(int(self.cfg.seed)),
            persistent_workers=(nw > 0),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None, "Call setup('fit') first."
        # Full-volume sliding-window inference is CPU-bound (unlike training's
        # patch sampling), so val gets its own worker count to keep the GPU fed.
        nw = 4
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=nw,
            pin_memory=(nw > 0),
            worker_init_fn=seed_worker if nw > 0 else None,
            persistent_workers=(nw > 0),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, "Call setup('test') first."
        nw = int(self.cfg.training.num_workers)
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=nw,
            pin_memory=(nw > 0),
            worker_init_fn=seed_worker if nw > 0 else None,
            persistent_workers=(nw > 0),
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
