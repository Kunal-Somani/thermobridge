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

Worker-safety notes (fixes DataLoader hang with num_workers>0):
- __init__ converts ALL OmegaConf values to plain Python primitives immediately.
  OmegaConf DictConfig/ListConfig objects are NOT pickle-safe across fork/spawn
  boundaries and cause silent worker deadlocks.
- setup() reads JSON files in the main process only; datasets store only plain
  Python dicts/lists/str/int/float — all pickle-safe.
- DataLoader uses multiprocessing_context="spawn" when num_workers>0 to avoid
  fork-with-open-file-handle deadlocks common in container environments.
- persistent_workers=True when num_workers>0 (avoids worker re-init overhead).
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
from src.data.datamodule import seed_worker
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


def _to_plain_dict(obj: Any) -> dict:
    """Recursively convert OmegaConf / any mapping to a plain Python dict.

    OmegaConf DictConfig objects are NOT safely picklable across fork/spawn
    worker boundaries.  Calling dict() on them is NOT sufficient — nested
    values remain DictConfig.  This helper walks the tree and produces a
    fully plain structure (dict/list/int/float/str/bool/None only).
    """
    # Try OmegaConf's own to_container first (handles structured configs too)
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True, throw_on_missing=False)  # type: ignore[return-value]
    except ImportError:
        pass
    # Fallback: plain dict/list handling
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_dict(v) for v in obj]  # type: ignore[return-value]
    return obj


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
        # Store as plain dicts (already guaranteed by callers, but be explicit)
        self.anatomy_to_idx = dict(anatomy_to_idx)
        self.modality_to_idx = dict(modality_to_idx)

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
        cfg:                     Fully-resolved OmegaConf config (converted to
                                 plain Python primitives in __init__ — do NOT
                                 store the raw OmegaConf object on self to
                                 avoid worker-process pickling deadlocks).
        splits_path:              outputs/splits.json for SynthRAD2023 (ADR-007).
        manifest_path:             outputs/preprocessed/manifest.json for SynthRAD2023.
        synthrad2025_splits_path:  Challenge pre-split JSON for SynthRAD2025 (ADR-012).
        manifest_2025_path:        outputs/preprocessed_2025/manifest_2025.json.
        num_workers:               Override num_workers (default: from cfg, 0 = safe).
    """

    def __init__(
        self,
        cfg: Any,
        splits_path: Path | None = None,
        manifest_path: Path | None = None,
        synthrad2025_splits_path: Path | None = None,
        manifest_2025_path: Path | None = None,
        num_workers: int | None = None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Convert ALL config values to plain Python primitives NOW.
        # OmegaConf DictConfig is not safely picklable across fork/spawn
        # worker process boundaries — storing self.cfg = cfg directly causes
        # silent DataLoader worker deadlocks when num_workers > 0.
        # ------------------------------------------------------------------
        cfg_plain = _to_plain_dict(cfg)

        self.splits_path = splits_path or (_REPO_ROOT / "outputs" / "splits.json")
        self.manifest_path = manifest_path or (_REPO_ROOT / "outputs" / "preprocessed" / "manifest.json")
        self.synthrad2025_splits_path = synthrad2025_splits_path or (
            _REPO_ROOT / "outputs" / "splits_synthrad2025.json"
        )
        self.manifest_2025_path = manifest_2025_path or (
            _REPO_ROOT / "outputs" / "preprocessed_2025" / "manifest_2025.json"
        )

        # Extract all needed values as plain Python primitives
        combined_cfg = cfg_plain.get("data", {}).get("combined", None)
        if combined_cfg is not None:
            self.anatomy_to_idx: dict[str, int] = dict(combined_cfg["anatomy_to_idx"])
            raw_mod = combined_cfg["modality_to_idx"]
            self.modality_to_idx: dict[str, int] = {
                _CONFIG_TO_DATASET_MODALITY_KEY[k]: int(v)
                for k, v in raw_mod.items()
            }
        else:
            self.anatomy_to_idx = dict(DEFAULT_ANATOMY_TO_IDX)
            self.modality_to_idx = dict(DEFAULT_MODALITY_TO_IDX)

        patch_cfg = cfg_plain.get("patch", {})
        self._patch_size: list[int] = [int(x) for x in patch_cfg.get("size", [96, 96, 96])]
        self._samples_per_volume: int = int(patch_cfg.get("samples_per_volume", 2))
        self._fg_fraction: float = float(patch_cfg.get("foreground_fraction", 0.8))
        self._base_seed: int = int(cfg_plain.get("seed", 42))

        training_cfg = cfg_plain.get("training", {})
        self._batch_size: int = int(training_cfg.get("batch_size", 6))
        # num_workers: explicit override wins, else from config, else 0
        if num_workers is not None:
            self._num_workers: int = int(num_workers)
        else:
            self._num_workers = int(training_cfg.get("num_workers", 0))

        self.train_ds: ConcatDataset | None = None
        self.val_ds: ConcatDataset | None = None
        self.test_ds: ConcatDataset | None = None

    # ------------------------------------------------------------------
    # Classmethod convenience constructor (worker-safe alternative to
    # passing a raw cfg object)
    # ------------------------------------------------------------------

    @classmethod
    def from_primitives(
        cls,
        *,
        splits_path: Path,
        manifest_path: Path,
        synthrad2025_splits_path: Path,
        manifest_2025_path: Path,
        anatomy_to_idx: dict[str, int] | None = None,
        modality_to_idx: dict[str, int] | None = None,
        patch_size: list[int] | None = None,
        samples_per_volume: int = 2,
        fg_fraction: float = 0.8,
        base_seed: int = 42,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> "CombinedDataModule":
        """Construct CombinedDataModule from plain Python primitives only.

        This is the recommended entry point for scripts and tests where no
        OmegaConf config is available.  All arguments are plain Python types
        so the DataModule and its child datasets are fully pickle-safe.
        """
        # Build a minimal plain-dict cfg so __init__ can extract values
        plain_cfg: dict = {
            "seed": base_seed,
            "patch": {
                "size": patch_size or [96, 96, 96],
                "samples_per_volume": samples_per_volume,
                "foreground_fraction": fg_fraction,
            },
            "training": {
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
            "data": {
                "combined": {
                    "anatomy_to_idx": anatomy_to_idx or dict(DEFAULT_ANATOMY_TO_IDX),
                    # from_primitives callers supply dataset-key modality map directly
                    # We must invert to config-key format for __init__ to re-invert.
                    # Easier: just patch self after construction.
                    "modality_to_idx": {
                        # Invert _CONFIG_TO_DATASET_MODALITY_KEY to produce MRI/CT/CBCT keys
                        "MRI": (modality_to_idx or DEFAULT_MODALITY_TO_IDX).get("mr", 0),
                        "CT":  (modality_to_idx or DEFAULT_MODALITY_TO_IDX).get("ct", 1),
                        "CBCT": (modality_to_idx or DEFAULT_MODALITY_TO_IDX).get("cbct", 2),
                    },
                }
            },
        }
        instance = cls(
            cfg=plain_cfg,
            splits_path=splits_path,
            manifest_path=manifest_path,
            synthrad2025_splits_path=synthrad2025_splits_path,
            manifest_2025_path=manifest_2025_path,
            num_workers=num_workers,
        )
        return instance

    # ------------------------------------------------------------------
    # setup — file I/O happens here (main process only).
    # Datasets store only plain Python dicts/lists/str/int/float so they
    # are safely picklable when forked/spawned into worker processes.
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        # All file reads happen in the main process inside setup().
        # The resulting plain-Python dicts (splits, manifests) are passed
        # to Dataset constructors which store them as-is.  No OmegaConf,
        # no open file handles, no lambdas cross the worker boundary.
        splits_2023 = load_splits(self.splits_path)
        with open(self.manifest_path) as f:
            manifest_2023: dict = json.load(f)

        splits_2025 = load_splits(self.synthrad2025_splits_path)
        with open(self.manifest_2025_path) as f:
            manifest_2025: dict = json.load(f)

        # Use pre-extracted plain primitives (set in __init__)
        patch_size = self._patch_size
        samples_per_volume = self._samples_per_volume
        fg_fraction = self._fg_fraction
        base_seed = self._base_seed

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

            # Val uses the same patch-sampling datasets as train (patch-based
            # val loss, see LitBridge.validation_step) — full-volume eval
            # datasets are reserved for evaluate_full() via test_dataloader().
            ds_2023_val = ThermoBridgeTrainDataset(
                patient_ids=splits_2023["val"], manifest=manifest_2023,
                patch_size=patch_size, samples_per_volume=samples_per_volume,
                fg_fraction=fg_fraction, base_seed=base_seed,
            )
            ds_2025_val = SynthRAD2025Dataset(
                patient_ids=splits_2025["val"], manifest=manifest_2025,
                patch_size=patch_size, samples_per_volume=samples_per_volume,
                fg_fraction=fg_fraction, modality_to_idx=self.modality_to_idx,
                anatomy_to_idx=self.anatomy_to_idx, base_seed=base_seed,
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

    def _make_dataloader(
        self,
        dataset: ConcatDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        """Shared DataLoader factory.

        Uses multiprocessing_context='spawn' when num_workers>0 to avoid
        fork-with-open-file-handle deadlocks that occur in container/cloud
        environments (the documented root cause of the training hang).
        persistent_workers=True avoids per-epoch worker restart overhead.
        """
        nw = self._num_workers
        kwargs: dict[str, Any] = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            drop_last=drop_last,
            pin_memory=(nw > 0),
            worker_init_fn=seed_worker if nw > 0 else None,
            persistent_workers=(nw > 0),
        )
        if nw > 0:
            # 'spawn' is safe across all Linux container setups.
            # 'fork' can deadlock if the parent holds open file handles
            # (e.g., from json.load in setup()) or if OmegaConf's C++
            # extension is loaded.
            kwargs["multiprocessing_context"] = "spawn"
            kwargs["prefetch_factor"] = 2
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "Call setup('fit') first."
        return self._make_dataloader(
            self.train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None, "Call setup('fit') first."
        return self._make_dataloader(
            self.val_ds,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, "Call setup('test') first."
        return self._make_dataloader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            drop_last=False,
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
