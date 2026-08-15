"""SynthRAD2025 dataset — Task1 (MRI<->CT) and Task2 (CBCT<->CT), ThermoBridge
Phase 1 (ADR-012, Chunk N1).

Reads manifest_2025.json entries produced by scripts/run_preprocess_2025.py
(schema documented there: patient_id, anatomy, task, center, modality_src,
src_path, ct_path, mask_path, src_norm_params, ct_norm_params, ...).

Each patient yields TWO samples (ADR-014, dual-direction batching):
    Task1: (MRI->CT,  m_s=MRI, m_t=CT ) direction_id=0
           (CT->MRI,  m_s=CT,  m_t=MRI) direction_id=1
    Task2: (CBCT->CT, m_s=CBCT,m_t=CT ) direction_id=0
           (CT->CBCT, m_s=CT,  m_t=CBCT) direction_id=1
direction_id follows the repo-wide convention: 0 = source-modality -> CT,
1 = CT -> source-modality (matches src/data/dataset.py's MR<->CT direction_id).

Batch schema (matches src/data/dataset.py's ThermoBridgeTrainDataset, plus
the m_s/m_t modality indices this multi-modality dataset requires):
    {source, target, mask, direction_id, m_s, m_t, patient_id, anatomy}
`anatomy` is an integer index (for the routing gate), not a string — the
mapping is configurable via `anatomy_to_idx` (default HN=0, TH=1, AB=2;
CombinedDataModule overrides it with the authoritative combined mapping).
`m_s`/`m_t` default to the global modality convention MRI=0, CT=1, CBCT=2,
also overridable via `modality_to_idx`.

Reuses foreground_biased_crop() from src/data/dataset.py — no duplicated
cropping logic; src/data/dataset.py itself is not modified.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import foreground_biased_crop

_DEFAULT_MODALITY_TO_IDX = {"mr": 0, "ct": 1, "cbct": 2}
_DEFAULT_ANATOMY_TO_IDX = {"HN": 0, "TH": 1, "AB": 2}


def _to_tensor_1chw(arr: np.ndarray) -> torch.Tensor:
    """Add channel dim and convert to float32 tensor: (Z,Y,X) -> (1,Z,Y,X)."""
    return torch.from_numpy(arr[np.newaxis].astype(np.float32))


def _load_patient_2025(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (src, ct, mask) float32 arrays from a manifest_2025.json entry."""
    src  = np.load(entry["src_path"])["data"]
    ct   = np.load(entry["ct_path"])["data"]
    mask = np.load(entry["mask_path"])["data"].astype(np.float32)
    return src, ct, mask


def _build_items(
    patient_ids: list[str],
    manifest: dict[str, Any],
    anatomy_to_idx: dict[str, int],
    modality_to_idx: dict[str, int],
    samples_per_volume: int | None,
) -> list[dict[str, Any]]:
    """Expand (patient, direction[, sample]) rows — shared by train/eval variants."""
    items: list[dict[str, Any]] = []
    m_ct = modality_to_idx["ct"]

    for pid in sorted(patient_ids):
        entry = manifest[pid]
        modality_src = entry["modality_src"]  # "mr" or "cbct"
        m_src = modality_to_idx[modality_src]
        anatomy_idx = anatomy_to_idx[entry["anatomy"]]

        for direction_id, (m_s, m_t) in enumerate([(m_src, m_ct), (m_ct, m_src)]):
            base = {
                "pid": pid,
                "anatomy_idx": anatomy_idx,
                "src_path": entry["src_path"],
                "ct_path": entry["ct_path"],
                "mask_path": entry["mask_path"],
                "direction_id": direction_id,
                "m_s": m_s,
                "m_t": m_t,
            }
            if samples_per_volume is None:
                items.append(base)
            else:
                for sample_idx in range(samples_per_volume):
                    items.append({**base, "sample_idx": sample_idx})
    return items


# ---------------------------------------------------------------------------
# Train dataset — random foreground-biased patches, dual-direction
# ---------------------------------------------------------------------------


class SynthRAD2025Dataset(Dataset):
    """Patch-based train dataset for SynthRAD2025 (Task1 MRI<->CT, Task2 CBCT<->CT).

    Dataset length = len(patient_ids) x 2 directions x samples_per_volume.
    """

    def __init__(
        self,
        patient_ids: list[str],
        manifest: dict[str, Any],
        patch_size: tuple[int, int, int],
        samples_per_volume: int,
        fg_fraction: float,
        modality_to_idx: dict[str, int] | None = None,
        anatomy_to_idx: dict[str, int] | None = None,
        base_seed: int = 42,
    ) -> None:
        self.patch_size = tuple(patch_size)
        self.samples_per_volume = samples_per_volume
        self.fg_fraction = fg_fraction
        self.base_seed = base_seed
        self.modality_to_idx = dict(modality_to_idx or _DEFAULT_MODALITY_TO_IDX)
        self.anatomy_to_idx = dict(anatomy_to_idx or _DEFAULT_ANATOMY_TO_IDX)

        self.items = _build_items(
            patient_ids, manifest, self.anatomy_to_idx, self.modality_to_idx, samples_per_volume
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        src, ct, mask = _load_patient_2025(item)

        item_seed = (self.base_seed + idx * 1000003) % (2**32)
        rng = np.random.default_rng(item_seed)
        src_p, ct_p, mask_p = foreground_biased_crop(
            src, ct, mask, self.patch_size, self.fg_fraction, rng
        )

        if item["direction_id"] == 0:  # source-modality -> CT
            source, target = src_p, ct_p
        else:                          # CT -> source-modality
            source, target = ct_p, src_p

        return {
            "source":       _to_tensor_1chw(source),
            "target":       _to_tensor_1chw(target),
            "mask":         _to_tensor_1chw(mask_p),
            "direction_id": torch.tensor(item["direction_id"], dtype=torch.long),
            "m_s":          torch.tensor(item["m_s"], dtype=torch.long),
            "m_t":          torch.tensor(item["m_t"], dtype=torch.long),
            "patient_id":   item["pid"],
            "anatomy":      item["anatomy_idx"],
        }


# ---------------------------------------------------------------------------
# Eval dataset — FULL volumes, no random crop, dual-direction
# ---------------------------------------------------------------------------


class SynthRAD2025EvalDataset(Dataset):
    """Full-volume eval dataset (val/test) for SynthRAD2025.

    Mirrors src/data/dataset.py's ThermoBridgeEvalDataset: returns entire
    preprocessed volumes (sliding-window inference done by the evaluator),
    deterministic ordering (sorted by patient_id, then direction_id).

    Dataset length = len(patient_ids) x 2 directions.
    """

    def __init__(
        self,
        patient_ids: list[str],
        manifest: dict[str, Any],
        modality_to_idx: dict[str, int] | None = None,
        anatomy_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.modality_to_idx = dict(modality_to_idx or _DEFAULT_MODALITY_TO_IDX)
        self.anatomy_to_idx = dict(anatomy_to_idx or _DEFAULT_ANATOMY_TO_IDX)
        self.manifest = manifest

        self.items = _build_items(
            patient_ids, manifest, self.anatomy_to_idx, self.modality_to_idx, samples_per_volume=None
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        src, ct, mask = _load_patient_2025(item)
        entry = self.manifest[item["pid"]]

        if item["direction_id"] == 0:
            source, target = src, ct
        else:
            source, target = ct, src

        return {
            "source":         _to_tensor_1chw(source),
            "target":         _to_tensor_1chw(target),
            "mask":           _to_tensor_1chw(mask),
            "direction_id":   torch.tensor(item["direction_id"], dtype=torch.long),
            "m_s":            torch.tensor(item["m_s"], dtype=torch.long),
            "m_t":            torch.tensor(item["m_t"], dtype=torch.long),
            "patient_id":     item["pid"],
            "anatomy":        item["anatomy_idx"],
            "ct_norm_params": entry["ct_norm_params"],
            "src_norm_params": entry["src_norm_params"],
        }
