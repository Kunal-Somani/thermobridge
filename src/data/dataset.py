"""MONAI-backed 3D patch dataset for ThermoBridge bidirectional synthesis.

Train:  foreground-biased random 96³ patches; both MR→CT and CT→MR directions.
Val/Test: full preprocessed volumes (no crop); sliding-window inference done later.

Each item is a dict with keys:
    source      : float32 tensor  (1, [Z,Y,X] or pZ,pY,pX)
    target      : float32 tensor  (same shape as source)
    mask        : float32 tensor  (same shape, binary)
    direction_id: int  0=MR→CT  1=CT→MR
    patient_id  : str
    anatomy     : str  'brain' | 'pelvis'

Direction balance: by construction each patient appears with both direction IDs,
so direction 0 and 1 are always 50/50 within an epoch.
Anatomy balance:  brain=126 and pelvis=126 patients in train → 50/50 by stratum.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Foreground-biased 3-D random crop
# ---------------------------------------------------------------------------


def foreground_biased_crop(
    mr: np.ndarray,
    ct: np.ndarray,
    mask: np.ndarray,
    patch_size: tuple[int, int, int],
    fg_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a random 3-D patch, biased toward the foreground mask.

    With probability ``fg_fraction`` the patch is centred on a randomly chosen
    foreground voxel; otherwise the centre is drawn uniformly at random.  The
    same crop window is applied to all three arrays, guaranteeing spatial
    alignment of source, target, and mask (R2 / PAPER==CODE concern).

    Args:
        mr, ct, mask: Shape (Z, Y, X).
        patch_size:   (pZ, pY, pX).
        fg_fraction:  Probability of a foreground-centred crop (0–1).
        rng:          Seeded numpy Generator (R6).

    Returns:
        Cropped (mr, ct, mask) each of shape patch_size.
    """
    Z, Y, X   = mr.shape
    pZ, pY, pX = patch_size

    if Z < pZ or Y < pY or X < pX:
        raise ValueError(
            f"Volume {mr.shape} is smaller than patch {patch_size} on at least one axis."
        )

    max_z0 = Z - pZ
    max_y0 = Y - pY
    max_x0 = X - pX

    z0 = y0 = x0 = None

    if rng.random() < fg_fraction:
        fg = np.argwhere(mask > 0)
        if len(fg) > 0:
            # Keep only centres that yield a fully-valid crop window
            valid = fg[
                (fg[:, 0] >= pZ // 2) & (fg[:, 0] < Z - pZ // 2) &
                (fg[:, 1] >= pY // 2) & (fg[:, 1] < Y - pY // 2) &
                (fg[:, 2] >= pX // 2) & (fg[:, 2] < X - pX // 2)
            ]
            if len(valid) > 0:
                ctr = valid[rng.integers(len(valid))]
                z0 = int(np.clip(ctr[0] - pZ // 2, 0, max_z0))
                y0 = int(np.clip(ctr[1] - pY // 2, 0, max_y0))
                x0 = int(np.clip(ctr[2] - pX // 2, 0, max_x0))

    if z0 is None:  # fallback: uniform random crop
        z0 = int(rng.integers(0, max_z0 + 1))
        y0 = int(rng.integers(0, max_y0 + 1))
        x0 = int(rng.integers(0, max_x0 + 1))

    sl = np.s_[z0:z0+pZ, y0:y0+pY, x0:x0+pX]
    return mr[sl], ct[sl], mask[sl]


# ---------------------------------------------------------------------------
# Helper: load a patient's preprocessed npz trio
# ---------------------------------------------------------------------------


def _load_patient(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (mr, ct, mask) float32 arrays from .npz files."""
    mr   = np.load(entry["mr_path"])["data"]    # (Z, Y, X) float32 norm [0,1]
    ct   = np.load(entry["ct_path"])["data"]    # (Z, Y, X) float32 norm [-1,1]
    mask = np.load(entry["mask_path"])["data"].astype(np.float32)  # (Z, Y, X) binary
    return mr, ct, mask


def _to_tensor_1chw(arr: np.ndarray) -> torch.Tensor:
    """Add channel dim and convert to float32 tensor: (Z,Y,X) → (1,Z,Y,X)."""
    return torch.from_numpy(arr[np.newaxis].astype(np.float32))


# ---------------------------------------------------------------------------
# Train dataset  — random foreground-biased patches, bidirectional
# ---------------------------------------------------------------------------


class ThermoBridgeTrainDataset(Dataset):
    """Patch-based train dataset for bidirectional MR↔CT synthesis.

    Dataset length = len(patient_ids) × 2 directions × samples_per_volume.
    Each __getitem__ loads ONE preprocessed volume and extracts ONE random patch.
    The datalist is expanded so that shuffling during training naturally mixes
    both directions and both anatomies across batches.
    """

    def __init__(
        self,
        patient_ids: list[str],
        manifest: dict[str, Any],
        patch_size: tuple[int, int, int],
        samples_per_volume: int,
        fg_fraction: float,
        base_seed: int = 42,
    ) -> None:
        self.patch_size         = tuple(patch_size)
        self.samples_per_volume = samples_per_volume
        self.fg_fraction        = fg_fraction
        self.base_seed          = base_seed

        # Expand: (pid, direction_id, sample_idx) — one row per dataset item
        self.items: list[dict[str, Any]] = []
        for pid in sorted(patient_ids):
            entry = manifest[pid]
            for direction_id in (0, 1):
                for sample_idx in range(samples_per_volume):
                    self.items.append({
                        "pid":          pid,
                        "anatomy":      entry["anatomy"],
                        "mr_path":      entry["mr_path"],
                        "ct_path":      entry["ct_path"],
                        "mask_path":    entry["mask_path"],
                        "direction_id": direction_id,
                        "sample_idx":   sample_idx,
                    })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]

        mr, ct, mask = _load_patient(item)

        # Per-item deterministic-but-varied seed to avoid repeated crops
        # (combined with worker seed set in worker_init_fn for full reproducibility)
        item_seed = (self.base_seed + idx * 1000003) % (2**32)
        rng = np.random.default_rng(item_seed)

        mr_p, ct_p, mask_p = foreground_biased_crop(
            mr, ct, mask, self.patch_size, self.fg_fraction, rng
        )

        # Swap source/target roles based on direction
        if item["direction_id"] == 0:   # MR → CT
            source, target = mr_p, ct_p
        else:                            # CT → MR
            source, target = ct_p, mr_p

        return {
            "source":       _to_tensor_1chw(source),
            "target":       _to_tensor_1chw(target),
            "mask":         _to_tensor_1chw(mask_p),
            "direction_id": torch.tensor(item["direction_id"], dtype=torch.long),
            "patient_id":   item["pid"],
            "anatomy":      item["anatomy"],
        }


# ---------------------------------------------------------------------------
# Eval dataset  — FULL volumes, no random crop, bidirectional
# ---------------------------------------------------------------------------


class ThermoBridgeEvalDataset(Dataset):
    """Full-volume eval dataset (val / test).

    Returns the entire preprocessed volume — no random cropping.
    Sliding-window patch inference is performed by the evaluator (chunk-5).
    Ordering is deterministic (sorted by patient_id, then direction_id).

    Dataset length = len(patient_ids) × 2 directions.
    """

    def __init__(
        self,
        patient_ids: list[str],
        manifest: dict[str, Any],
    ) -> None:
        self.items: list[dict[str, Any]] = []
        for pid in sorted(patient_ids):
            entry = manifest[pid]
            for direction_id in (0, 1):
                self.items.append({
                    "pid":          pid,
                    "anatomy":      entry["anatomy"],
                    "mr_path":      entry["mr_path"],
                    "ct_path":      entry["ct_path"],
                    "mask_path":    entry["mask_path"],
                    "direction_id": direction_id,
                    "ct_norm_params": entry["ct_norm_params"],
                    "mr_norm_params": entry["mr_norm_params"],
                })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        mr, ct, mask = _load_patient(item)

        if item["direction_id"] == 0:   # MR → CT
            source, target = mr, ct
        else:                            # CT → MR
            source, target = ct, mr

        return {
            "source":         _to_tensor_1chw(source),
            "target":         _to_tensor_1chw(target),
            "mask":           _to_tensor_1chw(mask),
            "direction_id":   torch.tensor(item["direction_id"], dtype=torch.long),
            "patient_id":     item["pid"],
            "anatomy":        item["anatomy"],
            "ct_norm_params": item["ct_norm_params"],
            "mr_norm_params": item["mr_norm_params"],
        }
