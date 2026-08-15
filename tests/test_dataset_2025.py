"""Unit tests for SynthRAD2025 dataset + combined index config (Chunk N1).

No real .npz/manifest data is needed: tests build tiny synthetic .npz volumes
in tmp_path and a hand-built manifest dict, matching manifest_2025.json's
schema (see src/data/dataset_2025.py / scripts/run_preprocess_2025.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.combined_datamodule import DEFAULT_ANATOMY_TO_IDX, DEFAULT_MODALITY_TO_IDX
from src.data.dataset_2025 import SynthRAD2025Dataset
from src.utils.config import load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_npz(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), data=arr.astype(np.float32))


def _make_manifest_entry(tmp_path: Path, pid: str, anatomy: str, modality_src: str) -> dict:
    shape = (12, 12, 12)
    rng = np.random.default_rng(0)
    src = rng.random(shape)
    ct = rng.random(shape) * 2 - 1
    mask = (rng.random(shape) > 0.3).astype(np.float32)

    out_dir = tmp_path / pid
    src_path = out_dir / f"{modality_src}.npz"
    ct_path = out_dir / "ct.npz"
    mask_path = out_dir / "mask.npz"
    _write_npz(src_path, src)
    _write_npz(ct_path, ct)
    _write_npz(mask_path, mask)

    return {
        "patient_id": pid,
        "anatomy": anatomy,
        "task": "task1" if modality_src == "mr" else "task2",
        "center": "A",
        "modality_src": modality_src,
        "src_path": str(src_path),
        "ct_path": str(ct_path),
        "mask_path": str(mask_path),
        "src_norm_params": {"method": "dummy"},
        "ct_norm_params": {"method": "dummy"},
    }


@pytest.fixture
def task1_manifest(tmp_path):
    pid = "1HNA001"
    entry = _make_manifest_entry(tmp_path, pid, "HN", "mr")
    return pid, {pid: entry}


@pytest.fixture
def task2_manifest(tmp_path):
    pid = "2ABA001"
    entry = _make_manifest_entry(tmp_path, pid, "AB", "cbct")
    return pid, {pid: entry}


def _build_dataset(pid: str, manifest: dict) -> SynthRAD2025Dataset:
    return SynthRAD2025Dataset(
        patient_ids=[pid],
        manifest=manifest,
        patch_size=(8, 8, 8),
        samples_per_volume=1,
        fg_fraction=0.8,
    )


# ---------------------------------------------------------------------------
# Config index tests
# ---------------------------------------------------------------------------


def test_modality_indices():
    cfg = load_config(_REPO_ROOT / "configs" / "default.yaml")
    m = cfg.data.combined.modality_to_idx
    assert m["MRI"] == 0
    assert m["CT"] == 1
    assert m["CBCT"] == 2


def test_anatomy_indices():
    cfg = load_config(_REPO_ROOT / "configs" / "default.yaml")
    a = cfg.data.combined.anatomy_to_idx
    assert a["brain"] == 0
    assert a["pelvis"] == 1
    assert a["HN"] == 2
    assert a["TH"] == 3
    assert a["AB"] == 4


def test_no_modality_index_collision():
    idxs = list(DEFAULT_MODALITY_TO_IDX.values())
    assert len(idxs) == len(set(idxs))


def test_combined_anatomy_count():
    assert len(DEFAULT_ANATOMY_TO_IDX) == 5
    assert set(DEFAULT_ANATOMY_TO_IDX.values()) == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Dataset behavior tests
# ---------------------------------------------------------------------------


def test_dual_direction_task1(task1_manifest):
    pid, manifest = task1_manifest
    ds = _build_dataset(pid, manifest)

    pairs = {(int(item["m_s"]), int(item["m_t"])) for item in ds.items}
    assert (0, 1) in pairs  # MRI -> CT
    assert (1, 0) in pairs  # CT -> MRI


def test_dual_direction_task2(task2_manifest):
    pid, manifest = task2_manifest
    ds = _build_dataset(pid, manifest)

    pairs = {(int(item["m_s"]), int(item["m_t"])) for item in ds.items}
    assert (2, 1) in pairs  # CBCT -> CT
    assert (1, 2) in pairs  # CT -> CBCT


def test_batch_keys(task1_manifest):
    pid, manifest = task1_manifest
    ds = _build_dataset(pid, manifest)
    batch = ds[0]

    for key in ("source", "target", "mask", "anatomy", "patient_id", "m_s", "m_t", "direction_id"):
        assert key in batch, f"missing key: {key}"
    assert batch["source"].shape == (1, 8, 8, 8)
    assert batch["target"].shape == (1, 8, 8, 8)
    assert batch["mask"].shape == (1, 8, 8, 8)
