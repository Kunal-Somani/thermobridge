"""Tests for patient-level split integrity — ThermoBridge (R3/R7).

Run::
    pytest tests/test_splits.py -v
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.splits import assert_no_leakage, load_splits, make_splits

_MANIFEST_PATH = _REPO_ROOT / "outputs" / "preprocessed" / "manifest.json"
_SPLITS_PATH   = _REPO_ROOT / "outputs" / "splits.json"

_TRAIN_FRAC = 0.70
_VAL_FRAC   = 0.15
_TEST_FRAC  = 0.15
_SEED       = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not _MANIFEST_PATH.exists():
        pytest.skip("manifest.json not found — run preprocess.py first.")
    with open(_MANIFEST_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def splits(manifest: dict) -> dict:
    """Generate splits fresh (does NOT depend on the saved file for isolation)."""
    return make_splits(manifest, _SEED, _TRAIN_FRAC, _VAL_FRAC, _TEST_FRAC)


@pytest.fixture(scope="module")
def saved_splits() -> dict:
    if not _SPLITS_PATH.exists():
        pytest.skip("splits.json not found — run splits.py first.")
    return load_splits(_SPLITS_PATH)


# ---------------------------------------------------------------------------
# 1. Zero patient leakage  (R3)
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_train_val_disjoint(self, splits: dict) -> None:
        overlap = set(splits["train"]) & set(splits["val"])
        assert overlap == set(), f"Train∩Val overlap: {overlap}"

    def test_train_test_disjoint(self, splits: dict) -> None:
        overlap = set(splits["train"]) & set(splits["test"])
        assert overlap == set(), f"Train∩Test overlap: {overlap}"

    def test_val_test_disjoint(self, splits: dict) -> None:
        overlap = set(splits["val"]) & set(splits["test"])
        assert overlap == set(), f"Val∩Test overlap: {overlap}"

    def test_assert_no_leakage_helper_passes(self, splits: dict) -> None:
        assert_no_leakage(splits)  # must not raise


# ---------------------------------------------------------------------------
# 2. Total count & uniqueness
# ---------------------------------------------------------------------------


class TestTotalCounts:
    def test_total_is_360(self, splits: dict) -> None:
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 360, f"Expected 360 total patients, got {total}"

    def test_no_duplicate_patient_ids(self, splits: dict) -> None:
        all_ids = splits["train"] + splits["val"] + splits["test"]
        assert len(all_ids) == len(set(all_ids)), "Duplicate patient IDs detected."

    def test_expected_train_size(self, splits: dict) -> None:
        # 5 strata: 3×42 + 84 + 42 = 252
        assert len(splits["train"]) == 252, f"Expected 252 train, got {len(splits['train'])}"

    def test_expected_val_size(self, splits: dict) -> None:
        # 5 strata: 3×9 + 18 + 9 = 54
        assert len(splits["val"]) == 54, f"Expected 54 val, got {len(splits['val'])}"

    def test_expected_test_size(self, splits: dict) -> None:
        assert len(splits["test"]) == 54, f"Expected 54 test, got {len(splits['test'])}"


# ---------------------------------------------------------------------------
# 3. Both anatomies present in every split
# ---------------------------------------------------------------------------


class TestAnatomyPresence:
    def test_both_anatomies_in_train(self, splits: dict, manifest: dict) -> None:
        anat = {manifest[p]["anatomy"] for p in splits["train"]}
        assert "brain"  in anat, "Brain missing from train split."
        assert "pelvis" in anat, "Pelvis missing from train split."

    def test_both_anatomies_in_val(self, splits: dict, manifest: dict) -> None:
        anat = {manifest[p]["anatomy"] for p in splits["val"]}
        assert "brain"  in anat, "Brain missing from val split."
        assert "pelvis" in anat, "Pelvis missing from val split."

    def test_both_anatomies_in_test(self, splits: dict, manifest: dict) -> None:
        anat = {manifest[p]["anatomy"] for p in splits["test"]}
        assert "brain"  in anat, "Brain missing from test split."
        assert "pelvis" in anat, "Pelvis missing from test split."


# ---------------------------------------------------------------------------
# 4. Per-stratum split ratios
# ---------------------------------------------------------------------------


class TestStratumRatios:
    """Each (anatomy, center) stratum should have the correct per-split counts."""

    # Expected counts from the 5 non-empty strata at 70/15/15
    _EXPECTED: dict[tuple[str, str], dict[str, int]] = {
        ("brain",  "A"): {"train": 42, "val": 9, "test": 9},
        ("brain",  "B"): {"train": 42, "val": 9, "test": 9},
        ("brain",  "C"): {"train": 42, "val": 9, "test": 9},
        ("pelvis", "A"): {"train": 84, "val": 18, "test": 18},
        ("pelvis", "C"): {"train": 42, "val": 9,  "test": 9},
    }

    def _stratum_counts(
        self, splits: dict, manifest: dict
    ) -> dict[tuple[str, str], dict[str, int]]:
        counts: dict = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
        for split_name, pids in splits.items():
            for pid in pids:
                e = manifest[pid]
                counts[(e["anatomy"], e["center"])][split_name] += 1
        return dict(counts)

    def test_per_stratum_train_counts(self, splits: dict, manifest: dict) -> None:
        counts = self._stratum_counts(splits, manifest)
        for key, exp in self._EXPECTED.items():
            got = counts.get(key, {}).get("train", 0)
            assert got == exp["train"], (
                f"Stratum {key} train: expected {exp['train']}, got {got}"
            )

    def test_per_stratum_val_counts(self, splits: dict, manifest: dict) -> None:
        counts = self._stratum_counts(splits, manifest)
        for key, exp in self._EXPECTED.items():
            got = counts.get(key, {}).get("val", 0)
            assert got == exp["val"], (
                f"Stratum {key} val: expected {exp['val']}, got {got}"
            )

    def test_per_stratum_test_counts(self, splits: dict, manifest: dict) -> None:
        counts = self._stratum_counts(splits, manifest)
        for key, exp in self._EXPECTED.items():
            got = counts.get(key, {}).get("test", 0)
            assert got == exp["test"], (
                f"Stratum {key} test: expected {exp['test']}, got {got}"
            )

    def test_pelvis_has_no_center_b(self, splits: dict, manifest: dict) -> None:
        """Sanity: the pelvis-B stratum must not appear in any split."""
        counts = self._stratum_counts(splits, manifest)
        assert ("pelvis", "B") not in counts, (
            "Unexpected pelvis-B stratum — data structure changed?"
        )


# ---------------------------------------------------------------------------
# 5. Determinism  (R6)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_result_on_repeated_call(self, manifest: dict) -> None:
        s1 = make_splits(manifest, _SEED, _TRAIN_FRAC, _VAL_FRAC, _TEST_FRAC)
        s2 = make_splits(manifest, _SEED, _TRAIN_FRAC, _VAL_FRAC, _TEST_FRAC)
        assert s1 == s2, "make_splits() is not deterministic — seed not working."

    def test_different_seed_gives_different_split(self, manifest: dict) -> None:
        s1 = make_splits(manifest, _SEED, _TRAIN_FRAC, _VAL_FRAC, _TEST_FRAC)
        s2 = make_splits(manifest, _SEED + 1, _TRAIN_FRAC, _VAL_FRAC, _TEST_FRAC)
        assert s1["train"] != s2["train"], (
            "Different seeds produced identical splits — suspicious."
        )

    def test_saved_splits_match_generated(self, splits: dict, saved_splits: dict) -> None:
        """Splits.json on disk must match freshly generated splits (R6 frozen)."""
        assert splits["train"] == saved_splits["train"], "Train split mismatch between saved and generated."
        assert splits["val"]   == saved_splits["val"],   "Val split mismatch."
        assert splits["test"]  == saved_splits["test"],  "Test split mismatch."
