"""Deterministic patient-level stratified split for SynthRAD2023 Task1.

Strata (non-empty cells only — pelvis has NO center B):
    brain-A:60  brain-B:60  brain-C:60  pelvis-A:120  pelvis-C:60

Each stratum is split independently at (train/val/test) fractions from config,
then unioned.  Expected: ~252 / 54 / 54.

CLI::
    python src/data/splits.py --config configs/default.yaml

Outputs (R5 — never under data/):
    outputs/splits.json

Rules: R3 (patient-level, zero overlap), R6 (seed from config, deterministic).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import load_config  # noqa: E402

_MANIFEST_PATH = _REPO_ROOT / "outputs" / "preprocessed" / "manifest.json"
_SPLITS_PATH   = _REPO_ROOT / "outputs" / "splits.json"


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------


def make_splits(
    manifest: dict,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> dict[str, list[str]]:
    """Stratify patients by (anatomy, center), split each stratum independently.

    Args:
        manifest:    manifest.json dict keyed by patient_id.
        seed:        Global RNG seed (R6).
        train_frac:  Fraction for training set.
        val_frac:    Fraction for validation set.
        test_frac:   Fraction for test set.

    Returns:
        Dict with keys 'train', 'val', 'test', each a sorted list of patient_ids.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0, got {train_frac+val_frac+test_frac:.4f}"
        )

    # Group patients by stratum
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, entry in manifest.items():
        key = (entry["anatomy"], entry["center"])
        strata[key].append(pid)

    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids:   list[str] = []
    test_ids:  list[str] = []

    for key in sorted(strata.keys()):
        pids = sorted(strata[key])  # deterministic order before shuffle
        # Each stratum gets its own sub-seed derived from the global seed
        sub_seed = int(rng.integers(2**32))
        pids_shuffled: list[str] = np.random.default_rng(sub_seed).permutation(pids).tolist()

        n = len(pids_shuffled)
        n_val   = round(n * val_frac)
        n_test  = round(n * test_frac)
        n_train = n - n_val - n_test  # remainder to train (avoids off-by-one)

        train_ids.extend(pids_shuffled[:n_train])
        val_ids.extend(pids_shuffled[n_train : n_train + n_val])
        test_ids.extend(pids_shuffled[n_train + n_val :])

    return {
        "train": sorted(train_ids),
        "val":   sorted(val_ids),
        "test":  sorted(test_ids),
    }


def assert_no_leakage(splits: dict[str, list[str]]) -> None:
    """Assert zero patient overlap across splits (R3)."""
    train = set(splits["train"])
    val   = set(splits["val"])
    test  = set(splits["test"])
    tv = train & val
    tt = train & test
    vt = val   & test
    if tv or tt or vt:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED — train∩val={len(tv)}, "
            f"train∩test={len(tt)}, val∩test={len(vt)}"
        )


def print_summary(splits: dict[str, list[str]], manifest: dict) -> None:
    """Print per-stratum and per-split counts."""
    print("\n── Per-stratum split counts ──")
    strata_all: dict[tuple, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for split_name, pids in splits.items():
        for pid in pids:
            e = manifest[pid]
            strata_all[(e["anatomy"], e["center"])][split_name] += 1

    header = f"  {'stratum':<18}  {'train':>6}  {'val':>5}  {'test':>5}  {'total':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in sorted(strata_all.keys()):
        c = strata_all[key]
        tot = c["train"] + c["val"] + c["test"]
        print(f"  {key[0]+'-'+key[1]:<18}  {c['train']:>6}  {c['val']:>5}  {c['test']:>5}  {tot:>6}")

    print("\n── Per-split totals ──")
    for name, pids in splits.items():
        anatomies = {manifest[p]["anatomy"] for p in pids}
        print(f"  {name:<8}: {len(pids):>4} patients  anatomies present: {sorted(anatomies)}")

    total = sum(len(v) for v in splits.values())
    print(f"\n  TOTAL: {total} patients across all splits")


def save_splits(splits: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: Path) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ThermoBridge — deterministic patient splits (R3/R6).")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--manifest", type=Path, default=_MANIFEST_PATH)
    p.add_argument("--out",      type=Path, default=_SPLITS_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not args.manifest.exists():
        sys.exit(f"ERROR: manifest not found at {args.manifest}. Run preprocess.py first.")

    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"Loaded manifest: {len(manifest)} patients.")

    splits = make_splits(
        manifest,
        seed=int(cfg.data.split.seed),
        train_frac=float(cfg.data.split.train),
        val_frac=float(cfg.data.split.val),
        test_frac=float(cfg.data.split.test),
    )

    assert_no_leakage(splits)
    print("✓ Zero patient overlap across splits.")

    print_summary(splits, manifest)
    save_splits(splits, args.out)
    print(f"\nSplits written → {args.out}")


if __name__ == "__main__":
    main()
