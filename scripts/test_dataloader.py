"""Smoke test for CombinedDataModule DataLoader worker hang fix.

Verifies that:
1. num_workers=0  → 3 batches iterate cleanly (basic sanity)
2. num_workers=2  → 3 batches iterate without hanging within 60 s

Usage::
    python scripts/test_dataloader.py \
        --manifest-2023 outputs/preprocessed/manifest.json \
        --splits-2023   outputs/splits.json \
        --manifest-2025 outputs/preprocessed_2025/manifest_2025.json \
        --splits-2025   outputs/splits_synthrad2025.json

Exit code 0 = all good, non-zero = failure.
"""

from __future__ import annotations

import argparse
import multiprocessing
import signal
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path regardless of where the script is invoked
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DataLoader smoke test for CombinedDataModule")
    p.add_argument("--manifest-2023", type=Path, required=True,
                   help="Path to outputs/preprocessed/manifest.json")
    p.add_argument("--splits-2023", type=Path, required=True,
                   help="Path to outputs/splits.json")
    p.add_argument("--manifest-2025", type=Path, required=True,
                   help="Path to outputs/preprocessed_2025/manifest_2025.json")
    p.add_argument("--splits-2025", type=Path, required=True,
                   help="Path to outputs/splits_synthrad2025.json")
    p.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96],
                   metavar=("D", "H", "W"),
                   help="Patch size (default: 96 96 96)")
    p.add_argument("--samples-per-volume", type=int, default=2,
                   help="Samples per volume (default: 2)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for smoke test (default: 1)")
    p.add_argument("--n-batches", type=int, default=3,
                   help="Number of batches to iterate (default: 3)")
    p.add_argument("--timeout", type=int, default=60,
                   help="Per-run timeout in seconds (default: 60)")
    p.add_argument("--skip-multiworker", action="store_true",
                   help="Skip the num_workers=2 test (for CI with no fork)")
    return p.parse_args()


def _run_dataloader(
    manifest_2023: Path,
    splits_2023: Path,
    manifest_2025: Path,
    splits_2025: Path,
    patch_size: list[int],
    samples_per_volume: int,
    batch_size: int,
    num_workers: int,
    n_batches: int,
    result_queue,   # multiprocessing.Queue
) -> None:
    """Worker function: builds DM, iterates n_batches, puts results in queue."""
    try:
        from src.data.combined_datamodule import CombinedDataModule

        dm = CombinedDataModule.from_primitives(
            splits_path=splits_2023,
            manifest_path=manifest_2023,
            synthrad2025_splits_path=splits_2025,
            manifest_2025_path=manifest_2025,
            patch_size=patch_size,
            samples_per_volume=samples_per_volume,
            fg_fraction=0.8,
            base_seed=42,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        dm.setup("fit")

        loader = dm.train_dataloader()
        shapes = []
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            shape_info = {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")}
            shapes.append(shape_info)

        result_queue.put(("ok", shapes))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def _run_with_timeout(
    *,
    manifest_2023: Path,
    splits_2023: Path,
    manifest_2025: Path,
    splits_2025: Path,
    patch_size: list[int],
    samples_per_volume: int,
    batch_size: int,
    num_workers: int,
    n_batches: int,
    timeout: int,
    label: str,
) -> bool:
    """Runs the dataloader test in a subprocess with a wall-clock timeout.

    Returns True on success, False on failure/timeout.
    """
    print(f"\n{'='*60}")
    print(f"  TEST: {label}  (timeout={timeout}s)")
    print(f"{'='*60}")

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_run_dataloader,
        args=(
            manifest_2023, splits_2023, manifest_2025, splits_2025,
            patch_size, samples_per_volume, batch_size, num_workers, n_batches, q,
        ),
    )

    t0 = time.monotonic()
    p.start()
    p.join(timeout=timeout)
    elapsed = time.monotonic() - t0

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        print(f"  [FAIL] TIMED OUT after {elapsed:.1f}s — DataLoader hang confirmed!", flush=True)
        return False

    if not q.empty():
        status, payload = q.get_nowait()
        if status == "ok":
            print(f"  [PASS] Completed in {elapsed:.1f}s", flush=True)
            print(f"  Batch shapes ({n_batches} batches):", flush=True)
            for i, shape_info in enumerate(payload):
                print(f"    batch {i}: {shape_info}", flush=True)
            return True
        else:
            print(f"  [FAIL] Exception in worker: {payload}", flush=True)
            return False
    else:
        exit_code = p.exitcode
        print(f"  [FAIL] Process exited with code {exit_code} and no result (elapsed={elapsed:.1f}s)", flush=True)
        return False


def main() -> None:
    # Must use 'spawn' start method at program entry on Linux to avoid
    # issues with any existing fork state from import-time side effects.
    multiprocessing.set_start_method("spawn", force=True)

    args = _parse_args()

    # Validate paths
    for path_arg, name in [
        (args.manifest_2023, "--manifest-2023"),
        (args.splits_2023,   "--splits-2023"),
        (args.manifest_2025, "--manifest-2025"),
        (args.splits_2025,   "--splits-2025"),
    ]:
        if not path_arg.exists():
            print(f"ERROR: {name} path does not exist: {path_arg}", file=sys.stderr)
            sys.exit(1)

    common = dict(
        manifest_2023=args.manifest_2023,
        splits_2023=args.splits_2023,
        manifest_2025=args.manifest_2025,
        splits_2025=args.splits_2025,
        patch_size=args.patch_size,
        samples_per_volume=args.samples_per_volume,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        timeout=args.timeout,
    )

    results = []

    # --- Test 1: num_workers=0 (must always work) ---
    ok0 = _run_with_timeout(
        num_workers=0,
        label="num_workers=0 (single-process, baseline)",
        **common,
    )
    results.append(("num_workers=0", ok0))

    # --- Test 2: num_workers=2 (catches worker hang) ---
    if not args.skip_multiworker:
        ok2 = _run_with_timeout(
            num_workers=2,
            label="num_workers=2 (multi-process, hang detection)",
            **common,
        )
        results.append(("num_workers=2", ok2))
    else:
        print("\n[SKIP] num_workers=2 test skipped (--skip-multiworker)", flush=True)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for label, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All DataLoader smoke tests passed.\n")
        sys.exit(0)
    else:
        print("\n✗ One or more DataLoader smoke tests FAILED.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
