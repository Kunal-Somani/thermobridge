"""VRAM profiler for ThermoBridge 3-D models.

Sweeps patch sizes [64, 96, 128] cubed at configurable batch sizes,
runs one forward + backward pass, records peak VRAM, and prints a table.

This profiles the U-Net only (a fast proxy). The final binding patch-size
decision will be re-profiled on the transformer denoiser (chunk-8).

Usage::
    python src/utils/vram_profile.py --config configs/default.yaml

Output: a table + outputs/reports/vram_profile.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.unet3d import build_unet3d
from src.utils.config import load_config


def _gb(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def profile_one(
    model: "torch.nn.Module",
    patch_size: tuple[int, int, int],
    batch_size: int,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Run one forward + backward, return VRAM stats.

    Returns:
        Dict with keys: patch_size, batch_size, peak_vram_gb, status.
    """
    pZ, pY, pX = patch_size
    result = {
        "patch_size": f"{pZ}x{pY}x{pX}",
        "batch_size": batch_size,
        "peak_vram_gb": None,
        "status": "ok",
    }

    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        source = torch.randn(batch_size, 1, pZ, pY, pX, device=device, requires_grad=False)
        dir_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        target = torch.randn(batch_size, 1, pZ, pY, pX, device=device)

        model.train()
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(source, dir_id)
            loss = torch.nn.functional.l1_loss(pred, target)

        # Backward pass (scaler-free; just want the memory footprint)
        loss.backward()
        model.zero_grad(set_to_none=True)

        peak = torch.cuda.max_memory_allocated(device)
        result["peak_vram_gb"] = round(_gb(peak), 3)

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        torch.cuda.empty_cache()

    except Exception as exc:
        result["status"] = f"ERROR: {exc}"

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="ThermoBridge VRAM profiler.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--patch-sizes", nargs="+", type=int, default=[64, 96, 128],
                   help="Cube side lengths to sweep (e.g. 64 96 128).")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2],
                   help="Batch sizes to test.")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "outputs" / "reports")
    args = p.parse_args()

    cfg   = load_config(args.config)
    model = build_unet3d(cfg.training)

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU found. VRAM profiling requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    model  = model.to(device)
    use_amp = not args.no_amp

    total_vram = _gb(torch.cuda.get_device_properties(device).total_memory)
    name       = torch.cuda.get_device_name(device)
    print(f"\nGPU: {name}  |  Total VRAM: {total_vram:.2f} GB")
    print(f"Model: UNet3D  channels={list(cfg.training.unet_channels)}")
    print(f"AMP: {use_amp}\n")

    header = f"  {'patch':>10}  {'batch':>6}  {'peak_GB':>9}  {'status':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows: list[dict] = []
    for ps in args.patch_sizes:
        patch_size = (ps, ps, ps)
        for bs in args.batch_sizes:
            res = profile_one(model, patch_size, bs, device, use_amp)
            rows.append(res)
            peak_str = f"{res['peak_vram_gb']:.3f}" if res["peak_vram_gb"] else " ---"
            fits = "✓ fits" if res["status"] == "ok" else res["status"]
            print(f"  {res['patch_size']:>10}  {bs:>6}  {peak_str:>9}  {fits:>8}")

    # Write CSV
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "vram_profile.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch_size", "batch_size", "peak_vram_gb", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTable written → {csv_path}")

    # Summary recommendation
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        best = max(ok_rows, key=lambda r: int(r["patch_size"].split("x")[0]))
        print(f"\nLargest fitting patch: {best['patch_size']}  "
              f"batch={best['batch_size']}  peak={best['peak_vram_gb']:.3f} GB")
        print("NOTE: this is a U-Net proxy. Re-profile with the transformer denoiser (chunk-8).")


if __name__ == "__main__":
    main()
