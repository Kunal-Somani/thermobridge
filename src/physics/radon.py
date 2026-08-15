"""Radon-domain consistency loss for CT/CBCT targets — Method Spec §7 (L_rad row),
ADR-011.

Radiotherapy dose calculation depends on line integrals of attenuation
(Radon / forward-projection space). ADR-011 makes synthetic CT/CBCT
consistent with ground truth in that space — the direct analogue of IC3's
k-space data-consistency term, and the clinically relevant quantity rather
than a cosmetic image-space metaphor.

FastRadonProjector uses a fast approximation: summed intensity projections
along the 3 orthogonal axes (axial, coronal, sagittal) — essentially 3
digitally-reconstructed-radiograph (DRR) views — rather than a full
sinogram (projections at many angles via the real Radon transform). This is
intentional: it is cheap, fully differentiable, and sufficient to penalize
gross attenuation-integral mismatches for a training-time consistency loss;
it is NOT a substitute for a full forward-projection operator in a dose
calculation pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastRadonProjector(nn.Module):
    """Differentiable 3D forward-projection approximation (§7, ADR-011).

    Fast approximation: sum projections along the 3 orthogonal axes (axial,
    coronal, sagittal) instead of a full sinogram. No learnable parameters —
    purely analytical.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, D, H, W) in [-1, 1] -> projections: (B, 3, H, W)."""
        axial = x.sum(dim=2)      # sum over D -> (B, 1, H, W)
        coronal = x.sum(dim=3)    # sum over H -> (B, 1, D, W)
        sagittal = x.sum(dim=4)   # sum over W -> (B, 1, D, H)

        # coronal/sagittal are (D, W) / (D, H) shaped; resize to the axial
        # view's (H, W) so the three DRRs can be stacked into one tensor.
        # For the cubic patches ThermoBridge trains on (D == H == W) this is
        # a no-op; interpolate keeps it well-defined (and differentiable)
        # for non-cubic volumes too (e.g. full-volume validation).
        target_hw = axial.shape[-2:]
        if coronal.shape[-2:] != target_hw:
            coronal = F.interpolate(coronal, size=target_hw, mode="bilinear", align_corners=False)
        if sagittal.shape[-2:] != target_hw:
            sagittal = F.interpolate(sagittal, size=target_hw, mode="bilinear", align_corners=False)

        return torch.cat([axial, coronal, sagittal], dim=1)  # (B, 3, H, W)


class RadonConsistencyLoss(nn.Module):
    """L_rad = ||R(x_hat_0) - R(x_0)||_1, applied only when the target is CT/CBCT (§7).

    Args:
        projector: A FastRadonProjector (or compatible) instance.
    """

    def __init__(self, projector: FastRadonProjector) -> None:
        super().__init__()
        self.projector = projector

    def forward(
        self,
        x_hat_0: torch.Tensor,
        x_0: torch.Tensor,
        is_ct_or_cbct_mask: torch.Tensor | bool | float,
    ) -> torch.Tensor:
        """
        Args:
            x_hat_0: (B, 1, D, H, W) predicted volume.
            x_0:     (B, 1, D, H, W) ground-truth target volume.
            is_ct_or_cbct_mask: per-sample (B,) mask, or a single bool/float
                gate for the whole batch. Samples/batches where this is 0
                (target is MRI) contribute 0 to the loss and receive no
                gradient from this term.

        Returns:
            Scalar tensor (always a tensor — 0.0 as a tensor, never None or
            a plain Python float, so loss arithmetic in the caller never breaks).
        """
        proj_hat = self.projector(x_hat_0)
        proj_gt = self.projector(x_0)
        l1 = (proj_hat - proj_gt).abs()  # (B, 3, H, W)

        mask = torch.as_tensor(is_ct_or_cbct_mask, dtype=l1.dtype, device=l1.device)
        if mask.dim() == 0:
            mask = mask.expand(l1.shape[0])
        mask = mask.view(-1, 1, 1, 1)

        n_active = mask.sum()
        if n_active.item() == 0:
            return torch.zeros((), device=l1.device, dtype=l1.dtype)

        per_sample_elems = l1.shape[1] * l1.shape[2] * l1.shape[3]
        return (l1 * mask).sum() / (n_active * per_sample_elems)
