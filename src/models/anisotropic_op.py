"""Learnable 3D anisotropic-diffusion operator — local mixer, Method Spec §6
(CONTRIBUTION B, ADR-004).

Perona-Malik edge-stopping diffusion:
    g(s)   = 1 / (1 + (s/K)^2)          K learnable, per-channel
    dI/dt  = div( g(||grad_I||) * grad_I )

Implemented as `num_steps` explicit Euler steps on 3D spatial gradients. Each
step uses the classic 6-neighbor (±x, ±y, ±z) discretization: for every
direction, compute the directional finite difference to that neighbor ("the
gradient at that face"), pass its magnitude through the conductance g(), and
sum the six conductance-weighted directional differences into the discrete
divergence (flux-conservative form; forward diff in the + direction, backward
diff in the - direction, each zeroed at the volume boundary — a no-flux /
Neumann boundary condition, i.e. one-sided differencing at the edges).

Motivation (§6): the I2SB forward corruption is isotropic; this operator is
anisotropic (stops diffusing across edges). Targets the bone/soft-tissue
boundary failure mode (IC3 Fig. 9). A standard 3D conv (`ConvLocalMixer`) is
provided for the B1/B2 ablation (operator vs. plain conv).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def perona_malik_conductance(s: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """g(s) = 1 / (1 + (s/K)^2)  (§6). K must already be strictly positive."""
    return 1.0 / (1.0 + (s / K) ** 2)


def _inverse_softplus(y: float) -> float:
    """raw such that softplus(raw) == y, for initializing a softplus-parameterized value."""
    return math.log(math.expm1(y))


def _directional_diff(I: torch.Tensor, dim: int, forward: bool) -> torch.Tensor:
    """Neighbor difference along `dim`; zeroed at the boundary (no-flux BC)."""
    shift = -1 if forward else 1
    neighbor = torch.roll(I, shifts=shift, dims=dim)
    diff = neighbor - I
    boundary_index = -1 if forward else 0
    idx = [slice(None)] * I.dim()
    idx[dim] = boundary_index
    diff = diff.clone()
    diff[tuple(idx)] = 0.0
    return diff


class AnisotropicDiffusionOp(nn.Module):
    """Perona-Malik 3D anisotropic diffusion, num_steps explicit Euler steps (§6).

    Args:
        num_channels:      C — number of feature channels.
        num_steps:         Number of explicit Euler steps per forward call.
        per_channel_k:     If True, K has shape (C,); else a single shared K.
        init_conductance_k: Initial value of K (learnable; kept positive via softplus).
        init_step_size:     Initial value of dt (learnable; kept positive via softplus).
    """

    def __init__(
        self,
        num_channels: int,
        num_steps: int,
        per_channel_k: bool,
        init_conductance_k: float,
        init_step_size: float,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_steps = num_steps
        self.per_channel_k = per_channel_k

        k_shape = (num_channels,) if per_channel_k else (1,)
        # Raw (unconstrained) parameters; forward() applies softplus so K, dt > 0 always.
        self.K_raw = nn.Parameter(torch.full(k_shape, _inverse_softplus(init_conductance_k)))
        self.dt_raw = nn.Parameter(torch.full((1,), _inverse_softplus(init_step_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W) -> same shape."""
        B, C, D, H, W = x.shape

        K = F.softplus(self.K_raw)
        K = K.view(1, C, 1, 1, 1) if self.per_channel_k else K.view(1, 1, 1, 1, 1)
        dt = F.softplus(self.dt_raw).view(1, 1, 1, 1, 1)

        I = x
        for _ in range(self.num_steps):
            divergence = torch.zeros_like(I)
            for dim in (2, 3, 4):  # D, H, W — 3D spatial gradients, not 2D slices
                for forward in (True, False):
                    diff = _directional_diff(I, dim, forward)
                    conductance = perona_malik_conductance(diff.abs(), K)
                    divergence = divergence + conductance * diff
            I = I + dt * divergence
        return I


class ConvLocalMixer(nn.Module):
    """Standard 3D conv local mixer — ablation B1/B2 baseline (§6, ADR-004).

    Depthwise 3x3x3 conv + pointwise 1x1x1 conv, same in/out channels.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels
        )
        self.pointwise = nn.Conv3d(num_channels, num_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W) -> same shape."""
        return self.pointwise(self.depthwise(x))
