"""I2SB (Image-to-Image Schrödinger Bridge) generative process — Method Spec §3.

Reference: Liu et al., ICML 2023, arXiv:2302.05872 (ADR-013). Simulation-free,
tractable, analytically computable marginals. This module implements the
forward marginal, the x0-parameterized training loss, and a deterministic
DDIM-style reverse sampler. It wraps (but does not modify) a denoiser with
the ThermoBridgeDenoiser interface: f_theta(x_t, t, m_s, m_t, alpha) -> x_hat_0.

Do not implement the score objective from scratch — this follows I2SB's
forward marginal and training loss exactly, per §3.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

# f_theta(x_t, t, m_s, m_t, alpha) -> x_hat_0
DenoiserFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _reshape_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a (B,) tensor to (B, 1, 1, 1, 1) matching x's rank."""
    return t.view(-1, *([1] * (x.dim() - 1)))


class I2SBProcess:
    """I2SB forward marginal, training loss, and deterministic reverse sampler (§3).

    Args:
        max_variance_s: s — max-variance hyperparameter (open value, ADR pending sweep).
        num_steps:      Default number of reverse-sampling steps (open value).
        time_weighting: One of "constant", "snr", "cosine" (open value, §3).
    """

    def __init__(self, max_variance_s: float, num_steps: int, time_weighting: str) -> None:
        if time_weighting not in ("constant", "snr", "cosine"):
            raise ValueError(f"Unknown time_weighting: {time_weighting!r}")
        self.s = float(max_variance_s)
        self.num_steps = int(num_steps)
        self.time_weighting_name = time_weighting

    # ------------------------------------------------------------------
    # Forward marginal — q(x_t | x_0, x_T) = N(mu_t, sigma_t^2 * I)  (§3)
    # ------------------------------------------------------------------

    def marginal_params(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """t: (B,) float in [0,1] (T=1.0, so t_bar == t). Returns (t_bar, sigma_t_sq), each (B,)."""
        t_bar = t
        sigma_t_sq = 2.0 * self.s * t_bar * (1.0 - t_bar)
        return t_bar, sigma_t_sq

    def forward_marginal(
        self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0, x_T) (§3).

        Args:
            x_0: (B, 1, D, H, W) target volume.
            x_T: (B, 1, D, H, W) source volume.
            t:   (B,) float timestep in [0, 1].

        Returns:
            (x_t, noise): sampled bridge state and the standard-normal noise used.
        """
        t_bar, sigma_t_sq = self.marginal_params(t)
        t_bar_b = _reshape_like(t_bar, x_0)
        sigma_t_b = _reshape_like(torch.sqrt(sigma_t_sq.clamp(min=0.0)), x_0)

        mu_t = (1.0 - t_bar_b) * x_0 + t_bar_b * x_T
        noise = torch.randn_like(x_0)
        x_t = mu_t + sigma_t_b * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Time weighting w(t)  (§3)
    # ------------------------------------------------------------------

    def time_weighting(self, t: torch.Tensor) -> torch.Tensor:
        """Returns w_t: (B,) per the configured schedule."""
        t_bar, sigma_t_sq = self.marginal_params(t)
        if self.time_weighting_name == "constant":
            return torch.ones_like(t_bar)
        if self.time_weighting_name == "snr":
            return 1.0 / (sigma_t_sq + 1e-8)
        # cosine
        return 0.5 * (1.0 + torch.cos(math.pi * t_bar))

    # ------------------------------------------------------------------
    # Training loss — x0-parameterization (§3)
    # ------------------------------------------------------------------

    def bridge_loss(
        self,
        f_theta: DenoiserFn,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        t: torch.Tensor,
        m_s: torch.Tensor,
        m_t: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """L_bridge = E[w(t) * ||f_theta(x_t, t, m_s, m_t, alpha) - x_0||_1]  (§3).

        Note: this is the L_rec term only (mask-weighting and the other loss
        terms — L_bnd, L_rad, L_ent, L_bal, L_cls — are separate §7 terms
        applied by the training module, not this bridge process).
        """
        x_t, _noise = self.forward_marginal(x_0, x_T, t)
        x_hat_0 = f_theta(x_t, t, m_s, m_t, alpha)

        w_t = self.time_weighting(t)  # (B,)
        l1_per_sample = (x_hat_0 - x_0).abs().flatten(1).mean(dim=1)  # (B,)
        return (w_t * l1_per_sample).mean()

    # ------------------------------------------------------------------
    # Reverse sampling — deterministic DDIM-style traversal (§3)
    # ------------------------------------------------------------------

    def reverse_sample(
        self,
        f_theta: DenoiserFn,
        x_T: torch.Tensor,
        m_s: torch.Tensor,
        m_t: torch.Tensor,
        alpha: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Deterministic reverse traversal from x_T (source) to x_hat_0 (target).

        No stochastic noise injection at inference (§3): at each step we predict
        x_hat_0 directly, then move to the next timestep using the analytic
        bridge mean (mu_t), i.e. no randn() call anywhere in this method.
        """
        steps = int(num_steps) if num_steps is not None else self.num_steps
        B = x_T.shape[0]
        device = x_T.device

        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

        x_t = x_T
        x_hat_0 = x_T
        for i in range(steps):
            t_cur = ts[i].expand(B)
            x_hat_0 = f_theta(x_t, t_cur, m_s, m_t, alpha)

            t_next = ts[i + 1]
            if float(t_next) <= 0.0:
                x_t = x_hat_0
            else:
                t_next_b = t_next.expand(B)
                t_next_r = _reshape_like(t_next_b, x_T)
                x_t = (1.0 - t_next_r) * x_hat_0 + t_next_r * x_T

        return x_hat_0
