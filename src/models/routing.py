"""N-way sparse anatomy routing gate — Method Spec §5 (CORE CONTRIBUTION A, ADR-003).

Fixes four flaws of the IC3 sigmoid-scalar gate (§5 table):
    - softmax over the simplex Delta^{A-1} instead of a 2-branch sigmoid
    - top-k sparse selection (k in {1,2}) instead of always evaluating all A branches
    - temperature annealing (soft -> sharp) instead of a BCE soft/hard contradiction
    - entropy + load-balance losses instead of requiring anatomy labels

Gate computation (§5):
    h          = MLP( GAP( Conv3D_small(x_T) ) )      # logits in R^A
    alpha_soft = softmax( h / tau )                    # simplex Delta^{A-1}
    S          = top-k indices of alpha_soft           # k in {1,2}; default k=2
    alpha_i    = alpha_soft_i / sum_{j in S} alpha_soft_j  for i in S; 0 otherwise

Top-k selection is made differentiable via renormalization (as specified in
§5), not Gumbel-Softmax straight-through — that is a separate ablation (C2).

Expert combination (§5):
    y = f_shared(x) + sum_{i in S} alpha_i * Adapter_i(f_shared(x))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gate: AnatomyRouter
# ---------------------------------------------------------------------------


class AnatomyRouter(nn.Module):
    """N-way sparse anatomy routing gate (§5).

    Args:
        in_channels:    Channels of the volume the gate reads (x_T has 1).
        hidden_dim:     Width of the gate's small conv/MLP (independent of
                         the denoiser's transformer hidden_dim).
        num_anatomies:  A — number of anatomy experts.
        top_k:          k — number of experts kept per sample (k in {1,2}).
        adapter_rank:   Bottleneck rank of the per-anatomy adapters. Not used
                         by the gate itself; kept here for config cohesion so
                         a single AnatomyRouter instance carries every
                         routing-related constant needed to build the
                         matching RoutedAdapterBlocks.
        tau_max, tau_min, total_epochs: temperature-annealing schedule (§5).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_anatomies: int,
        top_k: int,
        adapter_rank: int,
        tau_max: float,
        tau_min: float,
        total_epochs: int,
    ) -> None:
        super().__init__()
        self.num_anatomies = num_anatomies
        self.top_k = min(top_k, num_anatomies)
        self.adapter_rank = adapter_rank
        self.tau_max = float(tau_max)
        self.tau_min = float(tau_min)
        self.total_epochs = int(total_epochs)

        # Current temperature — updated externally each epoch via
        # `router.tau = router.tau_schedule(epoch)`; starts at tau_max (softest).
        self.tau = self.tau_max

        self.conv_small = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_anatomies),
        )

    def tau_schedule(self, epoch: int) -> float:
        """tau(e) = tau_max * (tau_min/tau_max)^(e/E)  (§5)."""
        e = max(0, min(epoch, self.total_epochs))
        ratio = self.tau_min / self.tau_max
        return self.tau_max * (ratio ** (e / self.total_epochs))

    def compute_alpha_soft(self, x_T: torch.Tensor) -> torch.Tensor:
        """h = MLP(GAP(Conv3D_small(x_T))); alpha_soft = softmax(h / tau)."""
        feat = self.conv_small(x_T)          # (B, hidden_dim, D', H', W')
        pooled = feat.mean(dim=(2, 3, 4))     # GAP -> (B, hidden_dim)
        h = self.mlp(pooled)                  # (B, A) logits
        return torch.softmax(h / self.tau, dim=-1)

    def forward(self, x_T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (alpha, S): sparse renormalized weights (B, A) and top-k indices (B, k)."""
        alpha_soft = self.compute_alpha_soft(x_T)

        topk_vals, topk_idx = torch.topk(alpha_soft, self.top_k, dim=-1)  # (B, k)
        denom = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        renorm_vals = topk_vals / denom

        alpha = torch.zeros_like(alpha_soft)
        alpha.scatter_(1, topk_idx, renorm_vals)
        return alpha, topk_idx


# ---------------------------------------------------------------------------
# Per-anatomy low-rank adapter
# ---------------------------------------------------------------------------


class AnatomyAdapter(nn.Module):
    """Low-rank bottleneck adapter: Linear(d, r) -> GELU -> Linear(r, d)  (§5)."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


# ---------------------------------------------------------------------------
# Routed adapter block — replaces the Identity() anatomy-adapter slot
# ---------------------------------------------------------------------------


class RoutedAdapterBlock(nn.Module):
    """y = x + sum_{i in S} alpha_i * Adapter_i(x)  (§5).

    Every adapter is evaluated and weighted by alpha_i; alpha_i == 0 for
    experts outside the top-k selection S, so the sum is mathematically
    identical to summing over S only. True O(k) sparse compute (skipping
    the zero-weighted experts' matmuls) is a future optimization — not
    required for correctness here.
    """

    def __init__(self, dim: int, num_anatomies: int, adapter_rank: int) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [AnatomyAdapter(dim, adapter_rank) for _ in range(num_anatomies)]
        )

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """x: (B, N, dim); alpha: (B, A)."""
        y = x
        for i, adapter in enumerate(self.adapters):
            w = alpha[:, i].view(-1, 1, 1)  # (B, 1, 1)
            y = y + w * adapter(x)
        return y


# ---------------------------------------------------------------------------
# Routing losses (§7: L_ent, L_bal, L_cls)
# ---------------------------------------------------------------------------


class RoutingLoss:
    """Stateless routing loss terms (§7). No learnable parameters."""

    @staticmethod
    def entropy_loss(alpha: torch.Tensor) -> torch.Tensor:
        """L_ent = -sum_i alpha_i * log(alpha_i + 1e-8), mean over batch."""
        per_sample = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1)
        return per_sample.mean()

    @staticmethod
    def balance_loss(alpha: torch.Tensor) -> torch.Tensor:
        """L_bal = A * sum_i f_i * P_i.

        f_i = fraction of the batch routed to expert i (alpha_i > 0).
        P_i = mean gate weight assigned to expert i across the batch.
        """
        A = alpha.shape[-1]
        f_i = (alpha > 0).float().mean(dim=0)  # (A,)
        p_i = alpha.mean(dim=0)                # (A,)
        return A * (f_i * p_i).sum()

    @staticmethod
    def cls_loss(alpha_soft: torch.Tensor | None, anatomy_labels: torch.Tensor | None) -> torch.Tensor:
        """L_cls = CE(alpha_soft, anatomy_labels); optional (lambda_cls may be 0, ADR-003).

        alpha_soft already sums to 1 per sample (post-softmax probabilities),
        so cross-entropy is computed as NLL on log(alpha_soft) rather than
        F.cross_entropy (which would apply softmax a second time).
        """
        if anatomy_labels is None:
            device = alpha_soft.device if alpha_soft is not None else "cpu"
            return torch.zeros((), device=device)
        log_probs = torch.log(alpha_soft.clamp_min(1e-8))
        return F.nll_loss(log_probs, anatomy_labels)
