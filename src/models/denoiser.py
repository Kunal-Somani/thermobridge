"""Hybrid transformer denoiser for the ThermoBridge I2SB bridge (Method Spec §4).

Architecture (§4):
    x_T (source) -> patchify -> [ Denoiser Block x L ] -> unpatchify -> x_hat_0

Each Denoiser Block:
    adaLN(c) -> global mixer (3D self-attention) -> adaLN(c)
    -> local mixer slot (§6, Chunk 9b) -> anatomy adapter slot (§5, Chunk 9)

Conditioning (§4.1):
    c = e_t (sinusoidal timestep -> MLP) + e_m (modality-pair embedding)
    injected via adaLN-Zero (scale/shift/gate zero-initialized -> identity at init).

The local mixer and anatomy adapter are nn.Identity() placeholders here — Chunks
9 and 9b replace them via set_local_mixer()/set_adapters() without touching the
rest of the model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .routing import AnatomyRouter, RoutedAdapterBlock


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding of a scalar timestep t in [0, 1], followed by an MLP."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float -> e_t: (B, hidden_dim)."""
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half)
        if emb.shape[1] < self.hidden_dim:
            emb = torch.nn.functional.pad(emb, (0, self.hidden_dim - emb.shape[1]))
        return self.mlp(emb)


class ModalityPairEmbedding(nn.Module):
    """e_m = Linear(concat(E_src[m_s], E_tgt[m_t])) (§4.1)."""

    def __init__(self, num_modalities: int, modality_embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.e_src = nn.Embedding(num_modalities, modality_embed_dim)
        self.e_tgt = nn.Embedding(num_modalities, modality_embed_dim)
        self.proj = nn.Linear(2 * modality_embed_dim, hidden_dim)

    def forward(self, m_s: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        """m_s, m_t: (B,) long -> e_m: (B, hidden_dim)."""
        pair = torch.cat([self.e_src(m_s), self.e_tgt(m_t)], dim=-1)
        return self.proj(pair)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x: (B, N, C); shift, scale: (B, C)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Denoiser block
# ---------------------------------------------------------------------------


class DenoiserBlock(nn.Module):
    """adaLN(c) -> 3D self-attention (global mixer) -> adaLN(c).

    The local mixer and anatomy adapter slots live at the top-level model
    (as nn.ModuleList) so they can be swapped per-block by Chunks 9/9b; this
    block only implements the adaLN-Zero-gated global attention mixer.
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # adaLN-Zero: predicts (shift1, scale1, gate1, shift2, scale2, gate2).
        # Zero-initialized so all scale/shift/gate start at zero -> identity block.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, N, hidden_dim); c: (B, hidden_dim)."""
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=-1)

        h = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out

        # Second adaLN-Zero residual; local mixer / anatomy adapter slots are
        # applied by the parent model after this block returns.
        h2 = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * h2
        return x


# ---------------------------------------------------------------------------
# Full denoiser
# ---------------------------------------------------------------------------

_MAX_TOKENS = 8192  # generous upper bound for the learned positional table


class ThermoBridgeDenoiser(nn.Module):
    """Hybrid transformer denoiser (§4): patchify -> L blocks -> unpatchify.

    forward(x_T, t, m_s, m_t, alpha) -> x_hat_0, all shapes as documented in
    the class docstring of the surrounding module.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        patch_size: tuple[int, int, int],
        num_modalities: int,
        modality_embed_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patch_size = tuple(patch_size)

        self.patchify = nn.Conv3d(1, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.unpatchify = nn.ConvTranspose3d(
            hidden_dim, 1, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.pos_embed = nn.Parameter(torch.randn(1, _MAX_TOKENS, hidden_dim) * 0.02)

        self.timestep_embed = SinusoidalTimestepEmbedding(hidden_dim)
        self.modality_embed = ModalityPairEmbedding(num_modalities, modality_embed_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [DenoiserBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        # Placeholders — replaced wholesale by Chunk 9b (local mixer, §6) and
        # Chunk 9 (anatomy-routed adapters, §5) via set_local_mixer/set_adapters.
        self.local_mixers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        self.anatomy_adapters = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        # Set by set_adapters() (§5, Chunk 9). While None, anatomy_adapters
        # stay Identity() and the routing gate is never evaluated.
        self.router: AnatomyRouter | None = None

    def set_local_mixer(self, mixers: list[nn.Module]) -> None:
        """Replace the local mixer slots (§6, anisotropic operator — Chunk 9b)."""
        assert len(mixers) == self.num_layers
        self.local_mixers = nn.ModuleList(mixers)

    def set_adapters(self, router: AnatomyRouter, adapter_blocks: nn.ModuleList) -> None:
        """Replace the Identity() adapter slots with routed anatomy adapters (§5, Chunk 9).

        After this call, forward() computes alpha from x_T via `router` at
        the start of each forward pass (the `alpha` argument to forward() is
        then ignored, exactly as it was ignored before this call).
        """
        assert len(adapter_blocks) == self.num_layers
        self.router = router
        self.anatomy_adapters = nn.ModuleList(adapter_blocks)

    def forward(
        self,
        x_T: torch.Tensor,
        t: torch.Tensor,
        m_s: torch.Tensor,
        m_t: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_T: (B, 1, D, H, W) source volume in [-1, 1]
            t:   (B,) float timestep in [0, 1]
            m_s: (B,) int source modality index
            m_t: (B,) int target modality index
            alpha: (B, A) float routing weights. Ignored once set_adapters()
                has installed a router — alpha is then computed from x_T
                internally (§5). Kept in the signature for interface
                stability with callers built before Chunk 9 (e.g. the I2SB
                bridge process, which always passes some alpha tensor).

        Returns:
            x_hat_0: (B, 1, D, H, W) predicted target volume
        """
        if self.router is not None:
            alpha, _S = self.router(x_T)
        else:
            del alpha  # no router installed yet; Identity() adapters ignore it anyway

        B, _, D, H, W = x_T.shape
        tokens = self.patchify(x_T)  # (B, hidden_dim, D', H', W')
        _, C, Dp, Hp, Wp = tokens.shape
        N = Dp * Hp * Wp

        x = tokens.flatten(2).transpose(1, 2)  # (B, N, hidden_dim)
        assert N <= self.pos_embed.shape[1], (
            f"Sequence length {N} exceeds positional embedding capacity {self.pos_embed.shape[1]}"
        )
        x = x + self.pos_embed[:, :N, :]

        c = self.timestep_embed(t) + self.modality_embed(m_s, m_t)  # (B, hidden_dim)

        for block, local_mixer, anatomy_adapter in zip(
            self.blocks, self.local_mixers, self.anatomy_adapters
        ):
            x = block(x, c)
            x = local_mixer(x)
            if isinstance(anatomy_adapter, RoutedAdapterBlock):
                x = anatomy_adapter(x, alpha)
            else:
                x = anatomy_adapter(x)

        x = x.transpose(1, 2).reshape(B, C, Dp, Hp, Wp)
        x_hat_0 = self.unpatchify(x)
        return x_hat_0
