"""3-D Conditional U-Net for bidirectional MR<->CT synthesis — ThermoBridge baseline.

Architecture
------------
- Input: 2 channels — (source patch, direction patch) where the direction plane
  is filled with 0.0 (MR→CT) or 1.0 (CT→MR).
- 4-level encoder / decoder with skip connections.
- Encoder level i: ConvBlock(→channels[i]) → skip; strided-conv downsample.
- Bottleneck: ConvBlock(channels[-2] → channels[-1]).
- Decoder level i: ConvTranspose3d upsample → cat(skip) → ConvBlock.
- Output: Conv3d(channels[0], 1, 1) — LINEAR, no activation, spans [-1,1].

Channel flow (channels=[c0,c1,c2,c3]):
    Enc 0: in → c0 skip, c0 downsampled
    Enc 1: c0 → c1 skip, c1 downsampled
    Enc 2: c1 → c2 skip, c2 downsampled
    Bottleneck: c2 → c3
    Dec 0: c3 up-c3, cat(skip₂=c2) → c3+c2 → ConvBlock → c2
    Dec 1: c2 up-c2, cat(skip₁=c1) → c2+c1 → ConvBlock → c1
    Dec 2: c1 up-c1, cat(skip₀=c0) → c1+c0 → ConvBlock → c0
    Out:   c0 → 1

Rules:
    R1 — direction channel explicit (not hidden).
    R7 — fully typed + docstrings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Two × (Conv3d → InstanceNorm3d → LeakyReLU)."""

    def __init__(self, in_ch: int, out_ch: int, negative_slope: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock → skip; strided Conv for downsampling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (skip, downsampled); skip has out_ch channels."""
        skip = self.conv(x)
        return skip, self.down(skip)


class DecoderBlock(nn.Module):
    """ConvTranspose3d → cat(skip) → ConvBlock."""

    def __init__(self, from_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        # Upsample keeps from_ch channels; after cat: from_ch + skip_ch → out_ch
        self.up   = nn.ConvTranspose3d(from_ch, from_ch, kernel_size=2, stride=2, bias=False)
        self.conv = ConvBlock(from_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x if spatial mismatch due to odd input dims
        if x.shape[2:] != skip.shape[2:]:
            diff = [skip.shape[i + 2] - x.shape[i + 2] for i in range(3)]
            x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------


class UNet3D(nn.Module):
    """3-D Conditional U-Net for ThermoBridge MR<->CT synthesis.

    Args:
        in_channels:    Source modality channels (default 1).
        channels:       Feature map sizes per level, e.g. [32, 64, 128, 256].
                        len(channels) − 1 encoder levels + 1 bottleneck.
        negative_slope: LeakyReLU slope.

    Input / Output:
        source:       (B, 1,  Z, Y, X) — normalised source patch
        direction_id: (B,)  long      — 0 = MR→CT, 1 = CT→MR
        output:       (B, 1,  Z, Y, X) — normalised prediction
    """

    def __init__(
        self,
        in_channels:    int = 1,
        channels:       list[int] | None = None,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        assert len(channels) >= 2, "Need ≥2 channel entries."

        n_enc = len(channels) - 1   # number of encoder/decoder levels

        # 2 input channels: source (1) + direction (1)
        net_in = in_channels + 1

        # Encoder
        self.encoders = nn.ModuleList()
        prev = net_in
        for ch in channels[:-1]:
            self.encoders.append(EncoderBlock(prev, ch))
            prev = ch

        # Bottleneck
        self.bottleneck = ConvBlock(prev, channels[-1])

        # Decoder — n_enc blocks, from deepest to shallowest
        self.decoders = nn.ModuleList()
        from_ch = channels[-1]
        for i in range(n_enc - 1, -1, -1):
            skip_ch = channels[i]
            out_ch  = channels[i]
            self.decoders.append(DecoderBlock(from_ch, skip_ch, out_ch))
            from_ch = out_ch

        # Linear output head — no activation
        self.out_conv = nn.Conv3d(channels[0], 1, kernel_size=1)

    def forward(self, source: torch.Tensor, direction_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source:       (B, 1, Z, Y, X)
            direction_id: (B,) long, values 0 or 1

        Returns:
            Predicted target (B, 1, Z, Y, X) in normalised space.
        """
        B, _, Z, Y, X = source.shape

        # Explicit direction channel (R1 — auditable)
        dir_ch = direction_id.float().view(B, 1, 1, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([source, dir_ch], dim=1)   # (B, 2, Z, Y, X)

        # Encoder — collect skips
        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder — consume skips deepest-first
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.out_conv(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_unet3d(cfg_training) -> UNet3D:
    """Instantiate UNet3D from the OmegaConf training config node."""
    channels = list(cfg_training.unet_channels)
    return UNet3D(in_channels=1, channels=channels)
