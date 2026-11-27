"""
total_ct_unet.py
================
Pure 3D Convolutional U-Net for total full-scan CT dose estimation
directly from patient CT density volume (rho).

No angle conditioning (FiLM) required since the 36-view CT rotation
is integrated into the total cumulative dose.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """Residual block with two Conv3d-InstanceNorm-LeakyReLU layers and an identity skip.

    Args:
        channels: Number of input (and output) channels.
    Returns:
        [B, channels, D, H, W] tensor with same spatial size as input.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)


class TotalCTDoseUNet(nn.Module):
    """
    3D U-Net taking [B, in_channels, D, H, W] -> [B, 1, D, H, W]
    Predicts 3D total CT scan dose from 3D CT density.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        grid_shape: tuple[int, int, int] = (32, 72, 128),
    ):
        super().__init__()
        self.grid_shape = grid_shape
        c = base_channels

        # Level 0 (Full res: 32 x 72 x 128)
        self.in_conv = nn.Conv3d(in_channels, c, kernel_size=3, padding=1)
        self.res0 = ResBlock3D(c)

        # Level 1 (Down 2x: 16 x 36 x 64)
        self.down1 = nn.Conv3d(c, c * 2, kernel_size=3, stride=2, padding=1)
        self.res1 = ResBlock3D(c * 2)

        # Level 2 (Down 4x: 8 x 18 x 32)
        self.down2 = nn.Conv3d(c * 2, c * 4, kernel_size=3, stride=2, padding=1)
        self.res2 = ResBlock3D(c * 4)

        # Bottleneck (Down 8x: 4 x 9 x 16)
        self.down3 = nn.Conv3d(c * 4, c * 8, kernel_size=3, stride=2, padding=1)
        self.bottleneck = ResBlock3D(c * 8)

        # Upsample 3 -> Level 2
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec_res2 = ResBlock3D(c * 4)

        # Upsample 2 -> Level 1
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec_res1 = ResBlock3D(c * 2)

        # Upsample 1 -> Level 0
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec_res0 = ResBlock3D(c)

        # Output head
        self.out_head = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rho: [B, 1, D, H, W] CT mass density
        Returns:
            dose: [B, 1, D, H, W] Total predicted 3D CT scan dose (non-negative)
        """
        # Encoder
        x0 = self.res0(self.in_conv(rho))                 # [B, c, 32, 72, 128]
        x1 = self.res1(self.down1(x0))                    # [B, 2c, 16, 36, 64]
        x2 = self.res2(self.down2(x1))                    # [B, 4c, 8, 18, 32]
        b = self.bottleneck(self.down3(x2))               # [B, 8c, 4, 9, 16]

        # Decoder with skip connections
        u2 = self.up3(b) + x2                             # [B, 4c, 8, 18, 32]
        d2 = self.dec_res2(u2)

        u1 = self.up2(d2) + x1                            # [B, 2c, 16, 36, 64]
        d1 = self.dec_res1(u1)

        u0 = self.up1(d1) + x0                            # [B, c, 32, 72, 128]
        d0 = self.dec_res0(u0)

        return self.out_head(d0)
