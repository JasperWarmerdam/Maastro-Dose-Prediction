from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierAngleEmbedding(nn.Module):
    """Encodes continuous beam angles into periodic harmonic Fourier features."""

    def __init__(self, num_bands: int = 4):
        super().__init__()
        self.num_bands = num_bands

    def forward(self, theta_deg: torch.Tensor) -> torch.Tensor:
        theta_rad = torch.deg2rad(theta_deg).view(-1, 1)
        freqs = (2.0 ** torch.arange(self.num_bands, device=theta_rad.device, dtype=theta_rad.dtype)).view(1, -1)
        angles = theta_rad * freqs
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class FiLMBlock3D(nn.Module):
    """LayerNorm over channels (per voxel) + FiLM with zero-initialized scaling."""

    def __init__(self, angle_embed_dim: int, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.film_gen = nn.Sequential(
            nn.Linear(angle_embed_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

    def forward(self, h: torch.Tensor, angle_embed: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h.movedim(1, -1)).movedim(-1, 1)
        gamma, beta = self.film_gen(angle_embed).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return (1.0 + gamma) * h_norm + beta


class FiLMConvBlock(nn.Module):
    """Conv3d -> FiLM -> activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        angle_embed_dim: int,
        kernel_size: int = 3,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.film = FiLMBlock3D(angle_embed_dim, out_channels)
        self.act = activation()

    def forward(self, x: torch.Tensor, angle_embed: torch.Tensor) -> torch.Tensor:
        return self.act(self.film(self.conv(x), angle_embed))


class ConvEncoder(nn.Module):
    """Stack of (Conv + FiLM + act) blocks, each followed by 2x spatial downsampling."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        depth: int,
        angle_embed_dim: int,
        kernel_size: int = 3,
        pool_type: str = "max",
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        if pool_type not in ("max", "avg"):
            raise ValueError(f"Unknown pool_type: {pool_type!r}")
        self.depth = depth
        self.blocks = nn.ModuleList(
            FiLMConvBlock(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                angle_embed_dim,
                kernel_size,
                activation,
            )
            for i in range(depth)
        )
        pool_cls = nn.MaxPool3d if pool_type == "max" else nn.AvgPool3d
        self.pool = pool_cls(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor, angle_embed: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, angle_embed)
            skips.append(x)
            x = self.pool(x)
        return x, skips


class SkipConnection3D(nn.Module):
    """Merges an upsampled decoder tensor with the matching encoder skip connection.

    Args:
        channels: Feature channel count (same for both inputs).
        mode: 'add' for elementwise sum; 'concat' for channel-concat followed by 1x1x1 conv.
    """
    def __init__(self, channels: int, mode: str = "add"):
        super().__init__()
        if mode not in ("add", "concat"):
            raise ValueError(f"Unknown skip_connection mode: {mode!r}")
        self.mode = mode
        if mode == "concat":
            self.proj = nn.Conv3d(channels * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        if self.mode == "add":
            return x + skip_x
        return self.proj(torch.cat([x, skip_x], dim=1))


class ConvDecoder(nn.Module):
    """Mirrors ConvEncoder: upsample, merge skip, then Conv + FiLM + act."""

    def __init__(
        self,
        hidden_channels: int,
        depth: int,
        angle_embed_dim: int,
        kernel_size: int = 3,
        skip_mode: str = "add",
        upsample_mode: str = "nearest",
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        if upsample_mode not in ("nearest", "transpose"):
            raise ValueError(f"Unknown upsample_mode: {upsample_mode!r}")
        self.depth = depth
        self.upsample_mode = upsample_mode
        if upsample_mode == "transpose":
            self.ups = nn.ModuleList(
                nn.ConvTranspose3d(hidden_channels, hidden_channels, kernel_size=2, stride=2) for _ in range(depth)
            )
        self.skips = nn.ModuleList(SkipConnection3D(hidden_channels, mode=skip_mode) for _ in range(depth))
        self.blocks = nn.ModuleList(
            FiLMConvBlock(hidden_channels, hidden_channels, angle_embed_dim, kernel_size, activation)
            for _ in range(depth)
        )

    def _upsample(self, x: torch.Tensor, level: int, size: torch.Size) -> torch.Tensor:
        """Upsample x to match the spatial size of the corresponding encoder skip tensor.

        Args:
            x: [B, C, D', H', W'] bottleneck or previous decoder output.
            level: Decoder depth index (selects the transpose-conv layer when applicable).
            size: Target spatial size (D, H, W) taken from the encoder skip tensor.
        Returns:
            [B, C, D, H, W] upsampled tensor.
        """
        if self.upsample_mode == "transpose":
            x = self.ups[level](x)
            return x[..., : size[0], : size[1], : size[2]]
        return F.interpolate(x, size=tuple(size), mode="nearest")

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor], angle_embed: torch.Tensor) -> torch.Tensor:
        for level in reversed(range(self.depth)):
            skip = skips[level]
            up = self._upsample(x, level, skip.shape[2:])
            x = self.skips[level](up, skip)
            x = self.blocks[level](x, angle_embed)
        return x


class PhysicsPrimaryDoseRayTracer(nn.Module):
    """
    Analytical primary ray-traced dose for 50 keV photons:
    Beer-Lambert exponential depth attenuation along WEPL + lateral Gaussian collimation + photoelectric bone boost.
    """

    def __init__(self, mu_eff: float = 0.22, beam_sigma: float = 1.8, bone_enhancement: float = 3.5):
        super().__init__()
        import numpy as np
        self.log_mu = nn.Parameter(torch.tensor(float(np.log(mu_eff))))
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(beam_sigma))))
        self.bone_boost = nn.Parameter(torch.tensor(float(bone_enhancement)))
        self.primary_scale = nn.Parameter(torch.tensor(0.8))

    def forward(self, rho: torch.Tensor, d_perp: torch.Tensor, wepl: torch.Tensor) -> torch.Tensor:
        """Compute analytical primary dose using Beer-Lambert + lateral Gaussian + bone boost.

        Args:
            rho: [B, N] or [N] voxel mass density (g/cm³).
            d_perp: [B, N] or [N] lateral perpendicular distance from beam axis (cm).
            wepl: [B, N] or [N] water-equivalent path length from source (cm).
        Returns:
            [B, N] or [N] primary dose estimate (non-negative).
        """
        mu = torch.exp(self.log_mu)
        sigma = torch.exp(self.log_sigma)

        lat = torch.exp(-0.5 * (d_perp / sigma) ** 2)
        trans = torch.exp(-mu * wepl)
        bone_factor = 1.0 + torch.relu(self.bone_boost) * torch.clamp((rho - 1.0) / 0.85, min=0.0, max=1.5)

        return torch.relu(self.primary_scale) * lat * trans * bone_factor


class RhoThetaDoseUNet(nn.Module):
    """Ordinary 3D U-Net with FiLM angle conditioning and optional physics residual ray tracing."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int] = (50, 25, 25),
        in_node_features: int = 4,  # [rho, d_parallel, d_perp, radiological_depth]
        hidden_channels: int = 32,
        depth: int = 3,
        num_harmonic_bands: int = 4,
        skip_mode: str = "add",
        activation: type[nn.Module] = nn.ReLU,
        kernel_size: int = 3,
        pool_type: str = "max",
        upsample_mode: str = "nearest",
        use_physics_residual: bool = True,
    ):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.num_nodes = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        self.depth = depth
        self.use_physics_residual = use_physics_residual

        self.angle_encoder = FourierAngleEmbedding(num_bands=num_harmonic_bands)
        angle_embed_dim = 2 * num_harmonic_bands

        if self.use_physics_residual:
            self.physics_tracer = PhysicsPrimaryDoseRayTracer()

        self.encoder = ConvEncoder(
            in_channels=in_node_features,
            hidden_channels=hidden_channels,
            depth=depth,
            angle_embed_dim=angle_embed_dim,
            kernel_size=kernel_size,
            pool_type=pool_type,
            activation=activation,
        )

        self.bottleneck = FiLMConvBlock(hidden_channels, hidden_channels, angle_embed_dim, kernel_size, activation)

        self.decoder = ConvDecoder(
            hidden_channels=hidden_channels,
            depth=depth,
            angle_embed_dim=angle_embed_dim,
            kernel_size=kernel_size,
            skip_mode=skip_mode,
            upsample_mode=upsample_mode,
            activation=activation,
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            activation(),
            nn.Linear(hidden_channels, 1),
        )

    def _to_nodes(self, f: torch.Tensor) -> torch.Tensor:
        """Extract a single-sample feature map and flatten it to [N, C] node format.

        Args:
            f: [B, C, D, H, W] volumetric feature tensor; only the first sample is used.
        Returns:
            [N, C] node feature matrix where N = D * H * W.
        """
        f = f[:1]
        if tuple(f.shape[2:]) != self.grid_shape:
            f = F.interpolate(f, size=self.grid_shape, mode="nearest")
        return f[0].flatten(1).t()

    def forward(
        self,
        x: torch.Tensor,
        theta_deg: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run a forward pass and return predicted per-voxel dose.

        Args:
            x: [B, N, C] or [N, C] node feature tensor (C = in_node_features).
            theta_deg: [B] or scalar beam gantry angle in degrees.
            edge_index: unused; accepted for API compatibility.
            return_features: If True, also return a dict of intermediate feature maps.
        Returns:
            dose: [B, N] or [N] non-negative dose predictions.
            features (optional): dict mapping layer names to [N, C] feature matrices.
        """
        batched = x.ndim == 3
        if not batched:
            x = x.unsqueeze(0)
        B, N, C = x.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes for grid {self.grid_shape}, got {N}")

        theta = theta_deg.reshape(-1)
        if theta.numel() == 1 and B > 1:
            theta = theta.expand(B)
        angle_embed = self.angle_encoder(theta)

        grid = x.reshape(B, *self.grid_shape, C).permute(0, 4, 1, 2, 3).contiguous()

        features: dict[str, torch.Tensor] = {}

        bottleneck_in, skips = self.encoder(grid, angle_embed)
        if return_features:
            for i, s in enumerate(skips):
                features[f"encoder_level_{i}"] = self._to_nodes(s)

        h = self.bottleneck(bottleneck_in, angle_embed)
        if return_features:
            features["bottleneck"] = self._to_nodes(h)

        x_recon = self.decoder(h, skips, angle_embed)
        if return_features:
            features["decoder_output"] = self._to_nodes(x_recon)

        scatter = self.readout(x_recon.movedim(1, -1)).squeeze(-1).reshape(B, N)

        if self.use_physics_residual and C >= 4:
            rho = x[..., 0]
            d_perp = x[..., 2]
            wepl = x[..., 3]
            primary = self.physics_tracer(rho, d_perp, wepl)
            dose = F.softplus(primary + scatter)
        else:
            dose = F.softplus(scatter)

        if not batched:
            dose = dose.squeeze(0)

        if return_features:
            return dose, features
        return dose
