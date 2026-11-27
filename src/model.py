from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GraphConv
except (ModuleNotFoundError, ImportError):
    # Pure PyTorch robust fallback (matches exact torch_geometric.nn.GraphConv formula & weights)
    class GraphConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, aggr: str = "add"):
            super().__init__()
            self.lin_rel = nn.Linear(in_channels, out_channels, bias=True)
            self.lin_root = nn.Linear(in_channels, out_channels, bias=False)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
            src, dst = edge_index[0], edge_index[1]
            msg = x[src]
            if edge_weight is not None:
                msg = msg * edge_weight.unsqueeze(-1)
            aggr = torch.zeros_like(x)
            aggr.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(-1)), msg)
            out = self.lin_rel(aggr) + self.lin_root(x)
            return out


class FourierAngleEmbedding(nn.Module):
    """Encodes continuous beam angles into periodic harmonic Fourier features."""

    def __init__(self, num_bands: int = 4):
        super().__init__()
        self.num_bands = num_bands

    def forward(self, theta_deg: torch.Tensor) -> torch.Tensor:
        theta_rad = torch.deg2rad(theta_deg).view(-1, 1)
        freqs = (2.0 ** torch.arange(self.num_bands, device=theta_rad.device, dtype=theta_rad.dtype)).view(1, -1)
        angles = theta_rad * freqs
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # [B, 2 * num_bands]


class FiLMBlock(nn.Module):
    """Normalized Feature-wise Linear Modulation with zero-initialized scaling."""

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
        h_norm = self.norm(h)
        film = self.film_gen(angle_embed)
        gamma, beta = film.chunk(2, dim=-1)

        if h.ndim == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return (1.0 + gamma) * h_norm + beta


class HierarchicalGridGNN(nn.Module):
    """
    Multi-Scale Hierarchical Graph Neural Network for Volumetric Radiotherapy Dose.
    Features 3 resolution levels (Fine -> Mid -> Bottleneck Coarse) to overcome
    the local 3-hop receptive field bottleneck while maintaining linear scaling.
    """
    def __init__(
        self,
        grid_shape: tuple[int, int, int] = (32, 72, 128),
        in_node_features: int = 8,
        hidden_channels: int = 32,
        num_harmonic_bands: int = 4,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        nz, ny, nx = grid_shape
        self.shape0 = (nz, ny, nx)
        self.shape1 = (nz // 2, ny // 2, nx // 2)
        self.shape2 = (nz // 4, ny // 4, nx // 4)

        from src.utils import build_grid_graph
        self.register_buffer("edge_index0", build_grid_graph(self.shape0))
        self.register_buffer("edge_index1", build_grid_graph(self.shape1))
        self.register_buffer("edge_index2", build_grid_graph(self.shape2))

        self.angle_encoder = FourierAngleEmbedding(num_bands=num_harmonic_bands)
        angle_dim = 2 * num_harmonic_bands

        self.conv0_in = GraphConv(in_node_features, hidden_channels)
        self.film0_in = FiLMBlock(angle_dim, hidden_channels)

        self.conv1_down = GraphConv(hidden_channels, hidden_channels * 2)
        self.film1_down = FiLMBlock(angle_dim, hidden_channels * 2)

        self.conv2_bot = GraphConv(hidden_channels * 2, hidden_channels * 2)
        self.film2_bot = FiLMBlock(angle_dim, hidden_channels * 2)

        self.conv1_up = GraphConv(hidden_channels * 2, hidden_channels)
        self.film1_up = FiLMBlock(angle_dim, hidden_channels)

        self.conv0_out = GraphConv(hidden_channels, hidden_channels)
        self.film0_out = FiLMBlock(angle_dim, hidden_channels)

        # High-Frequency Local Physical Skip Branch:
        # Directly models single-voxel stopping-power interactions (e.g. sharp cortical bone peaks)
        # without suffering from multi-layer graph smoothing/diffusion
        self.local_physics_head = nn.Sequential(
            nn.Linear(in_node_features, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

        # Global volumetric scatter readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

        # Fast pre-cached batched edge indices buffer (eliminates dynamic GPU allocation per step)
        self._edge_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _pool(self, h: torch.Tensor, shape_in: tuple[int, int, int], shape_out: tuple[int, int, int]) -> torch.Tensor:
        """Average-pool node features from a fine grid to a coarser grid using 3D avg pooling.

        Args:
            h: [B, N_in, C] node features at the finer resolution.
            shape_in: (nz, ny, nx) spatial shape of the input grid.
            shape_out: unused; output shape is inferred from 2x stride pooling.
        Returns:
            [B, N_out, C] node features at the coarser resolution.
        """
        B, N, C = h.shape
        nz, ny, nx = shape_in
        vol = h.view(B, nz, ny, nx, C).permute(0, 4, 1, 2, 3)
        pooled = F.avg_pool3d(vol, kernel_size=2, stride=2)
        return pooled.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

    def _unpool(self, h: torch.Tensor, shape_out: tuple[int, int, int]) -> torch.Tensor:
        """Trilinear upsample node features from a coarse grid to a finer grid.

        Args:
            h: [B, N_in, C] node features at the coarser resolution (assumed 2x downsampled).
            shape_out: (nz, ny, nx) spatial shape of the target finer resolution.
        Returns:
            [B, N_out, C] node features at the finer resolution.
        """
        B, N, C = h.shape
        nz, ny, nx = shape_out
        vol = h.view(B, nz // 2, ny // 2, nx // 2, C).permute(0, 4, 1, 2, 3)
        up = F.interpolate(vol, size=(nz, ny, nx), mode="trilinear", align_corners=False)
        return up.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

    def _get_batched_edges(self, edge_index: torch.Tensor, B: int, N: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (and cache) a batched disjoint-graph edge index for a mini-batch of B graphs.

        Args:
            edge_index: [2, E] single-graph edge index.
            B: batch size.
            N: number of nodes per graph.
            device: target device.
        Returns:
            (disjoint_edge_index [2, B*E], batch [B*N]) — node-to-graph assignment vector.
        """
        cache_key = (id(edge_index), B)
        if cache_key not in self._edge_cache or self._edge_cache[cache_key][0].device != device:
            offsets = torch.arange(B, device=device).unsqueeze(1) * N
            src = (edge_index[0].unsqueeze(0) + offsets).reshape(-1)
            dst = (edge_index[1].unsqueeze(0) + offsets).reshape(-1)
            disjoint_e = torch.stack([src, dst], dim=0)
            batch = torch.arange(B, device=device).repeat_interleave(N)
            self._edge_cache[cache_key] = (disjoint_e, batch)
        return self._edge_cache[cache_key]

    def _apply_graph_conv(
        self,
        conv: nn.Module,
        film: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        angle_embed: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one GraphConv + FiLM conditioning + ReLU to a batched node feature tensor.

        Args:
            conv: GraphConv layer.
            film: FiLMBlock layer for angle conditioning.
            x: [B, N, C] batched node features.
            edge_index: [2, E] single-graph edge index (batching handled internally).
            angle_embed: [B, angle_dim] beam angle embeddings.
            edge_weight: optional [B*E] per-edge weights.
        Returns:
            [B, N, C_out] updated node features.
        """
        B, N, C = x.shape
        disjoint_e, batch = self._get_batched_edges(edge_index, B, N, x.device)
        x_flat = x.reshape(B * N, C)

        h = conv(x_flat, disjoint_e, edge_weight) if edge_weight is not None else conv(x_flat, disjoint_e)
        h = film(h, angle_embed[batch])
        h = F.relu(h)
        return h.view(B, N, -1)

    def forward(
        self,
        x: torch.Tensor,
        theta_deg: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict per-voxel radiation dose from physical node features and beam angle.

        Args:
            x: [B, N, in_node_features] or [N, in_node_features] node feature tensor.
            theta_deg: [B] or scalar beam gantry angle in degrees.
            edge_index: unused (kept for API compatibility; built-in buffers are used).
        Returns:
            [B, N] or [N] predicted dose values (non-negative via softplus).
        """
        batched = x.ndim == 3
        if not batched:
            x = x.unsqueeze(0)

        theta = theta_deg.reshape(-1)
        B, N, _ = x.shape
        if theta.numel() == 1 and B > 1:
            theta = theta.expand(B)

        angle_embed = self.angle_encoder(theta)

        # Level 0 Conductance: Boundary-Aware Edge Weighting (no-grad to avoid storing 13M edge autograd graph)
        # Prevents message passing from washing out high-density bone peaks into surrounding tissue
        with torch.no_grad():
            disjoint_e0, _ = self._get_batched_edges(self.edge_index0, B, N, x.device)
            rho_nodes = x[:, :, 0].reshape(-1)
            delta_rho = torch.abs(rho_nodes[disjoint_e0[0]] - rho_nodes[disjoint_e0[1]])
            conductance0 = torch.exp(-2.0 * delta_rho)

        # Level 0 (Fine)
        h0 = self._apply_graph_conv(self.conv0_in, self.film0_in, x, self.edge_index0, angle_embed, edge_weight=conductance0)

        # Level 1 (Mid: 2x downsample)
        h1_in = self._pool(h0, self.shape0, self.shape1)
        h1 = self._apply_graph_conv(self.conv1_down, self.film1_down, h1_in, self.edge_index1, angle_embed)

        # Level 2 (Bottleneck: 4x downsample, covers entire anatomy)
        h2_in = self._pool(h1, self.shape1, self.shape2)
        h2 = self._apply_graph_conv(self.conv2_bot, self.film2_bot, h2_in, self.edge_index2, angle_embed)

        # Upsample to Level 1 + Skip Connection
        h1_up = self._unpool(h2, self.shape1) + h1
        h1_out = self._apply_graph_conv(self.conv1_up, self.film1_up, h1_up, self.edge_index1, angle_embed)

        # Upsample to Level 0 + Skip Connection
        h0_up = self._unpool(h1_out, self.shape0) + h0
        h0_out = self._apply_graph_conv(self.conv0_out, self.film0_out, h0_up, self.edge_index0, angle_embed, edge_weight=conductance0)

        # Dual-Stream Physics Readout:
        # 1. Volumetric scatter dose from multi-scale hierarchical graph pyramid
        scatter_dose = self.readout(h0_out).squeeze(-1)

        # 2. Local collision stopping-power dose directly from input physical features
        local_dose = self.local_physics_head(x).squeeze(-1)

        dose_pred = F.softplus(scatter_dose + local_dose)
        return dose_pred if batched else dose_pred.squeeze(0)
