from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

def build_grid_graph(shape: tuple[int, int, int]) -> torch.Tensor:
    """6-connectivity adjacency for a (nz, ny, nx) voxel grid, node id = row-major flat index."""
    nz, ny, nx = shape
    node_id = np.arange(nz * ny * nx).reshape(nz, ny, nx)

    def axis_edges(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.stack([a.ravel(), b.ravel()], axis=0)

    forward = np.concatenate(
        [
            axis_edges(node_id[:, :, :-1], node_id[:, :, 1:]),  # x-neighbors
            axis_edges(node_id[:, :-1, :], node_id[:, 1:, :]),  # y-neighbors
            axis_edges(node_id[:-1, :, :], node_id[1:, :, :]),  # z-neighbors
        ],
        axis=1,
    )
    undirected = np.concatenate([forward, forward[::-1]], axis=1)
    return torch.as_tensor(undirected, dtype=torch.long)


def compute_voxel_coordinates(
    grid_shape: tuple[int, int, int],
    x_bounds_cm: np.ndarray,
    y_bounds_cm: np.ndarray,
    z_bounds_cm: np.ndarray,
) -> torch.Tensor:
    """Compute physical (x, y, z) centers for all voxels matching row-major flat node index."""
    nz, ny, nx = grid_shape
    x_centers = 0.5 * (x_bounds_cm[:-1] + x_bounds_cm[1:])
    y_centers = 0.5 * (y_bounds_cm[:-1] + y_bounds_cm[1:])
    z_centers = 0.5 * (z_bounds_cm[:-1] + z_bounds_cm[1:])

    grid_z, grid_y, grid_x = np.meshgrid(z_centers, y_centers, x_centers, indexing="ij")
    coords = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)
    return torch.as_tensor(coords, dtype=torch.float32)


def compute_beam_projection_features(coords: torch.Tensor, theta_deg: torch.Tensor) -> torch.Tensor:
    """
    Computes depth along beam axis (d_parallel) and lateral perpendicular distance (d_perp)
    for a beam rotating in the X-Z plane around the longitudinal Y-axis.

    coords: [N, 3] continuous (x, y, z) coordinates
    theta_deg: [B] or scalar beam angles in degrees
    Returns: [B, N, 2] tensor containing (d_parallel, d_perp)
    """
    if theta_deg.ndim == 0:
        theta_deg = theta_deg.view(1)

    theta_rad = torch.deg2rad(theta_deg).view(-1, 1)  # [B, 1]
    sin_th = torch.sin(theta_rad)
    cos_th = torch.cos(theta_rad)

    x = coords[:, 0].unsqueeze(0)  # [1, N]
    y = coords[:, 1].unsqueeze(0)  # [1, N]
    z = coords[:, 2].unsqueeze(0)  # [1, N]

    d_parallel = x * sin_th + z * cos_th  # [B, N]
    d_perp = torch.sqrt((x * cos_th - z * sin_th) ** 2 + y ** 2)  # [B, N]

    return torch.stack([d_parallel, d_perp], dim=-1)  # [B, N, 2]


def compute_radiological_depth(
    rho_grid: torch.Tensor,     # [nz, ny, nx] or [B, nz, ny, nx]
    coords_cm: torch.Tensor,    # [N, 3] (x, y, z) voxel centres
    theta_deg: torch.Tensor,    # [B]
    bounds_cm: tuple,           # outer edges (x0, x1, y0, y1, z0, z1)
    n_steps: int = 64,
    source_sign: float = 1.0,
    depth_scale: float | None = None,
) -> torch.Tensor:              # [B, N, 1]
    """Fully batched radiological depth computation without per-sample loops."""
    device = coords_cm.device
    if theta_deg.ndim == 0:
        theta_deg = theta_deg.view(1)
    B = theta_deg.numel()

    x0, x1, y0, y1, z0, z1 = bounds_cm
    lo = torch.tensor([x0, y0, z0], dtype=torch.float32, device=device)
    hi = torch.tensor([x1, y1, z1], dtype=torch.float32, device=device)

    vol = rho_grid.float()
    if vol.ndim == 3:
        vol = vol.unsqueeze(0).unsqueeze(0)                   # [1, 1, nz, ny, nx]
    elif vol.ndim == 4:
        vol = vol.unsqueeze(1)                                # [B, 1, nz, ny, nx]
    if vol.shape[0] == 1 and B > 1:
        vol = vol.expand(B, -1, -1, -1, -1)                   # [B, 1, nz, ny, nx]

    path_len = float(((x1 - x0) ** 2 + (z1 - z0) ** 2) ** 0.5)
    ds = path_len / n_steps
    t = (torch.arange(n_steps, device=device, dtype=torch.float32) + 0.5) * ds  # [K]

    th = torch.deg2rad(theta_deg.float())
    u = source_sign * torch.stack([torch.sin(th), torch.zeros_like(th), torch.cos(th)], dim=-1)  # [B, 3]

    with torch.no_grad():
        # [B, N, K, 3] = [1, N, 1, 3] + [B, 1, 1, 3] * [1, 1, K, 1]
        pts = coords_cm.unsqueeze(0).unsqueeze(2) + (u.unsqueeze(1).unsqueeze(2) * t.view(1, 1, -1, 1))
        grid = (2.0 * (pts - lo) / (hi - lo) - 1.0).view(B, -1, n_steps, 1, 3)

        samp = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        depth = (samp.view(B, -1, n_steps).sum(dim=-1) * ds).unsqueeze(-1)  # [B, N, 1]

    return depth / (depth_scale if depth_scale is not None else path_len)


def load_theta_dataset(npz_path: str | Path) -> dict:
    """Load dataset, compute continuous physical coordinates, and return structured tensors."""
    data = np.load(npz_path)
    nz, ny, nx = data["phantom_density"].shape
    grid_shape = (nz, ny, nx)
    n_nodes = nz * ny * nx

    x_bounds = data["x_bounds_cm"]
    y_bounds = data["y_bounds_cm"]
    z_bounds = data["z_bounds_cm"]

    coords = compute_voxel_coordinates(grid_shape, x_bounds, y_bounds, z_bounds)

    coords_norm = coords.clone()
    coords_norm[:, 0] /= float(np.max(np.abs(x_bounds)))
    coords_norm[:, 1] /= float(np.max(np.abs(y_bounds)))
    coords_norm[:, 2] /= float(np.max(np.abs(z_bounds)))

    return {
        "rho": torch.as_tensor(data["phantom_density"].reshape(n_nodes, 1), dtype=torch.float32),
        "theta": torch.as_tensor(data["theta"], dtype=torch.float32),
        "dose": torch.as_tensor(data["dose"].reshape(-1, n_nodes), dtype=torch.float32),
        "uncertainty": torch.as_tensor(data["uncertainty"].reshape(-1, n_nodes), dtype=torch.float32),
        "coords": coords,
        "coords_norm": coords_norm,
        "grid_shape": grid_shape,
        "x_bounds_cm": x_bounds,
        "y_bounds_cm": y_bounds,
        "z_bounds_cm": z_bounds,
    }


def compute_zeff_from_rho(rho: torch.Tensor) -> torch.Tensor:
    """Stoichiometric effective atomic number Z_eff / 14.0 derived directly from density rho."""
    zeff = torch.where(
        rho <= 1.05,
        torch.full_like(rho, 7.42),
        7.42 + (13.8 - 7.42) * torch.clamp((rho - 1.05) / (1.85 - 1.05), 0.0, 1.0),
    )
    return zeff / 14.0


def construct_batch_features(
    rho, rho_grid, coords_norm, coords_cm, theta_deg, bounds_cm, in_node_features: int = 4, include_coords: bool = False
):
    """Build the full per-voxel input feature tensor for a batch of beam angles.

    Args:
        rho: [N, 1] voxel mass density (g/cm³).
        rho_grid: [nz, ny, nx] or [B, nz, ny, nx] density grid for radiological depth.
        coords_norm: [N, 3] normalized (x, y, z) voxel centres in [-1, 1].
        coords_cm: [N, 3] physical (x, y, z) voxel centres in cm.
        theta_deg: [B] beam gantry angles in degrees.
        bounds_cm: (x0, x1, y0, y1, z0, z1) physical volume edges in cm.
        in_node_features: Total feature channels. Controls which feature set is returned:
            4 -> [rho, d_parallel, d_perp, wepl]
            5 -> [rho, zeff, d_parallel, d_perp, wepl]
            7 -> [rho, x_norm, y_norm, z_norm, d_parallel, d_perp, wepl]
            8 -> [rho, zeff, x_norm, y_norm, z_norm, d_parallel, d_perp, wepl]
        include_coords: If True, always include normalized coordinates (equivalent to in_node_features=7).
    Returns:
        [B, N, in_node_features] feature tensor on the same device as rho.
    """
    proj = compute_beam_projection_features(coords_cm, theta_deg)                 # [B, N, 2] (d_parallel, d_perp)
    depth = compute_radiological_depth(rho_grid, coords_cm, theta_deg, bounds_cm)  # [B, N, 1] (WEPL)
    B = proj.shape[0]

    if include_coords or in_node_features == 7:
        static = torch.cat([rho, coords_norm], dim=-1).unsqueeze(0).expand(B, -1, -1)  # [B, N, 4]
        return torch.cat([static, proj, depth], dim=-1)                                 # [B, N, 7]

    if in_node_features == 8:
        zeff = compute_zeff_from_rho(rho)
        static_8ch = torch.cat([rho, zeff, coords_norm], dim=-1).unsqueeze(0).expand(B, -1, -1)  # [B, N, 5]
        return torch.cat([static_8ch, proj, depth], dim=-1)                                       # [B, N, 8]

    if in_node_features == 5:
        zeff = compute_zeff_from_rho(rho)
        static_5ch = torch.cat([rho, zeff], dim=-1).unsqueeze(0).expand(B, -1, -1)      # [B, N, 2]
        return torch.cat([static_5ch, proj, depth], dim=-1)                             # [B, N, 5]

    static = rho.unsqueeze(0).expand(B, -1, -1)                                         # [B, N, 1]
    return torch.cat([static, proj, depth], dim=-1)                                     # [B, N, 4]




def build_body_mask(phantom_density: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """Flattened boolean mask of in-body voxels, for masking the loss to exclude air."""
    return phantom_density.reshape(-1) > threshold


def split_dataset_indices(
    n_samples: int, train_frac: float, val_frac: float, test_frac: float, seed: int
) -> dict[str, np.ndarray]:
    """Random, non-overlapping train/val/test sample indices (fractions must sum to 1.0)."""
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    perm = np.random.default_rng(seed).permutation(n_samples)
    n_train = int(round(train_frac * n_samples))
    n_val = int(round(val_frac * n_samples))
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def split_dataset_by_phantom(
    path_sample_counts: list[tuple[Path | str, int]],
    test_phantom_patterns: list[str],
    val_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Partitions sample indices strictly by phantom identity for true OOD patient testing.
    Datasets matching test_phantom_patterns are assigned 100% to test.
    The remaining training phantoms have their beam angles split into train & val by val_frac.
    """
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    current_idx = 0
    for p_path, count in path_sample_counts:
        p_str = str(p_path)
        is_test = any(pat in p_str for pat in test_phantom_patterns)
        sample_range = np.arange(current_idx, current_idx + count)

        if is_test:
            test_indices.extend(sample_range.tolist())
        else:
            shuffled = rng.permutation(sample_range)
            n_val = int(round(val_frac * count))
            val_indices.extend(shuffled[:n_val].tolist())
            train_indices.extend(shuffled[n_val:].tolist())

        current_idx += count

    return {
        "train": np.array(train_indices, dtype=np.int64),
        "val": np.array(val_indices, dtype=np.int64),
        "test": np.array(test_indices, dtype=np.int64),
    }


def create_logger(log_path: Path) -> tuple[Callable[[str], None], Callable[[], None]]:
    """Return an emit function and a flush function to persist terminal output."""
    started_at = datetime.now()
    log_path.write_text(f"[{started_at:%Y-%m-%d %H:%M:%S}] Process started\n", encoding="utf-8")
    flushed = False

    def emit(message: str) -> None:
        print(message)
        timestamp = datetime.now()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp:%Y-%m-%d %H:%M:%S}] {message}\n")

    def flush() -> None:
        nonlocal flushed
        if not flushed:
            ended_at = datetime.now()
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{ended_at:%Y-%m-%d %H:%M:%S}] Process ended (elapsed: {ended_at - started_at})\n")
            flushed = True

    return emit, flush