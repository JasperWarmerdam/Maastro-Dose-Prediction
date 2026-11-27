"""
benchmark_inference.py
======================
Standalone GPU inference latency benchmark for all 3 trained models:
  1. 3D U-Net (Single-Beam, RhoThetaDoseUNet)
  2. Hierarchical GNN (Single-Beam, HierarchicalGridGNN)
  3. Total CT Dose U-Net (Full-Scan, TotalCTDoseUNet)

Measures both single-sample AND batched inference latency.
  - Single-sample: batch_size = 1  (as before)
  - Batched:       batch_sizes = [4, 8, 16, 32] — reports per-sample latency
                   (total_time / batch_size) for a fair comparison.

Uses CUDA synchronisation around each forward pass (10 warmup + 50 timed runs).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Pure PyTorch GraphConv fallback if torch_geometric is not installed
try:
    from torch_geometric.nn import GraphConv  # noqa: F401
except (ModuleNotFoundError, ImportError):
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
            return self.lin_rel(aggr) + self.lin_root(x)

    import src.model
    src.model.GraphConv = GraphConv

from src.U_net import RhoThetaDoseUNet
from src.model import HierarchicalGridGNN
from src.total_ct_unet import TotalCTDoseUNet


def benchmark_model(fn, warmup: int = 10, runs: int = 50) -> tuple[float, float]:
    """Run fn() warmup+runs times with CUDA sync; return (mean_ms, std_ms)."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    return float(np.mean(times_ms)), float(np.std(times_ms))


def print_row(label: str, mean_ms: float, std_ms: float, batch_size: int = 1,
              speedup_ref_h: float | None = None) -> None:
    """Format and print a single benchmark result row with per-sample latency and optional MC speedup.

    Args:
        label: Row label (e.g., batch size description).
        mean_ms: Mean total batch forward time in milliseconds.
        std_ms: Standard deviation of batch forward time in milliseconds.
        batch_size: Number of samples in the batch.
        speedup_ref_h: Optional Monte Carlo reference runtime in hours to compute speedup factor.
    Returns:
        None. Prints formatted line to stdout.
    """
    per_sample = mean_ms / batch_size
    per_sample_std = std_ms / batch_size
    extra = ""
    if speedup_ref_h is not None:
        speedup = (speedup_ref_h * 3600.0) / (per_sample / 1000.0)
        extra = f"  (~{per_sample / 1000.0:.4f} s/sample,  speedup vs {speedup_ref_h:.0f}h MC: ~{speedup:,.0f}x)"
    print(f"  {label:<40s}: {per_sample:.2f} +/- {per_sample_std:.2f} ms/sample{extra}")


def main():
    """Benchmark single-sample and batched GPU inference latency for all three trained models.

    Args:
        None.
    Returns:
        None. Loads checkpoints from outputs/ and prints latency and MC speedup tables to stdout.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    BATCH_SIZES = [1, 4, 8, 16, 32]
    MC_REF_HOURS = 12.0

    grid_shape = (32, 72, 128)
    n_voxels = 32 * 72 * 128  # 294,912

    # ------------------------------------------------------------------ #
    # 1. 3D U-Net
    # ------------------------------------------------------------------ #
    unet_ckpt = BASE_DIR / "outputs" / "unet_supercomputer" / "unet_clinical_best.pt"
    u_sd = torch.load(unet_ckpt, map_location=device, weights_only=False)["state_dict"]
    unet = RhoThetaDoseUNet(
        grid_shape=grid_shape,
        in_node_features=5,
        hidden_channels=32,
        depth=3,
        num_harmonic_bands=4,
        use_physics_residual=False,
    ).to(device)
    unet.load_state_dict(u_sd)
    unet.eval()

    print("=" * 75)
    print(f"INFERENCE LATENCY BENCHMARK | Device: {device} ({gpu_name})")
    print("=" * 75)
    print(f"\n{'3D U-Net (Single-Beam)'}")
    print(f"  {'Batch':<40s}  per-sample latency")
    for bs in BATCH_SIZES:
        x_u = torch.randn(bs, n_voxels, 5, device=device)
        theta = torch.full((bs,), 180.0, device=device)
        mean_ms, std_ms = benchmark_model(lambda: unet(x_u, theta))
        print_row(f"batch_size={bs}", mean_ms, std_ms, batch_size=bs)

    # ------------------------------------------------------------------ #
    # 2. Hierarchical GNN
    # ------------------------------------------------------------------ #
    gnn_ckpt = BASE_DIR / "outputs" / "gnn_hierarchical" / "gnn_hierarchical_best.pt"
    g_sd = torch.load(gnn_ckpt, map_location=device, weights_only=False)["state_dict"]
    gnn = HierarchicalGridGNN(
        grid_shape=grid_shape,
        in_node_features=8,
        hidden_channels=32,
        num_harmonic_bands=4,
    ).to(device)
    gnn.load_state_dict(g_sd)
    gnn.eval()

    print(f"\n{'Hierarchical GNN (Single-Beam)'}")
    print(f"  {'Batch':<40s}  per-sample latency")
    for bs in BATCH_SIZES:
        x_g = torch.randn(bs, n_voxels, 8, device=device)
        theta = torch.full((bs,), 180.0, device=device)
        mean_ms, std_ms = benchmark_model(lambda: gnn(x_g, theta))
        print_row(f"batch_size={bs}", mean_ms, std_ms, batch_size=bs)

    # ------------------------------------------------------------------ #
    # 3. Total CT Dose U-Net
    # ------------------------------------------------------------------ #
    tot_ckpt = BASE_DIR / "outputs" / "unet_total_dose" / "unet_total_dose_best.pt"
    t_sd = torch.load(tot_ckpt, map_location=device, weights_only=False)["state_dict"]
    tot_unet = TotalCTDoseUNet(in_channels=3, base_channels=32).to(device)
    tot_unet.load_state_dict(t_sd)
    tot_unet.eval()

    print(f"\n{'Total CT Dose U-Net (Full-Scan)'}")
    print(f"  {'Batch':<40s}  per-sample latency")
    for bs in BATCH_SIZES:
        x_t = torch.randn(bs, 3, *grid_shape, device=device)
        mean_ms, std_ms = benchmark_model(lambda: tot_unet(x_t))
        print_row(f"batch_size={bs}", mean_ms, std_ms, batch_size=bs,
                  speedup_ref_h=MC_REF_HOURS if bs == 1 else None)

    print("\n" + "=" * 75)
    print("NOTE: per-sample latency = total_forward_time / batch_size")
    print("      Batch inference amortises fixed GPU overhead (kernel launch,")
    print("      memory transfer) across multiple samples simultaneously.")
    print("=" * 75)


if __name__ == "__main__":
    main()
