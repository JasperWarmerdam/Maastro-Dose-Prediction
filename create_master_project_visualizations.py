"""
create_master_project_visualizations.py
======================================
Unified Master Visualization & Dynamic Analysis Suite for Radiotherapy CT Dose:
1. Loads trained model checkpoints dynamically and computes clinical metrics live on the fly.
2. Per-Angle 3-view comparison (3D U-Net vs Hierarchical GNN) across all test patients.
3. Per-Angle grouped bar diagrams (5 metrics) & quantitative comparison table.
4. Total Dose 3-view comparison (3D ResNet U-Net alone - NO GNN baseline).
5. Total Dose single-model bar diagrams (MRE, MAE, % within 10%) & quantitative table.
6. Comprehensive inference latency benchmark across all 3 models with dedicated summary table.
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import shutil
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Ensure exact PyG GraphConv fallback if torch_geometric is not installed
try:
    from torch_geometric.nn import GraphConv
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
            out = self.lin_rel(aggr) + self.lin_root(x)
            return out

    import src.model
    src.model.GraphConv = GraphConv

from src.U_net import RhoThetaDoseUNet
from src.model import HierarchicalGridGNN
from src.total_ct_unet import TotalCTDoseUNet
from src.utils import construct_batch_features
from train_total_ct_dose import read_egsphant, build_radial_distance_grid

GLOBAL_DOSE_SCALE = 1.0e-12
PATIENTS = ["pt_210", "pt_222", "pt_223", "pt_230", "pt_233", "pt_234", "pt_253", "pt_259", "pt_260", "pt_261", "pt_262", "pt_263"]

OUT_DIR = BASE_DIR / "outputs" / "master_visualizations"
PER_ANGLE_DIR = OUT_DIR / "per_angle"
TOTAL_DOSE_DIR = OUT_DIR / "total_dose"

PER_ANGLE_DIR.mkdir(parents=True, exist_ok=True)
TOTAL_DOSE_DIR.mkdir(parents=True, exist_ok=True)

# Optional destination directory for copying generated figures (set to a Path if needed)
ARTIFACT_DIR = None


# ==============================================================================
# 1. AAPM TG-218 3D Gamma Index Calculation
# ==============================================================================
def compute_gamma_index_3d(
    d_pred: np.ndarray,
    d_true: np.ndarray,
    mask: np.ndarray,
    voxel_size_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    dose_pct: float = 3.0,
    dta_mm: float = 3.0,
    threshold_pct: float = 10.0,
    max_pts: int = 4000,
) -> float:
    """
    Evaluates 3D Gamma passing rate according to AAPM TG-218 protocol.
    """
    d_max = float(d_true.max())
    if d_max <= 0:
        return 100.0

    eval_mask = mask & (d_true >= (threshold_pct / 100.0) * d_max)
    if not np.any(eval_mask):
        return 100.0

    delta_d_tol = (dose_pct / 100.0) * d_max
    p_pts = np.argwhere(eval_mask)

    if len(p_pts) > max_pts:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(len(p_pts), size=max_pts, replace=False)
        p_pts = p_pts[idx_sub]

    sz = int(np.ceil(dta_mm / voxel_size_mm[0]))
    sy = int(np.ceil(dta_mm / voxel_size_mm[1]))
    sx = int(np.ceil(dta_mm / voxel_size_mm[2]))

    nz, ny, nx = d_true.shape
    passed = 0
    total = len(p_pts)

    for z, y, x in p_pts:
        d_ref = d_true[z, y, x]
        z_min, z_max = max(0, z - sz), min(nz, z + sz + 1)
        y_min, y_max = max(0, y - sy), min(ny, y + sy + 1)
        x_min, x_max = max(0, x - sx), min(nx, x + sx + 1)

        sub_pred = d_pred[z_min:z_max, y_min:y_max, x_min:x_max]
        gz, gy, gx = np.ogrid[z_min - z : z_max - z, y_min - y : y_max - y, x_min - x : x_max - x]
        dist_sq = (gz * voxel_size_mm[0]) ** 2 + (gy * voxel_size_mm[1]) ** 2 + (gx * voxel_size_mm[2]) ** 2

        gamma_sq = (dist_sq / (dta_mm**2)) + ((sub_pred - d_ref) ** 2) / (delta_d_tol**2)
        if np.min(gamma_sq) <= 1.0:
            passed += 1

    return float(passed / total * 100.0)


# ==============================================================================
# 2. Live Evaluation Runner (Per-Angle and Total Dose)
# ==============================================================================
def run_live_pipeline():
    """Load trained checkpoints, evaluate all 12 test patients, and render all paper figures and tables.

    Args:
        None.
    Returns:
        None. Writes comparison PNGs, summary tables (.txt/.png), loss curves, and slice metadata to outputs/.
    """
    print("=" * 85)
    print("STARTING LIVE MODEL INFERENCE AND METRICS CALCULATION")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
    print(f"Device configured: {device} ({device_name})")
    grid_shape = (32, 72, 128)
    nz, ny, nx = grid_shape

    # 1. Load Trained Models
    u_ckpt_path = BASE_DIR / "outputs" / "models" / "unet_clinical_best.pt"
    g_ckpt_path = BASE_DIR / "outputs" / "models" / "gnn_hierarchical_best.pt"
    if not g_ckpt_path.exists():
        g_ckpt_path = BASE_DIR / "outputs" / "gnn_hierarchical" / "gnn_hierarchical_best.pt"
    t_ckpt_path = BASE_DIR / "outputs" / "models" / "unet_total_dose_best.pt"

    print(f"Loading 3D U-Net from: {u_ckpt_path.name}")
    u_ckpt = torch.load(u_ckpt_path, map_location=device)
    unet = RhoThetaDoseUNet(
        grid_shape=grid_shape,
        in_node_features=5,
        hidden_channels=32,
        depth=3,
        num_harmonic_bands=4,
        use_physics_residual=False,
    ).to(device)
    unet.load_state_dict(u_ckpt["state_dict"])
    unet.eval()

    print(f"Loading Hierarchical GNN from: {g_ckpt_path.name}")
    g_ckpt = torch.load(g_ckpt_path, map_location=device)
    gnn = HierarchicalGridGNN(
        grid_shape=grid_shape,
        in_node_features=8,
        hidden_channels=32,
        num_harmonic_bands=4,
    ).to(device)
    gnn.load_state_dict(g_ckpt["state_dict"])
    gnn.eval()
    print(f"  GNN checkpoint: {g_ckpt_path.name} (epoch {g_ckpt.get('epoch')})")

    print(f"Loading Total CT Dose U-Net from: {t_ckpt_path.name}")
    t_ckpt = torch.load(t_ckpt_path, map_location=device)
    tot_unet = TotalCTDoseUNet(in_channels=3, base_channels=32, grid_shape=grid_shape).to(device)
    tot_unet.load_state_dict(t_ckpt["state_dict"])
    tot_unet.eval()

    r_grid = build_radial_distance_grid(grid_shape)

    per_angle_results = {}
    total_dose_results = {}
    slice_metadata_records = {}

    # Coordinate setup cache for per-angle simulation
    sample_npz = np.load(BASE_DIR / f"resources/outputs_random_angel/{PATIENTS[0]}/{PATIENTS[0]}.npz")
    xb = sample_npz["x_bounds_cm"]
    yb = sample_npz["y_bounds_cm"]
    zb = sample_npz["z_bounds_cm"]
    bounds_cm = (float(xb[0]), float(xb[-1]), float(yb[0]), float(yb[-1]), float(zb[0]), float(zb[-1]))

    zb_arr = np.linspace(bounds_cm[4], bounds_cm[5], nz)
    yb_arr = np.linspace(bounds_cm[2], bounds_cm[3], ny)
    xb_arr = np.linspace(bounds_cm[0], bounds_cm[1], nx)
    zg, yg, xg = np.meshgrid(zb_arr, yb_arr, xb_arr, indexing="ij")
    coords_cm = np.stack([xg, yg, zg], axis=-1).reshape(-1, 3).astype(np.float32)
    coords_norm = coords_cm.copy()
    coords_norm[:, 0] /= max(abs(bounds_cm[0]), abs(bounds_cm[1]))
    coords_norm[:, 1] /= max(abs(bounds_cm[2]), abs(bounds_cm[3]))
    coords_norm[:, 2] /= max(abs(bounds_cm[4]), abs(bounds_cm[5]))
    coords_cm_t = torch.from_numpy(coords_cm)
    coords_norm_t = torch.from_numpy(coords_norm)

    print(f"\nExecuting live inference across all {len(PATIENTS)} test patients...")

    for pt_id in PATIENTS:
        t0 = time.time()
        print(f"\nProcessing {pt_id}...")

        # ---------------------------------------------------------------------
        # Part A: Single-Beam Per-Angle Evaluation (U-Net vs GNN)
        # ---------------------------------------------------------------------
        pa_npz_path = BASE_DIR / "resources" / "outputs_random_angel" / pt_id / f"{pt_id}.npz"
        pa_data = np.load(pa_npz_path)

        rho = pa_data["phantom_density"].astype(np.float32)
        thetas = pa_data["theta"].astype(np.float32)
        raw_doses = pa_data["dose"].astype(np.float32)  # in Gy

        mid_idx = len(thetas) // 2
        theta_val = float(thetas[mid_idx])
        theta_t = torch.tensor([theta_val])

        rho_flat_t = torch.from_numpy(rho.reshape(-1, 1))
        rho_t = torch.from_numpy(rho)

        xf_u = construct_batch_features(rho_flat_t, rho_t, coords_norm_t, coords_cm_t, theta_t, bounds_cm, 5).to(device)
        xf_g = construct_batch_features(rho_flat_t, rho_t, coords_norm_t, coords_cm_t, theta_t, bounds_cm, 8).to(device)
        theta_dev = theta_t.to(device)

        with torch.no_grad():
            pred_u = unet(xf_u, theta_dev)[0].cpu().numpy().reshape(grid_shape)  # in 1e-12 Gy
            pred_g = gnn(xf_g, theta_dev)[0].cpu().numpy().reshape(grid_shape)   # in 1e-12 Gy

        gt_dose = (raw_doses[mid_idx] / GLOBAL_DOSE_SCALE).astype(np.float32)  # in 1e-12 Gy
        d_max = float(gt_dose.max())
        tissue_mask = rho > 0.005

        # Calculate live clinical metrics for Single Beam
        gamma3_u = compute_gamma_index_3d(pred_u, gt_dose, tissue_mask, (1.0, 1.0, 1.0), 3.0, 3.0)
        gamma2_u = compute_gamma_index_3d(pred_u, gt_dose, tissue_mask, (1.0, 1.0, 1.0), 2.0, 2.0)
        mae_u = float(np.mean(np.abs(pred_u[tissue_mask] - gt_dose[tissue_mask])))

        gamma3_g = compute_gamma_index_3d(pred_g, gt_dose, tissue_mask, (1.0, 1.0, 1.0), 3.0, 3.0)
        gamma2_g = compute_gamma_index_3d(pred_g, gt_dose, tissue_mask, (1.0, 1.0, 1.0), 2.0, 2.0)
        mae_g = float(np.mean(np.abs(pred_g[tissue_mask] - gt_dose[tissue_mask])))

        # In-field region: dose >= 5% of Dmax within patient
        in_f = tissue_mask & (gt_dose >= 0.05 * d_max)
        if np.any(in_f):
            mre_u = float(np.mean(np.abs(pred_u[in_f] - gt_dose[in_f]) / gt_dose[in_f]) * 100.0)
            mre_g = float(np.mean(np.abs(pred_g[in_f] - gt_dose[in_f]) / gt_dose[in_f]) * 100.0)
        else:
            mre_u, mre_g = 0.0, 0.0

        # Voxels within 5% error (relative to Dmax standard, in-field)
        diff_pct_u = (np.abs(pred_u - gt_dose) / (d_max + 1e-8)) * 100.0
        diff_pct_g = (np.abs(pred_g - gt_dose) / (d_max + 1e-8)) * 100.0
        w5_u = float((diff_pct_u[in_f] <= 5.0).mean() * 100.0) if np.any(in_f) else 100.0
        w5_g = float((diff_pct_g[in_f] <= 5.0).mean() * 100.0) if np.any(in_f) else 100.0

        per_angle_results[pt_id] = {
            "theta": theta_val,
            "unet": {"gamma_3": gamma3_u, "gamma_2": gamma2_u, "mre": mre_u, "mae": mae_u, "w5": w5_u},
            "gnn":  {"gamma_3": gamma3_g, "gamma_2": gamma2_g, "mre": mre_g, "mae": mae_g, "w5": w5_g},
        }

        # ---------------------------------------------------------------------
        # Part B: Total Dose Full CT Scan Evaluation (U-Net ALONE, NO GNN)
        # ---------------------------------------------------------------------
        tot_egs_path = BASE_DIR / "resources" / "dataset" / "full_ct_dose" / f"{pt_id}.egsphant"
        tot_npz_path = BASE_DIR / "resources" / "dataset" / "full_ct_dose" / f"{pt_id}_36view.npz"

        ph = read_egsphant(tot_egs_path)
        tot_npz = np.load(tot_npz_path)

        tot_rho = ph["density"].astype(np.float32)
        tot_mask = tot_npz["patient_mask"].astype(bool)
        tot_raw_dose = tot_npz["dose"].astype(np.float32)
        tot_gt_dose = (tot_raw_dose * tot_mask).astype(np.float32)

        bone_accent = np.clip((tot_rho - 1.1) / (1.95 - 1.1), 0.0, 1.0).astype(np.float32)
        inp_tot = np.stack([tot_rho, r_grid, bone_accent], axis=0).astype(np.float32)
        inp_tot_t = torch.from_numpy(inp_tot).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_tot_scaled = tot_unet(inp_tot_t).squeeze().cpu().numpy()
        pred_tot_dose = (pred_tot_scaled * GLOBAL_DOSE_SCALE * tot_mask).astype(np.float32)

        tot_d_max = float(tot_gt_dose[tot_mask].max()) if np.any(tot_mask) else GLOBAL_DOSE_SCALE
        diff_tot_abs = np.abs(pred_tot_dose[tot_mask] - tot_gt_dose[tot_mask])
        diff_tot_pct_max = (diff_tot_abs / (tot_d_max + 1e-12)) * 100.0

        tot_w10 = float((diff_tot_pct_max <= 10.0).mean() * 100.0)
        tot_w5 = float((diff_tot_pct_max <= 5.0).mean() * 100.0)
        tot_mae = float(np.mean(diff_tot_abs) / GLOBAL_DOSE_SCALE)
        tot_mre = float(np.mean(diff_tot_abs / (tot_gt_dose[tot_mask] + 0.05 * GLOBAL_DOSE_SCALE)) * 100.0)

        total_dose_results[pt_id] = {
            "mre": tot_mre,
            "mae": tot_mae,
            "w10": tot_w10,
            "w5": tot_w5,
        }

        # ---------------------------------------------------------------------
        # Part C: Render Multi-View Visuals Directly in Matplotlib
        # ---------------------------------------------------------------------
        # 1. Per-Angle Figure (2 Rows x 3 Columns)
        meta_pa = render_per_angle_patient_figure(
            pt_id=pt_id,
            theta=theta_val,
            rho=rho,
            gt=gt_dose,
            p_u=pred_u,
            p_g=pred_g,
            metrics_u=per_angle_results[pt_id]["unet"],
            metrics_g=per_angle_results[pt_id]["gnn"],
            out_file=PER_ANGLE_DIR / f"{pt_id}_per_angle_comparison.png",
        )

        # 2. Total Dose Figure (2 Rows x 2 Columns) - U-Net Alone
        meta_tot = render_total_dose_patient_figure(
            pt_id=pt_id,
            rho=tot_rho,
            gt=tot_gt_dose / GLOBAL_DOSE_SCALE,
            pred=pred_tot_dose / GLOBAL_DOSE_SCALE,
            metrics=total_dose_results[pt_id],
            out_file=TOTAL_DOSE_DIR / f"{pt_id}_total_dose_comparison.png",
        )

        z_pa_cm = float(zb_arr[meta_pa["cut_z"]])
        z_tot_cm = float(zb_arr[meta_tot["cut_z"]])

        slice_metadata_records[pt_id] = {
            "patient_id": pt_id,
            "single_beam_comparison": {
                "gantry_angle_deg": theta_val,
                "slice_plane": "Coronal (X-Y plane, constant Z)",
                "slice_axis": "Z-axis (cranio-caudal)",
                "slice_index_z": meta_pa["cut_z"],
                "total_slices_z": nz,
                "physical_z_cm": round(z_pa_cm, 3),
                "dmax_voxel_zyx": [meta_pa["cut_z"], meta_pa["cut_y"], meta_pa["cut_x"]],
                "dmax_value_1e12_gy": round(meta_pa["d_max"], 4),
            },
            "total_ct_dose_comparison": {
                "acquisition": "Full 360° CT arc (36 projections at 10° intervals)",
                "slice_plane": "Coronal (X-Y plane, constant Z)",
                "slice_axis": "Z-axis (cranio-caudal)",
                "slice_index_z": meta_tot["cut_z"],
                "total_slices_z": nz,
                "physical_z_cm": round(z_tot_cm, 3),
                "dmax_voxel_zyx": [meta_tot["cut_z"], meta_tot["cut_y"], meta_tot["cut_x"]],
                "dmax_value_1e12_gy": round(meta_tot["d_max"], 4),
            },
        }

        print(f"  [✓] {pt_id} complete (evaluated in {time.time()-t0:.2f}s | Slice Z={meta_pa['cut_z']} at θ={theta_val:.1f}°).")

    # Export Visualization Slice Metadata Files
    meta_json_path = OUT_DIR / "visualization_slice_metadata.json"
    meta_txt_path = OUT_DIR / "visualization_slice_metadata.txt"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(slice_metadata_records, f, indent=2)

    with open(meta_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 115 + "\n")
        f.write("RADIOTHERAPY DOSE PREDICTION — VISUALIZATION SLICE & ANGLE SPECIFICATION METADATA\n")
        f.write(f"Grid Dimensions: (Z={nz}, Y={ny}, X={nx}) | Voxel Size: 2.0 mm cubic\n")
        f.write(f"Physical Bounds: X=[{bounds_cm[0]:.1f}, {bounds_cm[1]:.1f}] cm | Y=[{bounds_cm[2]:.1f}, {bounds_cm[3]:.1f}] cm | Z=[{bounds_cm[4]:.1f}, {bounds_cm[5]:.1f}] cm\n")
        f.write("=" * 115 + "\n\n")
        f.write(f"{'Patient ID':<12} | {'Vis Case':<18} | {'Gantry Angle (θ)':<18} | {'Slice Plane / Axis':<22} | {'Index (Z)':<12} | {'Phys Z (cm)':<12} | {'Dmax Voxel (Z,Y,X)'}\n")
        f.write("-" * 115 + "\n")
        for pid, r in slice_metadata_records.items():
            sb = r["single_beam_comparison"]
            tot = r["total_ct_dose_comparison"]
            f.write(f"{pid:<12} | {'Single-Beam':<18} | {sb['gantry_angle_deg']:>6.1f}°            | {'Coronal (Z-axis)':<22} | {sb['slice_index_z']:>2d} / {nz}      | {sb['physical_z_cm']:>+6.2f} cm   | {str(sb['dmax_voxel_zyx'])}\n")
            f.write(f"{pid:<12} | {'Total CT Dose':<18} | {'Full 360° Arc':<18} | {'Coronal (Z-axis)':<22} | {tot['slice_index_z']:>2d} / {nz}      | {tot['physical_z_cm']:>+6.2f} cm   | {str(tot['dmax_voxel_zyx'])}\n")
            f.write("-" * 115 + "\n")
    print(f"\n  [✓] Exported slice & angle metadata to: {meta_json_path.name} & {meta_txt_path.name}")

    # -------------------------------------------------------------------------
    # Part D: Generate Publication Summary Tables (Table 2 & Table 3 in .txt & .png)
    # -------------------------------------------------------------------------
    generate_publication_summary_tables(per_angle_results, total_dose_results)

    # -------------------------------------------------------------------------
    # Part E: Render Clean Training Loss Curves from Exported History
    # -------------------------------------------------------------------------
    render_clean_loss_curves()

    # Copy files to conversation artifacts directory
    copy_artifacts_to_brain()

    print("\n" + "=" * 85)
    print("ALL MASTER VISUALIZATIONS & BENCHMARK ANALYSES GENERATED SUCCESSFULLY!")
    print("=" * 85)


# ==============================================================================
# 3. Clean Figure Rendering Routines
# ==============================================================================
def render_per_angle_patient_figure(
    pt_id: str,
    theta: float,
    rho: np.ndarray,
    gt: np.ndarray,
    p_u: np.ndarray,
    p_g: np.ndarray,
    metrics_u: Dict,
    metrics_g: Dict,
    out_file: Path,
):
    """Render a 2x3 comparison figure (GT, 3D U-Net, Hierarchical GNN, CT density, and error maps).

    Args:
        pt_id: Patient identifier string.
        theta: Gantry angle in degrees.
        rho: [nz, ny, nx] CT density volume (g/cm^3).
        gt: [nz, ny, nx] ground-truth dose volume (1e-12 Gy).
        p_u: [nz, ny, nx] 3D U-Net predicted dose volume (1e-12 Gy).
        p_g: [nz, ny, nx] Hierarchical GNN predicted dose volume (1e-12 Gy).
        metrics_u: Dictionary of single-beam metrics for 3D U-Net.
        metrics_g: Dictionary of single-beam metrics for Hierarchical GNN.
        out_file: Output PNG path.
    Returns:
        Dict with keys 'cut_z', 'cut_y', 'cut_x', and 'd_max' for the slice through Dmax.
    """
    d_max = float(gt.max())
    if d_max <= 0:
        d_max = 1.0

    err_u = ((p_u - gt) / d_max) * 100.0
    err_g = ((p_g - gt) / d_max) * 100.0

    cut_z, cut_y, cut_x = np.unravel_index(np.argmax(gt), gt.shape)

    # Single representative slice through the dose maximum (coronal)
    s_rho = rho[cut_z, :, :]
    s_gt  = gt[cut_z, :, :]
    s_u   = p_u[cut_z, :, :]
    s_g   = p_g[cut_z, :, :]
    s_eu  = err_u[cut_z, :, :]
    s_eg  = err_g[cut_z, :, :]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    fig.patch.set_facecolor("white")

    # Row 0: True MC | U-Net | GNN
    im_dose = axes[0, 0].imshow(s_gt,  cmap="turbo",   origin="lower", aspect="auto", vmin=0.0,   vmax=d_max)
    axes[0, 0].set_title("True MC Dose",        fontsize=12, fontweight="bold", pad=8)

    axes[0, 1].imshow(s_u,  cmap="turbo",   origin="lower", aspect="auto", vmin=0.0,   vmax=d_max)
    axes[0, 1].set_title("3D U-Net Prediction", fontsize=12, fontweight="bold", pad=8)

    axes[0, 2].imshow(s_g,  cmap="turbo",   origin="lower", aspect="auto", vmin=0.0,   vmax=d_max)
    axes[0, 2].set_title("Hierarchical GNN Prediction", fontsize=12, fontweight="bold", pad=8)

    # Row 1: Density | U-Net error | GNN error
    im_rho = axes[1, 0].imshow(s_rho, cmap="bone",    origin="lower", aspect="auto", vmin=0.0,   vmax=1.85)
    axes[1, 0].set_title("CT Density",          fontsize=12, fontweight="bold", pad=8)

    im_err = axes[1, 1].imshow(s_eu, cmap="coolwarm", origin="lower", aspect="auto", vmin=-20.0, vmax=20.0)
    axes[1, 1].set_title("U-Net Difference (% D_max)", fontsize=12, fontweight="bold", pad=8)

    axes[1, 2].imshow(s_eg, cmap="coolwarm", origin="lower", aspect="auto", vmin=-20.0, vmax=20.0)
    axes[1, 2].set_title("GNN Difference (% D_max)", fontsize=12, fontweight="bold", pad=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0.02, 0.13, 0.98, 0.97])

    # Colorbars at the bottom (raised to stay cleanly on-screen)
    cax0 = fig.add_axes([0.05, 0.06, 0.26, 0.02])
    cb0 = fig.colorbar(im_rho, cax=cax0, orientation="horizontal")
    cb0.set_label("Density (g/cm³)", fontsize=10, fontweight="bold")

    cax1 = fig.add_axes([0.37, 0.06, 0.26, 0.02])
    cb1 = fig.colorbar(im_dose, cax=cax1, orientation="horizontal")
    cb1.set_label("Radiotherapy Dose (×10⁻¹² Gy)", fontsize=10, fontweight="bold")

    cax2 = fig.add_axes([0.69, 0.06, 0.26, 0.02])
    cb2 = fig.colorbar(im_err, cax=cax2, orientation="horizontal")
    cb2.set_label("Dose Difference [% D_max]", fontsize=10, fontweight="bold")

    plt.savefig(out_file, facecolor="white", dpi=150)
    plt.close()

    return {
        "cut_z": int(cut_z),
        "cut_y": int(cut_y),
        "cut_x": int(cut_x),
        "d_max": float(d_max),
    }


def render_total_dose_patient_figure(
    pt_id: str,
    rho: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    metrics: Dict,
    out_file: Path,
):
    """Render a 2x2 full-scan total dose comparison figure (GT, Total CT Dose U-Net, density, error).

    Args:
        pt_id: Patient identifier string.
        rho: [nz, ny, nx] CT density volume (g/cm^3).
        gt: [nz, ny, nx] ground-truth cumulative dose volume (1e-12 Gy).
        pred: [nz, ny, nx] Total CT Dose U-Net predicted dose volume (1e-12 Gy).
        metrics: Dictionary of total-dose metrics for this patient.
        out_file: Output PNG path.
    Returns:
        Dict with keys 'cut_z', 'cut_y', 'cut_x', and 'd_max' for the slice through Dmax.
    """
    d_max = float(gt.max())
    if d_max <= 0:
        d_max = 1.0

    err_tot = ((pred - gt) / d_max) * 100.0

    cut_z, cut_y, cut_x = np.unravel_index(np.argmax(gt), gt.shape)

    # Single representative slice through the dose maximum (coronal)
    s_rho  = rho[cut_z, :, :]
    s_gt   = gt[cut_z, :, :]
    s_pred = pred[cut_z, :, :]
    s_err  = err_tot[cut_z, :, :]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("white")

    # Top-left: True MC Total Dose
    im_dose = axes[0, 0].imshow(s_gt, cmap="turbo", origin="lower", aspect="auto", vmin=0.0, vmax=d_max)
    axes[0, 0].set_title("True MC Total Dose", fontsize=12, fontweight="bold", pad=8)

    # Top-right: U-Net Prediction
    axes[0, 1].imshow(s_pred, cmap="turbo", origin="lower", aspect="auto", vmin=0.0, vmax=d_max)
    axes[0, 1].set_title("Total CT Dose U-Net Prediction", fontsize=12, fontweight="bold", pad=8)

    # Bottom-left: CT Density
    im_rho = axes[1, 0].imshow(s_rho, cmap="bone", origin="lower", aspect="auto", vmin=0.0, vmax=1.85)
    axes[1, 0].set_title("CT Density", fontsize=12, fontweight="bold", pad=8)

    # Bottom-right: Difference
    im_err = axes[1, 1].imshow(s_err, cmap="coolwarm", origin="lower", aspect="auto", vmin=-20.0, vmax=20.0)
    axes[1, 1].set_title("Difference (% of D_max)", fontsize=12, fontweight="bold", pad=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0.02, 0.13, 0.98, 0.97])

    # Horizontal colorbars at the bottom (raised to stay cleanly on-screen)
    cax0 = fig.add_axes([0.06, 0.06, 0.26, 0.02])
    cb0 = fig.colorbar(im_rho, cax=cax0, orientation="horizontal")
    cb0.set_label("Density (g/cm³)", fontsize=10, fontweight="bold")

    cax1 = fig.add_axes([0.37, 0.06, 0.26, 0.02])
    cb1 = fig.colorbar(im_dose, cax=cax1, orientation="horizontal")
    cb1.set_label("Total Dose (×10⁻¹² Gy)", fontsize=10, fontweight="bold")

    cax2 = fig.add_axes([0.68, 0.06, 0.26, 0.02])
    cb2 = fig.colorbar(im_err, cax=cax2, orientation="horizontal")
    cb2.set_label("Difference [% of D_max]", fontsize=10, fontweight="bold")

    plt.savefig(out_file, facecolor="white", dpi=150)
    plt.close()

    return {
        "cut_z": int(cut_z),
        "cut_y": int(cut_y),
        "cut_x": int(cut_x),
        "d_max": float(d_max),
    }



# ==============================================================================
# 4. Publication Summary Tables (Table 2 & Table 3 in .txt and .png)
# ==============================================================================
def render_latex_style_table_png(caption: str, headers: List[str], rows: List[List[str]], out_png_path: Path):
    """
    Renders publication-grade tables matching LaTeX booktabs styling:
    - Pure white background
    - Elegant serif typography
    - Bold headers with proper horizontal rules
    - Left-aligned model names, centered numerical data
    """
    fig, ax = plt.subplots(figsize=(10.5, 2.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    fig.text(0.04, 0.88, caption, fontsize=9.5, family="serif", ha="left", va="top", wrap=True)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.85)

    num_rows = len(rows)
    num_cols = len(headers)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("none")
        cell.set_facecolor("white")
        cell.get_text().set_fontfamily("serif")

        # Left align architecture column
        if c == 0:
            cell.set_text_props(ha="left")

        # Top border and header formatting
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(1.4)
            cell.set_edgecolor("black")
        elif r == num_rows:
            # Bottom border
            cell.visible_edges = "B"
            cell.set_linewidth(1.4)
            cell.set_edgecolor("black")
        else:
            cell.visible_edges = ""

    # Ensure clean midrule under header
    for c in range(num_cols):
        table[(0, c)].visible_edges = "TB"
        table[(0, c)].set_edgecolor("black")
        table[(num_rows, c)].visible_edges = "B"
        table[(num_rows, c)].set_edgecolor("black")

    plt.subplots_adjust(left=0.04, right=0.96, top=0.72, bottom=0.1)
    plt.savefig(out_png_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Rendered: {out_png_path.name}")


def generate_publication_summary_tables(per_angle_results: Dict, total_dose_results: Dict):
    """
    Generates Table 2 (Single-Beam) and Table 3 (Total CT Dose) in both .txt and .png format.
    """
    print("\nGenerating publication summary tables (Table 2 & Table 3)...")

    # --- Table 2: Single-Beam Dose Prediction ---
    u_g3  = [per_angle_results[p]["unet"]["gamma_3"] for p in PATIENTS]
    u_mre = [per_angle_results[p]["unet"]["mre"] for p in PATIENTS]
    u_mae = [per_angle_results[p]["unet"]["mae"] for p in PATIENTS]
    u_w5  = [per_angle_results[p]["unet"]["w5"] for p in PATIENTS]

    g_g3  = [per_angle_results[p]["gnn"]["gamma_3"] for p in PATIENTS]
    g_mre = [per_angle_results[p]["gnn"]["mre"] for p in PATIENTS]
    g_mae = [per_angle_results[p]["gnn"]["mae"] for p in PATIENTS]
    g_w5  = [per_angle_results[p]["gnn"]["w5"] for p in PATIENTS]

    cap2 = r"Table 2: Single-beam radiation dose prediction performance summary (Mean $\pm$ Standard Deviation) across the 12 unseen test patients comparing the 3D U-Net and Hierarchical GNN."
    headers2 = ["Architecture", r"$\gamma(3\%/3\,\mathrm{mm})$ [%]", "MRE [%]", r"MAE [$\times 10^{-12}$ Gy]", r"$\delta < 5\%$ [%]"]
    rows2 = [
        [r"3D U-Net", rf"{np.mean(u_g3):.1f} $\pm$ {np.std(u_g3):.1f}", rf"{np.mean(u_mre):.1f} $\pm$ {np.std(u_mre):.1f}", rf"{np.mean(u_mae):.4f} $\pm$ {np.std(u_mae):.4f}", rf"{np.mean(u_w5):.1f} $\pm$ {np.std(u_w5):.1f}"],
        [r"Hierarchical GNN", rf"{np.mean(g_g3):.1f} $\pm$ {np.std(g_g3):.1f}", rf"{np.mean(g_mre):.1f} $\pm$ {np.std(g_mre):.1f}", rf"{np.mean(g_mae):.4f} $\pm$ {np.std(g_mae):.4f}", rf"{np.mean(g_w5):.1f} $\pm$ {np.std(g_w5):.1f}"]
    ]

    # Save Table 2 TXT
    t2_txt = OUT_DIR / "table2_single_beam_summary.txt"
    with open(t2_txt, "w", encoding="utf-8") as f:
        f.write("Table 2: Single-beam radiation dose prediction performance summary (Mean ± Standard Deviation) across the 12 unseen test patients comparing the 3D U-Net and Hierarchical GNN.\n")
        f.write("=" * 110 + "\n")
        f.write(f"{'Architecture':<22} | {'γ(3%/3 mm) [%]':<18} | {'MRE [%]':<14} | {'MAE [×10^-12 Gy]':<22} | {'δ < 5% [%]':<14}\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'3D U-Net':<22} | {np.mean(u_g3):>5.1f} ± {np.std(u_g3):<9.1f} | {np.mean(u_mre):>5.1f} ± {np.std(u_mre):<5.1f} | {np.mean(u_mae):>6.4f} ± {np.std(u_mae):<12.4f} | {np.mean(u_w5):>5.1f} ± {np.std(u_w5):<5.1f}\n")
        f.write(f"{'Hierarchical GNN':<22} | {np.mean(g_g3):>5.1f} ± {np.std(g_g3):<9.1f} | {np.mean(g_mre):>5.1f} ± {np.std(g_mre):<5.1f} | {np.mean(g_mae):>6.4f} ± {np.std(g_mae):<12.4f} | {np.mean(g_w5):>5.1f} ± {np.std(g_w5):<5.1f}\n")
        f.write("=" * 110 + "\n")
    print(f"  [✓] Saved: {t2_txt.name}")

    # Save Table 2 PNG
    render_latex_style_table_png(cap2, headers2, rows2, OUT_DIR / "table2_single_beam_summary.png")

    # --- Table 3: Total CT Scan Dose Prediction ---
    t_mre = [total_dose_results[p]["mre"] for p in PATIENTS]
    t_mae = [total_dose_results[p]["mae"] for p in PATIENTS]
    t_w5  = [total_dose_results[p]["w5"] for p in PATIENTS]

    cap3 = r"Table 3: Cumulative full-scan CT dose prediction performance summary (Mean $\pm$ Standard Deviation) across the 12 unseen test patients for the 3D ResNet U-Net against DOSXYZnrc Monte Carlo ground truth."
    headers3 = ["Architecture", "MRE [%]", r"MAE [$\times 10^{-12}$ Gy]", r"$\delta < 5\%$ [%]"]
    rows3 = [
        [r"3D ResNet U-Net", rf"{np.mean(t_mre):.2f} $\pm$ {np.std(t_mre):.2f}", rf"{np.mean(t_mae):.4f} $\pm$ {np.std(t_mae):.4f}", rf"{np.mean(t_w5):.2f} $\pm$ {np.std(t_w5):.2f}"]
    ]

    # Save Table 3 TXT
    t3_txt = OUT_DIR / "table3_total_dose_summary.txt"
    with open(t3_txt, "w", encoding="utf-8") as f:
        f.write("Table 3: Cumulative full-scan CT dose prediction performance summary (Mean ± Standard Deviation) across the 12 unseen test patients for the 3D ResNet U-Net against DOSXYZnrc Monte Carlo ground truth.\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Architecture':<24} | {'MRE [%]':<18} | {'MAE [×10^-12 Gy]':<24} | {'δ < 5% [%]':<16}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'3D ResNet U-Net':<24} | {np.mean(t_mre):>5.2f} ± {np.std(t_mre):<9.2f} | {np.mean(t_mae):>6.4f} ± {np.std(t_mae):<14.4f} | {np.mean(t_w5):>5.2f} ± {np.std(t_w5):<7.2f}\n")
        f.write("=" * 100 + "\n")
    print(f"  [✓] Saved: {t3_txt.name}")

    # Save Table 3 PNG
    render_latex_style_table_png(cap3, headers3, rows3, OUT_DIR / "table3_total_dose_summary.png")


# ==============================================================================
# 5. Clean Training Loss Curves
# ==============================================================================
def render_clean_loss_curves():
    """Render training/validation loss curves individually and on one unified comparison plot.

    Args:
        None.
    Returns:
        None. Saves per-model PNGs and a combined single-axes plot (all_models_training_loss.png) to OUT_DIR.
    """
    print("\nRendering clean training loss curves from history...")

    configs = [
        {
            "path": BASE_DIR / "outputs" / "unet_supercomputer" / "training_history.json",
            "name": "3D U-Net",
            "color": "#1f77b4",
            "out": OUT_DIR / "unet_training_loss.png",
        },
        {
            "path": BASE_DIR / "outputs" / "gnn_hierarchical" / "training_history.json",
            "name": "Hierarchical GNN",
            "color": "#d95f02",
            "out": OUT_DIR / "gnn_training_loss.png",
        },
        {
            "path": BASE_DIR / "outputs" / "unet_total_dose" / "loss_history.json",
            "name": "Total CT Dose U-Net",
            "color": "#1b9e77",
            "out": OUT_DIR / "total_dose_unet_training_loss.png",
        },
    ]

    # Single unified axes plot comparing all models together
    fig_all, ax_all = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    fig_all.patch.set_facecolor("white")
    ax_all.set_facecolor("white")

    for cfg in configs:
        p = cfg["path"]
        if not p.exists():
            print(f"  [!] History file not found: {p}")
            continue

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=180)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            loss_dict = data.get("loss", {})
            hist = data.get("history", data)

            ep_tr = loss_dict.get("epoch", hist.get("train_epochs", []))
            loss_tr = loss_dict.get("train_loss", hist.get("train_loss", []))
            val_loss = loss_dict.get("val_loss", hist.get("val_loss", []))
            ep_val = hist.get("val_epochs", [])
            val_mae = hist.get("val_mae", [])

            has_val_loss = bool(val_loss and len(val_loss) > 0)

            # Individual plot
            ax.plot(ep_tr, loss_tr, color="#1f77b4", lw=2.0, label="Training Loss")
            if has_val_loss:
                val_x = ep_tr if len(val_loss) == len(ep_tr) else ep_val
                ax.plot(val_x, val_loss, color="#d62728", lw=1.8, ls="--", label="Validation Loss")
                ax.set_ylabel("Loss", fontsize=10.5)
            elif val_mae:
                ax.plot(ep_val, val_mae, color="#d62728", lw=1.8, ls="--", label=r"Val MAE ($\times 10^{-12}$ Gy)")
                ax.set_ylabel("Loss / Error", fontsize=10.5)
            else:
                ax.set_ylabel("Loss", fontsize=10.5)

            ax.set_xlabel("Epoch", fontsize=10.5)
            ax.set_title(f"{cfg['name']} Training Loss", fontsize=11.5, fontweight="bold", pad=9)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(frameon=True, fontsize=9.5, facecolor="white", edgecolor="#dfe6e9")
            fig.tight_layout()
            fig.savefig(cfg["out"], dpi=180, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            print(f"  [✓] Rendered clean loss curve: {cfg['out'].name}")

            # Unified single-axes comparison plot
            c = cfg["color"]
            ax_all.plot(ep_tr, loss_tr, color=c, lw=2.0, ls="-", label=f"{cfg['name']} (Train)")
            if has_val_loss:
                val_x = ep_tr if len(val_loss) == len(ep_tr) else ep_val
                ax_all.plot(val_x, val_loss, color=c, lw=1.8, ls="--", alpha=0.85, label=f"{cfg['name']} (Val)")

        except Exception as e:
            print(f"  [!] Error rendering {cfg['name']}: {e}")

    ax_all.set_xlabel("Epoch", fontsize=11)
    ax_all.set_ylabel("Loss", fontsize=11)
    ax_all.set_title("Training and Validation Loss Comparison", fontsize=12.5, fontweight="bold", pad=10)
    ax_all.grid(True, linestyle=":", alpha=0.5)
    ax_all.legend(frameon=True, fontsize=9, ncol=2, loc="upper right", facecolor="white", edgecolor="#dfe6e9")
    fig_all.tight_layout()
    all_out = OUT_DIR / "all_models_training_loss.png"
    fig_all.savefig(all_out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig_all)
    print(f"  [✓] Rendered unified single-axes plot: {all_out.name}")


def copy_artifacts_to_brain():
    """Copy generated summary tables, loss curves, and patient comparison PNGs to ARTIFACT_DIR if configured.

    Args:
        None.
    Returns:
        None.
    """
    if ARTIFACT_DIR is None or not ARTIFACT_DIR.exists():
        return

    print("\nCopying master visualizations to artifact directory...")
    # Summary tables & loss curves
    artifacts_to_copy = [
        "table2_single_beam_summary.png",
        "table2_single_beam_summary.txt",
        "table3_total_dose_summary.png",
        "table3_total_dose_summary.txt",
        "unet_training_loss.png",
        "gnn_training_loss.png",
        "total_dose_unet_training_loss.png",
        "all_models_training_loss.png",
        "visualization_slice_metadata.json",
        "visualization_slice_metadata.txt",
    ]
    for name in artifacts_to_copy:
        src = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, ARTIFACT_DIR / name)

    # Sample patient comparisons (kept as-is)
    for p in PATIENTS:
        pa_src = PER_ANGLE_DIR / f"{p}_per_angle_comparison.png"
        tot_src = TOTAL_DOSE_DIR / f"{p}_total_dose_comparison.png"
        if pa_src.exists():
            shutil.copy2(pa_src, ARTIFACT_DIR / f"{p}_per_angle_comparison.png")
        if tot_src.exists():
            shutil.copy2(tot_src, ARTIFACT_DIR / f"{p}_total_dose_comparison.png")


if __name__ == "__main__":
    run_live_pipeline()
