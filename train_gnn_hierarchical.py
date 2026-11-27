"""
train_gnn_hierarchical.py
=========================
Trains the improved Multi-Scale Hierarchical Graph Neural Network (HierarchicalGridGNN)
on the supercomputer clinical OpenKBP CT cohort (outputs_random_angel):
  - Overcomes the 3 mm receptive field bottleneck via a 3-level graph pyramid
  - 8-Channel physical features: [rho, zeff, coords_norm(x, y, z), d_parallel, d_perp, wepl]
  - Method 2 (Air Regularization Loss): Penalizes stray dose in room air
  - Multi-Objective Radiotherapy Loss (L1, MSE, in-field relative error, bone, 3D gradient)
  - Evaluates Gamma 3%/3mm and Gamma 2%/2mm on held-out test patients
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import HierarchicalGridGNN
from src.utils import construct_batch_features

GLOBAL_DOSE_SCALE = 1.0e-12

DEFAULT_TEST_PATIENTS = {"pt_210", "pt_222", "pt_223", "pt_230", "pt_233", "pt_234", "pt_253", "pt_259", "pt_260", "pt_261", "pt_262", "pt_263"}
DEFAULT_VAL_PATIENTS = {"pt_201", "pt_216", "pt_225", "pt_239", "pt_246", "pt_251", "pt_252", "pt_255"}


# ==============================================================================
# 1. Multi-Objective Loss with Air Suppression
# ==============================================================================
def compute_clinical_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    rho: torch.Tensor,
    mask: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    w_rel: float = 0.5,
    w_bone: float = 0.5,
    w_grad: float = 0.1,
    w_air: float = 1.0,
) -> torch.Tensor:
    """Compute composite radiotherapy loss combining L1, MSE, in-field relative, focal bone, gradient, and air terms.

    Args:
        pred: [B, N] predicted dose tensor.
        target: [B, N] ground-truth dose tensor.
        rho: [B, N] voxel mass density tensor (g/cm^3).
        mask: [N] boolean mask of in-body voxels.
        grid_shape: (nz, ny, nx) spatial dimensions of the 3D volume.
        w_rel: Weight for in-field relative error term.
        w_bone: Weight for high-density bone and focal bone-peak term.
        w_grad: Weight for 3D spatial gradient L1 term.
        w_air: Weight for ambient air suppression MSE term.
    Returns:
        Scalar composite loss tensor.
    """
    masked_pred = pred[:, mask]
    masked_target = target[:, mask]
    masked_rho = rho[:, mask]

    l1 = F.l1_loss(masked_pred, masked_target)
    mse = F.mse_loss(masked_pred, masked_target)

    in_field = masked_target >= 0.05 * masked_target.max()
    if in_field.any():
        rel_diff = torch.abs(masked_pred[in_field] - masked_target[in_field]) / (masked_target[in_field] + 0.05)
        rel_loss = torch.mean(rel_diff)
    else:
        rel_loss = torch.tensor(0.0, device=pred.device)

    bone_mask = masked_rho > 1.2
    if bone_mask.any():
        bone_loss = F.l1_loss(masked_pred[bone_mask], masked_target[bone_mask])
        # Focal Bone Peak Supervision: heavily penalizes truncating high-dose cortical bone peaks
        bone_peak_mask = bone_mask & (masked_target >= 0.25 * masked_target.max())
        if bone_peak_mask.any():
            bone_peak_rel = torch.abs(masked_pred[bone_peak_mask] - masked_target[bone_peak_mask]) / (masked_target[bone_peak_mask] + 0.05)
            bone_loss = bone_loss + 1.2 * torch.mean(bone_peak_rel)
    else:
        bone_loss = torch.tensor(0.0, device=pred.device)

    B = pred.shape[0]
    p_grid = pred.view(B, *grid_shape)
    t_grid = target.view(B, *grid_shape)
    dp_dz = torch.abs(p_grid[:, 1:, :, :] - p_grid[:, :-1, :, :])
    dt_dz = torch.abs(t_grid[:, 1:, :, :] - t_grid[:, :-1, :, :])
    dp_dy = torch.abs(p_grid[:, :, 1:, :] - p_grid[:, :, :-1, :])
    dt_dy = torch.abs(t_grid[:, :, 1:, :] - t_grid[:, :, :-1, :])
    dp_dx = torch.abs(p_grid[:, :, :, 1:] - p_grid[:, :, :, :-1])
    dt_dx = torch.abs(t_grid[:, :, :, 1:] - t_grid[:, :, :, :-1])
    grad_loss = (F.l1_loss(dp_dz, dt_dz) + F.l1_loss(dp_dy, dt_dy) + F.l1_loss(dp_dx, dt_dx)) / 3.0

    air_mask = ~mask
    if air_mask.any():
        air_pred = pred[:, air_mask]
        air_loss = F.mse_loss(air_pred, torch.zeros_like(air_pred))
    else:
        air_loss = torch.tensor(0.0, device=pred.device)

    return l1 + mse + w_rel * rel_loss + w_bone * bone_loss + w_grad * grad_loss + w_air * air_loss


# ==============================================================================
# 2. Fast 3D Gamma Index
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
    """Compute the 3D Gamma passing rate (%) between predicted and reference dose volumes.

    Args:
        d_pred: 3D predicted dose array.
        d_true: 3D reference dose array.
        mask: 3D boolean body mask.
        voxel_size_mm: (dz, dy, dx) voxel spacing in mm.
        dose_pct: Dose-difference tolerance as percentage of global Dmax.
        dta_mm: Distance-to-agreement tolerance in mm.
        threshold_pct: Minimum reference dose (% of Dmax) included in evaluation.
        max_pts: Maximum number of sampled voxels to evaluate.
    Returns:
        Gamma passing rate in percent (0.0 to 100.0).
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
# 3. Patient Dataset Loader
# ==============================================================================
class SupercomputerPatientCache:
    """Loads patient .npz volumes and caches precomputed physical beam features in host RAM."""

    def __init__(self, grid_shape: Tuple[int, int, int], bounds_cm: Tuple[float, ...], device: torch.device):
        """Initialize coordinate grids and device references for a fixed patient volume geometry.

        Args:
            grid_shape: (nz, ny, nx) voxel grid dimensions.
            bounds_cm: (x_min, x_max, y_min, y_max, z_min, z_max) physical bounds in cm.
            device: Target PyTorch device.
        """
        self.device = device
        self.grid_shape = grid_shape
        self.bounds_cm = bounds_cm
        nz, ny, nx = grid_shape
        x_min, x_max, y_min, y_max, z_min, z_max = bounds_cm

        zb = np.linspace(z_min, z_max, nz)
        yb = np.linspace(y_min, y_max, ny)
        xb = np.linspace(x_min, x_max, nx)
        zg, yg, xg = np.meshgrid(zb, yb, xb, indexing="ij")
        coords_cm = np.stack([xg, yg, zg], axis=-1).reshape(-1, 3).astype(np.float32)
        coords_norm = coords_cm.copy()
        coords_norm[:, 0] /= max(abs(x_min), abs(x_max))
        coords_norm[:, 1] /= max(abs(y_min), abs(y_max))
        coords_norm[:, 2] /= max(abs(z_min), abs(z_max))

        self.coords_cm_t = torch.from_numpy(coords_cm).to(device)
        self.coords_norm_t = torch.from_numpy(coords_norm).to(device)

    def load_patient_npz(self, npz_path: Path) -> Dict:
        """Load a patient .npz file into GPU/CPU tensors and normalized dose arrays.

        Args:
            npz_path: Path to the patient .npz dataset file.
        Returns:
            Dictionary containing density, normalized dose, angles, body mask, and raw arrays.
        """
        data = np.load(npz_path)
        rho = data["phantom_density"].astype(np.float32)
        dose = data["dose"].astype(np.float32) / GLOBAL_DOSE_SCALE
        thetas = data["theta"].astype(np.float32)

        rho_flat = rho.reshape(-1)
        mask = rho_flat > 0.005

        return {
            "name": npz_path.parent.name if npz_path.parent.name.startswith("pt_") else npz_path.stem,
            "rho_grid": torch.from_numpy(rho).to(self.device),
            "rho_flat": torch.from_numpy(rho_flat).unsqueeze(-1).to(self.device),
            "dose_flat": torch.from_numpy(dose.reshape(len(thetas), -1)).to(self.device),
            "thetas": torch.from_numpy(thetas).to(self.device),
            "mask": torch.from_numpy(mask).to(self.device),
            "n_angles": len(thetas),
            "raw_dose": data["dose"].astype(np.float32),
            "raw_rho": rho,
        }

    def precompute_patient_features(self, d: Dict, in_node_features: int = 8, chunk_size: int = 4) -> None:
        """Pre-computes and caches ray-tracing features (WEPL + beam geometry) in Host RAM
        using small angle chunks (chunk_size=4, < 950 MB VRAM) so peak GPU memory stays low,
        eliminating 100% of repeated ray-tracing overhead during training without CUDA OOM.
        """
        all_feats = []
        thetas = d["thetas"]
        n_angles = len(thetas)

        with torch.no_grad():
            for i in range(0, n_angles, chunk_size):
                sub_thetas = thetas[i : i + chunk_size]
                f_chunk = construct_batch_features(
                    rho=d["rho_flat"],
                    rho_grid=d["rho_grid"],
                    coords_norm=self.coords_norm_t,
                    coords_cm=self.coords_cm_t,
                    theta_deg=sub_thetas,
                    bounds_cm=self.bounds_cm,
                    in_node_features=in_node_features,
                )
                all_feats.append(f_chunk.cpu())
                del f_chunk

            feats = torch.cat(all_feats, dim=0)
            # Store in standard OS Host RAM (avoids cudaHostAlloc / WDDM pinned-memory exhaustion)
            d["cached_x_feats"] = feats.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def find_dataset_dir(base_dir: Path) -> Path:
    """Locate the per-angle patient .npz dataset directory from standard candidate paths.

    Args:
        base_dir: Project root directory to search from.
    Returns:
        Path to the discovered dataset directory.
    """
    candidates = [
        base_dir / "resources" / "dataset" / "outputs_random_angel",
        base_dir / "resources" / "dataset" / "outputs_random_angle",
        base_dir / "resources" / "dataset" / "openkbp_8x_random",
    ]
    for c in candidates:
        if c.exists() and len(list(c.glob("**/*.npz"))) > 0:
            return c
    found = list(base_dir.glob("**/outputs_random_ange*"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Could not find patient dataset directory in {base_dir}")


def collect_patient_files(dataset_dir: Path) -> Dict[str, Path]:
    """Scan dataset_dir recursively for .npz files and map patient IDs to file paths.

    Args:
        dataset_dir: Root directory containing patient .npz files.
    Returns:
        Dictionary mapping patient ID strings (e.g., 'pt_210') to .npz Path objects.
    """
    npz_files = sorted(list(dataset_dir.glob("**/*.npz")))
    patient_map = {}
    for f in npz_files:
        pt_id = f.parent.name if f.parent.name.startswith("pt_") else f.stem.split("_")[0] + "_" + f.stem.split("_")[1]
        patient_map[pt_id] = f
    return patient_map


# ==============================================================================
# 4. Training and Evaluation Loops
# ==============================================================================
def train_epoch(
    model: nn.Module,
    datasets: List[Dict],
    cache: SupercomputerPatientCache,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    batch_size: int = 4,
    accum_steps: int = 2,
    w_air: float = 1.0,
) -> float:
    """Run one training epoch with gradient accumulation across all patients and shuffled beam angles.

    Args:
        model: HierarchicalGridGNN model being trained.
        datasets: List of patient dataset dicts.
        cache: Patient feature/coordinate cache.
        optimizer: PyTorch optimizer.
        scaler: Optional AMP GradScaler for mixed precision.
        batch_size: Number of beam angles per forward pass.
        accum_steps: Number of micro-batches per optimizer step.
        w_air: Weight for ambient air suppression loss.
    Returns:
        Mean training loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    step_count = 0

    np.random.shuffle(datasets)
    optimizer.zero_grad(set_to_none=True)

    for d in datasets:
        n_angles = d["n_angles"]
        perm = np.random.permutation(n_angles)

        for b_start in range(0, n_angles, batch_size):
            b_idx = perm[b_start : b_start + batch_size]
            b_thetas = d["thetas"][b_idx]
            b_target = d["dose_flat"][b_idx]

            if "cached_x_feats" in d:
                x_feats = d["cached_x_feats"][b_idx].to(cache.device, non_blocking=True)
            else:
                x_feats = construct_batch_features(
                    rho=d["rho_flat"],
                    rho_grid=d["rho_grid"],
                    coords_norm=cache.coords_norm_t,
                    coords_cm=cache.coords_cm_t,
                    theta_deg=b_thetas,
                    bounds_cm=cache.bounds_cm,
                    in_node_features=8,
                )

            if scaler is not None and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    pred = model(x_feats, b_thetas)
                    loss = compute_clinical_loss(
                        pred,
                        b_target,
                        d["rho_flat"].t().expand(len(b_idx), -1),
                        d["mask"],
                        cache.grid_shape,
                        w_air=w_air,
                    )
                    loss_scaled = loss / accum_steps
                scaler.scale(loss_scaled).backward()
                step_count += 1
                if step_count % accum_steps == 0 or (b_start + batch_size >= n_angles):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                pred = model(x_feats, b_thetas)
                loss = compute_clinical_loss(
                    pred,
                    b_target,
                    d["rho_flat"].t().expand(len(b_idx), -1),
                    d["mask"],
                    cache.grid_shape,
                    w_air=w_air,
                )
                loss_scaled = loss / accum_steps
                loss_scaled.backward()
                step_count += 1
                if step_count % accum_steps == 0 or (b_start + batch_size >= n_angles):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(1, total_batches)


def compute_validation_loss(
    model: nn.Module,
    val_datasets: List[Dict],
    cache: SupercomputerPatientCache,
    batch_size: int = 4,
    w_air: float = 1.0,
) -> float:
    """
    Computes the exact same composite radiotherapy loss on the validation cohort
    (in eval mode without gradients) for direct, identical-scale comparison with training loss.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for d in val_datasets:
            n_angles = d["n_angles"]
            for b_start in range(0, n_angles, batch_size):
                b_idx = slice(b_start, min(b_start + batch_size, n_angles))
                b_thetas = d["thetas"][b_idx]
                b_target = d["dose_flat"][b_idx]

                if "cached_x_feats" in d:
                    x_feats = d["cached_x_feats"][b_idx].to(cache.device, non_blocking=True)
                else:
                    x_feats = construct_batch_features(
                        rho=d["rho_flat"],
                        rho_grid=d["rho_grid"],
                        coords_norm=cache.coords_norm_t,
                        coords_cm=cache.coords_cm_t,
                        theta_deg=b_thetas,
                        bounds_cm=cache.bounds_cm,
                        in_node_features=8,
                    )

                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        pred = model(x_feats, b_thetas)
                        loss = compute_clinical_loss(
                            pred,
                            b_target,
                            d["rho_flat"].t().expand(len(b_thetas), -1),
                            d["mask"],
                            cache.grid_shape,
                            w_air=w_air,
                        )
                else:
                    pred = model(x_feats, b_thetas)
                    loss = compute_clinical_loss(
                        pred,
                        b_target,
                        d["rho_flat"].t().expand(len(b_thetas), -1),
                        d["mask"],
                        cache.grid_shape,
                        w_air=w_air,
                    )

                total_loss += float(loss.item())
                total_batches += 1

    return total_loss / max(1, total_batches)


def evaluate_patients(
    model: nn.Module,
    patients: List[Dict],
    cache: SupercomputerPatientCache,
    batch_size: int = 8,
) -> List[Dict]:
    """Evaluate the model on a cohort of patients and compute Gamma, MAE, and in-field error.

    Args:
        model: Trained HierarchicalGridGNN model.
        patients: List of patient dataset dicts.
        cache: Patient feature/coordinate cache.
        batch_size: Number of beam angles per inference batch.
    Returns:
        List of per-patient metric dicts (gamma_3_3, gamma_2_2, mae_1e12_gy, air_mae_1e12_gy, in_field_err_pct).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for d in patients:
            n_angles = d["n_angles"]
            preds = []
            for b_start in range(0, n_angles, batch_size):
                b_thetas = d["thetas"][b_start : b_start + batch_size]
                if "cached_x_feats" in d:
                    x_feats = d["cached_x_feats"][b_start : b_start + batch_size].to(cache.device, non_blocking=True)
                else:
                    x_feats = construct_batch_features(
                        rho=d["rho_flat"],
                        rho_grid=d["rho_grid"],
                        coords_norm=cache.coords_norm_t,
                        coords_cm=cache.coords_cm_t,
                        theta_deg=b_thetas,
                        bounds_cm=cache.bounds_cm,
                        in_node_features=8,
                    )
                p = model(x_feats, b_thetas)
                preds.append(p)

            pred_all = torch.cat(preds, dim=0).cpu().numpy() * GLOBAL_DOSE_SCALE
            target_all = d["raw_dose"].reshape(n_angles, -1)
            mask = d["mask"].cpu().numpy()

            diff = np.abs(pred_all[:, mask] - target_all[:, mask])
            mae = float(diff.mean() * 1e12)

            air_mask = ~mask
            air_mae = float(np.abs(pred_all[:, air_mask]).mean() * 1e12) if np.any(air_mask) else 0.0

            in_f = target_all[:, mask] >= 0.05 * target_all.max()
            rel_err = float((diff[in_f] / target_all[:, mask][in_f]).mean() * 100.0) if in_f.any() else 0.0

            mid_idx = n_angles // 2
            p3d = pred_all[mid_idx].reshape(cache.grid_shape)
            t3d = d["raw_dose"][mid_idx]
            m3d = mask.reshape(cache.grid_shape)

            gamma3 = compute_gamma_index_3d(p3d, t3d, m3d, (1.0, 1.0, 1.0), 3.0, 3.0)
            gamma2 = compute_gamma_index_3d(p3d, t3d, m3d, (1.0, 1.0, 1.0), 2.0, 2.0)

            results.append({
                "patient": d["name"],
                "gamma_3_3": round(gamma3, 2),
                "gamma_2_2": round(gamma2, 2),
                "mae_1e12_gy": round(mae, 4),
                "air_mae_1e12_gy": round(air_mae, 4),
                "in_field_err_pct": round(rel_err, 2),
            })

    return results


def plot_loss_vs_val_mae(
    history: Dict[str, list],
    out_png_path: Path,
    model_name: str = "Hierarchical Multi-Scale GNN",
    best_epoch: int | None = None,
    best_val_mae: float | None = None,
) -> None:
    """Renders a publication-quality training loss vs validation MAE convergence curve

    matching the master visualization visual styling (pure white background, clean
    typography,
    dual-axis formatting with deep blue / coral red).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    ep_train = history.get("train_epochs", [])
    loss_train = history.get("train_loss", [])
    ep_val = history.get("val_epochs", [])
    mae_val = history.get("val_mae", [])

    # Left Y-axis: Training Loss (Composite)
    color_loss = "#0984e3"  # Deep clean blue
    line1 = ax1.plot(
        ep_train,
        loss_train,
        color=color_loss,
        linewidth=2.2,
        label="Training Loss (Composite)",
    )
    ax1.set_xlabel("Training Epoch", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel(
        "Training Loss (Composite)",
        fontsize=11,
        fontweight="bold",
        color=color_loss,
        labelpad=8,
    )
    ax1.tick_params(axis="y", labelcolor=color_loss, labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1")

    # Right Y-axis: Validation MAE
    ax2 = ax1.twinx()
    ax2.set_facecolor("white")
    color_mae = "#d63031"  # Crisp coral crimson
    line2 = ax2.plot(
        ep_val,
        mae_val,
        color=color_mae,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label=r"Validation MAE ($\times 10^{-12}$ Gy)",
    )
    ax2.set_ylabel(
        r"Validation Dose MAE ($\times 10^{-12}\ \mathrm{Gy}$)",
        fontsize=11,
        fontweight="bold",
        color=color_mae,
        labelpad=8,
    )
    ax2.tick_params(axis="y", labelcolor=color_mae, labelsize=10)

    # Best Epoch Callout
    if best_epoch is not None and best_val_mae is not None:
        ax2.plot(
            best_epoch,
            best_val_mae,
            marker="*",
            markersize=14,
            color="#fdcb6e",
            markeredgecolor="#d63031",
            markeredgewidth=1.5,
            zorder=10,
        )
        ax2.annotate(
            f"Best: {best_val_mae:.4f} (Ep {best_epoch})",
            xy=(best_epoch, best_val_mae),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="#2d3436",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#fff9e6",
                edgecolor="#fdcb6e",
                alpha=0.9,
            ),
            arrowprops=dict(arrowstyle="->", color="#d63031", lw=1.2),
        )

    # Combined Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper right",
        framealpha=0.95,
        facecolor="white",
        edgecolor="#dfe6e9",
        fontsize=9.5,
    )

    plt.title(
        f"{model_name}: Convergence (Training Loss vs Validation MAE)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved convergence curve: {out_png_path.name}")


# ==============================================================================
# 5. Main Execution Function
# ==============================================================================
def main():
    """Run the full Hierarchical GNN training, validation, and held-out test evaluation pipeline.

    Args:
        None (parses command-line arguments via argparse).
    Returns:
        None. Saves the best checkpoint, training history JSON, convergence curve, and test results JSON.
    """
    parser = argparse.ArgumentParser(description="Train Hierarchical GNN on Supercomputer OpenKBP Data")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per forward step (default: 4, safe for 12GB VRAM)")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps (default: 2, effective batch size 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--val_interval", type=int, default=4, help="Epoch interval between validation checks (default: 4)")
    parser.add_argument("--hidden_channels", type=int, default=32, help="GNN channels (default: 32)")
    parser.add_argument("--w_air", type=float, default=1.0, help="Weight for air suppression loss (default: 1.0)")
    parser.add_argument("--out_dir", type=str, default="outputs/gnn_hierarchical", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = find_dataset_dir(PROJECT_ROOT)
    patient_files = collect_patient_files(dataset_dir)
    print("=" * 85)
    print("HIERARCHICAL MULTI-SCALE GNN SUPERCOMPUTER TRAINING PIPELINE")
    print(f"Device: {device} | Discovered Dataset Directory: {dataset_dir.name}")
    print(f"Total Patient Folders Found: {len(patient_files)}")
    print("=" * 85, flush=True)

    sample_pt_id, sample_file = next(iter(patient_files.items()))
    sample_data = np.load(sample_file)
    grid_shape = sample_data["phantom_density"].shape
    xb = sample_data.get("x_bounds_cm", np.array([-12.8, 12.8]))
    yb = sample_data.get("y_bounds_cm", np.array([-7.2, 7.2]))
    zb = sample_data.get("z_bounds_cm", np.array([-3.2, 3.2]))
    bounds_cm = (float(xb[0]), float(xb[-1]), float(yb[0]), float(yb[-1]), float(zb[0]), float(zb[-1]))

    print(f"Patient Grid Shape: {grid_shape} ({np.prod(grid_shape):,} voxels)")
    print(f"Physical Volume Bounds: X=[{bounds_cm[0]}, {bounds_cm[1]}] Y=[{bounds_cm[2]}, {bounds_cm[3]}] Z=[{bounds_cm[4]}, {bounds_cm[5]}] cm\n")

    cache = SupercomputerPatientCache(grid_shape, bounds_cm, device)

    all_pt_ids = sorted(list(patient_files.keys()))
    test_ids = [p for p in all_pt_ids if p in DEFAULT_TEST_PATIENTS]
    val_ids = [p for p in all_pt_ids if p in DEFAULT_VAL_PATIENTS]
    train_ids = [p for p in all_pt_ids if p not in DEFAULT_TEST_PATIENTS and p not in DEFAULT_VAL_PATIENTS]

    print(f"Patient Split Allocation:")
    print(f"  Training Patients:   {len(train_ids)} ({train_ids[:5]} ... {train_ids[-3:]})")
    print(f"  Validation Patients: {len(val_ids)} ({val_ids})")
    print(f"  Test Patients:       {len(test_ids)} ({test_ids})")

    print(f"\nLoading Training Datasets into RAM ({len(train_ids)} patients)...", flush=True)
    train_datasets = [cache.load_patient_npz(patient_files[p]) for p in train_ids]
    val_datasets = [cache.load_patient_npz(patient_files[p]) for p in val_ids]
    test_datasets = [cache.load_patient_npz(patient_files[p]) for p in test_ids]

    total_train_beams = sum(d["n_angles"] for d in train_datasets)
    print(f"Loaded {total_train_beams} total training beam simulations.")

    # Pre-cache beam ray-tracing features (WEPL + beam geometry) into Host RAM
    print("\nPre-caching 3D ray-tracing features in Host RAM to eliminate repeated runtime overhead...", flush=True)
    t_pre = time.time()
    for d in train_datasets + val_datasets:
        cache.precompute_patient_features(d, in_node_features=8)
    total_cached = total_train_beams + sum(d["n_angles"] for d in val_datasets)
    print(f"  [✓] Pre-cached {total_cached} total beam projections across RAM in {time.time()-t_pre:.1f}s.")

    model = HierarchicalGridGNN(
        grid_shape=grid_shape,
        in_node_features=8,
        hidden_channels=args.hidden_channels,
        num_harmonic_bands=4,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_val_mae = float("inf")
    best_epoch = None
    best_ckpt_path = out_dir / "gnn_hierarchical_best.pt"

    history = {
        "train_epochs": [],
        "train_loss": [],
        "val_epochs": [],
        "val_loss": [],
        "val_mae": [],
        "val_gamma": [],
    }

    print("\n" + "=" * 85)
    print("STARTING HIERARCHICAL GNN TRAINING LOOP")
    print("=" * 85, flush=True)

    t0 = time.time()
    interval_t0 = time.time()
    last_val_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_datasets, cache, optimizer, scaler, batch_size=args.batch_size, accum_steps=args.accum_steps, w_air=args.w_air)
        scheduler.step()

        # Compute validation loss every epoch using the exact same clinical loss function
        val_loss = compute_validation_loss(model, val_datasets, cache, batch_size=args.batch_size, w_air=args.w_air)

        history["train_epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Report progress on Epoch 1 and every val_interval epochs thereafter
        if epoch % args.val_interval == 0 or epoch == args.epochs or epoch == 1:
            val_results = evaluate_patients(model, val_datasets, cache, args.batch_size)
            mean_val_mae = float(np.mean([r["mae_1e12_gy"] for r in val_results]))
            mean_val_gamma = float(np.mean([r["gamma_3_3"] for r in val_results]))

            history["val_epochs"].append(epoch)
            history["val_mae"].append(mean_val_mae)
            history["val_gamma"].append(mean_val_gamma)

            is_best = mean_val_mae < best_val_mae
            if is_best:
                best_val_mae = mean_val_mae
                best_epoch = epoch
                torch.save(
                    {
                        "model_type": "hierarchical_gnn",
                        "state_dict": model.state_dict(),
                        "grid_shape": grid_shape,
                        "dose_scale": GLOBAL_DOSE_SCALE,
                        "epoch": epoch,
                        "val_mae": best_val_mae,
                        "val_loss": val_loss,
                        "split": {"train": train_ids, "val": val_ids, "test": test_ids},
                    },
                    best_ckpt_path,
                )

            mark = "(*BEST*)" if is_best else ""
            interval_elapsed = time.time() - interval_t0
            n_ep = epoch - last_val_epoch
            avg_ep_s = interval_elapsed / max(1, n_ep)
            print(
                f"[Epoch {epoch:02d}/{args.epochs}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"Val Gamma 3%/3mm: {mean_val_gamma:.1f}% | Val MAE: {mean_val_mae:.4f} {mark} "
                f"({interval_elapsed:.1f}s across {n_ep} ep, ~{avg_ep_s:.1f}s/ep)",
                flush=True,
            )
            interval_t0 = time.time()
            last_val_epoch = epoch

    print(f"\n[TRAINING FINISHED] Total time: {(time.time()-t0)/60.0:.2f} min. Best model saved to: {best_ckpt_path.name}")

    # Export training metrics history to JSON for external visualization and archiving
    history_json = out_dir / "training_history.json"
    with open(history_json, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "hierarchical_gnn",
            "history": history,
            "loss": {
                "epoch": history["train_epochs"],
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
            },
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
        }, f, indent=2)
    print(f"  [✓] Exported training metrics history to: {history_json.name}")

    # Also save loss_history.json matching total dose format
    loss_json = out_dir / "loss_history.json"
    with open(loss_json, "w", encoding="utf-8") as f:
        json.dump({
            "loss": {
                "epoch": history["train_epochs"],
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
            },
            "val_metrics": {
                "epoch": history["val_epochs"],
                "val_mae": history["val_mae"],
                "val_gamma": history["val_gamma"],
            },
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
        }, f, indent=2)
    print(f"  [✓] Exported standardized loss history to: {loss_json.name}")

    # Generate and save publication-quality Training Loss vs Validation MAE curve
    curve_png_path = out_dir / "gnn_training_loss_vs_val_mae.png"
    plot_loss_vs_val_mae(
        history=history,
        out_png_path=curve_png_path,
        model_name="Hierarchical Multi-Scale GNN",
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
    )

    print("\n" + "=" * 95)
    print(f"FINAL CLINICAL EVALUATION ON {len(test_datasets)} UNSEEN TEST PATIENTS")
    print("=" * 95, flush=True)

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_results = evaluate_patients(model, test_datasets, cache, args.batch_size)

    print(f"{'Patient':<12} | {'Gamma 3%/3mm':<14} | {'Gamma 2%/2mm':<14} | {'In-Field Err (%)':<18} | {'Tissue MAE':<14} | {'Air MAE':<12}")
    print("-" * 95)
    for r in test_results:
        print(f"{r['patient']:<12} | {r['gamma_3_3']:>12.1f}% | {r['gamma_2_2']:>12.1f}% | {r['in_field_err_pct']:>16.2f}% | {r['mae_1e12_gy']:>12.4f} | {r['air_mae_1e12_gy']:>10.4f}")
    print("=" * 95)

    mean_test_gamma = float(np.mean([r["gamma_3_3"] for r in test_results]))
    mean_test_mae = float(np.mean([r["mae_1e12_gy"] for r in test_results]))
    mean_test_air = float(np.mean([r["air_mae_1e12_gy"] for r in test_results]))
    print(f"Overall Test Cohort Mean: Gamma 3%/3mm = {mean_test_gamma:.2f}% | Tissue MAE = {mean_test_mae:.4f} | Air MAE = {mean_test_air:.4f} x10^-12 Gy")

    results_json = out_dir / "test_evaluation_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"Detailed results saved to: {results_json}")


if __name__ == "__main__":
    main()
