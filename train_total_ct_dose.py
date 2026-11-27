"""
train_total_ct_dose.py
======================
Trains the TotalCTDoseUNet to predict full-scan total 3D CT dose
directly from patient CT density (rho) and scanner radial geometry (r).

Inputs:
  Channel 0: Continuous CT tissue density rho(x, y, z) [g/cm^3]
  Channel 1: Radial distance from scanner isocenter r = sqrt(x^2 + y^2) [normalized]

Target:
  Patient internal total cumulative 3D CT scan dose (zeroed outside body)
  Normalized by GLOBAL_DOSE_SCALE = 1.0e-12 Gy

Loss:
  Masked L1 + Relative Percentage Error (body only) + Air penalty

Metrics (No Gamma!):
  - Voxels within +-10% Dmax (%)
  - Voxels within +-5% Dmax (%)
  - Tissue Mean Absolute Error (x10^-12 Gy)
  - Mean Relative Percentage Error (%)
  - Tissue-specific MAE: Bone (rho >= 1.2), Soft Tissue (0.8 <= rho < 1.2), Low Density (rho < 0.8)
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
torch.backends.cudnn.benchmark = True
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.total_ct_unet import TotalCTDoseUNet

GLOBAL_DOSE_SCALE = 1.0e-12  # 1.0 x 10^-12 Gy (normalizes patient dose to ~0.0 - 1.5)

DEFAULT_TRAIN_PATIENTS = [
    "pt_202", "pt_204", "pt_205", "pt_206", "pt_207", "pt_208", "pt_211", "pt_212",
    "pt_213", "pt_214", "pt_215", "pt_217", "pt_220", "pt_221", "pt_226", "pt_227",
    "pt_228", "pt_229", "pt_231", "pt_232", "pt_235", "pt_236", "pt_237", "pt_238",
    "pt_240", "pt_241", "pt_242", "pt_243", "pt_244", "pt_245", "pt_247", "pt_248",
    "pt_250", "pt_256", "pt_257", "pt_258"
]

DEFAULT_VAL_PATIENTS = [
    "pt_201", "pt_216", "pt_225", "pt_239", "pt_246", "pt_251", "pt_252", "pt_255"
]

DEFAULT_TEST_PATIENTS = [
    "pt_210", "pt_222", "pt_223", "pt_230", "pt_233", "pt_234", "pt_253", "pt_259", "pt_260", "pt_261", "pt_262", "pt_263"
]


def read_egsphant(filepath: str | Path) -> dict:
    """Parses standard EGSnrc .egsphant file."""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        num_media_line = f.readline().strip()
        while not num_media_line:
            num_media_line = f.readline().strip()
        num_media = int(num_media_line)
        media_names = [f.readline().strip() for _ in range(num_media)]
        f.readline()  # ESTEPE
        dims = f.readline().strip().split()
        nx, ny, nz = map(int, dims)
        x_bounds = np.fromstring(f.readline(), sep=" ")
        y_bounds = np.fromstring(f.readline(), sep=" ")
        z_bounds = np.fromstring(f.readline(), sep=" ")
        media = np.zeros((nz, ny, nx), dtype=np.int32)
        for k in range(nz):
            for j in range(ny):
                line = f.readline().strip()
                while not line:
                    line = f.readline().strip()
                media[k, j, :] = [int(c) for c in line]
        density = np.zeros((nz, ny, nx), dtype=np.float32)
        for k in range(nz):
            for j in range(ny):
                line = f.readline().strip()
                while not line:
                    line = f.readline().strip()
                density[k, j, :] = np.fromstring(line, sep=" ")
    return {"media_names": media_names, "nx": nx, "ny": ny, "nz": nz, "media": media, "density": density}


def build_radial_distance_grid(grid_shape: tuple[int, int, int] = (32, 72, 128)) -> np.ndarray:
    """
    Builds normalized radial distance grid r = sqrt(x^2 + y^2) / r_max from scanner isocenter.
    Shape: [D, H, W] = [32, 72, 128]
    """
    nz, ny, nx = grid_shape
    # Physical bounds in cm: X [-12.8, 12.8], Y [-7.2, 7.2]
    xs = np.linspace(-12.8, 12.8, nx, dtype=np.float32)
    ys = np.linspace(-7.2, 7.2, ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)  # [ny, nx]
    r_2d = np.sqrt(xx**2 + yy**2)
    r_norm = (r_2d / r_2d.max()).astype(np.float32)  # [ny, nx] in [0, 1]
    # Broadcast across all Z planes
    return np.repeat(r_norm[np.newaxis, :, :], nz, axis=0)  # [32, 72, 128]


def load_patient_data(dataset_dir: Path, pt_id: str, r_grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Loads egsphant density, computes 2-channel input, and loads masked patient dose."""
    egs_path = dataset_dir / f"{pt_id}.egsphant"
    npz_path = dataset_dir / f"{pt_id}_36view.npz"

    if not egs_path.exists():
        raise FileNotFoundError(f"Missing phantom: {egs_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz: {npz_path}")

    ph = read_egsphant(egs_path)
    npz_data = np.load(npz_path)

    rho = ph["density"].astype(np.float32)          # [32, 72, 128]
    mask = npz_data["patient_mask"].astype(bool)     # [32, 72, 128]
    raw_dose = npz_data["dose"].astype(np.float32)  # [32, 72, 128]

    # Mask dose to patient body (zero out unattenuated primary air dose)
    dose = (raw_dose * mask).astype(np.float32)

    # 3-Channel Input:
    # Channel 0: Continuous Density rho (0.0 - 1.95 g/cm^3)
    # Channel 1: Radial distance r from scanner center (0.0 - 1.0)
    # Channel 2: High-contrast Bone Feature Accent (ReLU(rho - 1.1) / 0.85)
    bone_accent = np.clip((rho - 1.1) / (1.95 - 1.1), 0.0, 1.0).astype(np.float32)
    input_feats = np.stack([rho, r_grid, bone_accent], axis=0).astype(np.float32)  # [3, 32, 72, 128]

    return {
        "pt_id": pt_id,
        "input": input_feats,
        "rho": rho,
        "dose": dose,
        "mask": mask,
    }


def evaluate_cohort(
    model: nn.Module,
    patients_data: List[Dict[str, np.ndarray]],
    device: torch.device,
) -> List[Dict]:
    """Evaluates test/validation cohort on clinical dosimetric metrics (No Gamma!)."""
    model.eval()
    results = []

    with torch.no_grad():
        for item in patients_data:
            pt_id = item["pt_id"]
            inp = torch.from_numpy(item["input"]).unsqueeze(0).to(device)  # [1, 2, 32, 72, 128]
            true_dose = item["dose"]
            mask = item["mask"]
            rho = item["rho"]

            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                pred_scaled = model(inp).squeeze(0).squeeze(0).cpu().numpy()

            # Zero out any air predictions outside patient
            pred_dose = (pred_scaled * GLOBAL_DOSE_SCALE * mask).astype(np.float32)

            d_max = float(true_dose[mask].max()) if np.any(mask) else 1.0e-12

            # Differences inside patient body
            diff_abs = np.abs(pred_dose[mask] - true_dose[mask])
            diff_pct_max = (diff_abs / (d_max + 1e-12)) * 100.0

            # 1. Voxels within +-10% and +-5% Dmax
            within_10 = float((diff_pct_max <= 10.0).mean() * 100.0)
            within_5 = float((diff_pct_max <= 5.0).mean() * 100.0)

            # 2. Overall Body MAE (x10^-12 Gy)
            tissue_mae = float(np.mean(diff_abs) / GLOBAL_DOSE_SCALE)

            # 3. Mean Relative Percentage Error in body
            rel_err_pct = float(np.mean(diff_abs / (true_dose[mask] + 0.05 * GLOBAL_DOSE_SCALE)) * 100.0)

            # 4. Tissue-specific MAE breakdown
            bone_mask = mask & (rho >= 1.2)
            soft_mask = mask & (rho >= 0.8) & (rho < 1.2)
            air_cavity_mask = mask & (rho < 0.8)

            bone_mae = float(np.mean(np.abs(pred_dose[bone_mask] - true_dose[bone_mask])) / GLOBAL_DOSE_SCALE) if np.any(bone_mask) else 0.0
            soft_mae = float(np.mean(np.abs(pred_dose[soft_mask] - true_dose[soft_mask])) / GLOBAL_DOSE_SCALE) if np.any(soft_mask) else 0.0
            cavity_mae = float(np.mean(np.abs(pred_dose[air_cavity_mask] - true_dose[air_cavity_mask])) / GLOBAL_DOSE_SCALE) if np.any(air_cavity_mask) else 0.0

            results.append({
                "patient": pt_id,
                "within_10_pct": round(within_10, 2),
                "within_5_pct": round(within_5, 2),
                "tissue_mae_1e12_gy": round(tissue_mae, 4),
                "rel_err_pct": round(rel_err_pct, 2),
                "bone_mae_1e12_gy": round(bone_mae, 4),
                "soft_tissue_mae_1e12_gy": round(soft_mae, 4),
                "air_cavity_mae_1e12_gy": round(cavity_mae, 4),
            })

    return results


def compute_loss(
    pred: torch.Tensor,
    b_dose: torch.Tensor,
    b_mask: torch.Tensor,
    b_rho: torch.Tensor,
    w_bone: float,
    w_rel: float,
    w_air: float,
) -> torch.Tensor:
    """Bone-Weighted Masked L1 + Relative Loss + Air Penalty (shared by train and val)."""
    diff = (pred - b_dose).abs()

    # 1. Bone-Weighted Absolute L1 inside body (w_bone x on dense bone)
    bone_weight = torch.ones_like(b_dose)
    bone_weight[b_rho >= 1.15] = w_bone
    weighted_diff = diff * bone_weight
    loss_tissue = (weighted_diff * b_mask).sum() / ((bone_weight * b_mask).sum() + 1e-6)

    # 2. Relative percentage error inside body (protects deep low-dose organs)
    n_tissue = b_mask.sum() + 1e-6
    loss_rel = ((diff / (b_dose + 0.05)) * b_mask).sum() / n_tissue

    # 3. Air penalty outside body
    n_air = (~b_mask).sum() + 1e-6
    loss_air = (pred.abs() * (~b_mask)).sum() / n_air

    return loss_tissue + w_rel * loss_rel + w_air * loss_air


def validation_loss(
    model: nn.Module,
    val_data: List[Dict[str, np.ndarray]],
    batch_size: int,
    w_bone: float,
    w_rel: float,
    w_air: float,
    device: torch.device,
) -> float:
    """Same loss as training, computed on the validation cohort (eval mode, no gradients)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(val_data), batch_size):
            chunk = val_data[start : start + batch_size]
            b_input = torch.stack([torch.from_numpy(d["input"]) for d in chunk]).to(device)
            b_dose = torch.stack([torch.from_numpy(d["dose"] / GLOBAL_DOSE_SCALE).unsqueeze(0) for d in chunk]).to(device)
            b_mask = torch.stack([torch.from_numpy(d["mask"]).unsqueeze(0) for d in chunk]).to(device)
            b_rho = b_input[:, 0:1, :, :, :]
            with torch.amp.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
                pred = model(b_input)
                loss = compute_loss(pred, b_dose, b_mask, b_rho, w_bone, w_rel, w_air)
            total += float(loss.item()) * len(chunk)
            n += len(chunk)
    return total / max(1, n)


def plot_loss_vs_val_mae(
    history: Dict[str, list],
    metric_history: Dict[str, list],
    out_png_path: Path,
    model_name: str = "Total Dose 3D U-Net",
    best_epoch: int | None = None,
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

    ep_train = history.get("epoch", [])
    loss_train = history.get("train_loss", [])
    ep_val = metric_history.get("epoch", [])
    mae_val = metric_history.get("val_mae", [])

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
    if best_epoch is not None and best_epoch in ep_val:
        idx = ep_val.index(best_epoch)
        best_val_mae = mae_val[idx]
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


def save_loss_curves_pdf(history: Dict[str, list], metric_history: Dict[str, list], pdf_path: Path, best_epoch: int | None = None) -> None:
    """Page 1: train vs val loss (linear + log). Page 2: validation metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    ep = history["epoch"]
    with PdfPages(pdf_path) as pdf:
        # ---- Page 1: loss curves ----
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, log in zip(axes, (False, True)):
            ax.plot(ep, history["train_loss"], color="#1f77b4", lw=2, label="Train loss")
            ax.plot(ep, history["val_loss"], color="#d62728", lw=2, ls="--", label="Validation loss")
            if best_epoch is not None:
                ax.axvline(best_epoch, color="gray", ls=":", label=f"Best checkpoint (epoch {best_epoch})")
            if log:
                ax.set_yscale("log")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss" + (" (log scale)" if log else ""))
            ax.set_title("Loss (log scale)" if log else "Loss")
            ax.grid(alpha=0.3, which="both"); ax.legend()
        fig.suptitle("Total CT Dose 3D U-Net: training vs validation loss", fontsize=14)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- Page 2: validation metrics ----
        if metric_history["epoch"]:
            me = metric_history["epoch"]
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].plot(me, metric_history["val_mae"], color="#2ca02c", lw=2, marker="o", ms=3)
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation tissue MAE (x10$^{-12}$ Gy)")
            axes[0].set_title("Validation MAE"); axes[0].grid(alpha=0.3)
            axes[1].plot(me, metric_history["val_within_10"], lw=2, marker="o", ms=3, label="Within ±10% Dmax")
            axes[1].plot(me, metric_history["val_within_5"], lw=2, marker="o", ms=3, label="Within ±5% Dmax")
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Voxels (%)")
            axes[1].set_title("Validation dosimetric pass rates"); axes[1].grid(alpha=0.3); axes[1].legend()
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def train_epoch(
    model: nn.Module,
    train_data: List[Dict[str, np.ndarray]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    batch_size: int,
    w_bone: float,
    w_rel: float,
    w_air: float,
    device: torch.device,
) -> float:
    """Trains 1 epoch with Bone-Weighted Masked L1 + Relative Loss + Air Penalty."""
    model.train()
    indices = np.random.permutation(len(train_data))
    total_loss = 0.0
    num_batches = 0

    for start_idx in range(0, len(indices), batch_size):
        batch_ids = indices[start_idx : start_idx + batch_size]
        b_input = torch.stack([torch.from_numpy(train_data[i]["input"]) for i in batch_ids]).to(device)  # [B, 3, 32, 72, 128]
        b_dose = torch.stack([torch.from_numpy(train_data[i]["dose"] / GLOBAL_DOSE_SCALE).unsqueeze(0) for i in batch_ids]).to(device)
        b_mask = torch.stack([torch.from_numpy(train_data[i]["mask"]).unsqueeze(0) for i in batch_ids]).to(device)
        b_rho = b_input[:, 0:1, :, :, :]  # Density channel for bone identification

        optimizer.zero_grad()

        with torch.amp.autocast('cuda') if scaler is not None else torch.no_grad():
            pred = model(b_input)
            loss = compute_loss(pred, b_dose, b_mask, b_rho, w_bone, w_rel, w_air)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def main():
    """Run the full Total CT Dose 3D U-Net training, validation, and test evaluation pipeline.

    Args:
        None (parses command-line arguments via argparse).
    Returns:
        None. Saves the best model checkpoint, training history JSON, loss curves, and test evaluation JSON.
    """
    parser = argparse.ArgumentParser(description="Train 3-Channel Bone-Accented Total CT Dose 3D U-Net")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--val_interval", type=int, default=4, help="Validation interval (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (default: 3e-4)")
    parser.add_argument("--w_bone", type=float, default=4.0, help="Weight multiplier on bone error (default: 4.0)")
    parser.add_argument("--w_rel", type=float, default=0.2, help="Weight on relative error loss (default: 0.2)")
    parser.add_argument("--w_air", type=float, default=0.5, help="Weight on air leakage penalty (default: 0.5)")
    parser.add_argument("--base_channels", type=int, default=32, help="U-Net base channels (default: 32)")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="resources/dataset/full_ct_dose",
        help="Path to full_ct_dose directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/unet_total_dose",
        help="Output directory for checkpoints and metrics",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = BASE_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = BASE_DIR / args.dataset_dir
    if not dataset_path.exists():
        dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {args.dataset_dir}")

    print("=" * 95)
    print("TOTAL 3D CT DOSE U-NET TRAINING PIPELINE (3-CHANNEL BONE-ACCENTUATED)")
    print(f"Device: {device} | Dataset: {dataset_path.name}")
    print("Inputs: [Ch0: Density rho, Ch1: Radial Distance r, Ch2: High-Contrast Bone Accent]")
    print(f"Loss: Bone-Weighted L1 (w_bone={args.w_bone}x) + Relative Error + Air Penalty")
    print("Target: Total 36-View CT Scan Dose (Masked Patient Volume, 10^-12 Gy)")
    print("Metrics: Voxels within +-10% & +-5% Dmax | Tissue MAE | Bone & Soft Tissue Breakdown")
    print("=" * 95)

    # Precompute 3D radial distance grid
    grid_shape = (32, 72, 128)
    r_grid = build_radial_distance_grid(grid_shape)

    # Discover all patient files dynamically from dataset directory
    npz_files = list(dataset_path.glob("*_36view.npz"))
    discovered_pts = sorted([f.name.replace("_36view.npz", "") for f in npz_files])

    test_ids = [p for p in DEFAULT_TEST_PATIENTS if p in discovered_pts]
    val_ids = [p for p in DEFAULT_VAL_PATIENTS if p in discovered_pts]
    train_ids = [p for p in discovered_pts if p not in test_ids and p not in val_ids]

    print(f"\nDiscovered {len(discovered_pts)} Total Patients:")
    print(f"  Training Cohort:   {len(train_ids)} patients")
    print(f"  Validation Cohort: {len(val_ids)} patients")
    print(f"  Test Cohort:       {len(test_ids)} patients (Benchmark: {test_ids})")

    # Load all patients into memory
    print(f"\nLoading {len(discovered_pts)} Patient Volumes into RAM...")
    t_load = time.time()
    train_data = [load_patient_data(dataset_path, pid, r_grid) for pid in train_ids]
    val_data = [load_patient_data(dataset_path, pid, r_grid) for pid in val_ids]
    test_data = [load_patient_data(dataset_path, pid, r_grid) for pid in test_ids]
    print(f"Successfully loaded {len(train_data)} Train, {len(val_data)} Val, {len(test_data)} Test patients in {time.time()-t_load:.2f}s!")

    # Instantiate model with 3 input channels
    model = TotalCTDoseUNet(
        in_channels=3,
        base_channels=args.base_channels,
        grid_shape=grid_shape,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: TotalCTDoseUNet (3-in, 1-out) | Grid: {grid_shape} | Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_val_score = 0.0
    best_epoch = None
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    metric_history = {"epoch": [], "val_within_10": [], "val_within_5": [], "val_mae": []}
    best_ckpt_path = out_dir / "unet_total_dose_best.pt"

    print("\n" + "=" * 95)
    print("STARTING TRAINING LOOP")
    print("=" * 95, flush=True)

    t0 = time.time()
    interval_t0 = time.time()
    last_val_epoch = 0
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        train_loss = train_epoch(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            scaler=scaler,
            batch_size=args.batch_size,
            w_bone=args.w_bone,
            w_rel=args.w_rel,
            w_air=args.w_air,
            device=device,
        )
        scheduler.step()

        # Validation loss (same loss function as training), every epoch (cheap: 8 patients)
        val_loss = validation_loss(model, val_data, args.batch_size, args.w_bone, args.w_rel, args.w_air, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Validate every val_interval epochs and at final epoch
        if epoch % args.val_interval == 0 or epoch == args.epochs or epoch == 1:
            val_results = evaluate_cohort(model, val_data, device)
            mean_val_within_10 = float(np.mean([r["within_10_pct"] for r in val_results]))
            mean_val_within_5 = float(np.mean([r["within_5_pct"] for r in val_results]))
            mean_val_mae = float(np.mean([r["tissue_mae_1e12_gy"] for r in val_results]))
            mean_val_bone_mae = float(np.mean([r["bone_mae_1e12_gy"] for r in val_results]))

            metric_history["epoch"].append(epoch)
            metric_history["val_within_10"].append(mean_val_within_10)
            metric_history["val_within_5"].append(mean_val_within_5)
            metric_history["val_mae"].append(mean_val_mae)

            is_best = mean_val_within_10 > best_val_score
            if is_best:
                best_val_score = mean_val_within_10
                best_epoch = epoch
                torch.save(
                    {
                        "model_type": "total_ct_dose_unet_3channel",
                        "in_channels": 3,
                        "state_dict": model.state_dict(),
                        "grid_shape": grid_shape,
                        "dose_scale": GLOBAL_DOSE_SCALE,
                        "epoch": epoch,
                        "val_within_10": best_val_score,
                        "val_mae": mean_val_mae,
                        "split": {
                            "train": train_ids,
                            "val": val_ids,
                            "test": test_ids,
                        },
                    },
                    best_ckpt_path,
                )

            mark = "(*BEST*)" if is_best else ""
            interval_elapsed = time.time() - interval_t0
            n_ep = epoch - last_val_epoch
            avg_ep_s = interval_elapsed / max(1, n_ep)
            print(
                f"[Epoch {epoch:02d}/{args.epochs}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"Val Within +-10%: {mean_val_within_10:.1f}% | Within +-5%: {mean_val_within_5:.1f}% | "
                f"Val MAE: {mean_val_mae:.4f} x10^-12 Gy {mark} ({interval_elapsed:.1f}s across {n_ep} ep, ~{avg_ep_s:.1f}s/ep)",
                flush=True,
            )
            interval_t0 = time.time()
            last_val_epoch = epoch

    print(f"\n[TRAINING FINISHED] Total time: {(time.time()-t0)/60.0:.2f} min. Best model: {best_ckpt_path.name}")

    # Standardized JSON export for master visualization script
    best_mae_val = None
    if best_epoch is not None and best_epoch in metric_history["epoch"]:
        best_mae_val = metric_history["val_mae"][metric_history["epoch"].index(best_epoch)]

    training_history_data = {
        "model_type": "total_dose_unet",
        "history": {
            "train_epochs": history["epoch"],
            "train_loss": history["train_loss"],
            "val_epochs": metric_history["epoch"],
            "val_mae": metric_history["val_mae"],
            "val_within_10": metric_history["val_within_10"],
            "val_within_5": metric_history["val_within_5"],
        },
        "best_epoch": best_epoch,
        "best_val_mae": best_mae_val,
    }
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_history_data, f, indent=2)
    print("  [✓] Exported standardized training metrics history to: training_history.json")

    # Save loss history (re-plottable) and the loss-curve PDF
    with open(out_dir / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump({"loss": history, "val_metrics": metric_history, "best_epoch": best_epoch}, f, indent=2)
    try:
        png_path = out_dir / "total_dose_unet_loss_vs_val_mae.png"
        plot_loss_vs_val_mae(history, metric_history, png_path, model_name="Total Dose 3D U-Net", best_epoch=best_epoch)
        pdf_path = out_dir / "loss_curves.pdf"
        save_loss_curves_pdf(history, metric_history, pdf_path, best_epoch)
        print(f"Loss curves saved to: {png_path} and {pdf_path}")
    except Exception as e:  # never lose the test evaluation because of a plotting problem
        print(f"[WARN] Could not create loss-curve visualization: {e}")

    # Final Clinical Benchmark on all unseen test patients
    print("\n" + "=" * 105)
    print(f"FINAL CLINICAL DOSIMETRIC EVALUATION ON {len(test_data)} UNSEEN TEST PATIENTS")
    print("=" * 105, flush=True)

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_results = evaluate_cohort(model, test_data, device)

    print(f"{'Patient':<10} | {'Within +-10%':<14} | {'Within +-5%':<14} | {'Tissue MAE':<14} | {'Bone MAE':<12} | {'Soft Tissue MAE':<16} | {'Rel Err (%)'}")
    print("-" * 105)
    for r in test_results:
        print(
            f"{r['patient']:<10} | {r['within_10_pct']:>12.1f}% | {r['within_5_pct']:>12.1f}% | "
            f"{r['tissue_mae_1e12_gy']:>12.4f} | {r['bone_mae_1e12_gy']:>10.4f} | "
            f"{r['soft_tissue_mae_1e12_gy']:>14.4f} | {r['rel_err_pct']:>10.2f}%"
        )
    print("=" * 105)

    mean_within_10 = float(np.mean([r["within_10_pct"] for r in test_results]))
    mean_within_5 = float(np.mean([r["within_5_pct"] for r in test_results]))
    mean_mae = float(np.mean([r["tissue_mae_1e12_gy"] for r in test_results]))
    mean_bone_mae = float(np.mean([r["bone_mae_1e12_gy"] for r in test_results]))
    mean_soft_mae = float(np.mean([r["soft_tissue_mae_1e12_gy"] for r in test_results]))
    mean_rel_err = float(np.mean([r["rel_err_pct"] for r in test_results]))

    print(
        f"Overall Test Cohort Mean: Within +-10% = {mean_within_10:.2f}% | "
        f"Within +-5% = {mean_within_5:.2f}% | Tissue MAE = {mean_mae:.4f} x10^-12 Gy\n"
        f"Tissue Breakdown: Bone MAE = {mean_bone_mae:.4f} | Soft Tissue MAE = {mean_soft_mae:.4f} | Mean Rel Err = {mean_rel_err:.2f}%"
    )

    results_json = out_dir / "test_evaluation_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nDetailed dosimetric results saved to: {results_json}")


if __name__ == "__main__":
    main()
