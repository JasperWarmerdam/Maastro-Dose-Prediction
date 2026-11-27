# Deep Learning for Volumetric CT Dose Prediction

This repository contains the PyTorch implementations of three deep learning architectures for fast, patient-specific 3D radiotherapy/CT dose prediction against DOSXYZnrc Monte Carlo ground truth:

1. **Single-Beam 3D U-Net (`RhoThetaDoseUNet`)** — Predicts 3D dose for a single gantry angle $\theta$ using FiLM angle conditioning on a 5-channel physical input (`[rho, zeff, d_parallel, d_perp, wepl]`).
2. **Single-Beam Hierarchical GNN (`HierarchicalGridGNN`)** — Multi-scale 3-level graph neural network with boundary-aware edge conductance and FiLM angle conditioning on an 8-channel input (`[rho, zeff, x_norm, y_norm, z_norm, d_parallel, d_perp, wepl]`).
3. **Full-Scan Total CT Dose 3D U-Net (`TotalCTDoseUNet`)** — Predicts the cumulative 36-view 3D CT dose in a single forward pass from a 3-channel input (`[rho, radial_distance, bone_accent]`).

## Project Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── U_net.py                              # RhoThetaDoseUNet architecture
│   ├── model.py                              # HierarchicalGridGNN architecture (with pure-PyTorch fallback)
│   ├── total_ct_unet.py                      # TotalCTDoseUNet architecture
│   └── utils.py                              # Graph construction, beam geometry, radiological depth (WEPL)
├── train_unet_supercomputer.py               # Train & evaluate single-beam 3D U-Net
├── train_gnn_hierarchical.py                 # Train & evaluate single-beam Hierarchical GNN
├── train_total_ct_dose.py                    # Train & evaluate full-scan Total CT Dose U-Net
├── create_master_project_visualizations.py   # Generate comparison figures, loss curves, and summary tables
├── benchmark_inference.py                    # GPU inference latency benchmark (single-sample & batched)
├── requirements.txt
└── environment_cluster.yml
```

u## Dataset Layout

Place the preprocessed patient datasets under `resources/dataset/`:
* **Single-beam data:** `resources/dataset/outputs_random_angel/pt_*/pt_*.npz` (containing `phantom_density`, `dose`, `theta`, and physical bounds).
* **Full-scan 36-view CT dose data:** `resources/dataset/full_ct_dose/pt_*.egsphant` and `pt_*_36view.npz`.

## Usage

### 1. Train the models
```bash
# Single-beam 3D U-Net
python train_unet_supercomputer.py --epochs 50 --batch_size 8

# Single-beam Hierarchical GNN
python train_gnn_hierarchical.py --epochs 50 --batch_size 4 --accum_steps 2

# Full-scan Total CT Dose 3D U-Net
python train_total_ct_dose.py --epochs 60 --batch_size 4
```

Checkpoints, training histories (`.json`), convergence plots (`.png`), and held-out test evaluations are saved to `outputs/`.

### 2. Generate figures & summary tables
```bash
python create_master_project_visualizations.py
```

### 3. Benchmark GPU inference latency
```bash
python benchmark_inference.py
```
