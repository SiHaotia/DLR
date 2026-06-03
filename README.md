# DLRFusion

**Degradation-Oriented Region Localization and Restoration for Infrared-Visible Image Fusion**

A U-shaped Transformer framework for degraded IVIF, built around **OFE** (Object Feature Enhancement), **CGM** (Channel Gating Mechanism), **SMM** (Spatial Masking Mechanism), and **channel-spatial weighting matrix fusion**.

---

## ⚙️ Environment Setup

### Recommended stack

| Item | Version |
|------|---------|
| Python | 3.9 (recommended) |
| PyTorch | 2.3.1 |
| CUDA | Match your PyTorch build (11.8+ suggested) |

### Installation

```bash
# 1) Create environment
conda create -n DLRFusion python=3.9 -y
conda activate DLRFusion

# 2) Install PyTorch (pick the command for your CUDA version from pytorch.org)
# e.g. pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# 3) Project dependencies
cd DLRFusion
pip install -r requirements.txt

# 4) Extra packages (YAML, LMDB, I/O, viz)
pip install pyyaml lmdb pillow scikit-image matplotlib
```

### Project layout

```
DLRFusion/
├── DLR/                    # Core networks, data, training logic
├── basicsr/                # Training / testing backbone
├── ldm/                    # Diffusion modules
├── options/                # Train & test configs
│   ├── train/              # DLR_S1.yml, DLR_S2.yml
│   └── test/               # DLR_S2.yml
├── datasets/               # Data & visualization tools
├── experiments/            # Checkpoints, logs, OFE cache
├── train.py                # Training entry
└── test.py                 # Testing entry
```

---

## 📊 Datasets & Scale

### Benchmark scale (paper, four datasets combined)

Experiments use **MSRS**, **M3FD**, **TNO**, and **LLVIP**, with degraded pairs synthesized following protocols similar to Text-DiFuse / TG-ECNet:

| Split | Image pairs (approx.) |
|-------|------------------------|
| Train | **19,278** |
| Test  | **6,135**  |

> These counts are the merged total across four datasets in the paper. You may train on a single dataset (e.g. MSRS only) in practice.

### Default layout (MSRS example)

Paths in YAML configs use semantic placeholders — replace them before running. Suggested structure:

```
MSRS-main/
├── train/
│   ├── ir/              # Clean IR (GT)
│   ├── ir_degrade/      # Degraded IR (suffix in filename, see table below)
│   ├── vi/              # Clean VI (GT)
│   └── vi_degrade/      # Degraded VI
└── test/
    ├── ir/
    ├── ir_degrade/
    ├── vi/
    └── vi_degrade/
```

**Hybrid degraded test set** (`options/test/DLR_S2.yml`) uses relative paths under `datasets/Hybrid_Datasets/test/` with per-degradation subfolders.

### Degradation types

During training, the IR branch randomly samples **4** degradation types (`DLR/data/paired_image_ir_dataset.py`). VI / contrastive training may use additional types.

#### Infrared (IR) suffixes

| Suffix | Meaning | Description |
|--------|---------|-------------|
| `LC` | Low Contrast | Reduced thermal contrast |
| `Norm` | Normalization | Brightness / thermal inconsistency |
| `RN` | Random Noise | Additive random noise |
| `SN` | Stripe Noise | Sensor stripe artifacts |

Filename pattern: `00001D_LC.png` → `{basename}_{degradation}{ext}`.

#### Visible (VI) suffixes

| Suffix | Meaning | Description |
|--------|---------|-------------|
| `LL` | Low Light | Under-exposure |
| `Blur` | Blur | Motion / defocus blur |
| `Rain` | Rain | Rain streaks |
| `haze` | Haze | Atmospheric haze |
| `OE` | Over Exposure | Over-exposure |
| `RN` | Random Noise | Additive noise |
| `Norm` | Normalization | Illumination-style degradation |

#### Typical test pairings (paper-style scenarios)

| VI degradation | Common IR pairing |
|----------------|-------------------|
| Low light (`LL`) | `LC` / `SN` |
| Haze | `LC` / `RN` |
| Over exposure (`OE`) | `LC` |
| Rain | `SN` |

`SpatialMaskingRefinement` predicts **3 degradation masks + 1 clean-region mask** (T = 4), consistent with the paper.

---

## 📁 Path configuration (required)

Train YAML files use placeholders like `path to XXX`. **Replace them with real paths** before training, e.g.:

```yaml
dataroot_lq_ir: path to MSRS training degraded infrared images
# change to:
dataroot_lq_ir: /your/path/MSRS-main/train/ir_degrade
```

Pretrained weights example:

```yaml
pretrain_network_g: path to DLR_S1 pretrained fusion network checkpoint
# change to:
pretrain_network_g: experiments/DLR_S1/models/net_g_5000.pth
```

---

## 🚀 Two-stage training

Aligned with the paper: **Stage 1** learns degraded-region restoration; **Stage 2** fixes restoration modules and optimizes fusion (+ diffusion).

### Stage 1 — Restoration (`DLR_S1`)

| Item | Value |
|------|-------|
| Config | `options/train/DLR_S1.yml` |
| Trains | `network_sp`, `network_cp`, `Transformer_DLR` |
| Paper (suggested) | lr `1e-4`, ~`3e5` iterations |
| Default in repo | `total_iter: 50000` (increase for full runs) |

```bash
python train.py -opt options/train/DLR_S1.yml
```

Checkpoints: `experiments/DLR_S1/models/` → `net_g_*.pth`, `net_sp_*.pth`, `net_cp_*.pth`.

### Stage 2 — Fusion & diffusion (`DLR_S2`)

| Item | Value |
|------|-------|
| Config | `options/train/DLR_S2.yml` |
| Requires | Stage 1 weights for `net_g`, `net_sp`, `net_cp` |
| Paper (suggested) | lr `2e-4`, ~`2e5` iterations; CGM/SMM frozen |
| Default in repo | `total_iter: 5000` (debug scale — increase for production) |

```bash
# Fill Stage 1 paths under `path:` in DLR_S2.yml first
python train.py -opt options/train/DLR_S2.yml
```

### OFE foreground cache (recommended before training)

`ObjectFeatureEnhancement` loads from:

```
experiments/OB_feature/features.pth
```

If missing, extract features first (e.g. via `basicsr/test.py`):

```bash
python basicsr/test.py -opt options/test/DLR_S2.yml
# Pack outputs into experiments/OB_feature/features.pth
```

See also scripts under `datasets/visual/` for mask / channel `.pth` generation.

### Key YAML knobs

| Key | Where | Notes |
|-----|-------|-------|
| `gpu_ids` | Top of file | GPU id list |
| `num_gpu` | Top of file | `0` = CPU |
| `batch_size_per_gpu` | `datasets.train` | Batch size |
| `gt_size` / `gt_sizes` | `datasets.train` | Progressive patch sizes |
| `total_iter` | `train` | Total iterations |
| `save_checkpoint_freq` | `logger` | Save interval |

TensorBoard: `tb_logger/<experiment_name>/`.

---

## 🧪 Testing

```bash
python test.py -opt options/test/DLR_S2.yml
```

- Test sets are split by degradation in YAML (`test_vi_Blur`, `test_ir_SN`, etc.).
- Default output: `./Results/DLR/` (override per-dataset `save_folder`).
- Set pretrained paths under `path:` (replace placeholders).

---

## 🔧 Optional: synthesize degradations

**IR degradation:**

```bash
# Edit input/output folders at the bottom of utils/IR_degraded.py, then:
python utils/IR_degraded.py
```

**VI fusion weights:** see `utils/VWM.py` (relative paths under `datasets/MSRS/train/...`).

---

## 🧩 Paper modules ↔ code

| Paper module | Code class |
|--------------|------------|
| Object Feature Enhancement (OFE) | `ObjectFeatureEnhancement` |
| Channel Gating Mechanism (CGM) | `ChannelGatingMechanism` |
| Spatial Masking Mechanism (SMM) | `SpatialMaskingMechanism` / `SpatialMaskingRefinement` |
| Channel-spatial weighting fusion | `ChannelSpatialWeightingMatrixFusion` |

---

## ❓ FAQ

1. **`path to XXX` does not load data**  
   Placeholders are not valid paths — set real absolute or relative paths in YAML.

2. **Missing `features.pth`**  
   Create `experiments/OB_feature/` and run feature extraction, or adjust loading in `Transformer_DLR_arch.py`.

3. **Stage 2 OOM**  
   Lower `batch_size_per_gpu` / `gt_size`, or disable `diffusion_schedule.apply_ldm` for ablation.

4. **Windows paths**  
   Prefer forward slashes: `datasets/Hybrid_Datasets/test/ir`.

---

## 📖 Citation

If you use this code, please cite the DLRFusion paper (Anonymous Authors, under review).
