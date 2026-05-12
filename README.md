# Pointcept — Static LiDAR Semantic Segmentation

Semantic segmentation of static terrestrial LiDAR point clouds using **Point Transformer V3 (PTv3)** within the [Pointcept](https://github.com/Pointcept/Pointcept) framework.

This project implements a two-stage self-distillation pipeline:

- **V1** — baseline PTv3 model trained on XYZ + intensity (4 channels)
- **V2** — refined PTv3 model that additionally uses V1's predictions as a guidance channel (5 channels), allowing it to learn from and correct V1's mistakes

Both models support **Test-Time Augmentation (TTA)** at inference for improved prediction quality.

---

## Classes

| Train ID | Class Name | Description |
|----------|-----------|-------------|
| 0 | Unclassified | Unknown or ignored points |
| 1 | Ground | Terrain, road surface |
| 2 | Pole | Lamp posts, utility poles |
| 3 | Sign | Road signs, signage |
| 4 | Bollard | Bollards, posts |
| 5 | Trunk | Tree trunks |
| 6 | Vegetation | Trees, bushes, hedges |
| 7 | Building | Building facades, walls |
| 8 | Fence | Fences |
| 9 | Gate | Gates, barriers |

> Class 0 (Unclassified) and class 255 are ignored during training loss computation.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Project Structure](#2-project-structure)
3. [Understanding the Config File](#3-understanding-the-config-file)
4. [V1 — Baseline Model](#4-v1--baseline-model)
   - [Data Preparation](#41-data-preparation)
   - [Training](#42-training)
   - [Inference](#43-inference)
5. [V2 — Self-Distillation Refinement](#5-v2--self-distillation-refinement)
   - [What is V2 and Why](#51-what-is-v2-and-why)
   - [Data Preparation](#52-data-preparation)
   - [Training](#53-training)
   - [Inference](#54-inference)

---

## 1. Installation

### Prerequisites

- Linux or WSL2 (Windows Subsystem for Linux)
- NVIDIA GPU with CUDA 12.4
- Python 3.10
- Git

### Clone the Repository

```bash
git clone https://gitlab.geo-sat.com/rnd/research/pointcept.git
cd Pointcept
```

### Environment Setup

This project was developed using [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment management. `uv` is a fast Python package manager. However, the same steps work with `conda` or plain `pip` — just replace `uv venv` with `conda create` and `uv pip install` with `pip install`.

#### Using uv (recommended)

```bash
# Install uv if not already installed

# Step 1: Create and activate the environment
# Python 3.10 is required
uv venv .venv-pointcept --python 3.10
source .venv-pointcept/bin/activate

# Step 2: Install PyTorch 2.5.0 with CUDA 12.4
uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Step 3: Install system dependency
sudo apt-get update
sudo apt-get install libsparsehash-dev

# Step 4: Install standard Python dependencies
uv pip install ninja h5py pyyaml tensorboard tensorboardx wandb \
    yapf addict einops scipy plyfile termcolor timm ftfy regex \
    tqdm matplotlib black open3d mlflow

# Step 5: Install PyTorch Geometric libraries
uv pip install torch-cluster torch-scatter torch-sparse torch-geometric \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html

# Step 6: Install spconv and LoRA tools
uv pip install spconv-cu124 peft

# Step 7: Install Git-based dependencies
uv pip install git+https://github.com/octree-nn/ocnn-pytorch.git
uv pip install "setuptools<81"
uv pip install git+https://github.com/openai/CLIP.git --no-build-isolation

# ==============================================================================
# NOTE: Flash Attention requires the NVIDIA CUDA Compiler (nvcc) to build.
# First, check if your system has it by running: nvcc --version
# ==============================================================================

# SCENARIO A: If `nvcc` IS ALREADY INSTALLED (e.g., standard WSL or Local PC)
# Run this directly  
# Note: Pinned to v2.7.0.post2 to avoid compute_120 architecture build errors.
uv pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.0.post2 --no-build-isolation

# SCENARIO B: If `nvcc` IS MISSING (e.g., JupyterLab, Docker containers)
# Run these steps to download the compiler into your conda environment first:
# 1. Install CUDA 12.4 Toolkit via conda or sudo
conda install -c "nvidia/label/cuda-12.4.1" cuda-toolkit -y #using conda

sudo apt-get update #or you can use sudo route
sudo apt-get install nvidia-cuda-toolkit #or you can use sudo route

# 2. Point the system to the new compiler and restrict GPU architectures
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export MAX_JOBS=4

# 3. Install Flash Attention
uv pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.0.post2 --no-build-isolation

# Step 8: Compile Pointcept's custom CUDA operations (might take a few mins)
cd libs/pointops
python setup.py install
cd ../../

cd libs/pointgroup_ops
python setup.py install
cd ../../
```

#### Using conda (alternative)

```bash
conda create -n pointcept python=3.10
conda activate pointcept
# Then follow the same pip install commands above
# replacing "uv pip install" with "pip install"
```

### Register the Custom Dataset Classes

After installation, register the custom dataset loaders by adding the following lines to `pointcept/datasets/__init__.py`:

```python
from .lidar_dataset import *   # V1 dataset
from .lidar_dataset2 import *  # V2 dataset
```

---

## 2. Project Structure

```
Pointcept/                          ← root directory
│
├── configs/                        ← all training config files
│   ├── laz_static/
│   │   └── lidar_config.py         ← V1 training config
│   └── laz_staticV2/
│       └── lidar_config-v2.py      ← V2 training config
│
├── exp/                            ← all training runs (auto-created)
│   └── laz_static/
│       └── lidar_ptv3_run1/
│           ├── model/
│           │   ├── model_best.pth  ← best checkpoint by val mIoU
│           │   └── model_last.pth  ← last epoch checkpoint
│           └── ...
│
├── libs/                           ← compiled CUDA libraries
│
├── pointcept/                      ← core framework
│   └── datasets/
│       ├── lidar_dataset.py        ← V1 dataset class
│       └── lidar_dataset2.py       ← V2 dataset class
│
├── Data_Preprocessing/             ← all data-related scripts
│   ├── data_preprocessing.py       ← LAZ → NPY for V1
│   ├── data_preprocessing2.py      ← LAZ → NPY for V2 (adds v1_pred)
│   └── predict_lazv2.py            ← run V1 inference to generate
│                                      v1_pred field in LAZ files
│
├── scripts/
│   └── train.sh                    ← training launch script
│
├── tools/
│   └── train.py                    ← main training entry point
│
└── inference/                      ← all inference scripts
    ├── Predict_laz_V1_V2.py        ← V1 → V2 + TTA inference
    └── Predict_laz_self_feeding.py ← V2 self-feeding + TTA (no V1 needed)
```

---

## 3. Understanding the Config File

Both V1 and V2 share the same config structure. This section explains the most important parameters so you can adapt training to your data and hardware.

The config file lives at:
- V1: `configs/laz_static/lidar_config.py`
- V2: `configs/laz_staticV2/lidar_config-v2.py`

### Class Map

The class map defines how raw LAZ classification IDs map to training IDs. Edit this in your preprocessing config or script:

```python
class_map = {
    0:   0,    # Unclassified  → ignore during training
    1:   1,    # Ground        → 1
    2:   2,    # Pole          → 2
    3:   3,    # Sign          → 3
    4:   4,    # Bollard       → 4
    5:   5,    # Trunk         → 5
    6:   6,    # Vegetation    → 6
    7:   7,    # Building      → 7
    8:   255,  # RIEN          → ignore during training
    9:   8,    # Fence         → 8
    10:  9,    # Gate          → 9
}
```

> Any class mapped to `255` is ignored by the loss function during training.

### Most Important Parameters

#### Data Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `data_root` | Path to your NPY dataset folder | Your absolute path |
| `num_classes` | Total number of output classes | `10` |
| `ignore_index` | Class ID ignored in loss | `255` |
| `loop` | How many times to repeat dataset per eval epoch | `10` |

#### Training Parameters

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `batch_size` | Scenes per training step | `2` or `3` depending on VRAM |
| `batch_size_val` | Scenes per val step | Always keep at `1` |
| `eval_epoch` | This is your actual epoch count | `50` = 50 real epochs |
| `epoch` | Only used to compute loop: `epoch / eval_epoch` | Set to `eval_epoch × loop` |

> **Important:** `eval_epoch` is the true epoch count in this setup. `epoch` is only used internally to calculate how many times the dataset loops per evaluation cycle.

#### Model Parameters

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `in_channels` | Input feature dimensions | V1: `4`, V2: `5` |
| `grid_size` | Voxel size in metres | `0.03`–`0.05m` |
| `point_max` | Max points per training crop | `150k`–`200k` |
| `enc_patch_size` | Attention patch size per encoder stage | `48` without Flash Attention |
| `drop_path` | Stochastic depth rate | `0.3` |
| `enable_flash` | Use Flash Attention (faster, needs CUDA ≥ 11.6) | `False` if not installed |

#### Scheduler Parameters

| Parameter | Description | How to calculate |
|-----------|-------------|-----------------|
| `max_lr` | Peak learning rate | `0.001` (standard for PTv3) |
| `total_steps` | Must match actual training steps exactly | `num_scenes × loop / batch_size × eval_epoch` |

> **Critical:** If `total_steps` is wrong, the learning rate schedule will be miscalibrated. Recalculate every time you change `batch_size`, `loop`, or `eval_epoch`.

**Example calculation:**
```
244 scenes × 10 loop / 2 batch × 50 eval_epochs = 61,000
```

#### Loss Functions

```python
criteria = [
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=255),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=255),
]
```

- **CrossEntropy** — optimises per-point accuracy
- **Lovasz** — optimises IoU directly, improves region boundary quality
- Using both together gives better results than either alone

---

## 4. V1 — Baseline Model

V1 is the foundation model. It takes raw point cloud geometry and intensity as input and learns to classify every point into one of 10 classes.

```
Input: XYZ (3) + Intensity (1) = 4 channels
Output: Per-point class predictions (10 classes)
```

### 4.1 Data Preparation

V1 data preparation reads your annotated LAZ files and converts them to NPY format that Pointcept can load during training.

**Input:** A folder of LAZ files split into `train/` and `val/`

**Output:**
```
DataV1/
    train/
        tile_001/
            coord.npy       # (N, 3) float32  — XYZ coordinates
            color.npy       # (N, 1) float32  — intensity normalised to [0, 1]
            segment.npy     # (N,)   int16    — class labels
        tile_002/
            ...
    val/
        ...
```

**Run the preprocessing script:**

```bash
python Data_Preprocessing/data_preprocessing.py \
    --config   /Data_Preprocessing/config.yaml
```

> The script automatically reads the `train/` and `val/` subfolder split from your LAZ directory and mirrors that structure in the output. Just change your raw data root (input-laz), processed root (output-npy), and class map paths in the config file.

### 4.2 Training

Update `configs/laz_static/lidar_config.py`:

```python
data = dict(
    train = dict(
        data_root = "/path/to/DataV1",
        ...
    ),
    val = dict(
        data_root = "/path/to/DataV1",
        ...
    ),
)
```

Recalculate `total_steps` based on your scene count:

```python
# Example: 244 scenes, loop=10, batch=2, eval_epoch=50
total_steps = 244 * 10 // 2 * 50  # = 61,000
```

Launch training from the Pointcept root directory:

```bash
sh scripts/train.sh \
    -p python \
    -g 1 \
    -d laz_static \
    -c lidar_config \
    -n lidar_ptv3_run1
```

| Argument | Meaning |
|----------|---------|
| `-p python` | Python interpreter |
| `-g 1` | Number of GPUs |
| `-d laz_static` | Config folder name under `configs/` |
| `-c lidar_config` | Config file name (without `.py`) |
| `-n lidar_ptv3_run1` | Run name — results saved to `exp/laz_static/lidar_ptv3_run1/` |

**Monitor training:**

```bash
tensorboard --logdir exp/laz_static/lidar_ptv3_run1
```

**Checkpoints** are saved to `exp/laz_static/lidar_ptv3_run1/model/`:
- `model_best.pth` — best validation mIoU checkpoint ← use this for inference
- `model_last.pth` — most recent epoch checkpoint

### 4.3 Inference

V1 inference is handled by the V1→V2 pipeline script (see Section 5.4). However if you want to run V1 standalone, use:

```bash
python inference/Predict_laz.py \
    --input_dir   /path/to/input \
    --output_dir  /path/to/output \
    --weight      exp/laz_static/lidar_ptv3_run1/model/model_best.pth \
    --tile_size   Chunk size in metres, default 25x25, if you train your model with bigger laz chunks, then change it \
    --tile_threshold   Files larger than this bbox get chunked (default: 30) \
    --grid_size    Voxel size, must match the training (default: 0.05)
```

> For V1-only inference, you can add the post processing method like Test Time Augmentation as well. Just use the --tta option.

---

## 5. V2 — Self-Distillation Refinement

### 5.1 What is V2 and Why

V2 is a second PTv3 model that builds on top of V1. Instead of starting from scratch, V2 receives V1's prediction for each point as an additional input channel. This allows V2 to:

- Learn where V1 was confident and correct → reinforce those predictions
- Learn where V1 made systematic errors → correct those specific mistakes

This technique is called **self-distillation** or **iterative refinement**.

```
                    ┌─────────────────────────────────────┐
                    │          V2 TRAINING PIPELINE        │
                    └─────────────────────────────────────┘

  Annotated LAZ files
         │
         ▼
  ┌─────────────┐
  │ V1 Inference│  Run V1 on all training files using the Data_Preprocessing/predict_lazv2.py 
  │  (predict_  │  file. Save per-point class predictions
  │  lazv2.py)  │  into LAZ as extra field "v1_pred"
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │ Data Preparation │  data_preprocessing2.py
  │                  │  Reads: coord, intensity, segment, v1_pred
  │                  │  Normalises v1_pred: class_id / 9.0 → [0, 1]
  │                  │  Saves: coord.npy, color.npy (intensity + v1_pred),
  │                  │         segment.npy, v1_pred.npy
  └──────┬───────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │                    V2 MODEL                          │
  │                                                      │
  │  Input: XYZ(3) + Intensity(1) + V1_pred_norm(1)     │
  │         = 5 channels                                 │
  │                                                      │
  │  Encoder → Decoder → FC → 10 class predictions      │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  Better predictions, especially on rare/thin classes
```

### 5.2 Data Preparation

V2 data preparation requires two inputs:
- Your **ground truth** annotated LAZ files (same as V1)
- **V1 prediction** LAZ files (generated by running V1 inference on your training data)

#### Step 1 — Generate V1 predictions on training data

Run V1 inference on all your training and validation LAZ files. This creates LAZ files with an extra `v1_pred` field containing V1's per-point class predictions:

```bash
python Data_Preprocessing/predict_lazv2.py \
    --input_dir  /path/to/annotated_laz/train \
    --output_dir /path/to/v1_predictions/train \
    --weight     exp/laz_static/lidar_ptv3_run1/model/model_best.pth
```

Repeat for the val split:

```bash
python Data_Preprocessing/predict_lazv2.py \
    --input_dir  /path/to/annotated_laz/val \
    --output_dir /path/to/v1_predictions/val \
    --weight     exp/laz_static/lidar_ptv3_run1/model/model_best.pth
```

> The output LAZ files contain the original classification (ground truth for CloudCompare review) and the locked `v1_pred` field (used as V2 training feature).

#### Step 2 — Convert to NPY for V2 training

```bash
python Data_Preprocessing/data_preprocessing2.py \
    --config  /Data_Preprocessing/config.yaml \
```
> Just change your train_gt_dir, val_gt_dir, train_v1_dir, val_v1_dir and output_npy_dir paths in the config.yaml file in Data_Preprocessing folder.

**Output structure:**

```
DataV2/
    train/
        tile_001/
            coord.npy       # (N, 3) float32  — XYZ coordinates
            color.npy       # (N, 2) float32  — [intensity, v1_pred_norm]
            segment.npy     # (N,)   int16    — ground truth labels
            v1_pred.npy     # (N,)   int64    — raw V1 class predictions
        tile_002/
            ...
    val/
        ...
```

> `v1_pred_norm` inside `color.npy` is the raw V1 class ID divided by 9.0, normalising it to the same [0, 1] range as intensity.

#### Step 3 — Register the V2 dataset class

Ensure `pointcept/datasets/__init__.py` contains:

```python
from .lidar_dataset2 import *
```

### 5.3 Training

Update `configs/laz_staticV2/lidar_config-v2.py`:

```python
data = dict(
    train = dict(
        type      = "LidarDatasetV2",   # V2 dataset class
        data_root = "/path/to/DataV2",
        ...
    ),
    val = dict(
        type      = "LidarDatasetV2",
        data_root = "/path/to/DataV2",
        ...
    ),
)

model = dict(
    backbone = dict(
        in_channels = 5,   # XYZ(3) + intensity(1) + v1_pred_norm(1)
        ...
    ),
)
```

Recalculate `total_steps` as before, then launch:

```bash
sh scripts/train.sh \
    -p python \
    -g 1 \
    -d laz_staticV2 \
    -c lidar_config-v2 \
    -n lidar_ptv3_v2_run1
```

Checkpoints are saved to `exp/laz_staticV2/lidar_ptv3_v2_run1/model/`.

### 5.4 Inference

Two inference modes are available. Both support TTA (Test-Time Augmentation) which runs multiple augmented passes and averages the softmax probabilities, producing cleaner and more stable predictions.

---

#### Option A — Standard V1 → V2 + TTA Pipeline

Runs V1 first to generate the prediction hint, then feeds that into V2 with TTA. This is the highest quality option.

```
LAZ file → V1 single pass → v1_pred_norm → V2 TTA (8 passes) → Output LAZ
```

**Single file (test first):**

```bash
python inference/Predict_laz_V1_V2.py \
    --single_file /path/to/input.laz \
    --output_dir  /path/to/output \
    --weight_v1   exp/laz_static/lidar_ptv3_run1/model/model_best.pth \
    --weight_v2   exp/laz_staticV2/lidar_ptv3_v2_run1/model/model_best.pth \
    --tta_mode    normal
```

**Full directory:**

```bash
python inference/Predict_laz_V1_V2.py \
    --input_dir  /path/to/laz_files \
    --output_dir /path/to/output \
    --weight_v1  exp/laz_static/lidar_ptv3_run1/model/model_best.pth \
    --weight_v2  exp/laz_staticV2/lidar_ptv3_v2_run1/model/model_best.pth \
    --tta_mode   normal
```

---

#### Option B — Self-Feeding V2 + TTA (No V1 Needed)

Runs V2 only — no V1 weight required. V2 bootstraps itself over multiple iterations, using its own predictions from the previous pass as the hint channel for the next pass.

```
LAZ file
    → Pass 1: v1_pred = zeros → rough prediction
    → Pass 2: v1_pred = Pass 1 result → better prediction
    → Pass 3: v1_pred = Pass 2 result → good prediction
    → Final:  TTA (8 passes) → Output LAZ
```

**Single file (test first):**

```bash
python inference/Predict_laz_self_feeding.py \
    --single_file /path/to/input.laz \
    --output_dir  /path/to/output \
    --weight      exp/laz_staticV2/lidar_ptv3_v2_run1/model/model_best.pth \
    --iterations  3 \
    --tta_mode    normal
```

**Full directory:**

```bash
python inference/Predict_laz_self_feeding.py \
    --input_dir  /path/to/laz_files \
    --output_dir /path/to/output \
    --weight     exp/laz_staticV2/lidar_ptv3_v2_run1/model/model_best.pth \
    --iterations 3 \
    --tta_mode   normal
```

---

#### TTA Mode Options

| Mode | Passes | Speed | Quality |
|------|--------|-------|---------|
| `none` | 1 | Fastest | Baseline |
| `fast` | 4 | ~4× slower | Good |
| `normal` | 8 | ~8× slower | Best (recommended) |
| `full` | 16 | ~16× slower | Marginal gain over normal |

#### Inference Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--tta_mode` | `normal` | TTA intensity |
| `--iterations` | `3` | Self-feeding passes |
| `--grid_size` | `0.05` | Voxel size — must match training |
| `--tile_size` | `25.0` | Chunk size in metres for large files |
| `--tile_threshold` | `30.0` | Files larger than this get chunked |
| `--overlap` | `2.0` | Overlap between chunks to avoid boundary artifacts |

> **grid\_size must match the value used during training.** Mismatched grid size will produce poor results without any error message.

---

## Notes

- Always use `model_best.pth` for inference, not `model_last.pth`. The best checkpoint is selected by highest validation mIoU and is not necessarily the last epoch.
- `batch_size_val` should always remain `1`. Full scenes without SphereCrop are too large for batch > 1 on most GPUs.
- When fine-tuning from a checkpoint, set `weight = "/path/to/checkpoint.pth"` and `resume = False` in the config. Reduce `max_lr` to `0.0001` (10× lower than initial training).
- If training is slow, enabling `enable_amp = True` with `amp_dtype = "float16"` typically gives a 1.5–2× speedup with no quality loss.
- Every train run will automatically get logged in mlflow. If you do not want to send info, just change `use_mlflow = False` in the lidar_config.py or lidar_config-v2 file.
