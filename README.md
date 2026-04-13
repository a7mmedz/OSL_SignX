# OSL-SignX: Omani Sign Language Recognition

An implementation of the **SignX** architecture ([arXiv:2504.16315v3](https://arxiv.org/abs/2504.16315)) for **Continuous Sign Language Recognition (CSLR)** on the Omani Sign Language (OSL) dataset, producing ordered sequences of Arabic gloss labels from raw video.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1 — Pose2Gloss  (trains the latent space from pose features)     │
│                                                                         │
│  Video ──► Pose Extractor ──► Per-frame pose (258d / 1959d)             │
│             (MediaPipe or       │                                        │
│              full 5-modality)   ▼                                       │
│                          PoseFusionEncoder                              │
│                          (multi-head self-attn, 4 layers)               │
│                                 │                                        │
│                                 ▼  2048-dim latent                      │
│                          CodeBookDecoder                                 │
│                          (4096-entry codebook + Transformer decoder)     │
│                                 │                                        │
│                                 ▼                                        │
│                          Gloss logits ──► CE + word-match + contrastive │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2 — Video2Pose  (learns to predict latent from RGB)              │
│                                                                         │
│  Video ──► ViT (vit_base_patch16_224) ──► 768-dim per-frame            │
│                                          │                              │
│                                          ▼                              │
│                                    Linear projection                    │
│                                          │                              │
│                                          ▼  2048-dim predicted latent   │
│                           MSE ◄──── vs Stage 1 latent (frozen)         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 3 — CSLR  (continuous recognition in latent space)              │
│                                                                         │
│  Video ──► Stage 2 (frozen) ──► 2048-dim latent                        │
│                                       │                                  │
│                               ┌───────┴───────┐                        │
│                          LayerNorm      Adaptive pruning (Fisher)       │
│                               └───────┬───────┘                        │
│                                       ▼                                  │
│                            ResNet34-1D  (1-D residual blocks)           │
│                                       ▼                                  │
│                            TemporalConv  (stride=2 downsampling)        │
│                                       ▼                                  │
│                            BiLSTM (2 layers, 512 hidden)                │
│                                       ▼                                  │
│                            Transformer Encoder  (6 layers, dim=256)     │
│                               ├── CTC head  ──► CTC loss                │
│                               └── Transformer Decoder ──► CE loss       │
│                                                                         │
│                         KD loss (CTC teacher → decoder student)        │
└─────────────────────────────────────────────────────────────────────────┘

                  Inference
                  ─────────
  Video ──► Stage2 ──► Stage3 ──► CTC log-probs ──► Beam Search (size 8)
                                                          │
                                                          ▼
                                                   Arabic gloss sequence
```

---

## Dataset

| Property | Value |
|---|---|
| Language | Omani Sign Language (Arabic glosses) |
| Word-level videos | `final_split/{train,dev,test}/rgb/*.mp4` |
| Sentence-level videos | `dataset/OSL-Sentences/rgb_format/*.mp4` |
| Vocabulary | 801 sign glosses + 1 CTC blank = **802 tokens** |
| Filename format | `{ID}_{SYY}_{TZZ}.mp4` (ID=WordID, SYY=SignerID, TZZ=TakeID) |

The `final_split/` directory uses a signer-aware split to prevent data leakage.

---

## Dataset Path Configuration

The dataset root is set in `configs/paths.yaml` and **can be overridden without editing any file** via an environment variable:

```bash
export DATASET_ROOT=/path/to/your/dataset
```

### Path layout expected inside `DATASET_ROOT`

```
$DATASET_ROOT/
├── final_split/
│   ├── train/rgb/*.mp4
│   ├── dev/rgb/*.mp4
│   └── test/rgb/*.mp4
└── dataset/
    └── OSL-Sentences/
        └── rgb_format/*.mp4
```

The default (local WSL) path is:
```
/mnt/c/Users/Admin/Desktop/SQU/Spring_26/FYP/Models/Dataset
```

---

## Installation

### Option A — Local WSL (using `uv`)

```bash
# 1. Navigate to the project
cd /path/to/OSL_SignX

# 2. Create and activate the virtual environment
uv venv --python 3.10
source .venv/bin/activate

# 3. Install CUDA-enabled PyTorch first (match your CUDA version)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
uv pip install -r requirements.txt

# 5. Install the package in editable mode
uv pip install -e .
```

### Option B — Remote GPU Server (via SSH + Jupyter Lab)

SSH into the server and open a terminal in Jupyter Lab (or use the SSH terminal directly):

```bash
# 1. Navigate to the uploaded project
cd /path/to/OSL_SignX

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install CUDA-enabled PyTorch (check server CUDA version with: nvcc --version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Install the package in editable mode
pip install -e .
```

> **Check CUDA version**: Run `nvcc --version` or `nvidia-smi` on the server.  
> Replace `cu121` with `cu118`, `cu124`, etc. to match.

---

## Dataset Preparation

### 1. Upload the dataset to the server

Copy the dataset to the server (from your local machine):

```bash
# Example using scp (run on your local machine):
scp -r "/mnt/c/Users/Admin/Desktop/SQU/Spring_26/FYP/Models/Dataset" \
    user@server_ip:/home/user/data/OSL_Dataset

# Or use rsync for large transfers (shows progress, resumes if interrupted):
rsync -avz --progress \
    "/mnt/c/Users/Admin/Desktop/SQU/Spring_26/FYP/Models/Dataset/" \
    user@server_ip:/home/user/data/OSL_Dataset/
```

Then set `DATASET_ROOT` on the server:

```bash
export DATASET_ROOT=/home/user/data/OSL_Dataset
# Add to ~/.bashrc to make it permanent:
echo 'export DATASET_ROOT=/home/user/data/OSL_Dataset' >> ~/.bashrc
```

### 2. Validate paths

```bash
bash scripts/prepare_dataset.sh
```

This prints per-split video counts, class counts, and signer counts.

### 3. Vocabulary files

Place the following (already in the repo under `data/`):
```
data/gloss_vocab.txt    ← 802 lines: "0000 <blank>", "0001 مستشفى", ...
data/Words.txt          ← 800 lines: WordID → Arabic gloss
data/Sentences.txt      ← 444 lines: SentenceID → Arabic sentence text
```

### 4. Sentence gloss annotations (manual, for Stage 3)

Stage 3 sentence-level training requires `data/sentence_glosses.txt`. Format:

```
0001 0042 0007 0115 0003
0002 0088 0201 0007
```

Each line: `SentenceID gloss1 gloss2 ...` where each gloss is a WordID from `Words.txt`.

### 5. (Optional) Pre-extract pose features (~5× faster training)

```bash
SPLIT=train bash scripts/extract_poses.sh
SPLIT=dev   bash scripts/extract_poses.sh
```

Then in `configs/default.yaml` set:
```yaml
pose:
  backend: precomputed
  precomputed_dir: outputs/cache/pose_train
```

---

## Training

### Setting the dataset path (three ways — pick one)

**Option 1 — Environment variable (recommended for server):**
```bash
export DATASET_ROOT=/home/user/data/OSL_Dataset
```

**Option 2 — CLI flag:**
```bash
python -m signx.training.train_stage1 --dataset-root /home/user/data/OSL_Dataset
```

**Option 3 — Edit `configs/paths.yaml` directly:**
```yaml
dataset_root: /home/user/data/OSL_Dataset
```

---

### Stage 1 — Pose2Gloss

Trains the core latent space from pose features.

**Terminal / SSH:**
```bash
source .venv/bin/activate
export DATASET_ROOT=/home/user/data/OSL_Dataset
python -m signx.training.train_stage1 --config configs/stage1_pose2gloss.yaml
```

**Jupyter Lab cell:**
```python
import os
os.environ["DATASET_ROOT"] = "/home/user/data/OSL_Dataset"
os.chdir("/path/to/OSL_SignX")   # must run from project root

from signx.training.train_stage1 import main
main()
```

**Via the unified script:**
```bash
bash scripts/train.sh --stage 1 --dataset-root /home/user/data/OSL_Dataset
```

**Monitor**: `outputs/logs/stage1_pose2gloss/curves/` — PNG training curves saved after each epoch.  
**Expected**: several hours on a single GPU (800 epochs, batch 32 × 4 grad accum).

---

### Stage 2 — Video2Pose

Requires a trained Stage 1 checkpoint at `outputs/checkpoints/stage1_pose2gloss/best.pt`.

**Terminal:**
```bash
export DATASET_ROOT=/home/user/data/OSL_Dataset
python -m signx.training.train_stage2 --config configs/stage2_video2pose.yaml
```

**Jupyter Lab cell:**
```python
import os
os.environ["DATASET_ROOT"] = "/home/user/data/OSL_Dataset"
os.chdir("/path/to/OSL_SignX")

from signx.training.train_stage2 import main
main()
```

**Monitor**: `outputs/logs/stage2_video2pose/curves/` — PNG training curves.  
**Expected**: ~4–8 hours (300 epochs).

> Before running, verify `model.stage1_checkpoint` in `configs/stage2_video2pose.yaml` points to the correct checkpoint.

---

### Stage 3 — Continuous Recognition

Requires Stage 2. Populate `data/sentence_glosses.txt` first for sentence-level training.

**Terminal:**
```bash
export DATASET_ROOT=/home/user/data/OSL_Dataset
python -m signx.training.train_stage3 --config configs/stage3_cslr.yaml
```

**Jupyter Lab cell:**
```python
import os
os.environ["DATASET_ROOT"] = "/home/user/data/OSL_Dataset"
os.chdir("/path/to/OSL_SignX")

from signx.training.train_stage3 import main
main()
```

**Monitor**: `outputs/logs/stage3_cslr/curves/` — PNG WER and loss curves.  
**Expected**: 200 epochs on sentence-level data.

---

### Run all three stages sequentially

```bash
bash scripts/train.sh --dataset-root /home/user/data/OSL_Dataset
```

---

## Visualization / Monitoring

Training curves are saved automatically as PNG files after every epoch — **no wandb account needed**:

```
outputs/logs/
├── stage1_pose2gloss/
│   ├── train.log
│   └── curves/
│       ├── stage1_pose2gloss_loss.png
│       ├── stage1_pose2gloss_text.png
│       └── stage1_pose2gloss_wer.png
├── stage2_video2pose/
│   └── curves/
│       └── stage2_video2pose_mse.png
└── stage3_cslr/
    └── curves/
        ├── stage3_cslr_wer.png
        └── stage3_cslr_loss.png
```

### Enabling Weights & Biases (optional, for remote monitoring)

In `configs/default.yaml`:
```yaml
logging:
  wandb_enabled: true
  wandb_project: osl-signx
  wandb_entity: your-wandb-username
```

Then on the server, log in once:
```bash
wandb login
```

---

## Evaluation

```bash
export DATASET_ROOT=/home/user/data/OSL_Dataset
bash scripts/evaluate.sh \
    --stage2-checkpoint outputs/checkpoints/stage2_video2pose/best.pt \
    --stage3-checkpoint outputs/checkpoints/stage3_cslr/best.pt
```

Or from a Python script / notebook:
```bash
python -m signx.inference.evaluate \
    --dataset-root /home/user/data/OSL_Dataset \
    --stage2-checkpoint outputs/checkpoints/stage2_video2pose/best.pt \
    --stage3-checkpoint outputs/checkpoints/stage3_cslr/best.pt \
    --split test \
    --output outputs/eval_results.json
```

Results are written to `outputs/eval_results.json`.

### Metrics

| Metric | Description |
|---|---|
| **WER** | Word Error Rate (Levenshtein distance / reference length). Lower is better. |
| **BLEU** | Corpus-level BLEU-4 (sacrebleu). Higher is better. |
| **P-I Accuracy** | Position-Independent token match rate. Higher is better. |

---

## Inference on a Single Video

```bash
python -m signx.inference.predict \
    --video path/to/video.mp4 \
    --stage2-checkpoint outputs/checkpoints/stage2_video2pose/best.pt \
    --stage3-checkpoint outputs/checkpoints/stage3_cslr/best.pt \
    --beam-size 8 \
    --device cuda
```

---

## Configuration

All configs are in `configs/`. They use OmegaConf with a `defaults:` chain:

```
stage3_cslr.yaml
  → default.yaml
    → paths.yaml
```

### Key options

| Config key | Location | Description |
|---|---|---|
| `dataset_root` | `paths.yaml` | Root path to the dataset (override with `DATASET_ROOT` env var) |
| `pose.backend` | `default.yaml` | `mediapipe` / `precomputed` / `full5` |
| `pose.mediapipe_dim` | `default.yaml` | Output dim for MediaPipe (258) |
| `model.latent_dim` | stage configs | Shared latent space size (2048) |
| `train.batch_size` | stage configs | Batch size per GPU |
| `train.grad_accum_steps` | stage configs | Gradient accumulation steps |
| `logging.wandb_enabled` | `default.yaml` | Enable W&B logging |
| `checkpoint.save_top_k` | `default.yaml` | Top-K checkpoints to keep |
| `multi_gpu` | `default.yaml` | Set `true` for multi-GPU (DataParallel) |

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Tests use random dummy data — no dataset or GPU required.

---

## Known Limitations

1. **Sentence annotations missing**: `data/sentence_glosses.txt` is a placeholder. Stage 3 uses word-level data until you fill it in manually.

2. **Single-modality pose**: The default `mediapipe` backend uses 258-dim features. The paper uses a 1959-dim 5-modality concatenation (SMPLer-X + DWPose + MediaPipe + PrimeDepth + Sapiens). Switch `pose.backend: full5` only after installing those tools manually — they are not pip-installable.

3. **Dataset I/O over WSL**: The dataset is on the Windows filesystem (`/mnt/c/...`). Read speeds are lower than native Linux. Pre-extracting pose features to the Linux filesystem (`outputs/cache/`) helps significantly. On the server this is not an issue.

4. **Memory**: Stage 3 with sentence-level data can require 24 GB+ GPU RAM at the default `latent_dim=2048`. Reduce `model.latent_dim` or `train.batch_size` if OOM errors occur.

---

## Citation

```bibtex
@misc{signx2025,
  title  = {SignX: Continuous Sign Recognition in Compact Pose-Rich Latent Space},
  author = {},
  year   = {2025},
  note   = {arXiv:2504.16315v3}
}
```
