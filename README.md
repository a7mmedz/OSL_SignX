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
│                          (multi-head self-attn, 2 layers)               │
│                                 │                                        │
│                                 ▼  512-dim latent                       │
│                          CodeBookDecoder                                 │
│                          (1024-entry codebook + Transformer decoder)     │
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
│                                          ▼  512-dim predicted latent    │
│                           MSE ◄──── vs Stage 1 latent (frozen)         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 3 — CSLR  (continuous recognition in latent space)              │
│                                                                         │
│  Video ──► Stage 2 (frozen) ──► 512-dim latent                         │
│                                       │                                  │
│                               ┌───────┴───────┐                        │
│                          LayerNorm      Adaptive pruning (Fisher)       │
│                               └───────┬───────┘                        │
│                                       ▼                                  │
│                            TemporalConv  (stride=2 downsampling)        │
│                                       ▼                                  │
│                            BiLSTM (2 layers, 256 hidden)                │
│                                       ▼                                  │
│                            Transformer Encoder  (4 layers, dim=256)     │
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
cd /path/to/OSL_SignX
uv venv --python 3.10
source .venv/bin/activate

# Match cu121 to your CUDA version (check with: nvcc --version)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
uv pip install -e .
```

### Option B — Remote GPU Server (SSH + Jupyter Lab)

```bash
cd /path/to/OSL_SignX
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Match cu121 to your CUDA version (check with: nvcc --version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

> Run `nvcc --version` or `nvidia-smi` to find your CUDA version.  
> Replace `cu121` with `cu118`, `cu124`, etc. to match.

---

## Dataset Preparation

### 1. Upload the dataset to the server

```bash
# From your local WSL machine:
rsync -avz --progress \
    "/mnt/c/Users/Admin/Desktop/SQU/Spring_26/FYP/Models/Dataset/" \
    user@server_ip:/home/user/data/OSL_Dataset/
```

Then set `DATASET_ROOT` permanently on the server:

```bash
echo 'export DATASET_ROOT=/home/user/data/OSL_Dataset' >> ~/.bashrc
source ~/.bashrc
```

### 2. Validate paths

```bash
bash scripts/prepare_dataset.sh
```

### 3. Vocabulary files

Already present in the repo under `data/`:
```
data/gloss_vocab.txt       802 lines — "0000 <blank>", "0001 مستشفى", ...
data/Words.txt             800 lines — WordID → Arabic gloss
data/Sentences.txt         443 lines — SentenceID → Arabic sentence text
data/sentence_glosses.txt  164 annotated sentences for Stage 3
```

### 4. (Optional) Pre-extract pose features — ~5× faster training

Run once per split before training. Avoids re-running the pose extractor every epoch:

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

## Pose Extraction Backends

The system supports five pose extraction modalities that can be used individually or combined. The default is **MediaPipe** (easiest to install). Setting `backend: full5` uses all five concatenated (1959-dim), matching the original SignX paper.

| # | Backend key | Dims | What it captures | Install |
|---|---|---|---|---|
| 1 | `mediapipe` | 258 | Body + hand landmarks | `pip install mediapipe` |
| 2 | `dwpose` | 399 | Whole-body 133 keypoints | `pip install openmim && mim install mmpose` |
| 3 | `smplerx` | 432 | 3-D body mesh joints | Clone from GitHub (see below) |
| 4 | `primedepth` | 480 | Monocular depth features | Clone from GitHub (see below) |
| 5 | `sapiens` | 390 | Dense body keypoints | `pip install transformers accelerate` |
| — | `full5` | **1959** | All 5 concatenated | All of the above |
| — | `precomputed` | any | Load pre-extracted `.pt` | No extra install |

### Install DWPose (modality 2)

```bash
pip install openmim
mim install mmengine "mmcv>=2.0.0" mmdet "mmpose>=1.0.0"
```

### Install SMPLer-X (modality 3)

```bash
git clone https://github.com/caizhongang/SMPLer-X.git
cd SMPLer-X && pip install -v -e .
# Download SMPL-X body model from https://smpl-x.is.tue.mpg.de/
# Download checkpoint and set: export SMPLERX_CKPT=/path/to/checkpoint.pth.tar
```

### Install PrimeDepth (modality 4)

```bash
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro && pip install -e .
python -m depth_pro.cli.download_models
```

### Install Sapiens (modality 5)

```bash
pip install transformers accelerate
# Model weights (~1.2 GB) download automatically on first use
```

### Check which modalities are ready

```bash
python3 -c "
from signx.pose import check_full5_available
for name, ok in check_full5_available().items():
    print(f'  {name:12s}: {\"ready\" if ok else \"NOT installed\"}')"
```

### Switch backends

Edit `configs/default.yaml`:
```yaml
pose:
  backend: mediapipe     # single modality — 258-dim (default)
  # backend: full5       # all 5 methods  — 1959-dim (matches paper)
  # backend: precomputed # pre-extracted  — fastest for training
```

When switching from `mediapipe` to `full5`, also update `pose_input_dim` in `configs/stage1_pose2gloss.yaml`:
```yaml
model:
  pose_input_dim: 1959   # was 258 for mediapipe
```

---

## Training

### Step 0 — Set the dataset path

**Option 1 — Environment variable (recommended):**
```bash
export DATASET_ROOT=/home/user/data/OSL_Dataset
```

**Option 2 — CLI flag (per-run):**
```bash
python -m signx.training.train_stage1 --dataset-root /home/user/data/OSL_Dataset
```

**Option 3 — Edit `configs/paths.yaml` directly:**
```yaml
dataset_root: /home/user/data/OSL_Dataset
```

---

### GPU and memory settings

The default config is tuned for a **4× 24 GB GPU server**. Key settings in `configs/default.yaml`:

```yaml
multi_gpu: true    # DataParallel across all 4 GPUs automatically
use_amp:   true    # fp16 mixed precision — ~2× speed, ~50% VRAM
num_workers: 8     # 2 workers per GPU
```

#### If you get an OOM error

Work through this checklist in order — stop as soon as the error goes away:

| Step | What to change | Where |
|---|---|---|
| 1 | Reduce `batch_size` by half | stage config `train.batch_size` |
| 2 | Increase `grad_accum_steps` by 2× to compensate | stage config `train.grad_accum_steps` |
| 3 | Confirm `use_amp: true` | `configs/default.yaml` |
| 4 | Confirm `multi_gpu: true` and all 4 GPUs are visible (`nvidia-smi`) | `configs/default.yaml` |
| 5 | Reduce `video.max_frames` from 256 to 128 | `configs/default.yaml` |
| 6 | Reduce `num_workers` to 4 or even 0 | `configs/default.yaml` |

**Current batch sizes per stage** (effective batch = batch_size × grad_accum_steps):

| Stage | `batch_size` | `grad_accum_steps` | Effective batch | Per GPU |
|---|---|---|---|---|
| Stage 1 | 128 | 1 | 128 | 32 |
| Stage 2 | 64 | 1 | 64 | 16 |
| Stage 3 | 16 | 2 | 32 | 4 |

Batch sizes already account for 4 GPUs. If you only have 1 GPU, divide `batch_size` by 4 and set `multi_gpu: false`.

---

### Stage 1 — Pose2Gloss

Trains the latent space from pose features. Runs on **word-level videos**.

**Terminal / SSH:**
```bash
source .venv/bin/activate
export DATASET_ROOT=/home/user/data/OSL_Dataset
python -m signx.training.train_stage1 --config configs/stage1_pose2gloss.yaml
```

**Jupyter Lab cell:**
```python
import os, sys
os.environ["DATASET_ROOT"] = "/home/user/data/OSL_Dataset"
os.chdir("/path/to/OSL_SignX")   # must run from project root

from signx.training.train_stage1 import main
main()
```

**Via unified script:**
```bash
bash scripts/train.sh --stage 1 --dataset-root /home/user/data/OSL_Dataset
```

**Key config** (`configs/stage1_pose2gloss.yaml`):
```yaml
model:
  pose_input_dim: 258    # 258 for mediapipe / 1959 for full5
  latent_dim: 512
train:
  epochs: 800
  batch_size: 128        # 32 per GPU × 4 GPUs
  grad_accum_steps: 1
data:
  level: word
```

**Monitor:** `outputs/logs/stage1_pose2gloss/curves/` — PNG loss and accuracy plots saved every epoch.  
**Checkpoint:** `outputs/checkpoints/stage1_pose2gloss/best.pt`

---

### Stage 2 — Video2Pose

Requires Stage 1 to be complete. Freezes Stage 1 and trains a ViT to predict pose latents from raw RGB.

**Verify Stage 1 checkpoint exists first:**
```bash
ls outputs/checkpoints/stage1_pose2gloss/best.pt
```

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

**Key config** (`configs/stage2_video2pose.yaml`):
```yaml
model:
  latent_dim: 512        # must match stage1 latent_dim
  stage1_checkpoint: outputs/checkpoints/stage1_pose2gloss/best.pt
train:
  epochs: 300
  batch_size: 64         # 16 per GPU × 4 GPUs
```

**Monitor:** `outputs/logs/stage2_video2pose/curves/` — MSE curves.  
**Checkpoint:** `outputs/checkpoints/stage2_video2pose/best.pt`

---

### Stage 3 — Continuous Recognition (CSLR)

Requires Stage 2 to be complete. Trains on **sentence-level videos** using CTC + CE + knowledge distillation.

**Verify Stage 2 checkpoint exists first:**
```bash
ls outputs/checkpoints/stage2_video2pose/best.pt
```

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

**Key config** (`configs/stage3_cslr.yaml`):
```yaml
model:
  latent_dim: 512        # must match stage1/stage2 latent_dim
  stage2_checkpoint: outputs/checkpoints/stage2_video2pose/best.pt
train:
  epochs: 100            # 132 training sentences — converges fast
  batch_size: 16         # 4 per GPU × 4 GPUs
  grad_accum_steps: 2    # effective batch = 32
data:
  level: sentence
```

**Monitor:** `outputs/logs/stage3_cslr/curves/` — WER curves.  
**Checkpoint:** `outputs/checkpoints/stage3_cslr/best.pt`

> **Note:** `data/sentence_glosses.txt` contains 164 annotated sentences (132 train / 16 dev / 16 test). Stage 3 trains on these only.

---

### Run all three stages sequentially

```bash
bash scripts/train.sh --dataset-root /home/user/data/OSL_Dataset
```

---

## Training Curves and Visualization

Training curves are saved automatically as PNG files after every epoch — **no W&B account needed**:

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

### Stage 1 Report Figures

After Stage 1 training completes, generate visualizations for your report:

```bash
python -m signx.inference.visualize_stage1 \
    --checkpoint outputs/checkpoints/stage1_pose2gloss/best.pt \
    --config    configs/stage1_pose2gloss.yaml \
    --n-samples 5 \
    --split     test
```

Figures saved to `outputs/visualizations/stage1/`:

| File | Contents |
|---|---|
| `pose_features_<ID>.png` | Per-frame pose feature heatmap (time × feature dim) |
| `attention_<ID>.png` | Self-attention map from PoseFusionEncoder layer 1 |
| `predictions_<ID>.png` | Top-5 predicted glosses vs ground truth |
| `feature_norms_<ID>.png` | Feature vector L2 norm over time |
| `method_comparison.png` | P-I accuracy bar chart across pose backends |

To generate the method comparison chart after running all backends:
```bash
python -m signx.inference.visualize_stage1 \
    --method-results mediapipe=0.72 dwpose=0.75 smplerx=0.78 primedepth=0.69 sapiens=0.76 full5=0.82
```

### Enabling Weights & Biases (optional)

In `configs/default.yaml`:
```yaml
logging:
  wandb_enabled: true
  wandb_project: osl-signx
  wandb_entity: your-wandb-username
```

Log in once on the server:
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

Or directly:
```bash
python -m signx.inference.evaluate \
    --dataset-root /home/user/data/OSL_Dataset \
    --stage2-checkpoint outputs/checkpoints/stage2_video2pose/best.pt \
    --stage3-checkpoint outputs/checkpoints/stage3_cslr/best.pt \
    --split test \
    --output outputs/eval_results.json
```

### Metrics

| Metric | Description |
|---|---|
| **WER** | Word Error Rate (Levenshtein / reference length). Lower is better. |
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

## Configuration Reference

All configs are in `configs/`. They use OmegaConf with a `defaults:` chain:

```
stage3_cslr.yaml → default.yaml → paths.yaml
```

### Key options

| Config key | File | Description |
|---|---|---|
| `dataset_root` | `paths.yaml` | Dataset root — override with `DATASET_ROOT` env var |
| `pose.backend` | `default.yaml` | `mediapipe` / `full5` / `precomputed` |
| `pose.full_dim` | `default.yaml` | 1959 (MediaPipe 258 + DWPose 399 + SMPLerX 432 + PrimeDepth 480 + Sapiens 390) |
| `model.latent_dim` | stage configs | Latent space size — must match across all 3 stages (default: 512) |
| `model.pose_input_dim` | `stage1_pose2gloss.yaml` | 258 for mediapipe, 1959 for full5 |
| `train.batch_size` | stage configs | Per-run batch size (splits across GPUs automatically) |
| `train.grad_accum_steps` | stage configs | Gradient accumulation |
| `multi_gpu` | `default.yaml` | `true` = DataParallel across all visible GPUs |
| `use_amp` | `default.yaml` | `true` = fp16 mixed precision |
| `num_workers` | `default.yaml` | DataLoader workers (8 for server, 0 for debugging) |
| `logging.wandb_enabled` | `default.yaml` | Enable W&B logging |
| `checkpoint.save_top_k` | `default.yaml` | Number of best checkpoints to keep |

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

All 25 tests use random dummy data — no dataset or GPU required.

---

## Known Limitations

1. **SMPLer-X and PrimeDepth require manual installation** from GitHub. DWPose requires MMPose. Sapiens and MediaPipe are pip-installable. See the Pose Extraction Backends section for per-modality instructions.

2. **Dataset I/O over WSL**: Reading from the Windows filesystem (`/mnt/c/...`) is slow. Pre-extracting pose features to the Linux filesystem with `extract_poses.sh` and switching to `backend: precomputed` helps significantly. On the server this is not an issue.

3. **Stage 3 sentence dataset is small**: Only 164 annotated sentences (132 train). The model converges in ~50–80 epochs. Watch `val/wer` curves — stop early if validation WER stops improving.

4. **`latent_dim` must be consistent**: All three stage configs must use the same `model.latent_dim`. The default is 512. If you change it, update all three stage configs and retrain from Stage 1.

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
