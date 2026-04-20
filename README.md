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

## Dataset Layout on the Server

| Property | Value |
|---|---|
| Language | Omani Sign Language (Arabic glosses) |
| Root directory | `FYPproject/` |
| Word-level videos | `FYPproject/final_split/{train,dev,test}/rgb/*.mp4` |
| Sentence-level videos | `FYPproject/OSL-Sentences/OSL-Sentences/rgb_format/*.mp4` |
| Vocabulary | 801 sign glosses + 1 CTC blank = **802 tokens** |
| Filename format | `{ID}_{SYY}_{TZZ}.mp4` (ID=WordID, SYY=SignerID, TZZ=TakeID) |

Expected tree under `FYPproject/`:

```
FYPproject/
├── final_split/
│   ├── train/rgb/*.mp4
│   ├── dev/rgb/*.mp4
│   └── test/rgb/*.mp4
└── OSL-Sentences/
    └── OSL-Sentences/
        └── rgb_format/*.mp4
```

---

## Quick-Start: Full Training from Scratch

> All commands run in a Linux terminal. Every `$` is a shell prompt — do **not** type the `$`.

```
Step 1  Install project dependencies
Step 2  Install pose extractor dependencies (4 extra backends)
Step 3  Set dataset path
Step 4  Validate dataset paths
Step 5  (Optional) Pre-extract pose features for faster training
Step 6  Train Stage 1 — Pose2Gloss
Step 7  Train Stage 2 — Video2Pose
Step 8  Train Stage 3 — CSLR
Step 9  Evaluate
```

---

## Step 1 — Install Project Dependencies

```bash
$ cd /path/to/OSL_SignX
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
```

Check your CUDA version:

```bash
$ nvcc --version
# or
$ nvidia-smi
```

Install PyTorch matching your CUDA version (replace `cu121` with `cu118`, `cu124`, etc.):

```bash
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install all project requirements:

```bash
$ pip install -r requirements.txt
$ pip install -e .
```

`requirements.txt` installs:

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Deep learning framework |
| `numpy`, `scipy` | Numerical computing |
| `decord`, `opencv-python`, `Pillow` | Video and image I/O |
| `mediapipe` | Pose extraction backend 1 (default) |
| `PyYAML`, `omegaconf` | Config loading |
| `tqdm` | Progress bars |
| `wandb` | Optional experiment tracking |
| `editdistance`, `sacrebleu` | WER and BLEU metrics |
| `einops`, `timm` | Model utilities (ViT backbone) |

---

## Step 2 — Install Pose Extractor Backends

The system supports five pose modalities. **MediaPipe is already installed** via `requirements.txt`. Install the remaining four for `full5` mode (1959-dim), which matches the original SignX paper.

### Check which backends are ready

```bash
$ python3 -c "
from signx.pose import check_full5_available
for name, ok in check_full5_available().items():
    print(f'  {name:12s}: {\"ready\" if ok else \"NOT installed\"}')"
```

---

### Backend 2 — DWPose (whole-body 133 keypoints, 399-dim)

```bash
$ pip install openmim
$ python3 -m mim install mmengine
$ python3 -m mim install "mmcv>=2.0.0"
$ python3 -m mim install mmdet
$ python3 -m mim install "mmpose>=1.0.0"
```

Verify:

```bash
$ python3 -c "import mmpose; print('DWPose ready')"
```

---

### Backend 3 — SMPLer-X (3-D body mesh joints, 432-dim)

Clone the repo **outside** of OSL_SignX, then install:

```bash
$ cd ~
$ git clone https://github.com/caizhongang/SMPLer-X.git
$ cd SMPLer-X
$ pip install -v -e .
$ cd /path/to/OSL_SignX
```

Download the SMPL-X body model:
1. Register at <https://smpl-x.is.tue.mpg.de/> and download `SMPLX_NEUTRAL.npz`
2. Place it at `~/SMPLer-X/data/body_models/smplx/SMPLX_NEUTRAL.npz` (or wherever SMPLer-X expects it)

Download the SMPLer-X checkpoint:

```bash
# Instructions and links are in the SMPLer-X repo README
# After downloading, set the path:
$ export SMPLERX_CKPT=/path/to/smplerx_checkpoint.pth.tar
# Add to ~/.bashrc to make it permanent:
$ echo 'export SMPLERX_CKPT=/path/to/smplerx_checkpoint.pth.tar' >> ~/.bashrc
```

Verify:

```bash
$ python3 -c "import smpler_x; print('SMPLer-X ready')"
```

---

### Backend 4 — PrimeDepth / depth-pro (monocular depth features, 480-dim)

```bash
$ cd ~
$ git clone https://github.com/apple/ml-depth-pro.git
$ cd ml-depth-pro
$ pip install -e .
$ python3 -m depth_pro.cli.download_models
$ cd /path/to/OSL_SignX
```

Verify:

```bash
$ python3 -c "import depth_pro; print('PrimeDepth ready')"
```

---

### Backend 5 — Sapiens (dense body keypoints, 390-dim)

```bash
$ pip install transformers accelerate
```

Model weights (~1.2 GB) download automatically on first use.

Verify:

```bash
$ python3 -c "from transformers import pipeline; print('Sapiens ready')"
```

---

### Re-check all backends

```bash
$ python3 -c "
from signx.pose import check_full5_available
for name, ok in check_full5_available().items():
    print(f'  {name:12s}: {\"ready\" if ok else \"NOT installed\"}')"
```

Expected output when all are installed:

```
  mediapipe   : ready
  dwpose      : ready
  smplerx     : ready
  primedepth  : ready
  sapiens     : ready
```

---

### Switch pose backend

Edit `configs/default.yaml`:

```yaml
pose:
  backend: mediapipe     # single modality — 258-dim (default, lightest)
  # backend: full5       # all 5 modalities — 1959-dim (matches paper)
  # backend: precomputed # load pre-extracted .pt files (fastest for training)
```

When switching to `full5`, also update `pose_input_dim` in `configs/stage1_pose2gloss.yaml`:

```yaml
model:
  pose_input_dim: 1959   # 258 for mediapipe, 1959 for full5
```

---

## Step 3 — Set the Dataset Path

Set `DATASET_ROOT` to the directory that contains `final_split/` and `OSL-Sentences/`:

```bash
$ export DATASET_ROOT=/home/<your_username>/FYPproject
# Make it permanent:
$ echo 'export DATASET_ROOT=/home/<your_username>/FYPproject' >> ~/.bashrc
$ source ~/.bashrc
```

Verify the variable is set:

```bash
$ echo $DATASET_ROOT
/home/<your_username>/FYPproject
```

---

## Step 4 — Validate Dataset Paths

```bash
$ bash scripts/prepare_dataset.sh
```

This checks that all expected directories and at least a few video files are present.

### Vocabulary files

Already in the repo under `data/`:

```
data/gloss_vocab.txt         802 lines — "0000 <blank>", "0001 مستشفى", ...
data/Words.txt               800 lines — WordID → Arabic gloss
data/Sentences.txt           443 lines — SentenceID → Arabic sentence text
data/sentence_glosses.txt    164 annotated sentences for Stage 3
```

---

## Step 5 — (Optional) Pre-extract Pose Features

Run once before training to cache pose features on disk. This avoids re-running the pose extractor every epoch and gives ~5× faster training.

```bash
$ source .venv/bin/activate
$ export DATASET_ROOT=/home/<your_username>/FYPproject

$ SPLIT=train bash scripts/extract_poses.sh
$ SPLIT=dev   bash scripts/extract_poses.sh
$ SPLIT=test  bash scripts/extract_poses.sh
```

Then set `backend: precomputed` in `configs/default.yaml`:

```yaml
pose:
  backend: precomputed
  precomputed_dir: outputs/cache/pose_train
```

---

## GPU and Memory Settings

All configs are already tuned for a **single 24 GB GPU** with fp16 AMP enabled.

| Stage | `batch_size` | `grad_accum_steps` | Effective batch |
|---|---|---|---|
| Stage 1 | 32 | 4 | 128 |
| Stage 2 | 16 | 4 | 64 |
| Stage 3 | 4 | 8 | 32 |

Key settings already applied in `configs/default.yaml`:

```yaml
multi_gpu: false   # single GPU
use_amp:   true    # fp16 — ~2× speed, ~50% VRAM
num_workers: 4
```

### If you still get an OOM error

Work through this checklist in order — stop as soon as it clears:

| Step | Change | File |
|---|---|---|
| 1 | Halve `batch_size` | stage config |
| 2 | Double `grad_accum_steps` to keep effective batch the same | stage config |
| 3 | Confirm `use_amp: true` | `configs/default.yaml` |
| 4 | Reduce `video.max_frames` from 256 to 128 | `configs/default.yaml` |
| 5 | Reduce `num_workers` to 2 or 0 | `configs/default.yaml` |

---

## Step 6 — Train Stage 1 (Pose2Gloss)

Trains the latent space from pose features using **word-level videos**.

```bash
$ source .venv/bin/activate
$ export DATASET_ROOT=/home/<your_username>/FYPproject
$ python -m signx.training.train_stage1 --config configs/stage1_pose2gloss.yaml
```

**Monitor:** `outputs/logs/stage1_pose2gloss/curves/` — PNG loss and accuracy plots saved every epoch.  
**Checkpoint:** `outputs/checkpoints/stage1_pose2gloss/best.pt`

Key config (`configs/stage1_pose2gloss.yaml`):

```yaml
model:
  pose_input_dim: 258    # 258 for mediapipe / 1959 for full5
  latent_dim: 512
train:
  epochs: 800
  batch_size: 32         # single 24 GB GPU; effective batch = 128
  grad_accum_steps: 4
data:
  level: word
```

---

## Step 7 — Train Stage 2 (Video2Pose)

Freezes Stage 1 and trains a ViT to predict pose latents from RGB frames.

Verify Stage 1 completed first:

```bash
$ ls outputs/checkpoints/stage1_pose2gloss/best.pt
```

Then train:

```bash
$ source .venv/bin/activate
$ export DATASET_ROOT=/home/<your_username>/FYPproject
$ python -m signx.training.train_stage2 --config configs/stage2_video2pose.yaml
```

**Monitor:** `outputs/logs/stage2_video2pose/curves/` — MSE curves.  
**Checkpoint:** `outputs/checkpoints/stage2_video2pose/best.pt`

Key config (`configs/stage2_video2pose.yaml`):

```yaml
model:
  latent_dim: 512        # must match stage1
  stage1_checkpoint: outputs/checkpoints/stage1_pose2gloss/best.pt
train:
  epochs: 300
  batch_size: 16         # single 24 GB GPU; effective batch = 64
  grad_accum_steps: 4
```

---

## Step 8 — Train Stage 3 (CSLR)

Trains continuous recognition on **sentence-level videos** using CTC + CE + knowledge distillation.

Verify Stage 2 completed first:

```bash
$ ls outputs/checkpoints/stage2_video2pose/best.pt
```

Then train:

```bash
$ source .venv/bin/activate
$ export DATASET_ROOT=/home/<your_username>/FYPproject
$ python -m signx.training.train_stage3 --config configs/stage3_cslr.yaml
```

**Monitor:** `outputs/logs/stage3_cslr/curves/` — WER curves.  
**Checkpoint:** `outputs/checkpoints/stage3_cslr/best.pt`

Key config (`configs/stage3_cslr.yaml`):

```yaml
model:
  latent_dim: 512        # must match stage1/stage2
  stage2_checkpoint: outputs/checkpoints/stage2_video2pose/best.pt
train:
  epochs: 100
  batch_size: 4          # single 24 GB GPU; effective batch = 32
  grad_accum_steps: 8
data:
  level: sentence
```

> **Note:** `data/sentence_glosses.txt` has 164 annotated sentences (132 train / 16 dev / 16 test). Stage 3 converges in ~50–80 epochs.

---

## Run All Three Stages Sequentially

```bash
$ bash scripts/train.sh --dataset-root /home/<your_username>/FYPproject
```

---

## Step 9 — Evaluate

```bash
$ source .venv/bin/activate
$ export DATASET_ROOT=/home/<your_username>/FYPproject
$ python -m signx.inference.evaluate \
    --dataset-root $DATASET_ROOT \
    --stage2-checkpoint outputs/checkpoints/stage2_video2pose/best.pt \
    --stage3-checkpoint outputs/checkpoints/stage3_cslr/best.pt \
    --split test \
    --output outputs/eval_results.json
```

Or via the convenience script:

```bash
$ bash scripts/evaluate.sh \
    --stage2-checkpoint outputs/checkpoints/stage2_video2pose/best.pt \
    --stage3-checkpoint outputs/checkpoints/stage3_cslr/best.pt
```

### Metrics

| Metric | Description |
|---|---|
| **WER** | Word Error Rate (Levenshtein / reference length). Lower is better. |
| **BLEU** | Corpus-level BLEU-4 (sacrebleu). Higher is better. |
| **P-I Accuracy** | Position-Independent token match rate. Higher is better. |

---

## Training Curves

Saved automatically as PNG files after every epoch — no W&B account needed:

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

### Stage 1 Report Visualizations

After Stage 1 completes:

```bash
$ python -m signx.inference.visualize_stage1 \
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
$ python -m signx.inference.visualize_stage1 \
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
$ wandb login
```

---

## Inference on a Single Video

```bash
$ python -m signx.inference.predict \
    --video path/to/video.mp4 \
    --stage2-checkpoint outputs/checkpoints/stage2_video2pose/best.pt \
    --stage3-checkpoint outputs/checkpoints/stage3_cslr/best.pt \
    --beam-size 8 \
    --device cuda
```

---

## Configuration Reference

All configs live in `configs/`. They use OmegaConf with a `defaults:` chain:

```
stage3_cslr.yaml → default.yaml → paths.yaml
```

| Config key | File | Description |
|---|---|---|
| `dataset_root` | `paths.yaml` | Dataset root — override with `DATASET_ROOT` env var |
| `pose.backend` | `default.yaml` | `mediapipe` / `full5` / `precomputed` |
| `pose.full_dim` | `default.yaml` | 1959 (MediaPipe 258 + DWPose 399 + SMPLerX 432 + PrimeDepth 480 + Sapiens 390) |
| `model.latent_dim` | stage configs | Must match across all 3 stages (default: 512) |
| `model.pose_input_dim` | `stage1_pose2gloss.yaml` | 258 for mediapipe, 1959 for full5 |
| `train.batch_size` | stage configs | Per-run batch size |
| `train.grad_accum_steps` | stage configs | Gradient accumulation |
| `multi_gpu` | `default.yaml` | false = single GPU |
| `use_amp` | `default.yaml` | true = fp16 mixed precision |
| `num_workers` | `default.yaml` | DataLoader workers |
| `logging.wandb_enabled` | `default.yaml` | Enable W&B logging |

---

## Running Tests

```bash
$ source .venv/bin/activate
$ pytest tests/ -v
```

All 25 tests use random dummy data — no dataset or GPU required.

---

## Known Limitations

1. **SMPLer-X and PrimeDepth require manual installation** from GitHub. DWPose requires MMPose (via `openmim`). Sapiens and MediaPipe are pip-installable.

2. **Stage 3 sentence dataset is small**: Only 164 annotated sentences (132 train). Watch `val/wer` curves — stop early if validation WER stops improving after ~80 epochs.

3. **`latent_dim` must be consistent**: All three stage configs must use the same `model.latent_dim`. The default is 512. If you change it, update all three configs and retrain from Stage 1.

4. **Sentence split**: The sentence videos in `OSL-Sentences/OSL-Sentences/rgb_format/` are not pre-split. The code assigns them to train/dev/test by signer ID automatically. Once you have a manual split, place folders under `OSL-Sentences/` following the same `{train,dev,test}/rgb/` layout as `final_split/` and update `sentence_dir` in `configs/paths.yaml`.

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
