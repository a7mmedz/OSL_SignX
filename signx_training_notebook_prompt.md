# TASK: Create a Complete SignX Training Notebook for Omani Sign Language

Read through the entire OSL_SignX project codebase first. Understand every file, every config, every model component. Then create ONE comprehensive Jupyter notebook that covers the ENTIRE pipeline from environment setup to final evaluation.

## CRITICAL INSTRUCTIONS

1. **Read the full codebase first** — run `find /path/to/OSL_SignX -name "*.py" -o -name "*.yaml" | head -80` and read every relevant file before writing any code. Understand what has already been built, what configs exist, what model dimensions are used.

2. **The notebook must be SELF-CONTAINED** — someone should be able to open it and run cell by cell from top to bottom. Every cell must have a markdown header explaining what it does and why.

3. **Use the existing codebase** — import from the `signx` package that's already built. Do NOT rewrite the models from scratch. The notebook orchestrates the existing code.

4. **Save the notebook** to `OSL_SignX/notebooks/full_training_pipeline.ipynb`

---

## NOTEBOOK STRUCTURE (follow this EXACTLY)

### Part 0: Environment Setup
```
- Cell: Check Python version, CUDA availability, GPU info (name, memory, compute capability)
- Cell: Install/verify all dependencies using `uv pip install` (reference requirements.txt)
- Cell: Add project root to sys.path so `import signx` works
- Cell: Set all random seeds (torch, numpy, random) for reproducibility
- Cell: Print full environment summary table (package versions, GPU, OS, paths)
```

### Part 1: Dataset Exploration & Validation
```
- Cell: Load and display configs from paths.yaml — show all dataset paths
- Cell: Verify dataset paths exist, count files per split:
    - final_split/train/rgb/ → count .mp4 files
    - final_split/dev/rgb/ → count .mp4 files  
    - final_split/test/rgb/ → count .mp4 files
    - dataset/OSL-Sentences/rgb_format/ → count .mp4 files
- Cell: Load gloss_vocab.txt — display vocab size, sample entries, Arabic text verification
- Cell: Load Words.txt — show word distribution stats
- Cell: Load Sentences.txt — analyze sentence lengths, token stats
- Cell: Dataset statistics TABLE:
    | Split | Videos | Unique Words | Unique Signers | Avg Duration | Total Duration |
- Cell: Video duration analysis — histogram of video lengths per split
- Cell: Class distribution analysis — bar chart of samples per gloss (top 30 and bottom 30)
- Cell: Sample 5 random videos from train set — display:
    - Video metadata (resolution, fps, duration, frame count)
    - Thumbnail grid (first, middle, last frame)
    - Associated gloss label in Arabic
- Cell: Signer distribution analysis — how many videos per signer per split
```

### Part 2: Pose Extraction Demo (All 5 Methods)
```
This section demonstrates ALL 5 pose extraction methods from the SignX paper on a SINGLE sample video.

- Markdown cell explaining: SignX uses 5 complementary pose extraction methods to build a rich pose representation. Each captures different aspects of the signer's body.

- Cell: Select one clear sample video from the dataset (pick one with good lighting)
- Cell: Load the video, extract frames, display a representative frame

For EACH of the 5 methods below, create cells that:
  1. Install/import the method
  2. Run extraction on the sample video
  3. Visualize the output overlaid on the original frame
  4. Print the output dimensions and explain what they capture

- **Method 1: MediaPipe** (258-dim per frame)
    - Extract 33 body landmarks + 21 per hand + face mesh
    - Visualize: skeleton overlay on frame, hand landmarks closeup
    - Output: [T × 258] — show shape, sample values

- **Method 2: DWPose** (384-dim per frame)
    - 18 body + 21×2 hand + 68 facial keypoints + confidence scores
    - Visualize: full keypoint overlay with confidence coloring
    - Output: [T × 384] — show shape, sample values
    - Note: requires `pip install dwpose` or equivalent. If not available, explain installation and show expected output format.

- **Method 3: SMPLer-X** (165-dim per frame)
    - 3D body mesh parameters (pose + shape + expression)
    - Visualize: 3D mesh overlay or skeleton projection
    - Output: [T × 165] — show shape, sample values
    - Note: This is heavy. If GPU memory is insufficient, show how to run it and display expected output format with a placeholder.

- **Method 4: PrimeDepth** (576-dim per frame)
    - Monocular depth estimation
    - Visualize: depth map colorized (jet colormap) side by side with RGB
    - Output: [T × 576] — show shape, sample values
    - Note: Same as above — show installation and expected format if cannot run.

- **Method 5: Sapiens Segmentation** (576-dim per frame)
    - Fine-grained human body part segmentation
    - Visualize: segmentation mask with body part colors overlaid on frame
    - Output: [T × 576] — show shape, sample values

- Cell: COMPARISON TABLE:
    | Method | Dimensions | What It Captures | Speed (FPS) | GPU Memory |
    |--------|-----------|------------------|-------------|------------|
    | MediaPipe | 258 | 3D joints, hands, face | ~30 | CPU only |
    | DWPose | 384 | 2D keypoints + confidence | ~15 | ~2GB |
    | SMPLer-X | 165 | 3D body mesh params | ~3 | ~8GB |
    | PrimeDepth | 576 | Scene depth | ~10 | ~4GB |
    | Sapiens | 576 | Body part segmentation | ~8 | ~6GB |

- Cell: Show the CONCATENATED pose vector [1959-dim] for one frame
- Cell: Visualize all 5 methods side by side in a 2×3 grid (original + 5 methods) — THIS IS KEY, make it look like Figure 1 from the SignX paper

IMPORTANT: For methods that are too heavy to install/run in the current environment:
  - Still show the code that WOULD run them
  - Show the expected output tensor shapes
  - If pre-extracted .pkl files exist in the dataset, load and visualize those
  - Wrap heavy methods in try/except so the notebook doesn't crash
  - Mark cells as optional with "[OPTIONAL - requires X GPU memory]"
```

### Part 3: Data Pipeline Setup
```
- Cell: Initialize the Dataset class from signx.data.dataset for train/dev/test
- Cell: Show a batch — visualize what the dataloader produces:
    - Input tensor shape
    - Label tensor shape
    - Sample frames from the batch
- Cell: Show data augmentation examples — display same video with different augmentations side by side
- Cell: Verify collate function works with variable-length sequences
- Cell: Benchmark dataloader speed (samples/sec, batches/sec)
```

### Part 4: Model Architecture Inspection
```
- Cell: Instantiate the full SignX model from existing code
- Cell: Print model architecture summary (use torchinfo or manual)
    - Total parameters
    - Parameters per stage
    - Memory footprint estimate
- Cell: Architecture diagram — create a visual flow diagram showing:
    Input Video → ViT Backbone → Frame Features [768-dim]
    → Temporal Transformer → Pose Features [2048-dim]
    → ResNet34 → TemporalConv → BiLSTM [1024-dim]
    → Transformer Encoder-Decoder → Beam Search → Gloss Sequence
- Cell: Test forward pass with DUMMY random data (do not need real dataset):
    - Random tensor [batch=2, T=30, 3, 224, 224]
    - Verify output shape matches expected gloss sequence
    - Print intermediate tensor shapes at each stage
- Cell: Verify each sub-model independently:
    - Pose fusion module: input [B, T, 1959] → output [B, T, 2048]
    - Video2Pose module: input [B, T, 3, 224, 224] → output [B, T, 2048]
    - Temporal model: input [B, T, 2048] → output [B, T', 1024]
    - Transformer decoder: input [B, T', 1024] → output gloss sequence
```

### Part 5: Stage 1 Training — Pose2Gloss
```
- Markdown: Explain Stage 1 — we train the pose fusion layer to convert pose features into meaningful gloss text, ensuring the latent space is semantically meaningful.

- Cell: Load Stage 1 config from configs/stage1_pose2gloss.yaml, display all hyperparameters
- Cell: Setup training components:
    - Model (pose encoder + codebook decoder)
    - Optimizer (AdamW, lr=1e-3, weight_decay=0.01)
    - Scheduler (cosine decay)
    - Loss functions (text_loss + word_match_loss + contrastive_loss, all λ=1)
- Cell: OOM Prevention settings:
    - Set appropriate batch size based on available GPU memory
    - Enable gradient accumulation if needed
    - Enable mixed precision (torch.cuda.amp) if available
    - Set appropriate max_frames based on GPU memory
    - Print estimated memory usage before training starts
- Cell: Training loop with FULL logging:
    - Progress bar (tqdm)
    - Every N steps: log loss components (text_loss, word_match_loss, contrastive_loss, total_loss)
    - Every epoch: run validation, compute metrics
    - Save best checkpoint by validation loss
    - Save training curves data to a JSON file for later visualization
- Cell: TRAINING VISUALIZATION (run after training or load from saved data):
    - Loss curves: 4 subplots (total, text, word_match, contrastive) vs steps
    - Learning rate schedule plot
    - Validation loss vs epoch
    - Training vs validation loss comparison (overfitting check)
- Cell: Stage 1 results table:
    | Metric | Train | Val |
    |--------|-------|-----|
    | Total Loss | x.xx | x.xx |
    | Text Loss | x.xx | x.xx |
    | Word Match Loss | x.xx | x.xx |
    | Contrastive Loss | x.xx | x.xx |
```

### Part 6: Stage 2 Training — Video2Pose
```
- Markdown: Explain Stage 2 — freeze Stage 1, train ViT to predict pose features from raw RGB video.

- Cell: Load Stage 1 checkpoint, freeze all Stage 1 parameters
- Cell: Setup Stage 2 training:
    - Video2Pose model (ViT backbone + temporal transformer)
    - Optimizer (AdamW)
    - Loss: MSE(predicted_pose, gt_pose)
- Cell: Training loop with logging (similar to Stage 1)
- Cell: TRAINING VISUALIZATION:
    - MSE loss curve vs steps
    - Predicted vs ground-truth pose feature comparison (scatter plot of a few dimensions)
    - t-SNE visualization of predicted pose features colored by gloss label
- Cell: Stage 2 results table
```

### Part 7: Stage 3 Training — Continuous Sign Language Recognition
```
- Markdown: Explain Stage 3 — the core CSLR module. Temporal modeling + sequence refinement in latent space.

- Cell: Load Stage 2 checkpoint, setup Stage 3 components:
    - ResNet34 backbone + TemporalConv + BiLSTM
    - Transformer encoder-decoder (6 layers, dim 256)
    - CTC loss + cross-entropy + KD loss + latent regularizer
    - Noam scheduler (warmup=4000)
    - Beam search decoder (beam=8)
- Cell: Feature processing setup:
    - Adaptive feature pruning (monitor dimension variance, prune every K iterations)
    - Multi-scale temporal augmentation (10-fold: temporal scaling, spatial jitter, Gaussian noise)
    - Covariance whitening
    - Stochastic frame dropping
- Cell: Training loop with COMPREHENSIVE logging:
    - Loss components: CTC loss, CE loss, KD loss, latent regularizer
    - Metrics computed every N steps: WER on dev set
    - Checkpoint averaging (top-5 by validation WER)
    - Gradient norm monitoring
    - Feature pruning stats (how many dimensions active)
- Cell: TRAINING VISUALIZATIONS:
    - Loss curves (all components + total)
    - WER curve over training (dev set)
    - Learning rate schedule (Noam warmup)
    - Gradient norm over training
    - Active feature dimensions over training (pruning effect)
    - Attention heatmap from transformer decoder (sample)
    - CTC alignment visualization (sample video → predicted vs ground truth)
```

### Part 8: Evaluation & Metrics
```
- Cell: Load best checkpoint (averaged top-5)
- Cell: Run full evaluation on DEV set
- Cell: Run full evaluation on TEST set
- Cell: COMPREHENSIVE RESULTS TABLE (like Tables 1, 3, 4 from paper):
    | Dataset | Split | WER↓ | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE |
    |---------|-------|------|--------|--------|--------|--------|-------|
    | OSL     | Dev   | xx.x | xx.xx  | xx.xx  | xx.xx  | xx.xx  | xx.xx |
    | OSL     | Test  | xx.x | xx.xx  | xx.xx  | xx.xx  | xx.xx  | xx.xx |

- Cell: Per-class accuracy analysis:
    - Top 10 best recognized glosses (with accuracy)
    - Top 10 worst recognized glosses (with accuracy)
    - Confusion matrix heatmap (top 30 most common glosses)
- Cell: Error analysis:
    - Most common substitution errors (gloss A predicted as gloss B)
    - Most common insertion errors
    - Most common deletion errors
    - Error examples with video thumbnails
- Cell: Per-Instance (P-I) accuracy computation (for isolated word recognition)
- Cell: Inference speed benchmark:
    | Method | FPS↑ | Power (W)↓ | WER (%)↓ |
    (Compare latent space vs pixel space if possible)
```

### Part 9: Qualitative Results & Visualization
```
- Cell: Select 10 test samples — show for each:
    - Video thumbnail strip (key frames)
    - Ground truth gloss sequence (Arabic)
    - Predicted gloss sequence (Arabic)
    - Color-coded comparison (green=correct, red=error, yellow=substitution)
    - Like Table 2 in the paper

- Cell: Word-to-Frame Alignment visualization (like Figure 7 in paper):
    - Show attention peaks mapping generated words to video frames
    - Three-layer alignment: Gloss ↔ Feature Index ↔ Original Frames

- Cell: Pose dimension importance heatmap (like Figure 8):
    - Normalized importance of all 1959 pose dimensions across 5 pose types
    - Separated by red boundary lines for each pose type

- Cell: Latent space visualization:
    - t-SNE plot of latent features colored by gloss category
    - Cluster analysis — do similar signs cluster together?

- Cell: Attention map visualization from transformer decoder:
    - Cross-attention weights showing which latent features the decoder attends to
    - Self-attention patterns

- Cell: Training efficiency plot (like Figure 5):
    - Track convergence of loss components across training periods (Early/Middle/Late/Stable)
    - Temperature sensitivity analysis if applicable
```

### Part 10: Ablation Studies
```
- Cell: Component ablation (like Figure 4):
    - SLR only vs Full Pipeline (Latent+SLR+Refinement)
    - RGB-only latent vs With Pose
    - Latent vs Latent+SLR
    - Latent vs Refined Latent
    - Bar chart comparison for each metric (BLEU-1 through BLEU-4, ROUGE)

- Cell: Modality ablation (like Table 9):
    - Zero out each pose modality one at a time
    - Measure WER impact
    | Ablated Modality | WER (%) | ΔWER | Relative Degradation |
    |-----------------|---------|------|---------------------|
    | Full Pipeline   | xx.xx   | –    | –                   |
    | w/o SMPLer-X    | xx.xx   | +x.x | x.x%               |
    | w/o DWPose      | xx.xx   | +x.x | x.x%               |
    | ...             |         |      |                     |

NOTE: If full ablations take too long, structure the code so it CAN be run later. Use flags/configs to control which ablation to run. Save results to JSON for later visualization.
```

### Part 11: Export & Summary
```
- Cell: Save all results to a structured JSON file:
    {
        "training_history": {...},
        "eval_results": {...},
        "ablation_results": {...},
        "model_info": {...}
    }
- Cell: Generate a final summary report (print formatted):
    - Best model performance
    - Training time per stage
    - Total parameters
    - Key findings
- Cell: Export best model for inference:
    - Save model weights
    - Save vocab file
    - Save config
    - Print inference command example
```

---

## CRITICAL TECHNICAL REQUIREMENTS

### OOM Prevention (GPU Memory Management)
The notebook MUST handle limited GPU memory gracefully:
- Start each training section with GPU memory check
- Auto-calculate safe batch size based on available GPU memory
- Use `torch.cuda.amp.autocast()` for mixed precision training
- Use `torch.cuda.empty_cache()` between stages
- Use gradient accumulation for effective larger batch sizes
- Set `max_frames` config to limit video length based on GPU memory
- Add memory monitoring cells between major sections
- If a cell would OOM, catch the error, suggest reduced settings, and continue

### Visualization Requirements
- Use matplotlib with a clean, professional style (set style at the top)
- All plots must have: title, axis labels, legend, grid
- Use Arabic-compatible font for any text displaying Arabic glosses
- Save all figures to `notebooks/figures/` directory
- Make figures publication-quality (300 DPI, appropriate size)
- For Arabic text in matplotlib, add: `plt.rcParams['font.family'] = 'DejaVu Sans'` or use a font that supports Arabic

### Robustness
- Every training cell should have try/except for graceful error handling
- Auto-save checkpoints every N steps (not just every epoch)
- If training is interrupted, the notebook should be able to resume from the last checkpoint
- Add timing to every major cell (time.time() start/end, print duration)
- Log everything to both console and a log file

### Dataset Path
The dataset is at: `/mnt/c/Users/MOBPC/Downloads/FYP/FYPproject/OSL_Dataset`
DO NOT copy the dataset. Read directly from this path.
If this path doesn't work, add a cell at the top where the user can set the correct path.

---

## WHAT ALREADY EXISTS IN THE CODEBASE

Read the existing `OSL_SignX/` project that was already built. It contains:
- `signx/models/` — all model components (ViT, pose fusion, codebook decoder, temporal model, transformer decoder, losses)
- `signx/data/` — dataset classes, transforms, vocab handling, collate functions
- `signx/pose/` — pose extraction (mediapipe, feature compiler)
- `signx/training/` — training loops for all 3 stages, scheduler, metrics
- `signx/inference/` — beam search, prediction, evaluation
- `configs/` — YAML configs for all stages and paths
- `data/` — vocab files, word lists, sentence lists

The notebook should IMPORT from these modules, not reimplement them. If something is missing or broken in the existing code, fix it in the notebook or note what needs to be fixed.

---

## FINAL NOTES

- The notebook will be long. That's expected. Use clear section headers and markdown explanations.
- Prioritize WORKING code over perfect code. If something might fail, add fallbacks.
- The notebook should work even if only MediaPipe is available for pose extraction (the other 4 methods are optional/heavy).
- Display progress and status at every step — the user should always know what's happening.
- After creating the notebook, verify it by reading through it one more time to check for import errors, path issues, or logical inconsistencies.
