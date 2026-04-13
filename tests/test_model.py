"""Forward-pass smoke tests for all three SignX stages.

All models and tensors are placed on CUDA when available, falling back to CPU
so tests still pass in environments without a GPU.
"""
from __future__ import annotations

import torch
import pytest

# ---------------------------------------------------------------------------
# Device — CUDA preferred, CPU fallback
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _t(*shape, dtype=torch.float32) -> torch.Tensor:
    """Shorthand: create a random tensor on DEVICE."""
    return torch.randn(*shape, dtype=dtype, device=DEVICE)

def _i(*shape, low=1, high=None) -> torch.Tensor:
    """Shorthand: create a random int64 tensor on DEVICE."""
    return torch.randint(low, high, shape, device=DEVICE)

# ---------------------------------------------------------------------------
# Shared tiny config values
# ---------------------------------------------------------------------------
VOCAB_SIZE = 10
BLANK_ID = 0
BATCH = 2
T_VIDEO = 16    # video frames
T_POSE = 12     # pose frames (slightly different to test temporal mismatch)
H = W = 64
POSE_DIM = 32
LATENT_DIM = 64
CODEBOOK_SIZE = 16
CODEBOOK_DIM = 32
DECODER_LAYERS = 1
HEADS = 2
FUSION_LAYERS = 1
TCONV_CH = [64]
TCONV_K = [3]
TCONV_S = [1]
LSTM_H = 32
LSTM_LAYERS = 1
TRANSF_LAYERS = 1
TRANSF_DIM = 32
TRANSF_HEADS = 2
TRANSF_FFN = 64


# ---------------------------------------------------------------------------
# Stage 1: Pose2Gloss
# ---------------------------------------------------------------------------

def test_stage1_forward():
    from signx.models.signx_model import Stage1Model

    model = Stage1Model(
        pose_input_dim=POSE_DIM,
        latent_dim=LATENT_DIM,
        codebook_size=CODEBOOK_SIZE,
        codebook_dim=CODEBOOK_DIM,
        decoder_layers=DECODER_LAYERS,
        num_heads=HEADS,
        vocab_size=VOCAB_SIZE,
        num_fusion_layers=FUSION_LAYERS,
        dropout=0.0,
        max_target_len=8,
    ).to(DEVICE).eval()

    pose   = _t(BATCH, T_POSE, POSE_DIM)
    target = _i(BATCH, 5, high=VOCAB_SIZE)

    with torch.no_grad():
        out = model(pose, target)

    assert "latent" in out and "logits" in out
    assert out["latent"].shape == (BATCH, T_POSE, LATENT_DIM)
    assert out["logits"].shape  == (BATCH, 5, VOCAB_SIZE)
    assert out["latent"].device.type == DEVICE.type


def test_stage1_generate():
    from signx.models.signx_model import Stage1Model

    model = Stage1Model(
        pose_input_dim=POSE_DIM,
        latent_dim=LATENT_DIM,
        codebook_size=CODEBOOK_SIZE,
        codebook_dim=CODEBOOK_DIM,
        decoder_layers=DECODER_LAYERS,
        num_heads=HEADS,
        vocab_size=VOCAB_SIZE,
        num_fusion_layers=FUSION_LAYERS,
        dropout=0.0,
        max_target_len=8,
    ).to(DEVICE).eval()

    pose = _t(BATCH, T_POSE, POSE_DIM)
    with torch.no_grad():
        ids = model.decoder.generate(
            model.encode_pose(pose), bos_id=0, eos_id=0, max_len=6
        )
    assert ids.shape[0] == BATCH
    assert ids.shape[1] <= 6
    assert ids.device.type == DEVICE.type


# ---------------------------------------------------------------------------
# Stage 2: Video2Pose
# ---------------------------------------------------------------------------

def test_stage2_forward():
    from signx.models.video2pose import Video2PoseModel

    TEST_H = TEST_W = 64
    model = Video2PoseModel(
        vit_name="vit_base_patch16_224",
        vit_pretrained=False,
        latent_dim=LATENT_DIM,
        dropout=0.0,
        img_size=TEST_H,
    ).to(DEVICE).eval()

    video = _t(BATCH, T_VIDEO, 3, TEST_H, TEST_W)
    with torch.no_grad():
        latent = model(video)

    assert latent.shape == (BATCH, T_VIDEO, LATENT_DIM)
    assert latent.device.type == DEVICE.type


# ---------------------------------------------------------------------------
# Stage 3: CSLR
# ---------------------------------------------------------------------------

def _make_stage3() -> "Stage3Model":
    from signx.models.signx_model import Stage3Model
    return Stage3Model(
        latent_dim=LATENT_DIM,
        pruned_dim=LATENT_DIM,
        tconv_channels=TCONV_CH,
        tconv_kernels=TCONV_K,
        tconv_strides=TCONV_S,
        lstm_hidden=LSTM_H,
        lstm_layers=LSTM_LAYERS,
        bidirectional=True,
        transformer_layers=TRANSF_LAYERS,
        transformer_dim=TRANSF_DIM,
        transformer_heads=TRANSF_HEADS,
        transformer_ffn=TRANSF_FFN,
        vocab_size=VOCAB_SIZE,
        dropout_attn=0.0,
        dropout_relu=0.0,
        dropout_res=0.0,
        max_target_len=8,
    ).to(DEVICE).eval()


def test_stage3_ctc_output():
    model = _make_stage3()
    latent  = _t(BATCH, T_VIDEO, LATENT_DIM)
    lengths = torch.tensor([T_VIDEO, T_VIDEO // 2], device=DEVICE)

    with torch.no_grad():
        out = model(latent, lengths=lengths)

    assert "ctc_logits" in out
    B, Tp, V = out["ctc_logits"].shape
    assert B == BATCH
    assert V == VOCAB_SIZE
    assert out["ctc_logits"].device.type == DEVICE.type


def test_stage3_with_decoder_target():
    model  = _make_stage3()
    latent = _t(BATCH, T_VIDEO, LATENT_DIM)
    target = _i(BATCH, 4, high=VOCAB_SIZE)

    with torch.no_grad():
        out = model(latent, target=target)

    assert out["dec_logits"] is not None
    assert out["dec_logits"].shape == (BATCH, 4, VOCAB_SIZE)
    assert out["dec_logits"].device.type == DEVICE.type


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def test_beam_search_greedy():
    from signx.inference.beam_search import ctc_greedy_decode

    log_probs = torch.log_softmax(_t(20, VOCAB_SIZE), dim=-1)
    ids = ctc_greedy_decode(log_probs, blank_id=BLANK_ID)
    assert isinstance(ids, list)
    assert all(i != BLANK_ID for i in ids)


def test_beam_search_decode():
    from signx.inference.beam_search import BeamSearchDecoder

    decoder   = BeamSearchDecoder(vocab_size=VOCAB_SIZE, blank_id=BLANK_ID, beam_size=3)
    log_probs = torch.log_softmax(_t(10, VOCAB_SIZE), dim=-1)
    ids = decoder.decode(log_probs)
    assert isinstance(ids, list)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def test_contrastive_loss():
    from signx.models.losses import contrastive_loss

    a = _t(4, 16)
    b = _t(4, 16)
    loss = contrastive_loss(a, b)
    assert loss.item() >= 0
    assert loss.device.type == DEVICE.type


def test_distillation_loss():
    from signx.models.losses import distillation_loss

    teacher = _t(BATCH, 5, VOCAB_SIZE)
    student = _t(BATCH, 5, VOCAB_SIZE)
    loss = distillation_loss(student, teacher)
    assert loss.item() >= 0
    assert loss.device.type == DEVICE.type


# ---------------------------------------------------------------------------
# Metrics (CPU — these operate on Python lists, no tensors)
# ---------------------------------------------------------------------------

def test_wer_perfect():
    from signx.training.metrics import compute_wer
    assert compute_wer([[1, 2, 3], [4, 5]], [[1, 2, 3], [4, 5]]) == 0.0


def test_wer_all_wrong():
    from signx.training.metrics import compute_wer
    assert compute_wer([[1, 2, 3]], [[4, 5, 6]]) > 0.0


def test_pi_accuracy():
    from signx.training.metrics import compute_pi_accuracy
    assert compute_pi_accuracy([[1, 2, 3]], [[3, 2, 1]]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Feature compiler
# ---------------------------------------------------------------------------

def test_feature_compiler():
    from signx.pose.feature_compiler import PoseAwareFeatureCompiler

    compiler = PoseAwareFeatureCompiler(feature_dim=POSE_DIM, frame_dropout=0.0).to(DEVICE)
    x   = _t(BATCH, T_POSE, POSE_DIM)
    out = compiler(x)
    assert out.shape == x.shape
    assert out.device.type == DEVICE.type


def test_feature_compiler_training_dropout():
    from signx.pose.feature_compiler import PoseAwareFeatureCompiler

    compiler = PoseAwareFeatureCompiler(feature_dim=POSE_DIM, frame_dropout=0.5).to(DEVICE)
    compiler.train()
    x   = _t(BATCH, T_POSE, POSE_DIM)
    out = compiler(x)
    assert out.shape == x.shape
    assert out.device.type == DEVICE.type
