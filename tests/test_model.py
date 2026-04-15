import torch

from uniindex.layout import mask_logits
from uniindex.model import UnifiedDenoiser


def test_unified_denoiser_shape():
    model = UnifiedDenoiser(
        input_dim=32,
        seq_len=17,
        vocab_size=74,
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        dropout=0.0,
    )
    z_t = torch.randn(3, 17, 32)
    t = torch.rand(3)
    modality_ids = torch.tensor([0] * 16 + [1])
    logits = model(z_t, t, modality_ids)
    assert logits.shape == (3, 17, 74)


def test_unified_denoiser_accepts_positionwise_time():
    model = UnifiedDenoiser(
        input_dim=32,
        seq_len=17,
        vocab_size=74,
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        dropout=0.0,
    )
    z_t = torch.randn(3, 17, 32)
    t = torch.rand(3, 17)
    modality_ids = torch.tensor([0] * 16 + [1])
    logits = model(z_t, t, modality_ids)
    assert logits.shape == (3, 17, 74)


def test_mask_logits_keeps_valid_regions():
    logits = torch.zeros(2, 5, 12)
    masked = mask_logits(logits, image_seq_len=4, text_seq_len=1, codebook_size=10, text_vocab_size=2)
    assert torch.isneginf(masked[:, :4, 10:]).all()
    assert torch.isneginf(masked[:, 4:, :10]).all()
