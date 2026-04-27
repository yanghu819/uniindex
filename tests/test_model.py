import torch

from uniindex.layout import TaskLayout, mask_logits
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


def test_unified_denoiser_forward_features_shape():
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
    features = model.forward_features(z_t, t, modality_ids)
    assert features.shape == (3, 17, 64)


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


def test_image_summary_to_text_gate_tracks_i2t_conditioning():
    model = UnifiedDenoiser(
        input_dim=32,
        seq_len=6,
        vocab_size=74,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        dropout=0.0,
        image_summary_to_text=True,
    )
    modality_ids = torch.tensor([0, 0, 0, 0, 1, 1])

    i2t_t = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.25, 0.25]])
    t2i_t = torch.tensor([[0.25, 0.25, 0.25, 0.25, 1.0, 1.0]])

    assert torch.allclose(model._image_summary_gate(i2t_t, modality_ids), torch.tensor([0.75]))
    assert torch.allclose(model._image_summary_gate(t2i_t, modality_ids), torch.tensor([0.0]))


def test_unified_denoiser_can_return_image_semantic_tokens():
    model = UnifiedDenoiser(
        input_dim=32,
        seq_len=6,
        vocab_size=74,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        dropout=0.0,
        image_semantic_tokens=2,
    )
    z_t = torch.randn(3, 6, 32)
    t = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.25, 0.25]]).expand(3, -1)
    modality_ids = torch.tensor([0, 0, 0, 0, 1, 1])

    default_features = model.forward_features(z_t, t, modality_ids)
    extended_features = model.forward_features(z_t, t, modality_ids, include_extra_tokens=True)

    assert default_features.shape == (3, 6, 32)
    assert extended_features.shape == (3, 8, 32)
    assert model(z_t, t, modality_ids).shape == (3, 6, 74)


def test_mask_logits_keeps_valid_regions():
    layout = TaskLayout(image_seq_len=4, text_seq_len=1, codebook_size=10, text_vocab_size=2)
    logits = torch.zeros(2, layout.seq_len, layout.vocab_size)
    masked = mask_logits(logits, layout=layout)
    assert torch.isneginf(masked[:, :4, 10:]).all()
    assert torch.isneginf(masked[:, 4:, :10]).all()
