from types import SimpleNamespace

import torch
import torch.nn as nn

from uniindex.i2t_llm_decoder import SoftPrefixAdapter, freeze_module, score_candidate_token_ids
from uniindex.i2t_llm_decoder import extract_i2t_image_features
from uniindex.layout import TaskLayout
from uniindex.model import UnifiedDenoiser


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int, preferred_tokens: list[int]) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.preferred_tokens = preferred_tokens

    def forward(self, inputs_embeds, attention_mask=None):
        del attention_mask
        batch, seq_len, _ = inputs_embeds.shape
        logits = torch.full((batch, seq_len, self.vocab_size), -10.0, device=inputs_embeds.device)
        start = 2
        for offset, token_id in enumerate(self.preferred_tokens):
            logits[:, start + offset, token_id] = 10.0
        return SimpleNamespace(logits=logits)


class FixedFeatureDenoiser(nn.Module):
    def __init__(self, layout: TaskLayout, hidden_dim: int) -> None:
        super().__init__()
        self.layout = layout
        self.hidden_dim = hidden_dim

    def forward_features(self, z_t, t, modality_ids):
        del t, modality_ids
        hidden = torch.zeros(z_t.shape[0], self.layout.seq_len, self.hidden_dim, device=z_t.device)
        hidden[:, self.layout.image_slice] = 1.0
        hidden[:, self.layout.text_slice] = 3.0
        return hidden


def test_soft_prefix_adapter_shape():
    adapter = SoftPrefixAdapter(input_dim=4, hidden_dim=7, prefix_tokens=3, output_dim=5)
    output = adapter(torch.randn(2, 4))
    assert output.shape == (2, 3, 5)


def test_freeze_module_disables_trainable_parameters():
    module = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    stats = freeze_module(module)
    assert stats["total_parameters"] > 0
    assert stats["trainable_parameters"] == 0
    assert not any(parameter.requires_grad for parameter in module.parameters())


def test_score_candidate_token_ids_prefers_matching_candidate():
    embedding = nn.Embedding(16, 6)
    model = FakeCausalLM(vocab_size=16, preferred_tokens=[4, 5, 6])
    prefix = torch.randn(2, 2, 6)
    prompt = torch.tensor([3])
    candidates = torch.tensor(
        [
            [4, 5, 6],
            [4, 9, 6],
            [8, 5, 6],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([3, 3, 3], dtype=torch.long)

    scores = score_candidate_token_ids(
        model=model,
        embedding=embedding,
        prefix_embeds=prefix,
        prompt_token_ids=prompt,
        candidate_token_ids=candidates,
        candidate_lengths=lengths,
    )

    assert scores.shape == (2, 3)
    assert scores.argmax(dim=1).tolist() == [0, 0]


def test_extract_i2t_image_features_returns_backward_compatible_tensor():
    layout = TaskLayout(image_seq_len=2, text_seq_len=2, codebook_size=3, text_vocab_size=4)
    denoiser = UnifiedDenoiser(
        input_dim=layout.vocab_size,
        seq_len=layout.seq_len,
        vocab_size=layout.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        dropout=0.0,
    )
    config = SimpleNamespace(
        i2t_llm=SimpleNamespace(feature_progress=0.5),
        sampling=SimpleNamespace(
            image_time_power=1.0,
            text_time_power=1.0,
            image_to_text_text_time_power=1.0,
            image_to_text_text_time_schedule="power",
            image_to_text_logit_normal_loc=0.0,
            image_to_text_logit_normal_scale=1.0,
        ),
    )
    features = extract_i2t_image_features(
        denoiser=denoiser,
        config=config,
        layout=layout,
        schedule_tables={"kind": "power"},
        image_tokens=torch.tensor([[0, 1], [2, 0]]),
    )
    if hasattr(features, "is_inference"):
        assert not features.is_inference()

    adapter = SoftPrefixAdapter(input_dim=8, hidden_dim=8, prefix_tokens=2, output_dim=4)
    loss = adapter(features).sum()
    loss.backward()
    assert any(parameter.grad is not None for parameter in adapter.parameters())


def test_extract_i2t_image_features_can_pool_text_hidden():
    layout = TaskLayout(image_seq_len=2, text_seq_len=2, codebook_size=3, text_vocab_size=4)
    config = SimpleNamespace(
        i2t_llm=SimpleNamespace(feature_progress=0.5, feature_pool="text"),
        sampling=SimpleNamespace(
            image_time_power=1.0,
            text_time_power=1.0,
            image_to_text_text_time_power=1.0,
            image_to_text_text_time_schedule="power",
            image_to_text_logit_normal_loc=0.0,
            image_to_text_logit_normal_scale=1.0,
        ),
    )

    features = extract_i2t_image_features(
        denoiser=FixedFeatureDenoiser(layout, hidden_dim=5),
        config=config,
        layout=layout,
        schedule_tables={"kind": "power"},
        image_tokens=torch.tensor([[0, 1], [2, 0]]),
    )

    assert torch.equal(features, torch.full((2, 5), 3.0))
