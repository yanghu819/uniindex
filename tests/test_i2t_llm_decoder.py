from types import SimpleNamespace

import torch
import torch.nn as nn

from uniindex.i2t_llm_decoder import SoftPrefixAdapter, freeze_module, score_candidate_token_ids


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
