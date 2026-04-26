import torch

from uniindex.text import build_text_metadata
from uniindex.vq_text_decoder import (
    VQTextDecoder,
    masked_candidate_scores,
    text_sequence_loss,
)


def test_vq_text_decoder_shape():
    decoder = VQTextDecoder(
        codebook_size=17,
        image_seq_len=5,
        text_seq_len=4,
        text_vocab_size=9,
        d_model=16,
        n_heads=4,
        n_layers=1,
    )
    logits = decoder(torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]))
    assert logits.shape == (2, 4, 9)


def test_text_sequence_loss_ignores_pad_positions():
    logits = torch.zeros(1, 3, 4)
    targets = torch.tensor([[1, 2, 0]])
    loss = text_sequence_loss(logits, targets, pad_id=0)
    expected = torch.nn.functional.cross_entropy(logits[:, :2].reshape(-1, 4), targets[:, :2].reshape(-1))
    assert torch.allclose(loss, expected)


def test_masked_candidate_scores_ignores_pad_and_normalizes_length():
    metadata = build_text_metadata(
        kind="char",
        label_values=[0, 1],
        strings=["a", "bb"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    logits = torch.full((1, metadata.seq_len, metadata.vocab_size), -8.0)
    candidate = metadata.label_text_tokens[0]
    for position, token_id in enumerate(candidate.tolist()):
        if token_id != metadata.pad_id:
            logits[0, position, token_id] = 8.0
    scores = masked_candidate_scores(logits, metadata)
    assert scores.shape == (1, 2)
    assert scores.argmax(dim=1).tolist() == [0]
