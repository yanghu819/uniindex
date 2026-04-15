import torch

from uniindex.text import (
    build_text_metadata,
    decode_text_tokens,
    encode_labels,
    label_values_from_text_tokens,
    sequence_candidate_scores,
    shifted_label_text_tokens,
)


def test_text_metadata_roundtrip():
    metadata = build_text_metadata(
        kind="char",
        label_values=[0, 1, 2],
        strings=["zero", "one", "two"],
        pad_token="<pad>",
    )
    encoded = encode_labels(torch.tensor([2, 0]), metadata)
    decoded = decode_text_tokens(encoded, metadata)
    assert decoded == ["two", "zero"]
    restored = label_values_from_text_tokens(encoded, metadata)
    assert torch.equal(restored, torch.tensor([2, 0]))


def test_shifted_label_text_tokens_applies_offset():
    metadata = build_text_metadata(
        kind="char",
        label_values=[0, 1],
        strings=["zero", "one"],
        pad_token="<pad>",
    )
    shifted = shifted_label_text_tokens(metadata, token_offset=7)
    assert torch.equal(shifted, metadata.label_text_tokens + 7)


def test_sequence_candidate_scores_prefers_matching_sequence():
    logits = torch.full((1, 2, 5), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    candidate_tokens = torch.tensor([[1, 2], [3, 4]])
    scores = sequence_candidate_scores(logits, candidate_tokens)
    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]
