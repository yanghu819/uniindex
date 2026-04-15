import torch

from uniindex.text import build_text_metadata, decode_text_tokens, encode_labels, label_values_from_text_tokens


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
