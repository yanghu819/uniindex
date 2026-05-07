import torch

from uniindex.text import (
    build_text_metadata,
    decode_text_tokens,
    encode_labels,
    label_values_from_text_tokens,
    shifted_label_text_tokens,
    text_scoring_mask,
)


def test_text_metadata_roundtrip():
    metadata = build_text_metadata(
        kind="char",
        label_values=[0, 1, 2],
        strings=["zero", "one", "two"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    encoded = encode_labels(torch.tensor([2, 0]), metadata)
    decoded = decode_text_tokens(encoded, metadata)
    assert decoded == ["two", "zero"]
    restored = label_values_from_text_tokens(encoded, metadata)
    assert torch.equal(restored, torch.tensor([2, 0]))


def test_label_text_metadata_uses_atomic_label_tokens():
    metadata = build_text_metadata(
        kind="label",
        label_values=[0, 1, 2],
        strings=["zero", "one", "two"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )

    encoded = encode_labels(torch.tensor([2, 0]), metadata)

    assert metadata.seq_len == 3
    assert metadata.vocab_tokens == ("<pad>", "<bos>", "<eos>", "zero", "one", "two")
    assert torch.equal(
        encoded,
        torch.tensor(
            [
                [metadata.bos_id, metadata.vocab_tokens.index("two"), metadata.eos_id],
                [metadata.bos_id, metadata.vocab_tokens.index("zero"), metadata.eos_id],
            ]
        ),
    )
    assert decode_text_tokens(encoded, metadata) == ["two", "zero"]
    assert torch.equal(label_values_from_text_tokens(encoded, metadata), torch.tensor([2, 0]))


def test_shifted_label_text_tokens_applies_offset():
    metadata = build_text_metadata(
        kind="char",
        label_values=[0, 1],
        strings=["zero", "one"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    shifted = shifted_label_text_tokens(metadata, token_offset=7)
    assert torch.equal(shifted, metadata.label_text_tokens + 7)


def test_decode_text_tokens_stops_at_eos_and_skips_bos():
    metadata = build_text_metadata(
        kind="char",
        label_values=[0, 1],
        strings=["zero", "one"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tokens = torch.tensor([[metadata.bos_id, metadata.vocab_tokens.index("o"), metadata.eos_id, metadata.vocab_tokens.index("n")]])
    assert decode_text_tokens(tokens, metadata) == ["o"]


def test_label_values_from_text_tokens_uses_decoded_strings_with_eos():
    metadata = build_text_metadata(
        kind="char",
        label_values=[0, 1],
        strings=["zero", "one"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    row = torch.tensor(
        [[
            metadata.bos_id,
            metadata.vocab_tokens.index("o"),
            metadata.vocab_tokens.index("n"),
            metadata.vocab_tokens.index("e"),
            metadata.eos_id,
            metadata.vocab_tokens.index("z"),
        ]]
    )
    restored = label_values_from_text_tokens(row, metadata)
    assert torch.equal(restored, torch.tensor([1]))


def test_text_scoring_mask_excludes_bos_and_keeps_eos():
    metadata = build_text_metadata(
        kind="char",
        label_values=[0],
        strings=["one"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    mask = text_scoring_mask(metadata.label_text_tokens, metadata, include_bos=False, include_eos=True)
    assert torch.equal(mask[0], torch.tensor([False, True, True, True, True]))
