import torch

from uniindex.layout import TaskLayout, mask_logits


def test_task_layout_shapes_and_offsets():
    layout = TaskLayout(image_seq_len=4, text_seq_len=2, codebook_size=10, text_vocab_size=3)
    assert layout.seq_len == 6
    assert layout.vocab_size == 13
    assert layout.image_slice == slice(0, 4)
    assert layout.text_slice == slice(4, 6)
    assert layout.text_offset == 10


def test_position_valid_token_mask_accepts_layout():
    layout = TaskLayout(image_seq_len=2, text_seq_len=3, codebook_size=4, text_vocab_size=3)
    mask = layout.position_valid_token_mask()
    assert mask.shape == (5, 7)
    assert torch.equal(mask[0], torch.tensor([True, True, True, True, False, False, False]))
    assert torch.equal(mask[2], torch.tensor([False, False, False, False, True, True, True]))


def test_mask_logits_accepts_layout():
    layout = TaskLayout(image_seq_len=4, text_seq_len=1, codebook_size=10, text_vocab_size=2)
    logits = torch.zeros(2, layout.seq_len, layout.vocab_size)
    masked = mask_logits(logits, layout=layout)
    assert torch.isneginf(masked[:, layout.image_slice, layout.codebook_size :]).all()
    assert torch.isneginf(masked[:, layout.text_slice, : layout.codebook_size]).all()
