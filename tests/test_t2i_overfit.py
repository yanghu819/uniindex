import torch

from uniindex.layout import TaskLayout
from uniindex.t2i_overfit import (
    _image_token_accuracy,
    _image_ce_loss,
    _jsonable_counter,
    _label_values_from_class_indices,
)


def test_label_values_from_class_indices_maps_noncontiguous_values():
    label_values = torch.tensor([10, 30, 50])
    indices = torch.tensor([2, 0, 1])
    assert _label_values_from_class_indices(indices, label_values).tolist() == [50, 10, 30]


def test_jsonable_counter_sorts_and_stringifies_keys():
    assert _jsonable_counter({3: 2, 1: 4}) == {"1": 4, "3": 2}


def test_image_token_metrics_ignore_text_positions():
    layout = TaskLayout(image_seq_len=2, text_seq_len=1, codebook_size=4, text_vocab_size=3)
    targets = torch.tensor([[1, 2, 5]])
    logits = torch.zeros(1, layout.seq_len, layout.vocab_size)
    logits[0, 0, 1] = 3.0
    logits[0, 1, 0] = 3.0
    logits[0, 2, 6] = 99.0

    assert _image_token_accuracy(logits, targets, layout) == 0.5
    loss = _image_ce_loss(logits, targets, layout)
    assert loss.ndim == 0
