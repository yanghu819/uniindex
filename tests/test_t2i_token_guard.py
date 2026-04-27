import pytest
import torch

from uniindex.t2i_token_guard import (
    _label_value_to_class_indices,
    _labels_for_class_indices,
    confusion_matrix,
)


def test_confusion_matrix_counts_target_rows_and_prediction_columns():
    predicted = torch.tensor([0, 1, 1, 2, 0])
    target = torch.tensor([0, 0, 1, 2, 2])
    matrix = confusion_matrix(predicted, target, num_classes=3)
    assert matrix.tolist() == [
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 1],
    ]


def test_confusion_matrix_ignores_invalid_entries():
    predicted = torch.tensor([0, 9, 1, -1])
    target = torch.tensor([0, 1, -1, 1])
    matrix = confusion_matrix(predicted, target, num_classes=2)
    assert matrix.tolist() == [
        [1, 0],
        [0, 0],
    ]


def test_confusion_matrix_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        confusion_matrix(torch.tensor([0]), torch.tensor([0, 1]), num_classes=2)


def test_label_class_index_helpers_round_trip_noncontiguous_labels():
    label_values = torch.tensor([10, 30, 50])
    indices = torch.tensor([2, 0, 1])
    labels = _labels_for_class_indices(indices, label_values)
    assert labels.tolist() == [50, 10, 30]
    assert _label_value_to_class_indices(labels, label_values).tolist() == [2, 0, 1]
    assert _label_value_to_class_indices(torch.tensor([30, 99]), label_values).tolist() == [1, -1]
