import pytest
import torch

from uniindex.label_feature_probe import (
    PooledFeatureLabelProbe,
    VQTokenLabelProbe,
    labels_to_class_indices,
)


def test_vq_token_label_probe_shape():
    probe = VQTokenLabelProbe(codebook_size=11, seq_len=5, num_classes=3, d_model=8)
    logits = probe(torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]))
    assert logits.shape == (2, 3)


def test_pooled_feature_label_probe_shape():
    probe = PooledFeatureLabelProbe(input_dim=8, num_classes=4, hidden_dim=6)
    logits = probe(torch.randn(3, 8))
    assert logits.shape == (3, 4)


def test_labels_to_class_indices_maps_config_values():
    labels = torch.tensor([10, 30, 20, 10])
    label_values = torch.tensor([10, 20, 30])
    assert labels_to_class_indices(labels, label_values).tolist() == [0, 2, 1, 0]


def test_labels_to_class_indices_rejects_unknown_labels():
    with pytest.raises(ValueError, match="outside config.labels.values"):
        labels_to_class_indices(torch.tensor([0, 99]), torch.tensor([0, 1]))
