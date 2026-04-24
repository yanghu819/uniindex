from types import SimpleNamespace

import torch

from uniindex.i2t_sampler_state_ft import (
    _generate_i2t_sampler_states,
    _shuffled_image_contrast_loss,
    _trace_step_requests,
)
from uniindex.layout import TaskLayout
from uniindex.model import UnifiedDenoiser


def _sampler_config(**overrides):
    values = {
        "steps": 4,
        "temperature": 1.0,
        "image_time_power": 1.0,
        "text_time_power": 1.0,
        "image_to_text_text_time_power": 1.0,
        "image_to_text_text_time_schedule": "power",
        "image_to_text_logit_normal_loc": 0.0,
        "image_to_text_logit_normal_scale": 1.0,
        "integrator": "legacy_progress_euler",
        "image_to_text_projection": "none",
        "image_to_text_projection_progress": 0.5,
        "image_to_text_projection_progresses": None,
        "image_to_text_candidate_score_progress": None,
        "image_to_text_candidate_score_num_noise": 1,
        "image_to_text_candidate_score_blend_weight": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(sampling=SimpleNamespace(**values))


def test_trace_step_requests_selects_nearest_sampler_step():
    assert _trace_step_requests(steps=8, progress_values=(0.5, 0.74, 0.9)) == {
        4: [0.5],
        6: [0.74],
        7: [0.9],
    }


def test_generate_i2t_sampler_states_returns_requested_shapes():
    layout = TaskLayout(image_seq_len=2, text_seq_len=2, codebook_size=3, text_vocab_size=4)
    teacher = UnifiedDenoiser(
        input_dim=layout.vocab_size,
        seq_len=layout.seq_len,
        vocab_size=layout.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        dropout=0.0,
    )
    image_tokens = torch.tensor([[0, 1], [2, 0]])

    states = _generate_i2t_sampler_states(
        teacher=teacher,
        config=_sampler_config(),
        layout=layout,
        schedule_tables={"kind": "power"},
        text_metadata=None,
        image_tokens=image_tokens,
        progress_values=(0.5, 0.75),
    )

    assert [state.progress for state in states] == [0.5, 0.75]
    for state in states:
        assert state.z_t.shape == (2, layout.seq_len, layout.vocab_size)
        assert state.t_pos.shape == (2, layout.seq_len)
        assert torch.equal(state.t_pos[:, layout.image_slice], torch.ones(2, layout.image_seq_len))


def test_shuffled_image_contrast_loss_uses_only_label_changed_pairs():
    candidate_targets = torch.tensor([[3], [4]])
    text_targets = torch.tensor([[3], [4]])
    labels = torch.tensor([0, 1])
    shifted_labels = torch.tensor([1, 0])
    true_logits = torch.zeros(2, 1, 5)
    shuffled_logits = torch.zeros(2, 1, 5)
    true_logits[0, 0, 3] = 5.0
    true_logits[1, 0, 4] = 5.0
    shuffled_logits[0, 0, 3] = 1.0
    shuffled_logits[1, 0, 4] = 1.0

    low_loss = _shuffled_image_contrast_loss(
        true_text_logits=true_logits,
        shuffled_text_logits=shuffled_logits,
        text_targets=text_targets,
        labels=labels,
        shifted_labels=shifted_labels,
        candidate_text_targets=candidate_targets,
        margin=1.0,
    )
    high_loss = _shuffled_image_contrast_loss(
        true_text_logits=shuffled_logits,
        shuffled_text_logits=true_logits,
        text_targets=text_targets,
        labels=labels,
        shifted_labels=shifted_labels,
        candidate_text_targets=candidate_targets,
        margin=1.0,
    )

    assert low_loss.item() < 0.15
    assert high_loss.item() > 1.0
