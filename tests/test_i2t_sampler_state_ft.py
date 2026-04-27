from types import SimpleNamespace

import torch

from uniindex.i2t_sampler_state_ft import (
    _anchor_kl_loss,
    _generate_i2t_sampler_states,
    _masked_kl_divergence,
    _sampler_state_loss,
    _set_trainable_scope,
    _shuffled_image_contrast_loss,
    _trace_step_requests,
)
from uniindex.layout import TaskLayout, unified_targets
from uniindex.model import UnifiedDenoiser
from uniindex.state import build_flm_clean_state


def _sampler_config(**overrides):
    sampling_values = {
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
    sampling_values.update(overrides)
    train_values = {
        "image_time_power": 1.0,
        "text_time_power": 1.0,
        "image_to_text_text_time_power": 1.0,
    }
    return SimpleNamespace(sampling=SimpleNamespace(**sampling_values), train=SimpleNamespace(**train_values))


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


def test_sampler_state_loss_accepts_inference_mode_traces_for_backward():
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
    student = UnifiedDenoiser(
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
    state = _generate_i2t_sampler_states(
        teacher=teacher,
        config=_sampler_config(),
        layout=layout,
        schedule_tables={"kind": "power"},
        text_metadata=None,
        image_tokens=image_tokens,
        progress_values=(0.5,),
    )[0]
    if hasattr(state.z_t, "is_inference"):
        assert state.z_t.is_inference()

    loss, _, _ = _sampler_state_loss(
        student=student,
        state=state,
        layout=layout,
        modality_ids=layout.position_modalities(),
        text_targets=torch.tensor([[3, 4], [4, 5]]),
        text_pad_id=3,
        candidate_text_targets=torch.tensor([[3, 4], [4, 5]]),
        sequence_weight=1.0,
    )

    loss.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())


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


def test_trainable_scope_limits_updated_parameters():
    layout = TaskLayout(image_seq_len=2, text_seq_len=2, codebook_size=3, text_vocab_size=4)
    model = UnifiedDenoiser(
        input_dim=layout.vocab_size,
        seq_len=layout.seq_len,
        vocab_size=layout.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=2,
        mlp_ratio=2,
        dropout=0.0,
    )

    stats = _set_trainable_scope(model, "last_block")

    assert 0 < stats["trainable_parameters"] < stats["total_parameters"]
    assert any(parameter.requires_grad for parameter in model.transformer.layers[-1].parameters())
    assert not any(parameter.requires_grad for parameter in model.transformer.layers[0].parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_masked_kl_divergence_is_zero_for_identical_logits():
    layout = TaskLayout(image_seq_len=1, text_seq_len=1, codebook_size=2, text_vocab_size=3)
    logits = torch.randn(2, layout.seq_len, layout.vocab_size)

    loss = _masked_kl_divergence(
        student_logits=logits,
        teacher_logits=logits.clone(),
        valid_token_mask=layout.position_valid_token_mask(),
        temperature=1.0,
    )

    assert loss.item() < 1e-6


def test_anchor_kl_loss_is_zero_when_student_matches_teacher():
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
    student = UnifiedDenoiser(
        input_dim=layout.vocab_size,
        seq_len=layout.seq_len,
        vocab_size=layout.vocab_size,
        d_model=8,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        dropout=0.0,
    )
    student.load_state_dict(teacher.state_dict())
    image_tokens = torch.tensor([[0, 1], [2, 0]])
    text_tokens = torch.tensor([[0, 1], [1, 2]])
    targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
    x1 = build_flm_clean_state(targets, layout.vocab_size)

    loss = _anchor_kl_loss(
        teacher=teacher,
        student=student,
        config=_sampler_config(),
        x1=x1,
        progress=torch.tensor([0.25, 0.75]),
        layout=layout,
        modality_ids=layout.position_modalities(),
        schedule_tables={"kind": "power"},
        valid_token_mask=layout.position_valid_token_mask(),
        anchor_tasks=("joint", "text_to_image"),
        temperature=1.0,
    )

    assert loss.item() < 1e-6
