import math

import torch

from uniindex.eval import (
    _candidate_denoiser_score_text,
    _projection_step_index,
    _projection_step_indices,
    _sample_unified_with_logits,
)
from uniindex.layout import TaskLayout
from uniindex.state import build_flm_clean_state
from uniindex.text import build_text_metadata


class StaticEndpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        logits = torch.full((2, 4), -20.0)
        logits[0, 0] = math.log(0.8)
        logits[0, 1] = math.log(0.2)
        logits[1, 2] = math.log(0.7)
        logits[1, 3] = math.log(0.3)
        self.register_buffer("logits_template", logits)
        self.inputs: list[torch.Tensor] = []
        self.times: list[torch.Tensor] = []

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        del modality_ids
        self.inputs.append(z_t.detach().clone())
        self.times.append(t.detach().clone())
        return self.logits_template.to(z_t.device).unsqueeze(0).expand(z_t.shape[0], -1, -1)


class CandidatePreferenceModel(torch.nn.Module):
    def __init__(self, layout: TaskLayout, preferred_text_targets: torch.Tensor) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.layout = layout
        self.register_buffer("preferred_text_targets", preferred_text_targets.long())
        self.inputs: list[torch.Tensor] = []

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        del t, modality_ids
        self.inputs.append(z_t.detach().clone())
        logits = torch.full(
            (z_t.shape[0], self.layout.seq_len, self.layout.vocab_size),
            -10.0,
            device=z_t.device,
        )
        logits[:, self.layout.image_slice, 0] = 10.0
        for offset, token_id in enumerate(self.preferred_text_targets.tolist()):
            logits[:, self.layout.image_seq_len + offset, token_id] = 10.0
        return logits


def test_scheduled_euler_uses_physical_schedule_delta():
    torch.manual_seed(0)
    layout = TaskLayout(image_seq_len=1, text_seq_len=1, codebook_size=2, text_vocab_size=2)
    model = StaticEndpointModel()
    _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=2,
        image_time_power=1.0,
        text_time_power=0.5,
        image_to_text_text_time_power=None,
        integrator="scheduled_euler",
        final_decode="last_endpoint",
        batch_size=1,
    )

    assert len(model.inputs) == 2
    first_z = model.inputs[0]
    first_t = model.times[0]
    next_t = torch.tensor([[0.5, math.sqrt(0.5)]])
    endpoint = torch.tensor([[[0.8, 0.2, 0.0, 0.0], [0.0, 0.0, 0.7, 0.3]]])
    expected_second_z = first_z + (next_t - first_t).unsqueeze(-1) * (endpoint - first_z) / (
        1.0 - first_t
    ).unsqueeze(-1).clamp_min(1e-4)
    assert torch.allclose(model.inputs[1], expected_second_z, atol=1e-6)


def test_last_endpoint_decode_skips_extra_final_model_call():
    layout = TaskLayout(image_seq_len=1, text_seq_len=1, codebook_size=2, text_vocab_size=2)
    model = StaticEndpointModel()
    _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=3,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        integrator="scheduled_euler",
        final_decode="last_endpoint",
        batch_size=1,
    )
    assert len(model.inputs) == 3

    model_with_final_call = StaticEndpointModel()
    _sample_unified_with_logits(
        model=model_with_final_call,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=3,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        integrator="scheduled_euler",
        final_decode="final_model_call",
        batch_size=1,
    )
    assert len(model_with_final_call.inputs) == 4


def test_final_model_progress_controls_final_call_time():
    layout = TaskLayout(image_seq_len=1, text_seq_len=1, codebook_size=2, text_vocab_size=2)
    model = StaticEndpointModel()
    _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=1,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        integrator="legacy_progress_euler",
        final_decode="final_model_call",
        final_model_progress=0.5,
        batch_size=1,
    )

    assert len(model.times) == 2
    assert torch.allclose(model.times[-1], torch.full((1, layout.seq_len), 0.5))


def test_conditioned_image_slice_stays_clean_between_steps():
    torch.manual_seed(0)
    layout = TaskLayout(image_seq_len=1, text_seq_len=1, codebook_size=2, text_vocab_size=2)
    model = StaticEndpointModel()
    image_tokens = torch.tensor([[1]])
    _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=2,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        integrator="scheduled_euler",
        final_decode="last_endpoint",
        condition_image_tokens=image_tokens,
    )
    clean_image = build_flm_clean_state(image_tokens, layout.vocab_size)
    assert torch.equal(model.inputs[0][:, layout.image_slice], clean_image)
    assert torch.equal(model.inputs[1][:, layout.image_slice], clean_image)


def test_candidate_denoiser_score_selects_best_candidate_without_labels():
    torch.manual_seed(0)
    metadata = build_text_metadata(
        kind="char",
        label_values=[0, 1],
        strings=["a", "b"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    layout = TaskLayout(
        image_seq_len=1,
        text_seq_len=metadata.seq_len,
        codebook_size=2,
        text_vocab_size=metadata.vocab_size,
    )
    preferred = metadata.label_text_tokens[1] + layout.text_offset
    model = CandidatePreferenceModel(layout, preferred_text_targets=preferred)

    selected_text, selected_indices, scores = _candidate_denoiser_score_text(
        model=model,
        layout=layout,
        schedule_tables={"kind": "power"},
        image_tokens=torch.tensor([[0], [1]]),
        text_metadata=metadata,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        progress_values=[0.5],
        num_noise=2,
    )

    assert scores.shape == (2, 2)
    assert selected_indices.tolist() == [1, 1]
    assert torch.equal(selected_text, metadata.label_text_tokens[[1, 1]])


def test_projection_step_index_uses_nearest_sampler_step():
    assert _projection_step_index(32, 0.5) == 16
    assert _projection_step_index(32, 0.95) == 30


def test_projection_step_indices_deduplicates_sampler_steps():
    assert _projection_step_indices(32, [0.5, 0.51, 0.75]) == {16, 24}


def test_candidate_projection_replaces_text_state_before_final_call():
    torch.manual_seed(0)
    metadata = build_text_metadata(
        kind="char",
        label_values=[0, 1],
        strings=["a", "b"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    layout = TaskLayout(
        image_seq_len=1,
        text_seq_len=metadata.seq_len,
        codebook_size=2,
        text_vocab_size=metadata.vocab_size,
    )
    preferred = metadata.label_text_tokens[1] + layout.text_offset
    model = CandidatePreferenceModel(layout, preferred_text_targets=preferred)

    _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=1,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        integrator="legacy_progress_euler",
        final_decode="final_model_call",
        final_model_progress=1.0,
        image_to_text_projection="candidate_renoise",
        image_to_text_projection_progress=0.0,
        text_metadata=metadata,
        condition_image_tokens=torch.tensor([[0], [1]]),
    )

    expected_text_state = build_flm_clean_state(preferred.expand(2, -1), layout.vocab_size)
    assert torch.equal(model.inputs[-1][:, layout.text_slice], expected_text_state)


def test_candidate_projection_progresses_can_project_twice():
    torch.manual_seed(0)
    metadata = build_text_metadata(
        kind="char",
        label_values=[0, 1],
        strings=["a", "b"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    layout = TaskLayout(
        image_seq_len=1,
        text_seq_len=metadata.seq_len,
        codebook_size=2,
        text_vocab_size=metadata.vocab_size,
    )
    preferred = metadata.label_text_tokens[1] + layout.text_offset
    expected_text_state = build_flm_clean_state(preferred.expand(2, -1), layout.vocab_size)

    single_projection_model = CandidatePreferenceModel(layout, preferred_text_targets=preferred)
    _sample_unified_with_logits(
        model=single_projection_model,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=2,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        integrator="legacy_progress_euler",
        final_decode="final_model_call",
        final_model_progress=1.0,
        image_to_text_projection="candidate_renoise",
        image_to_text_projection_progresses=[0.0],
        text_metadata=metadata,
        condition_image_tokens=torch.tensor([[0], [1]]),
    )

    double_projection_model = CandidatePreferenceModel(layout, preferred_text_targets=preferred)
    _sample_unified_with_logits(
        model=double_projection_model,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=2,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        integrator="legacy_progress_euler",
        final_decode="final_model_call",
        final_model_progress=1.0,
        image_to_text_projection="candidate_renoise",
        image_to_text_projection_progresses=[0.0, 0.5],
        text_metadata=metadata,
        condition_image_tokens=torch.tensor([[0], [1]]),
    )

    assert not torch.equal(single_projection_model.inputs[-1][:, layout.text_slice], expected_text_state)
    assert torch.equal(double_projection_model.inputs[-1][:, layout.text_slice], expected_text_state)
