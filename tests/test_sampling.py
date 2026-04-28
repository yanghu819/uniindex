import math

import torch

from uniindex.eval import (
    _candidate_denoiser_score_text,
    _EvalSamplingRngStreams,
    _eval_sampling_rng,
    _logit_normal_gamma_from_progress,
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


class CrossModalPreferenceModel(torch.nn.Module):
    def __init__(self, layout: TaskLayout) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.layout = layout

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        del t, modality_ids
        logits = torch.full(
            (z_t.shape[0], self.layout.seq_len, self.layout.vocab_size),
            -10.0,
            device=z_t.device,
        )
        logits[:, self.layout.image_slice, 1] = 5.0
        logits[:, self.layout.image_slice, self.layout.text_offset] = 10.0
        logits[:, self.layout.text_slice, self.layout.text_offset + 1] = 5.0
        logits[:, self.layout.text_slice, 0] = 10.0
        return logits


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


class CandidateScoreProjectionModel(torch.nn.Module):
    def __init__(self, layout: TaskLayout, sampler_text_targets: torch.Tensor, score_text_targets: torch.Tensor) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.layout = layout
        self.register_buffer("sampler_text_targets", sampler_text_targets.long())
        self.register_buffer("score_text_targets", score_text_targets.long())
        self.inputs: list[torch.Tensor] = []

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        del modality_ids
        self.inputs.append(z_t.detach().clone())
        logits = torch.full(
            (z_t.shape[0], self.layout.seq_len, self.layout.vocab_size),
            -10.0,
            device=z_t.device,
        )
        logits[:, self.layout.image_slice, 0] = 10.0
        score_context = t[:, self.layout.text_slice].amin(dim=1) >= 0.99
        for row in range(z_t.shape[0]):
            targets = self.score_text_targets if bool(score_context[row]) else self.sampler_text_targets
            for offset, token_id in enumerate(targets.tolist()):
                logits[row, self.layout.image_seq_len + offset, token_id] = 10.0
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


def test_sampling_logit_mask_none_allows_full_vocab_argmax():
    layout = TaskLayout(image_seq_len=1, text_seq_len=1, codebook_size=2, text_vocab_size=2)
    model = CrossModalPreferenceModel(layout)

    unmasked, _ = _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=1,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        integrator="legacy_progress_euler",
        final_decode="last_endpoint",
        batch_size=1,
        noise_support="full_vocab",
        logit_mask="none",
    )
    masked, _ = _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=1,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        integrator="legacy_progress_euler",
        final_decode="last_endpoint",
        batch_size=1,
        noise_support="full_vocab",
        logit_mask="modality",
    )

    assert unmasked.tolist() == [[layout.text_offset, 0]]
    assert masked.tolist() == [[1, layout.text_offset + 1]]


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


def test_eval_sampling_rng_restores_outer_rng_and_splits_branches():
    device = torch.device("cpu")
    torch.manual_seed(123)
    before = torch.rand(3)
    with _eval_sampling_rng(enabled=True, base_seed=42, device=device, branch="image_to_text", batch_index=0):
        image_to_text_draw = torch.rand(4)
    after = torch.rand(3)

    torch.manual_seed(123)
    expected_before = torch.rand(3)
    expected_after = torch.rand(3)
    assert torch.equal(before, expected_before)
    assert torch.equal(after, expected_after)

    with _eval_sampling_rng(enabled=True, base_seed=42, device=device, branch="text_to_image", batch_index=0):
        text_to_image_draw = torch.rand(4)
    with _eval_sampling_rng(enabled=True, base_seed=42, device=device, branch="text_to_image", batch_index=0):
        repeated_text_to_image_draw = torch.rand(4)
    assert torch.equal(text_to_image_draw, repeated_text_to_image_draw)
    assert not torch.equal(image_to_text_draw, text_to_image_draw)


def test_eval_sampling_rng_streams_continue_per_branch():
    device = torch.device("cpu")
    streams = _EvalSamplingRngStreams(enabled=True, base_seed=42, device=device)

    with streams.branch("text_to_image"):
        first_draw = torch.rand(2)
    with streams.branch("text_to_image"):
        second_draw = torch.rand(2)

    torch.manual_seed(42 + 20_000)
    expected_first = torch.rand(2)
    expected_second = torch.rand(2)
    assert torch.equal(first_draw, expected_first)
    assert torch.equal(second_draw, expected_second)


def test_logit_normal_gamma_from_progress_controls_midpoint():
    progress = torch.tensor([0.0, 0.5, 1.0])
    gamma = _logit_normal_gamma_from_progress(progress, loc=-2.0, scale=1.0)
    assert torch.equal(gamma[[0, 2]], torch.tensor([0.0, 1.0]))
    assert torch.allclose(gamma[1], torch.sigmoid(torch.tensor(-2.0)))


def test_logit_normal_i2t_schedule_sets_text_time_directly():
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
        image_to_text_text_time_power=4.0,
        image_to_text_text_time_schedule="logit_normal",
        image_to_text_logit_normal_loc=-2.0,
        image_to_text_logit_normal_scale=1.0,
        integrator="legacy_progress_euler",
        final_decode="final_model_call",
        final_model_progress=0.5,
        condition_image_tokens=torch.tensor([[1]]),
    )

    expected_text_gamma = torch.sigmoid(torch.tensor(-2.0))
    assert torch.allclose(model.times[-1][:, layout.text_slice], expected_text_gamma.reshape(1, 1))


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


def test_candidate_score_projection_uses_denoiser_score_before_renoise():
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
    sampler_preferred = metadata.label_text_tokens[0] + layout.text_offset
    score_preferred = metadata.label_text_tokens[1] + layout.text_offset
    model = CandidateScoreProjectionModel(
        layout,
        sampler_text_targets=sampler_preferred,
        score_text_targets=score_preferred,
    )

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
        image_to_text_projection="candidate_score_renoise",
        image_to_text_projection_progress=0.0,
        image_to_text_candidate_score_progress=[1.0],
        image_to_text_candidate_score_num_noise=1,
        text_metadata=metadata,
        condition_image_tokens=torch.tensor([[0], [1]]),
    )

    expected_text_state = build_flm_clean_state(score_preferred.expand(2, -1), layout.vocab_size)
    assert torch.equal(model.inputs[-1][:, layout.text_slice], expected_text_state)


def test_candidate_score_blend_projection_keeps_sampler_score_primary():
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
    sampler_preferred = metadata.label_text_tokens[0] + layout.text_offset
    score_preferred = metadata.label_text_tokens[1] + layout.text_offset

    sampler_only_model = CandidateScoreProjectionModel(
        layout,
        sampler_text_targets=sampler_preferred,
        score_text_targets=score_preferred,
    )
    _sample_unified_with_logits(
        model=sampler_only_model,
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
        image_to_text_projection="candidate_score_blend_renoise",
        image_to_text_projection_progress=0.0,
        image_to_text_candidate_score_progress=[1.0],
        image_to_text_candidate_score_num_noise=1,
        image_to_text_candidate_score_blend_weight=0.0,
        text_metadata=metadata,
        condition_image_tokens=torch.tensor([[0], [1]]),
    )
    expected_sampler_state = build_flm_clean_state(sampler_preferred.expand(2, -1), layout.vocab_size)
    assert torch.equal(sampler_only_model.inputs[-1][:, layout.text_slice], expected_sampler_state)

    blended_model = CandidateScoreProjectionModel(
        layout,
        sampler_text_targets=sampler_preferred,
        score_text_targets=score_preferred,
    )
    _sample_unified_with_logits(
        model=blended_model,
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
        image_to_text_projection="candidate_score_blend_renoise",
        image_to_text_projection_progress=0.0,
        image_to_text_candidate_score_progress=[1.0],
        image_to_text_candidate_score_num_noise=1,
        image_to_text_candidate_score_blend_weight=2.0,
        text_metadata=metadata,
        condition_image_tokens=torch.tensor([[0], [1]]),
    )
    expected_score_state = build_flm_clean_state(score_preferred.expand(2, -1), layout.vocab_size)
    assert torch.equal(blended_model.inputs[-1][:, layout.text_slice], expected_score_state)
