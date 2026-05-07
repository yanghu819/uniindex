import math

import torch

from uniindex.eval import _EvalSamplingRngStreams, _logit_normal_gamma_from_progress, _sample_unified_with_logits
from uniindex.layout import TaskLayout
from uniindex.state import build_flm_clean_state


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


def test_sampler_uses_physical_schedule_delta():
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
        batch_size=1,
    )

    assert len(model.inputs) == 3
    first_z = model.inputs[0]
    first_t = model.times[0]
    next_t = torch.tensor([[0.5, math.sqrt(0.5)]])
    endpoint = torch.tensor([[[0.8, 0.2, 0.0, 0.0], [0.0, 0.0, 0.7, 0.3]]])
    expected_second_z = first_z + (next_t - first_t).unsqueeze(-1) * (endpoint - first_z) / (
        1.0 - first_t
    ).unsqueeze(-1).clamp_min(1e-4)
    assert torch.allclose(model.inputs[1], expected_second_z, atol=1e-6)


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
        condition_image_tokens=image_tokens,
    )
    clean_image = build_flm_clean_state(image_tokens, layout.vocab_size)
    assert torch.equal(model.inputs[0][:, layout.image_slice], clean_image)
    assert torch.equal(model.inputs[1][:, layout.image_slice], clean_image)


def test_conditioned_text_slice_stays_clean_between_steps():
    torch.manual_seed(0)
    layout = TaskLayout(image_seq_len=1, text_seq_len=1, codebook_size=2, text_vocab_size=2)
    model = StaticEndpointModel()
    text_tokens = torch.tensor([[1]])
    _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables={"kind": "power"},
        temperature=1.0,
        steps=2,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=None,
        condition_text_tokens=text_tokens,
    )
    clean_text = build_flm_clean_state(text_tokens + layout.text_offset, layout.vocab_size)
    assert torch.equal(model.inputs[0][:, layout.text_slice], clean_text)
    assert torch.equal(model.inputs[1][:, layout.text_slice], clean_text)


def test_eval_sampling_rng_streams_continue_per_branch():
    device = torch.device("cpu")
    streams = _EvalSamplingRngStreams(enabled=True, base_seed=42, device=device)

    torch.manual_seed(123)
    outer_before = torch.rand(2)
    with streams.branch("text_to_image"):
        first_draw = torch.rand(2)
    with streams.branch("text_to_image"):
        second_draw = torch.rand(2)
    outer_after = torch.rand(2)

    torch.manual_seed(123)
    assert torch.equal(outer_before, torch.rand(2))
    assert torch.equal(outer_after, torch.rand(2))

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
        steps=2,
        image_time_power=1.0,
        text_time_power=1.0,
        image_to_text_text_time_power=4.0,
        image_to_text_text_time_schedule="logit_normal",
        image_to_text_logit_normal_loc=-2.0,
        image_to_text_logit_normal_scale=1.0,
        condition_image_tokens=torch.tensor([[1]]),
    )

    expected_text_gamma = torch.sigmoid(torch.tensor(-2.0)).reshape(1)
    assert torch.allclose(model.times[1][:, layout.text_slice].reshape(-1), expected_text_gamma)
