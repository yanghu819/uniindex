import math

import torch

from uniindex.eval import _sample_unified_with_logits
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
