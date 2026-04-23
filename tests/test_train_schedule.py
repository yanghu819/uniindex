import torch

from uniindex.config import load_config
from uniindex.layout import TaskLayout, unified_targets
from uniindex.state import build_flm_clean_state
from uniindex.task_schedule import task_for_step
from uniindex.text import build_text_metadata, shifted_label_text_tokens
from uniindex.train import _apply_image_to_text_noise_policy, _image_text_mismatch_loss, _loss_for_task, _task_time_schedule


class ImageBoundTextModel(torch.nn.Module):
    def __init__(self, layout: TaskLayout, label_text_tokens: torch.Tensor) -> None:
        super().__init__()
        self.layout = layout
        self.register_buffer("label_text_tokens", label_text_tokens.long())

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        del t, modality_ids
        image_indices = z_t[:, self.layout.image_slice].argmax(dim=-1).squeeze(1)
        logits = torch.full(
            (z_t.shape[0], self.layout.seq_len, self.layout.vocab_size),
            -10.0,
            device=z_t.device,
        )
        logits[:, self.layout.image_slice, 0] = 10.0
        for row, image_index in enumerate(image_indices.tolist()):
            for offset, token_id in enumerate(self.label_text_tokens[image_index].tolist()):
                logits[row, self.layout.image_seq_len + offset, token_id] = 10.0
        return logits


class SelfPromptTextModel(torch.nn.Module):
    def __init__(self, layout: TaskLayout) -> None:
        super().__init__()
        self.layout = layout

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        del t, modality_ids
        logits = torch.full(
            (z_t.shape[0], self.layout.seq_len, self.layout.vocab_size),
            -10.0,
            device=z_t.device,
        )
        logits[:, self.layout.image_slice, 0] = 10.0
        text_indices = z_t[:, self.layout.text_slice].argmax(dim=-1)
        for row in range(z_t.shape[0]):
            for offset, token_id in enumerate(text_indices[row].tolist()):
                logits[row, self.layout.image_seq_len + offset, token_id] = 10.0
        return logits


def test_task_for_step_uses_configured_stage2_repeats():
    config = load_config("configs/flm_joint_work.yaml")
    tasks = [task_for_step(config, "stage2", step) for step in range(8)]
    assert tasks == [
        "joint",
        "joint",
        "text_to_image",
        "text_to_image",
        "image_to_text",
        "image_to_text",
        "image_to_text",
        "image_to_text",
    ]


def test_task_time_schedule_uses_image_to_text_override_for_empirical():
    config = load_config("configs/flm_joint_work_fullvocab.yaml")
    progress = torch.tensor([0.5])
    modality_ids = torch.tensor([0, 1])
    schedule_tables = {
        "kind": "empirical",
        "image": {
            "progress_grid": torch.tensor([0.0, 0.5, 1.0]),
            "t_grid": torch.tensor([0.0, 0.5, 1.0]),
        },
        "text": {
            "progress_grid": torch.tensor([0.0, 0.5, 1.0]),
            "t_grid": torch.tensor([0.0, 0.5, 1.0]),
        },
    }
    t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, "image_to_text")
    assert torch.allclose(t_pos, torch.tensor([[0.5, 0.0625]]), atol=1e-6)


def test_image_to_text_noise_policy_caps_text_time_only_for_i2t():
    layout = TaskLayout(image_seq_len=2, text_seq_len=3, codebook_size=4, text_vocab_size=5)
    t_pos = torch.full((2, layout.seq_len), 0.8)
    capped = _apply_image_to_text_noise_policy(
        t_pos,
        layout,
        "image_to_text",
        text_time_cap=0.25,
        noise_only_prob=0.0,
    )
    assert torch.equal(capped[:, layout.image_slice], t_pos[:, layout.image_slice])
    assert torch.equal(capped[:, layout.text_slice], torch.full((2, layout.text_seq_len), 0.25))

    unchanged = _apply_image_to_text_noise_policy(
        t_pos,
        layout,
        "joint",
        text_time_cap=0.25,
        noise_only_prob=1.0,
    )
    assert torch.equal(unchanged, t_pos)


def test_image_to_text_noise_policy_can_force_text_noise():
    layout = TaskLayout(image_seq_len=1, text_seq_len=2, codebook_size=4, text_vocab_size=5)
    t_pos = torch.full((2, layout.seq_len), 0.8)
    adjusted = _apply_image_to_text_noise_policy(
        t_pos,
        layout,
        "image_to_text",
        text_time_cap=None,
        noise_only_prob=1.0,
    )
    assert torch.equal(adjusted[:, layout.image_slice], t_pos[:, layout.image_slice])
    assert torch.equal(adjusted[:, layout.text_slice], torch.zeros(2, layout.text_seq_len))


def test_image_text_mismatch_loss_rewards_shifted_image_binding():
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
    image_tokens = torch.tensor([[0], [1]])
    text_targets = metadata.label_text_tokens.clone()
    targets = unified_targets(image_tokens, text_targets, layout.codebook_size)
    x1 = build_flm_clean_state(targets, layout.vocab_size)
    t_pos = torch.ones((2, layout.seq_len))
    modality_ids = layout.position_modalities()
    valid_token_mask = layout.position_valid_token_mask()
    candidate_text_targets = shifted_label_text_tokens(metadata, token_offset=layout.codebook_size)

    image_bound_loss = _image_text_mismatch_loss(
        model=ImageBoundTextModel(layout, candidate_text_targets),
        x1=x1,
        t_pos=t_pos,
        modality_ids=modality_ids,
        layout=layout,
        valid_token_mask=valid_token_mask,
        text_targets=targets[:, layout.text_slice],
        candidate_text_targets=candidate_text_targets,
        margin=1.0,
    )
    self_prompt_loss = _image_text_mismatch_loss(
        model=SelfPromptTextModel(layout),
        x1=x1,
        t_pos=t_pos,
        modality_ids=modality_ids,
        layout=layout,
        valid_token_mask=valid_token_mask,
        text_targets=targets[:, layout.text_slice],
        candidate_text_targets=candidate_text_targets,
        margin=1.0,
    )

    assert image_bound_loss.item() == 0.0
    assert self_prompt_loss.item() > 1.0


def test_image_text_mismatch_loss_skips_same_label_pairs():
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
    image_tokens = torch.tensor([[0], [1]])
    text_targets = metadata.label_text_tokens[[0, 0]]
    targets = unified_targets(image_tokens, text_targets, layout.codebook_size)
    x1 = build_flm_clean_state(targets, layout.vocab_size)

    loss = _image_text_mismatch_loss(
        model=SelfPromptTextModel(layout),
        x1=x1,
        t_pos=torch.ones((2, layout.seq_len)),
        modality_ids=layout.position_modalities(),
        layout=layout,
        valid_token_mask=layout.position_valid_token_mask(),
        text_targets=targets[:, layout.text_slice],
        candidate_text_targets=shifted_label_text_tokens(metadata, token_offset=layout.codebook_size),
        margin=1.0,
    )

    assert loss.item() == 0.0


def test_loss_for_task_does_not_apply_mismatch_loss_to_joint_task():
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
    targets = unified_targets(torch.tensor([[0], [1]]), metadata.label_text_tokens, layout.codebook_size)
    logits = torch.zeros((2, layout.seq_len, layout.vocab_size))

    _, parts = _loss_for_task(
        logits=logits,
        targets=targets,
        layout=layout,
        joint_weight=0.5,
        text_weight=1.0,
        text_pad_id=metadata.pad_id,
        label_text_tokens=shifted_label_text_tokens(metadata, token_offset=layout.codebook_size),
        text_sequence_weight=0.0,
        task="joint",
        image_to_text_mismatch_weight=0.1,
    )

    assert parts["mismatch_loss"] == 0.0
