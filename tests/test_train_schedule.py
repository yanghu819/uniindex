import torch

from uniindex.config import load_config
from uniindex.layout import TaskLayout, unified_targets
from uniindex.task_schedule import task_for_step
from uniindex.text import build_text_metadata
from uniindex.train import _apply_image_to_text_noise_policy, _loss_for_task, _task_time_schedule


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


def test_loss_for_task_uses_only_active_modality_for_conditional_tasks():
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

    joint_loss, joint_parts = _loss_for_task(
        logits=logits,
        targets=targets,
        layout=layout,
        joint_weight=0.5,
        text_weight=1.0,
        text_pad_id=metadata.pad_id,
        task="joint",
    )
    image_loss, image_parts = _loss_for_task(
        logits=logits,
        targets=targets,
        layout=layout,
        joint_weight=0.5,
        text_weight=1.0,
        text_pad_id=metadata.pad_id,
        task="text_to_image",
    )
    text_loss, text_parts = _loss_for_task(
        logits=logits,
        targets=targets,
        layout=layout,
        joint_weight=0.5,
        text_weight=1.0,
        text_pad_id=metadata.pad_id,
        task="image_to_text",
    )

    assert torch.allclose(image_loss, torch.tensor(image_parts["image_loss"]))
    assert torch.allclose(text_loss, torch.tensor(text_parts["text_loss"]))
    assert torch.allclose(joint_loss, 0.5 * torch.tensor(joint_parts["image_loss"]) + torch.tensor(joint_parts["text_loss"]))
