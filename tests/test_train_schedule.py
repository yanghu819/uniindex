import torch

from uniindex.config import load_config
from uniindex.task_schedule import task_for_step
from uniindex.train import _task_time_schedule


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
