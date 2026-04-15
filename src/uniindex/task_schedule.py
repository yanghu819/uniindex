from __future__ import annotations

from .config import ProjectConfig


def task_for_step(config: ProjectConfig, stage: str, step: int) -> str:
    if stage == "stage1":
        return "joint"
    cycle = (
        ["joint"] * max(config.train.stage2_joint_repeats, 0)
        + ["text_to_image"] * max(config.train.stage2_text_to_image_repeats, 0)
        + ["image_to_text"] * max(config.train.stage2_image_to_text_repeats, 0)
    )
    if not cycle:
        raise ValueError("stage2 task schedule is empty")
    return cycle[step % len(cycle)]
