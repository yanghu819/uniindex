from uniindex.config import load_config
from uniindex.task_schedule import task_for_step


def test_task_for_step_uses_configured_stage2_repeats():
    config = load_config("configs/flm_joint_work.yaml")
    tasks = [task_for_step(config, "stage2", step) for step in range(8)]
    assert tasks == [
        "joint",
        "joint",
        "label_to_image",
        "label_to_image",
        "image_to_label",
        "image_to_label",
        "image_to_label",
        "image_to_label",
    ]
