from pathlib import Path

from uniindex.i2t_repeats_sweep import (
    RepeatsSweepResult,
    _baseline_result,
    _build_promoted_long_config,
    _build_short_config,
    _repeats_tag,
)


def _result(repeats: int, exact: float, constrained: float, text_to_image: float) -> RepeatsSweepResult:
    metrics = {
        "image_to_text_exact_match": exact,
        "image_to_text_label_accuracy_constrained": constrained,
        "text_to_image_accuracy": text_to_image,
    }
    return RepeatsSweepResult(
        source_config_path=Path(f"source-{repeats}.yaml"),
        local_config_path=Path(f"local-{repeats}.yaml"),
        metrics_path=Path(f"metrics-{repeats}.json"),
        metrics=metrics,
        stage2_image_to_text_repeats=repeats,
    )


def test_repeats_tag_zero_pads_values():
    assert _repeats_tag(4) == "i2tr04"
    assert _repeats_tag(12) == "i2tr12"


def test_build_short_config_updates_paths_and_repeats():
    config = _build_short_config(
        short_base_config_path=Path("configs/flm_joint_work_fullvocab_short_i2tp40_tsw075.yaml"),
        snapshot_path=None,
        repeats=6,
    )
    assert config["project"]["name"] == "uniindex-fullvocab-short-i2tr06-tsw075"
    assert config["paths"]["artifacts_dir"] == "artifacts/fullvocab_short_shared"
    assert config["paths"]["models_dir"] == "models/fullvocab_short_i2tr06_tsw075"
    assert config["paths"]["runs_dir"] == "runs/fullvocab_short_i2tr06_tsw075"
    assert config["train"]["stage2_image_to_text_repeats"] == 6
    assert config["train"]["image_to_text_text_time_power"] == 4.0


def test_build_promoted_long_config_updates_paths_and_repeats():
    config = _build_promoted_long_config(
        long_config_path=Path("configs/flm_joint_work_fullvocab_tsw075.yaml"),
        snapshot_path=None,
        winner=_result(8, 0.1, 0.2, 0.3),
    )
    assert config["project"]["name"] == "uniindex-fullvocab-long-tsw075-i2tr08"
    assert config["paths"]["artifacts_dir"] == "artifacts/fullvocab_short_shared"
    assert config["paths"]["models_dir"] == "models/fullvocab_long_tsw075_i2tr08"
    assert config["paths"]["runs_dir"] == "runs/fullvocab_long_tsw075_i2tr08"
    assert config["train"]["stage2_image_to_text_repeats"] == 8


def test_baseline_result_matches_long_config_repeats():
    baseline = _baseline_result(
        [_result(4, 0.0, 0.1, 0.2), _result(6, 0.0, 0.2, 0.3), _result(8, 0.0, 0.1, 0.2)],
        Path("configs/flm_joint_work_fullvocab_tsw075.yaml"),
    )
    assert baseline.stage2_image_to_text_repeats == 6
