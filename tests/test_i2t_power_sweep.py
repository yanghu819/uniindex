from pathlib import Path

from uniindex.i2t_power_sweep import (
    SweepResult,
    _build_promoted_long_config,
    _metric_tuple,
    _winner_beats_baseline,
)


def _result(power: float, exact: float, constrained: float, text_to_image: float) -> SweepResult:
    metrics = {
        "image_to_text_exact_match": exact,
        "image_to_text_label_accuracy_constrained": constrained,
        "text_to_image_accuracy": text_to_image,
    }
    return SweepResult(
        source_config_path=Path(f"source-{power}.yaml"),
        local_config_path=Path(f"local-{power}.yaml"),
        metrics_path=Path(f"metrics-{power}.json"),
        metrics=metrics,
        image_to_text_text_time_power=power,
    )


def test_metric_tuple_uses_exact_then_constrained_then_text_to_image():
    assert _metric_tuple(
        {
            "image_to_text_exact_match": 0.1,
            "image_to_text_label_accuracy_constrained": 0.2,
            "text_to_image_accuracy": 0.3,
        }
    ) == (0.1, 0.2, 0.3)


def test_winner_beats_baseline_requires_strict_metric_improvement():
    baseline = _result(4.0, 0.0, 0.10, 0.20)
    better = _result(6.0, 0.0, 0.11, 0.20)
    tied = _result(2.0, 0.0, 0.10, 0.20)
    assert _winner_beats_baseline(better, baseline) is True
    assert _winner_beats_baseline(tied, baseline) is False


def test_build_promoted_long_config_updates_paths_and_power():
    config = _build_promoted_long_config(
        long_config_path=Path("configs/flm_joint_work_fullvocab_tsw075.yaml"),
        snapshot_path=None,
        winner=_result(6.0, 0.0, 0.11, 0.20),
    )
    assert config["project"]["name"] == "uniindex-fullvocab-long-tsw075-i2tp60"
    assert config["paths"]["artifacts_dir"] == "artifacts/fullvocab_short_shared"
    assert config["paths"]["models_dir"] == "models/fullvocab_long_tsw075_i2tp60"
    assert config["train"]["image_to_text_text_time_power"] == 6.0
    assert config["sampling"]["image_to_text_text_time_power"] == 6.0
