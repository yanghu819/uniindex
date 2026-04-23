from pathlib import Path

import yaml

from uniindex.config import load_config


def test_load_smoke_config():
    config = load_config("configs/smoke.yaml")
    assert config.tokenizer.kind == "dummy"
    assert config.model.d_model == 64
    assert config.tokenizer.compact_vocab is True
    assert config.train.text_time_power == 0.5
    assert config.train.image_to_text_text_time_power is None
    assert config.train.image_to_text_text_time_cap is None
    assert config.train.image_to_text_noise_only_prob == 0.0
    assert config.text.strings[0] == "zero"
    assert config.text.bos_token == "<bos>"
    assert config.text.eos_token == "<eos>"
    assert config.schedule.kind == "empirical"
    assert config.train.text_sequence_weight == 0.25
    assert config.sampling.integrator == "legacy_progress_euler"
    assert config.sampling.final_decode == "final_model_call"
    assert config.sampling.final_model_progress == 1.0
    assert config.sampling.image_to_text_decoder == "sample"
    assert config.sampling.candidate_score_progress is None
    assert config.sampling.candidate_score_num_noise == 4
    assert config.sampling.image_to_text_projection == "none"
    assert config.sampling.image_to_text_projection_progress == 0.5
    assert config.sampling.image_to_text_projection_progresses is None
    assert config.sampling.image_to_text_candidate_score_progress is None
    assert config.sampling.image_to_text_candidate_score_num_noise == 1
    assert config.sampling.image_to_text_candidate_score_blend_weight == 0.0
    assert config.sampling.image_to_text_text_time_schedule == "power"
    assert config.sampling.image_to_text_logit_normal_loc == 0.0
    assert config.sampling.image_to_text_logit_normal_scale == 1.0
    assert config.eval.isolate_sampling_rng is False
    assert config.eval.sampling_seed is None


def test_load_understanding_config():
    config = load_config("configs/flm_understanding.yaml")
    assert config.train.image_to_text_text_time_power == 4.0
    assert config.sampling.image_to_text_text_time_power == 4.0


def test_load_joint_work_config():
    config = load_config("configs/flm_joint_work.yaml")
    assert config.train.stage2_joint_repeats == 2
    assert config.train.stage2_text_to_image_repeats == 2
    assert config.train.stage2_image_to_text_repeats == 4
    assert config.train.text_sequence_weight == 0.5


def test_load_fullvocab_clean_config():
    config = load_config("configs/smoke_fullvocab_clean.yaml")
    assert config.tokenizer.compact_vocab is False
    assert config.paths.models_dir.name == "smoke_fullvocab_clean"


def test_load_imageheavy_config():
    config = load_config("configs/flm_joint_work_imageheavy.yaml")
    assert config.train.stage2_joint_repeats == 2
    assert config.train.stage2_text_to_image_repeats == 1
    assert config.train.stage2_image_to_text_repeats == 8


def test_load_smoke_cifar10_config():
    config = load_config("configs/smoke_cifar10.yaml")
    assert config.dataset.name == "cifar10"
    assert config.tokenizer.kind == "dummy"
    assert config.text.strings[0] == "airplane"


def test_load_cifar10_fullvocab_quick_config():
    config = load_config("configs/flm_cifar10_fullvocab_quick.yaml")
    assert config.dataset.name == "cifar10"
    assert config.tokenizer.compact_vocab is False
    assert config.train.batch_size == 2


def test_load_fullvocab_tsw075_config():
    config = load_config("configs/flm_joint_work_fullvocab_tsw075.yaml")
    assert config.train.text_sequence_weight == 0.75
    assert config.train.image_to_text_text_time_power == 4.0
    assert config.train.stage2_image_to_text_repeats == 6
    assert config.sampling.temperature == 0.7
    assert config.sampling.image_to_text_text_time_power == 4.0
    assert config.sampling.integrator == "legacy_progress_euler"
    assert config.sampling.final_decode == "final_model_call"
    assert config.sampling.final_model_progress == 0.95
    assert config.sampling.image_to_text_decoder == "sample"
    assert config.sampling.image_to_text_projection == "candidate_renoise"
    assert config.sampling.image_to_text_projection_progress == 0.5
    assert config.sampling.image_to_text_projection_progresses is None
    assert config.paths.artifacts_dir.name == "fullvocab_short_shared"
    assert config.paths.runs_dir.name == "fullvocab_long_tsw075_i2tr06"


def test_load_candidate_projection_config():
    config = load_config("configs/flm_joint_work_fullvocab_tsw075_candidate_proj_p050.yaml")
    assert config.sampling.image_to_text_projection == "candidate_renoise"
    assert config.sampling.image_to_text_projection_progress == 0.5
    assert config.sampling.image_to_text_projection_progresses is None
    assert config.paths.models_dir.name == "fullvocab_long_tsw075_i2tr06"
    assert config.paths.runs_dir.name == "fullvocab_long_tsw075_i2tr06_candidate_proj_p050"


def test_load_projection_progresses_config(tmp_path):
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
    raw["sampling"]["image_to_text_projection"] = "candidate_renoise"
    raw["sampling"]["image_to_text_projection_progresses"] = [0.5, "0.75"]
    raw["sampling"]["image_to_text_candidate_score_progress"] = ["0.5", 0.75]
    raw["sampling"]["image_to_text_candidate_score_num_noise"] = "2"
    raw["sampling"]["image_to_text_candidate_score_blend_weight"] = "0.25"
    raw["sampling"]["image_to_text_text_time_schedule"] = "logit_normal"
    raw["sampling"]["image_to_text_logit_normal_loc"] = "-2.5"
    raw["sampling"]["image_to_text_logit_normal_scale"] = 1.5
    raw["eval"]["isolate_sampling_rng"] = True
    raw["eval"]["sampling_seed"] = 12345
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "smoke_multi_projection.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.sampling.image_to_text_projection_progress == 0.5
    assert config.sampling.image_to_text_projection_progresses == [0.5, 0.75]
    assert config.sampling.image_to_text_candidate_score_progress == [0.5, 0.75]
    assert config.sampling.image_to_text_candidate_score_num_noise == 2
    assert config.sampling.image_to_text_candidate_score_blend_weight == 0.25
    assert config.sampling.image_to_text_text_time_schedule == "logit_normal"
    assert config.sampling.image_to_text_logit_normal_loc == -2.5
    assert config.sampling.image_to_text_logit_normal_scale == 1.5
    assert config.eval.isolate_sampling_rng is True
    assert config.eval.sampling_seed == 12345


def test_load_i2t_power_short_configs():
    expected = {
        "configs/flm_joint_work_fullvocab_short_i2tp20_tsw075.yaml": 2.0,
        "configs/flm_joint_work_fullvocab_short_i2tp40_tsw075.yaml": 4.0,
        "configs/flm_joint_work_fullvocab_short_i2tp60_tsw075.yaml": 6.0,
    }
    for path, power in expected.items():
        config = load_config(path)
        assert config.train.text_sequence_weight == 0.75
        assert config.train.stage1_steps == 200
        assert config.train.stage2_steps == 400
        assert config.train.image_to_text_text_time_power == power
        assert config.sampling.image_to_text_text_time_power == power


def test_load_lowt_short_config():
    config = load_config("configs/flm_joint_work_fullvocab_short_i2tp40_tsw075_lowt.yaml")
    assert config.train.image_to_text_text_time_power == 4.0
    assert config.train.image_to_text_text_time_cap == 0.5
    assert config.train.image_to_text_noise_only_prob == 0.5
    assert config.sampling.final_model_progress == 0.95
    assert config.paths.models_dir.name == "fullvocab_short_i2tp40_tsw075_lowt"
