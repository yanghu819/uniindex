from pathlib import Path

import pytest
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
    assert config.train.stage2_init_checkpoint is None
    assert config.text.strings[0] == "zero"
    assert config.text.bos_token == "<bos>"
    assert config.text.eos_token == "<eos>"
    assert config.schedule.kind == "empirical"
    assert config.sampling.image_to_text_text_time_schedule == "power"
    assert config.sampling.image_to_text_logit_normal_loc == 0.0
    assert config.sampling.image_to_text_logit_normal_scale == 1.0
    assert config.eval.isolate_sampling_rng is False
    assert config.eval.sampling_seed is None


def test_load_main_config_is_siglip_vq_label_text():
    config = load_config("configs/main.yaml")
    assert config.tokenizer.kind == "siglip_vq"
    assert config.tokenizer.model_name == "inclusionAI/LLaDA2.0-Uni"
    assert config.tokenizer.image_size == 512
    assert config.text.kind == "label"
    assert config.dataset.name == "mnist"
    assert config.train.stage2_image_to_text_repeats == 10
    assert config.train.text_time_power == 1.0
    assert config.train.image_to_text_text_time_power is None
    assert config.sampling.steps == 32
    assert config.sampling.text_time_power == 1.0
    assert config.sampling.image_to_text_text_time_power is None


def test_load_understanding_config_keeps_i2t_time_override():
    config = load_config("configs/flm_understanding.yaml")
    assert config.train.image_to_text_text_time_power == 4.0
    assert config.sampling.image_to_text_text_time_power == 4.0


def test_load_joint_work_config_uses_stage2_repeats():
    config = load_config("configs/flm_joint_work.yaml")
    assert config.train.stage2_joint_repeats == 2
    assert config.train.stage2_text_to_image_repeats == 2
    assert config.train.stage2_image_to_text_repeats == 4


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


def test_legacy_extra_yaml_keys_are_ignored(tmp_path):
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
    raw["model"]["old_unused_model_key"] = True
    raw["train"]["old_unused_train_key"] = 0.125
    raw["sampling"]["old_unused_sampling_key"] = "unused"
    raw["train"]["stage2_init_checkpoint"] = "models/active/checkpoints/stage2_latest.pt"
    raw["sampling"]["image_to_text_text_time_schedule"] = "logit_normal"
    raw["sampling"]["image_to_text_logit_normal_loc"] = "-2.5"
    raw["sampling"]["image_to_text_logit_normal_scale"] = 1.5
    raw["eval"]["isolate_sampling_rng"] = True
    raw["eval"]["sampling_seed"] = 12345
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "smoke_extra.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.train.stage2_init_checkpoint == "models/active/checkpoints/stage2_latest.pt"
    assert config.sampling.image_to_text_text_time_schedule == "logit_normal"
    assert config.sampling.image_to_text_logit_normal_loc == -2.5
    assert config.sampling.image_to_text_logit_normal_scale == 1.5
    assert config.eval.isolate_sampling_rng is True
    assert config.eval.sampling_seed == 12345


def test_invalid_i2t_schedule_is_rejected(tmp_path):
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
    raw["sampling"]["image_to_text_text_time_schedule"] = "bad"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "bad.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported image_to_text_text_time_schedule"):
        load_config(config_path)


def test_i2t_noise_policy_bounds_are_checked(tmp_path):
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
    raw["train"]["image_to_text_noise_only_prob"] = 1.2
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "bad_noise.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="image_to_text_noise_only_prob"):
        load_config(config_path)
