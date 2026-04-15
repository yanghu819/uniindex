from uniindex.config import load_config


def test_load_smoke_config():
    config = load_config("configs/smoke.yaml")
    assert config.tokenizer.kind == "dummy"
    assert config.model.d_model == 64
    assert config.tokenizer.compact_vocab is True
    assert config.train.text_time_power == 0.5
    assert config.train.image_to_text_text_time_power is None
    assert config.text.strings[0] == "zero"
    assert config.schedule.kind == "empirical"


def test_load_understanding_config():
    config = load_config("configs/flm_understanding.yaml")
    assert config.train.image_to_text_text_time_power == 4.0
    assert config.sampling.image_to_text_text_time_power == 4.0


def test_load_joint_work_config():
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
