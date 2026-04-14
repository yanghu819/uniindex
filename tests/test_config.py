from uniindex.config import load_config


def test_load_smoke_config():
    config = load_config("configs/smoke.yaml")
    assert config.tokenizer.kind == "dummy"
    assert config.model.d_model == 64
    assert config.tokenizer.compact_vocab is True
    assert config.train.label_time_power == 0.5
    assert config.train.image_to_label_label_time_power is None


def test_load_understanding_config():
    config = load_config("configs/flm_understanding.yaml")
    assert config.train.image_to_label_label_time_power == 4.0
    assert config.sampling.image_to_label_label_time_power == 4.0
