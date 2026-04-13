from uniindex.config import load_config


def test_load_smoke_config():
    config = load_config("configs/smoke.yaml")
    assert config.tokenizer.kind == "dummy"
    assert config.model.d_model == 64
    assert config.tokenizer.compact_vocab is True
    assert config.train.label_time_power == 0.5
