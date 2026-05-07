import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import yaml

SPEC = spec_from_file_location("unified", Path(__file__).resolve().parents[1] / "unified.py")
assert SPEC is not None and SPEC.loader is not None
unified = module_from_spec(SPEC)
sys.modules["unified"] = unified
SPEC.loader.exec_module(unified)


def test_mainline_is_siglip_vq_not_emu_or_qwen():
    config = unified.MAIN_CONFIG_DATA
    assert config["tokenizer"]["kind"] == "siglip_vq"
    assert config["tokenizer"]["model_name"] == "inclusionAI/LLaDA2.0-Uni"
    assert config["text"]["kind"] == "label"
    assert config["model"]["image_semantic_tokens"] == 1
    assert config["model"]["image_semantic_source"] == "vq_tokens"
    dumped = yaml.safe_dump(config).lower()
    assert "qwen" not in dumped
    assert "emu3" not in dumped


def test_best_understanding_result_is_the_siglip_vq_semantic_token_result():
    result = unified.BEST_UNDERSTANDING_RESULT
    assert result["code_sha"] == "b96d5f195340650e0082725b57af1aa9ad85420c"
    assert result["i2t_exact_at_progress_0_5"] == 0.8828125
    assert result["semantic_hidden_label_probe"] == 0.9609375


def test_write_config_roundtrip(tmp_path):
    out = tmp_path / "main.yaml"
    unified.write_config(out)
    loaded = yaml.safe_load(out.read_text())
    assert loaded == unified.MAIN_CONFIG_DATA


def test_unified_flm_shape_and_semantic_token():
    model = unified.UnifiedFLM(
        vocab_size=32,
        image_vocab_size=10,
        seq_len=6,
        image_seq_len=4,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
    )
    z = torch.zeros(3, 6, 32)
    z[:, :4, :10] = torch.nn.functional.one_hot(
        torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [4, 5, 6, 7]]),
        num_classes=10,
    ).float()
    z[:, 4:, 10:12] = 0.5
    t = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.25, 0.25]]).expand(3, -1)
    modality = torch.tensor([0, 0, 0, 0, 1, 1])

    assert model(z, t, modality).shape == (3, 6, 32)
    assert model.features(z, t, modality, include_semantic=True).shape == (3, 7, 32)


def test_reproduce_commands_are_single_mainline():
    commands = [" ".join(cmd) for cmd in unified.reproduce_commands()]
    assert commands[0].endswith("unified.py config --out configs/main.yaml")
    assert any("train --config configs/main.yaml --stage stage1" in command for command in commands)
    assert any("train --config configs/main.yaml --stage stage2" in command for command in commands)
    assert any("diagnose-i2t --config configs/main.yaml" in command for command in commands)
    assert not any("qwen" in command.lower() or "emu3" in command.lower() for command in commands)


def test_about_cli_smoke():
    result = subprocess.run(
        [sys.executable, "unified.py", "about"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "siglip_vq" in result.stdout
    assert "0.8828125" in result.stdout
