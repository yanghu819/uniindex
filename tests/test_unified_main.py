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
    assert config["train"]["text_time_power"] == 1.0
    assert config["train"]["image_to_text_text_time_power"] is None
    assert config["sampling"]["text_time_power"] == 1.0
    assert config["sampling"]["image_to_text_text_time_power"] is None
    assert "image_" + "summary_to_text" not in config["model"]
    assert "image_extra_tokens" not in config["model"]
    dumped = yaml.safe_dump(config).lower()
    assert "qwen" not in dumped
    assert "emu3" not in dumped
    assert "probe" not in dumped


def test_best_understanding_result_is_clean_unified_result():
    result = unified.BEST_UNDERSTANDING_RESULT
    assert result["code_sha"] == "7ff026bc1a939c2f7e5f377bb72b00012e4807f0"
    assert result["i2t_exact_at_progress_0_5"] == 0.8671875
    assert result["i2t_token_at_progress_0_5"] == 0.93359375


def test_generation_gate_is_not_silently_marked_solved():
    report = unified.acceptance_report()
    assert report["understanding"]["passed"] is True
    assert report["generation"]["passed"] is False
    assert report["overall_passed"] is False
    clean = report["generation"]["result"]
    assert clean["conditioned_token_label_accuracy"] < unified.ACCEPTANCE["generation_conditioned_token_label_min"]
    assert clean["generated_unique_token_count"] == 769


def test_write_config_roundtrip(tmp_path):
    out = tmp_path / "main.yaml"
    unified.write_config(out)
    loaded = yaml.safe_load(out.read_text())
    assert loaded == unified.MAIN_CONFIG_DATA


def test_unified_flm_shape():
    model = unified.UnifiedFLM(
        vocab_size=32,
        seq_len=6,
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
    assert model.features(z, t, modality).shape == (3, 6, 32)


def test_reproduce_commands_are_single_mainline():
    commands = [" ".join(cmd) for cmd in unified.reproduce_commands()]
    assert commands[0].endswith("unified.py config --out configs/main.yaml")
    assert any("train --config configs/main.yaml --stage stage1" in command for command in commands)
    assert any("train --config configs/main.yaml --stage stage2" in command for command in commands)
    assert any("eval --config configs/main.yaml" in command for command in commands)
    assert any("quick-visualize --config configs/main.yaml" in command for command in commands)
    assert any("visualize --config configs/main.yaml" in command for command in commands)
    assert not any("probe" in command.lower() for command in commands)
    assert not any("qwen" in command.lower() or "emu3" in command.lower() for command in commands)


def test_about_cli_smoke():
    result = subprocess.run(
        [sys.executable, "unified.py", "about"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "siglip_vq" in result.stdout
    assert "0.8671875" in result.stdout
