import json
from pathlib import Path

import torch
import yaml

from uniindex.config import load_config
from uniindex.layout import TaskLayout
from uniindex.text import build_text_metadata, text_state_dict
from uniindex.visualize import export_text_to_image_grid


def _write_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
    raw["project"]["name"] = "quick-visual-test"
    raw["paths"] = {
        "data_dir": "data",
        "artifacts_dir": "artifacts",
        "models_dir": "models",
        "runs_dir": "runs",
        "logs_dir": "logs",
        "cache_dir": ".cache",
    }
    raw["labels"]["values"] = [0, 1]
    raw["text"]["kind"] = "label"
    raw["text"]["strings"] = ["zero", "one"]
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "quick.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def test_quick_visualize_samples_prompt_grid_without_classifier(monkeypatch, tmp_path):
    config = load_config(_write_config(tmp_path))
    metadata = build_text_metadata(
        kind="label",
        label_values=[0, 1],
        strings=["zero", "one"],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tokenizer_state = {
        **text_state_dict(metadata),
        "label_values": torch.tensor(metadata.label_values),
        "image_seq_len": 2,
        "codebook_size": 4,
        "grid_shape": (1, 2),
        "image_size": 8,
    }
    layout = TaskLayout(
        image_seq_len=2,
        text_seq_len=metadata.seq_len,
        codebook_size=4,
        text_vocab_size=metadata.vocab_size,
    )
    captured = {}

    def fake_sample_unified(**kwargs):
        captured["condition_text_tokens"] = kwargs["condition_text_tokens"].detach().cpu()
        captured["steps"] = kwargs["steps"]
        captured["temperature"] = kwargs["temperature"]
        batch = kwargs["condition_text_tokens"].shape[0]
        return torch.zeros(batch, layout.seq_len, dtype=torch.long)

    def fail_classifier(*args, **kwargs):
        raise AssertionError("quick visualization must not use classifier-based generation scoring")

    monkeypatch.setattr("uniindex.visualize._load_stage2", lambda config, device: (object(), tokenizer_state, layout))
    monkeypatch.setattr("uniindex.visualize.build_schedule_tables", lambda *args, **kwargs: {"kind": "power"})
    monkeypatch.setattr("uniindex.visualize.build_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr("uniindex.visualize.sample_unified", fake_sample_unified)
    monkeypatch.setattr(
        "uniindex.visualize._decode_image_tokens",
        lambda tokenizer, image_tokens, state, grid_shape, device: torch.zeros(image_tokens.shape[0], 3, 8, 8),
    )
    monkeypatch.setattr("uniindex.visualize.load_classifier", fail_classifier)

    result = export_text_to_image_grid(config, seeds_per_label=2, steps=5, temperature=0.6)

    grid_path = Path(result["text_to_image_prompt_grid"])
    summary_path = grid_path.parent / "quick_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert grid_path.exists()
    assert summary["prompts"] == ["zero", "zero", "one", "one"]
    assert summary["sampling_steps"] == 5
    assert summary["temperature"] == 0.6
    assert captured["steps"] == 5
    assert captured["temperature"] == 0.6
    assert torch.equal(
        captured["condition_text_tokens"],
        metadata.label_text_tokens.repeat_interleave(2, dim=0),
    )
