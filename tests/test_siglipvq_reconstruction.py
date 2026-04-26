from types import SimpleNamespace

import torch

from uniindex.siglipvq_reconstruction import probe_siglipvq_reconstruction


class _RunContext:
    def __init__(self, root):
        self.root = root

    def set_device(self, device):
        self.device = device

    def log_path(self, name):
        return self.root / name


def test_siglipvq_reconstruction_prepare_runs_with_grad_enabled(monkeypatch, tmp_path):
    def fake_prepare_assets(config):
        assert torch.is_grad_enabled()

    monkeypatch.setattr("uniindex.siglipvq_reconstruction.ensure_project_dirs", lambda config: None)
    monkeypatch.setattr("uniindex.siglipvq_reconstruction.set_seed", lambda seed: None)
    monkeypatch.setattr("uniindex.siglipvq_reconstruction.prepare_assets", fake_prepare_assets)
    monkeypatch.setattr("uniindex.siglipvq_reconstruction.resolve_device", lambda device, gpu_index: torch.device("cpu"))
    monkeypatch.setattr(
        "uniindex.siglipvq_reconstruction.load_tokenizer_state",
        lambda config: {"grid_shape": (1, 1), "image_size": 8},
    )
    monkeypatch.setattr(
        "uniindex.siglipvq_reconstruction.metadata_from_state",
        lambda state: SimpleNamespace(label_values=[0], label_strings=["zero"]),
    )
    monkeypatch.setattr("uniindex.siglipvq_reconstruction.build_tokenizer", lambda config, device: object())
    monkeypatch.setattr("uniindex.siglipvq_reconstruction.classifier_path", lambda models_dir, dataset_name: tmp_path / "clf.pt")
    monkeypatch.setattr("uniindex.siglipvq_reconstruction.load_classifier", lambda path, dataset_name, device: object())
    monkeypatch.setattr("uniindex.siglipvq_reconstruction.split_path", lambda config, split: tmp_path / f"{split}.pt")
    monkeypatch.setattr(
        "uniindex.siglipvq_reconstruction.build_loader",
        lambda path, batch_size, shuffle, num_workers: iter(
            [{"image_tokens": torch.zeros(1, 1, dtype=torch.long), "label": torch.zeros(1, dtype=torch.long)}]
        ),
    )
    monkeypatch.setattr(
        "uniindex.siglipvq_reconstruction._decode_image_tokens",
        lambda tokenizer, image_tokens, tokenizer_state, grid_shape, device: torch.zeros(1, 3, 8, 8),
    )
    monkeypatch.setattr(
        "uniindex.siglipvq_reconstruction.classify_images",
        lambda classifier, decoded, dataset_name: torch.zeros(1, dtype=torch.long),
    )

    config = SimpleNamespace(
        tokenizer=SimpleNamespace(kind="siglip_vq"),
        train=SimpleNamespace(seed=1, device="cpu", gpu_index=0, num_workers=0),
        dataset=SimpleNamespace(name="mnist"),
        paths=SimpleNamespace(models_dir=tmp_path),
    )

    summary = probe_siglipvq_reconstruction(
        config=config,
        sample_count=1,
        run_context=_RunContext(tmp_path),
    )

    assert summary["accuracy"] == 1.0
