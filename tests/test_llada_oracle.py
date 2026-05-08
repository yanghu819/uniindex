import torch

from uniindex.llada_oracle import (
    OracleRunConfig,
    _load_official_model,
    digit_prompts,
    infer_grid_shape,
    normalize_image_token_rows,
)


def test_digit_prompts_formats_labels_and_digits():
    prompts = digit_prompts("digit {digit}: {label}")

    assert prompts[0] == "digit 0: zero"
    assert prompts[-1] == "digit 9: nine"
    assert len(prompts) == 10


def test_normalize_image_token_rows_accepts_tensor_dict_and_tuple():
    assert normalize_image_token_rows(torch.tensor([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    assert normalize_image_token_rows({"image_tokens": torch.tensor([5, 6])}) == [[5, 6]]
    assert normalize_image_token_rows((None, [7, 8, 9])) == [[7, 8, 9]]


def test_infer_grid_shape_prefers_requested_or_square():
    assert infer_grid_shape(256, 16, 16) == (16, 16)
    assert infer_grid_shape(256, None, None) == (16, 16)
    assert infer_grid_shape(192, 12, None) == (12, 16)


def test_official_model_loader_uses_repo_local_cache(monkeypatch, tmp_path):
    calls = {}

    class FakeModel:
        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            calls["model_eval"] = True

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["model"] = (model_name, kwargs)
            return FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["tokenizer"] = (model_name, kwargs)
            return object()

    import transformers
    import huggingface_hub

    def fake_snapshot_download(**kwargs):
        calls["snapshot"] = kwargs
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(transformers, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    config = OracleRunConfig(
        model_name="inclusionAI/LLaDA2.0-Uni",
        revision="abc123",
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / ".cache"),
        prompts=["zero"],
        image_h=512,
        image_w=512,
        token_grid_h=16,
        token_grid_w=16,
        generation_steps=32,
        block_length=None,
        temperature=1.0,
        cfg_scale=2.0,
        remasking="low_confidence",
        decode_steps=8,
        decode_mode="decoder-turbo",
        resolution_multiplier=2,
        device="cpu",
        dtype="bfloat16",
        local_files_only=True,
        trust_remote_code=True,
        device_map=None,
        download_workers=1,
        seed=45,
        skip_decode=False,
    )

    _load_official_model(config)

    expected_cache = str(tmp_path / ".cache" / "huggingface")
    assert calls["snapshot"]["repo_id"] == "inclusionAI/LLaDA2.0-Uni"
    assert calls["snapshot"]["revision"] == "abc123"
    assert calls["snapshot"]["cache_dir"] == expected_cache
    assert calls["snapshot"]["max_workers"] == 1
    assert calls["model"][0] == str(tmp_path / "snapshot")
    assert calls["model"][1]["cache_dir"] == expected_cache
    assert calls["model"][1]["local_files_only"] is True
    assert calls["tokenizer"][1]["cache_dir"] == expected_cache
    assert calls["tokenizer"][1]["local_files_only"] is True
    assert calls["model_device"] == "cpu"
