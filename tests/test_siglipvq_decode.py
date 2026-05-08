import sys
from types import ModuleType, SimpleNamespace

import torch
from PIL import Image

from uniindex.tokenizer import (
    LLADA2_UNI_DECODER_ASSET_FILES,
    LLADA2_UNI_IMAGE_TOKENIZER_FILES,
    SiglipVQVisionTokenizer,
    download_siglip_vq_decoder_assets,
    _install_diffusers_attention_dispatch_compat,
    _resolve_local_model_source,
)


def test_siglipvq_decode_token_batch_uses_llada_decoder(monkeypatch, tmp_path):
    calls = []

    def fake_download(*, cache_dir, model_name, allow_patterns):
        calls.append((cache_dir, model_name, tuple(allow_patterns)))
        return {"source_path": str(tmp_path / "image_tokenizer.py"), "model_dir": str(tmp_path / "snapshot")}

    def fake_load_decoder(cache_dir):
        assert cache_dir == tmp_path

        def decode_vq_tokens(token_ids, h, w, model_path, device, resolution_multiplier, num_steps, decode_mode):
            calls.append((tuple(token_ids), h, w, model_path, str(device), resolution_multiplier, num_steps, decode_mode))
            return Image.new("RGB", (16, 16), color=(255, 0, 0))

        return SimpleNamespace(decode_vq_tokens=decode_vq_tokens)

    monkeypatch.setattr("uniindex.tokenizer._download_siglip_vq_assets", fake_download)
    monkeypatch.setattr("uniindex.tokenizer._load_llada_decoder_module", fake_load_decoder)

    tokenizer = SiglipVQVisionTokenizer.__new__(SiglipVQVisionTokenizer)
    tokenizer.image_size = 8
    tokenizer.model_name = "inclusionAI/LLaDA2.0-Uni"
    tokenizer.cache_dir = tmp_path
    tokenizer.device = torch.device("cpu")
    tokenizer.dtype = torch.float32

    decoded = tokenizer.decode_token_batch(torch.tensor([[1, 2, 3, 4]]), grid_shape=(2, 2))

    assert decoded.shape == (1, 3, 8, 8)
    assert calls[0] == (tmp_path, "inclusionAI/LLaDA2.0-Uni", tuple(LLADA2_UNI_DECODER_ASSET_FILES))
    assert calls[1] == ((1, 2, 3, 4), 2, 2, str(tmp_path / "snapshot"), "cpu", 2, 8, "decoder-turbo")


def test_decoder_asset_download_extends_encoder_assets(monkeypatch, tmp_path):
    captured = {}

    def fake_ensure_decoder_source(cache_dir):
        captured["decoder_source_cache"] = cache_dir
        return cache_dir / "llada2_uni" / "decoder"

    def fake_download(*, cache_dir, model_name, allow_patterns):
        captured["download"] = (cache_dir, model_name, allow_patterns)
        return {"source_path": str(cache_dir / "encoder.py"), "model_dir": str(cache_dir / "snapshot")}

    monkeypatch.setattr("uniindex.tokenizer.ensure_llada_decoder_source", fake_ensure_decoder_source)
    monkeypatch.setattr("uniindex.tokenizer._download_siglip_vq_assets", fake_download)

    config = SimpleNamespace(
        paths=SimpleNamespace(cache_dir=tmp_path),
        tokenizer=SimpleNamespace(model_name="inclusionAI/LLaDA2.0-Uni"),
    )
    result = download_siglip_vq_decoder_assets(config)

    assert result["decoder_source_dir"].startswith(str(tmp_path))
    assert set(LLADA2_UNI_IMAGE_TOKENIZER_FILES).issubset(set(captured["download"][2]))
    assert "decoder-turbo/decoder_model.safetensors" in captured["download"][2]
    assert "vae/diffusion_pytorch_model.safetensors" in captured["download"][2]


def test_local_model_source_resolves_huggingface_snapshot(tmp_path):
    cache_root = tmp_path / ".cache" / "huggingface" / "hub"
    repo_cache = cache_root / "models--BAAI--Emu3.5-VisionTokenizer"
    snapshot = repo_cache / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_text("weights", encoding="utf-8")
    refs = repo_cache / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123", encoding="utf-8")

    source, resolved_cache = _resolve_local_model_source(
        "BAAI/Emu3.5-VisionTokenizer",
        tmp_path / ".cache",
        tmp_path,
    )

    assert source == str(snapshot)
    assert resolved_cache == str(cache_root)


def test_diffusers_attention_dispatch_compat_drops_parallel_config(monkeypatch):
    diffusers = ModuleType("diffusers")
    diffusers.__path__ = []
    models = ModuleType("diffusers.models")
    models.__path__ = []
    attention_processor = ModuleType("diffusers.models.attention_processor")
    calls = []

    def dispatch_attention_fn(query, *, scale=1.0):
        calls.append((float(query.item()), scale))
        return query * scale

    attention_processor.dispatch_attention_fn = dispatch_attention_fn
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.models", models)
    monkeypatch.setitem(sys.modules, "diffusers.models.attention_processor", attention_processor)

    _install_diffusers_attention_dispatch_compat()

    result = attention_processor.dispatch_attention_fn(torch.tensor(2.0), scale=3.0, parallel_config=object())

    assert result.item() == 6.0
    assert calls == [(2.0, 3.0)]
