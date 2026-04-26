from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import ProjectConfig


LLADA2_UNI_REPO_ID = "inclusionAI/LLaDA2.0-Uni"
LLADA2_UNI_HF_REVISION = "be6d052cfd7d8033903c0622ce839b99a6f40aa8"
LLADA2_UNI_GITHUB_REVISION = "3ad52317aec756d743c02b7d0a9e37b12f252837"
LLADA2_UNI_IMAGE_TOKENIZER_URL = (
    "https://raw.githubusercontent.com/inclusionAI/LLaDA2.0-Uni/"
    f"{LLADA2_UNI_GITHUB_REVISION}/encoder/image_tokenizer.py"
)
LLADA2_UNI_IMAGE_TOKENIZER_FILES = [
    "image_tokenizer/config.json",
    "image_tokenizer/preprocessor_config.json",
    "image_tokenizer/image_tokenizer.safetensors",
]


@dataclass
class TokenizerArtifacts:
    codebook: torch.Tensor
    codebook_size: int
    embed_dim: int
    grid_shape: tuple[int, int]
    image_size: int


class BaseVisionTokenizer:
    def encode_pil_batch(self, images: Sequence[Image.Image]) -> tuple[torch.Tensor, tuple[int, int]]:
        raise NotImplementedError

    def decode_token_batch(self, tokens: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
        raise NotImplementedError

    def artifacts(self) -> TokenizerArtifacts:
        raise NotImplementedError


class DummyVisionTokenizer(BaseVisionTokenizer):
    def __init__(self, image_size: int, codebook_size: int, embed_dim: int, grid_size: int) -> None:
        self.image_size = image_size
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        centers = torch.linspace(0.0, 1.0, codebook_size).unsqueeze(1)
        features = torch.linspace(-1.0, 1.0, embed_dim).unsqueeze(0)
        self._codebook = torch.cos(centers * features * torch.pi).float()

    def encode_pil_batch(self, images: Sequence[Image.Image]) -> tuple[torch.Tensor, tuple[int, int]]:
        tokens = []
        for image in images:
            arr = np.asarray(image.convert("L").resize((self.grid_size, self.grid_size), Image.BILINEAR), dtype=np.float32) / 255.0
            token = np.clip(np.round(arr * (self.codebook_size - 1)), 0, self.codebook_size - 1).astype(np.int64)
            tokens.append(torch.from_numpy(token).reshape(-1))
        return torch.stack(tokens, dim=0), (self.grid_size, self.grid_size)

    def decode_token_batch(self, tokens: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
        batch = tokens.shape[0]
        h, w = grid_shape
        pixels = tokens.float().view(batch, 1, h, w) / max(self.codebook_size - 1, 1)
        pixels = F.interpolate(pixels, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return pixels.repeat(1, 3, 1, 1).clamp(0.0, 1.0)

    def artifacts(self) -> TokenizerArtifacts:
        return TokenizerArtifacts(
            codebook=self._codebook.clone(),
            codebook_size=self.codebook_size,
            embed_dim=self.embed_dim,
            grid_shape=(self.grid_size, self.grid_size),
            image_size=self.image_size,
        )


class EmuVisionTokenizer(BaseVisionTokenizer):
    def __init__(self, model_name: str, trust_remote_code: bool, image_size: int, device: torch.device, dtype: torch.dtype) -> None:
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model.eval()
        self.model.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.image_size = image_size
        self._codebook = self.model.quantize.embedding.weight.detach().float().cpu()

    def _preprocess_pil_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = []
        for image in images:
            rgb = image.convert("RGB")
            if rgb.size != (self.image_size, self.image_size):
                rgb = rgb.resize((self.image_size, self.image_size), Image.BICUBIC)
            array = np.asarray(rgb, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            batch.append(tensor)
        pixel_values = torch.stack(batch, dim=0)
        return pixel_values.mul(2.0).sub(1.0)

    @torch.inference_mode()
    def encode_pil_batch(self, images: Sequence[Image.Image]) -> tuple[torch.Tensor, tuple[int, int]]:
        pixel_values = self._preprocess_pil_batch(images)
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        quant_embed, _, (_, _, token_ids) = self.model.encode(pixel_values)
        batch, _, height, width = quant_embed.shape
        tokens = token_ids.view(batch, height * width).detach().cpu()
        return tokens, (height, width)

    @torch.inference_mode()
    def decode_token_batch(self, tokens: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
        batch = tokens.shape[0]
        codes = tokens.to(device=self.device, dtype=torch.long).reshape(-1)
        decoded = self.model.decode_code(codes, shape=(batch, grid_shape[0], grid_shape[1]))
        decoded = decoded.float().detach().cpu()
        if decoded.min() < 0:
            decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0
        return decoded.clamp(0.0, 1.0)

    def artifacts(self) -> TokenizerArtifacts:
        h, w = self.model.decoder.z_shape[2], self.model.decoder.z_shape[3]
        return TokenizerArtifacts(
            codebook=self._codebook.clone(),
            codebook_size=self._codebook.shape[0],
            embed_dim=self._codebook.shape[1],
            grid_shape=(h, w),
            image_size=self.image_size,
        )


def _llada_code_cache_dir(cache_dir: Path) -> Path:
    return cache_dir / "llada2_uni" / LLADA2_UNI_GITHUB_REVISION / "encoder"


def ensure_llada_image_tokenizer_source(cache_dir: Path) -> Path:
    code_dir = _llada_code_cache_dir(cache_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "__init__.py").touch()
    source_path = code_dir / "image_tokenizer.py"
    if not source_path.exists():
        urlretrieve(LLADA2_UNI_IMAGE_TOKENIZER_URL, source_path)
    return source_path


def _load_llada_image_tokenizer_class(source_path: Path):
    try:
        import torchvision.transforms.v2.functional  # noqa: F401
    except ImportError:
        _install_torchvision_v2_functional_stub()
    spec = importlib.util.spec_from_file_location("uniindex_llada2_image_tokenizer", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import LLaDA image tokenizer source at {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ImageTokenizer


def _install_torchvision_v2_functional_stub() -> None:
    functional = types.ModuleType("torchvision.transforms.v2.functional")

    def to_image(image):
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim == 2:
            array = array[:, :, None]
        return torch.from_numpy(array.copy()).permute(2, 0, 1)

    def to_dtype(tensor, dtype, scale=False):
        tensor = tensor.to(dtype=dtype)
        if scale:
            tensor = tensor / 255.0
        return tensor

    functional.to_image = to_image
    functional.to_dtype = to_dtype
    torchvision_module = sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))
    transforms_module = sys.modules.setdefault("torchvision.transforms", types.ModuleType("torchvision.transforms"))
    v2_module = sys.modules.setdefault("torchvision.transforms.v2", types.ModuleType("torchvision.transforms.v2"))
    v2_module.functional = functional
    transforms_module.v2 = v2_module
    torchvision_module.transforms = transforms_module
    sys.modules["torchvision.transforms.v2.functional"] = functional


def _download_siglip_vq_assets(*, cache_dir: Path, model_name: str) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    source_path = ensure_llada_image_tokenizer_source(cache_dir)
    hf_cache_dir = cache_dir / "huggingface"
    local_snapshot = _local_siglip_vq_snapshot(hf_cache_dir, model_name)
    if local_snapshot is not None and os.environ.get("HF_HUB_OFFLINE") == "1":
        model_dir = str(local_snapshot)
    else:
        try:
            model_dir = snapshot_download(
                repo_id=model_name,
                revision=LLADA2_UNI_HF_REVISION,
                cache_dir=str(hf_cache_dir),
                allow_patterns=LLADA2_UNI_IMAGE_TOKENIZER_FILES,
            )
        except Exception:
            local_snapshot = _local_siglip_vq_snapshot(hf_cache_dir, model_name)
            if local_snapshot is None:
                raise
            model_dir = str(local_snapshot)
    return {
        "source_path": str(source_path),
        "model_dir": str(model_dir),
    }


def _local_siglip_vq_snapshot(hf_cache_dir: Path, model_name: str) -> Path | None:
    repo_cache = hf_cache_dir / f"models--{model_name.replace('/', '--')}"
    snapshot = repo_cache / "snapshots" / LLADA2_UNI_HF_REVISION
    if all((snapshot / filename).exists() for filename in LLADA2_UNI_IMAGE_TOKENIZER_FILES):
        return snapshot
    return None


def download_siglip_vq_assets(config: ProjectConfig) -> dict[str, str]:
    return _download_siglip_vq_assets(
        cache_dir=config.paths.cache_dir,
        model_name=str(config.tokenizer.model_name or LLADA2_UNI_REPO_ID),
    )


class SiglipVQVisionTokenizer(BaseVisionTokenizer):
    def __init__(self, model_name: str, image_size: int, device: torch.device, dtype: torch.dtype, cache_dir: Path) -> None:
        self.image_size = image_size
        assets = _download_siglip_vq_assets(cache_dir=cache_dir, model_name=model_name)
        image_tokenizer_cls = _load_llada_image_tokenizer_class(Path(assets["source_path"]))
        self.model = image_tokenizer_cls(model_path=assets["model_dir"], device=str(device), dtype=dtype)
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def encode_pil_batch(self, images: Sequence[Image.Image]) -> tuple[torch.Tensor, tuple[int, int]]:
        rows = []
        grid_shape: tuple[int, int] | None = None
        for image in images:
            info = self.model.encode_with_info(image)
            grid_thw = tuple(int(value) for value in info["grid_thw"])
            current_grid = (grid_thw[1], grid_thw[2])
            if grid_shape is None:
                grid_shape = current_grid
            elif grid_shape != current_grid:
                raise ValueError(f"SigLIP-VQ batch produced mixed grid shapes: {grid_shape} and {current_grid}")
            rows.append(torch.tensor(info["token_ids"], dtype=torch.long))
        if grid_shape is None:
            raise ValueError("encode_pil_batch requires at least one image")
        return torch.stack(rows, dim=0), grid_shape

    def decode_token_batch(self, tokens: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
        raise NotImplementedError(
            "siglip_vq currently wires the LLaDA2.0-Uni encoder only; use it for i2t/probe runs, not t2i eval."
        )

    def artifacts(self) -> TokenizerArtifacts:
        codebook = self.model.vqmodel.quantize.embedding.weight.detach().float().cpu()
        patch_size = int(self.model.image_processor.patch_size)
        return TokenizerArtifacts(
            codebook=codebook.clone(),
            codebook_size=int(self.model.codebook_size),
            embed_dim=int(self.model.embed_dim),
            grid_shape=(self.image_size // patch_size, self.image_size // patch_size),
            image_size=self.image_size,
        )


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[name]


def build_tokenizer(config: ProjectConfig, device: torch.device | None = None) -> BaseVisionTokenizer:
    tok_cfg = config.tokenizer
    if tok_cfg.kind == "dummy":
        return DummyVisionTokenizer(
            image_size=tok_cfg.image_size,
            codebook_size=int(tok_cfg.codebook_size),
            embed_dim=int(tok_cfg.embed_dim),
            grid_size=int(tok_cfg.grid_size),
        )
    resolved_device = device or torch.device("cpu")
    if tok_cfg.kind == "emu3p5":
        return EmuVisionTokenizer(
            model_name=str(tok_cfg.model_name),
            trust_remote_code=bool(tok_cfg.trust_remote_code),
            image_size=tok_cfg.image_size,
            device=resolved_device,
            dtype=torch_dtype_from_name(tok_cfg.dtype),
        )
    if tok_cfg.kind == "siglip_vq":
        return SiglipVQVisionTokenizer(
            model_name=str(tok_cfg.model_name or LLADA2_UNI_REPO_ID),
            image_size=tok_cfg.image_size,
            device=resolved_device,
            dtype=torch_dtype_from_name(tok_cfg.dtype),
            cache_dir=config.paths.cache_dir,
        )
    raise ValueError(f"unsupported tokenizer kind: {tok_cfg.kind}")
