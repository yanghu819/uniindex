from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
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
LLADA2_UNI_DECODER_SOURCE_FILES = [
    "decoder/__init__.py",
    "decoder/decode.py",
    "decoder/decoder_model.py",
    "decoder/sigvq.py",
    "decoder/smart_img_process.py",
    "decoder/utils.py",
    "decoder/transport/__init__.py",
    "decoder/transport/dpm_solver.py",
    "decoder/transport/integrators.py",
    "decoder/transport/path.py",
    "decoder/transport/transport.py",
    "decoder/transport/utils.py",
]
LLADA2_UNI_DECODER_ASSET_FILES = [
    *LLADA2_UNI_IMAGE_TOKENIZER_FILES,
    "image_tokenizer/sigvq_embedding.pt",
    "decoder-turbo/config.json",
    "decoder-turbo/decoder_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
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


def _llada_decoder_cache_dir(cache_dir: Path) -> Path:
    return cache_dir / "llada2_uni" / LLADA2_UNI_GITHUB_REVISION / "decoder"


def ensure_llada_image_tokenizer_source(cache_dir: Path) -> Path:
    code_dir = _llada_code_cache_dir(cache_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "__init__.py").touch()
    source_path = code_dir / "image_tokenizer.py"
    if not source_path.exists():
        urlretrieve(LLADA2_UNI_IMAGE_TOKENIZER_URL, source_path)
    return source_path


def ensure_llada_decoder_source(cache_dir: Path) -> Path:
    code_dir = _llada_decoder_cache_dir(cache_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in LLADA2_UNI_DECODER_SOURCE_FILES:
        output_path = code_dir / relative_path.removeprefix("decoder/")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists():
            source_url = (
                "https://raw.githubusercontent.com/inclusionAI/LLaDA2.0-Uni/"
                f"{LLADA2_UNI_GITHUB_REVISION}/{relative_path}"
            )
            urlretrieve(source_url, output_path)
    return code_dir


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
    transforms_functional = types.ModuleType("torchvision.transforms.functional")
    transforms_functional.__spec__ = ModuleSpec("torchvision.transforms.functional", loader=None)

    def to_pil_image(tensor):
        array = tensor.detach().cpu().clamp(0.0, 1.0)
        if array.dim() == 2:
            array = array.unsqueeze(0)
        if array.shape[0] == 1:
            array = array.repeat(3, 1, 1)
        array = (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array)

    transforms_functional.to_pil_image = to_pil_image

    functional = types.ModuleType("torchvision.transforms.v2.functional")
    functional.__spec__ = ModuleSpec("torchvision.transforms.v2.functional", loader=None)

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
    torchvision_module.__spec__ = ModuleSpec("torchvision", loader=None)
    transforms_module.__spec__ = ModuleSpec("torchvision.transforms", loader=None)
    v2_module.__spec__ = ModuleSpec("torchvision.transforms.v2", loader=None)
    transforms_module.functional = transforms_functional
    v2_module.functional = functional
    transforms_module.v2 = v2_module
    torchvision_module.transforms = transforms_module
    sys.modules["torchvision.transforms.functional"] = transforms_functional
    sys.modules["torchvision.transforms.v2.functional"] = functional


def _install_llada_decoder_dependency_stubs() -> None:
    try:
        import torchvision.transforms.functional  # noqa: F401
        import torchvision.transforms.v2.functional  # noqa: F401
    except ImportError:
        _install_torchvision_v2_functional_stub()

    if importlib.util.find_spec("flash_attn") is None:
        flash_attn = types.ModuleType("flash_attn")
        flash_attn.__spec__ = ModuleSpec("flash_attn", loader=None)

        def flash_attn_func(query, key, value, dropout_p=0.0, **_kwargs):
            if query.dim() != 4:
                raise ValueError(f"flash_attn_func stub expects 4D tensors, got {tuple(query.shape)}")
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=float(dropout_p))
            return output.transpose(1, 2)

        flash_attn.flash_attn_func = flash_attn_func
        sys.modules["flash_attn"] = flash_attn

    if importlib.util.find_spec("torchdiffeq") is None:
        torchdiffeq = types.ModuleType("torchdiffeq")
        torchdiffeq.__spec__ = ModuleSpec("torchdiffeq", loader=None)

        def odeint(func, y0, t, method=None, atol=None, rtol=None):  # noqa: ARG001
            values = [y0]
            y = y0
            for index in range(len(t) - 1):
                dt = t[index + 1] - t[index]
                y = y + dt * func(t[index], y)
                values.append(y)
            return torch.stack(values, dim=0)

        torchdiffeq.odeint = odeint
        sys.modules["torchdiffeq"] = torchdiffeq


def _load_llada_decoder_module(cache_dir: Path):
    _install_llada_decoder_dependency_stubs()
    code_dir = ensure_llada_decoder_source(cache_dir)
    package_name = "uniindex_llada2_decoder"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__spec__ = ModuleSpec(package_name, loader=None, is_package=True)
        package.__path__ = [str(code_dir)]
        sys.modules[package_name] = package
    module_name = f"{package_name}.decode"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, code_dir / "decode.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import LLaDA decoder source at {code_dir / 'decode.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _download_siglip_vq_assets(
    *,
    cache_dir: Path,
    model_name: str,
    allow_patterns: list[str],
) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    source_path = ensure_llada_image_tokenizer_source(cache_dir)
    hf_cache_dir = cache_dir / "huggingface"
    local_snapshot = _local_siglip_vq_snapshot(hf_cache_dir, model_name, required_files=allow_patterns)
    if local_snapshot is not None and os.environ.get("HF_HUB_OFFLINE") == "1":
        model_dir = str(local_snapshot)
    else:
        try:
            model_dir = snapshot_download(
                repo_id=model_name,
                revision=LLADA2_UNI_HF_REVISION,
                cache_dir=str(hf_cache_dir),
                allow_patterns=allow_patterns,
            )
        except Exception:
            local_snapshot = _local_siglip_vq_snapshot(hf_cache_dir, model_name, required_files=allow_patterns)
            if local_snapshot is None:
                raise
            model_dir = str(local_snapshot)
    return {
        "source_path": str(source_path),
        "model_dir": str(model_dir),
    }


def _local_siglip_vq_snapshot(
    hf_cache_dir: Path,
    model_name: str,
    *,
    required_files: list[str] | None = None,
) -> Path | None:
    repo_cache = hf_cache_dir / f"models--{model_name.replace('/', '--')}"
    snapshot = repo_cache / "snapshots" / LLADA2_UNI_HF_REVISION
    files = required_files or LLADA2_UNI_IMAGE_TOKENIZER_FILES
    if all((snapshot / filename).exists() for filename in files):
        return snapshot
    return None


def download_siglip_vq_assets(config: ProjectConfig) -> dict[str, str]:
    return _download_siglip_vq_assets(
        cache_dir=config.paths.cache_dir,
        model_name=str(config.tokenizer.model_name or LLADA2_UNI_REPO_ID),
        allow_patterns=LLADA2_UNI_IMAGE_TOKENIZER_FILES,
    )


def download_siglip_vq_decoder_assets(config: ProjectConfig) -> dict[str, str]:
    source_dir = ensure_llada_decoder_source(config.paths.cache_dir)
    assets = _download_siglip_vq_assets(
        cache_dir=config.paths.cache_dir,
        model_name=str(config.tokenizer.model_name or LLADA2_UNI_REPO_ID),
        allow_patterns=LLADA2_UNI_DECODER_ASSET_FILES,
    )
    return {
        **assets,
        "decoder_source_dir": str(source_dir),
    }


class SiglipVQVisionTokenizer(BaseVisionTokenizer):
    def __init__(self, model_name: str, image_size: int, device: torch.device, dtype: torch.dtype, cache_dir: Path) -> None:
        self.image_size = image_size
        self.model_name = model_name
        self.cache_dir = cache_dir
        assets = _download_siglip_vq_assets(
            cache_dir=cache_dir,
            model_name=model_name,
            allow_patterns=LLADA2_UNI_IMAGE_TOKENIZER_FILES,
        )
        image_tokenizer_cls = _load_llada_image_tokenizer_class(Path(assets["source_path"]))
        self.model = image_tokenizer_cls(model_path=assets["model_dir"], device=str(device), dtype=dtype)
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def encode_pil_batch(self, images: Sequence[Image.Image]) -> tuple[torch.Tensor, tuple[int, int]]:
        if not images:
            raise ValueError("encode_pil_batch requires at least one image")
        token_rows = self.model.encode_batch(list(images))
        rows = [torch.tensor(tokens, dtype=torch.long) for tokens in token_rows]
        patch_size = int(self.model.image_processor.patch_size)
        grid_shape = (self.image_size // patch_size, self.image_size // patch_size)
        return torch.stack(rows, dim=0), grid_shape

    def decode_token_batch(self, tokens: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
        if tokens.numel() == 0:
            return torch.empty(0, 3, self.image_size, self.image_size)
        assets = _download_siglip_vq_assets(
            cache_dir=self.cache_dir,
            model_name=self.model_name,
            allow_patterns=LLADA2_UNI_DECODER_ASSET_FILES,
        )
        decoder = _load_llada_decoder_module(self.cache_dir)
        decoded = []
        for row in tokens.detach().cpu().long():
            image = decoder.decode_vq_tokens(
                row.tolist(),
                h=int(grid_shape[0]),
                w=int(grid_shape[1]),
                model_path=assets["model_dir"],
                device=self.device,
                resolution_multiplier=1,
                num_steps=8,
                decode_mode="decoder-turbo",
            )
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            if tensor.shape[-2:] != (self.image_size, self.image_size):
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            decoded.append(tensor)
        return torch.stack(decoded, dim=0).clamp(0.0, 1.0)

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
