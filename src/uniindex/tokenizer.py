from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import ProjectConfig


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


def _model_cache_name(model_name: str) -> str:
    return "models--" + model_name.replace("/", "--")


def _cache_search_dirs(cache_dir: Path, repo_root: Path) -> list[Path]:
    roots = [cache_dir]
    if repo_root.parent.name == "worktrees":
        roots.append(repo_root.parent.parent / ".cache")

    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        for candidate in (
            root,
            root / "hub",
            root / "transformers",
            root / "huggingface" / "hub",
            root / "huggingface" / "transformers",
            root / "hf" / "hub",
            root / "hf" / "transformers",
        ):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    return candidates


def _has_model_files(snapshot_dir: Path) -> bool:
    if not (snapshot_dir / "config.json").exists():
        return False
    return any(
        (snapshot_dir / filename).exists()
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.bin",
        )
    )


def _snapshot_from_cache(cache_root: Path, model_name: str) -> Path | None:
    snapshots_dir = cache_root / _model_cache_name(model_name) / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_path = cache_root / _model_cache_name(model_name) / "refs" / "main"
    if ref_path.exists():
        snapshot = snapshots_dir / ref_path.read_text(encoding="utf-8").strip()
        if _has_model_files(snapshot):
            return snapshot

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir() and _has_model_files(path)]
    if not snapshots:
        return None
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def _resolve_local_model_source(model_name: str, cache_dir: Path, repo_root: Path) -> tuple[str, str | None]:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path.resolve()), None

    searched: list[str] = []
    for cache_root in _cache_search_dirs(cache_dir, repo_root):
        searched.append(str(cache_root))
        snapshot = _snapshot_from_cache(cache_root, model_name)
        if snapshot is not None:
            return str(snapshot), str(cache_root)

    message = (
        f"missing local vision tokenizer model '{model_name}'. "
        "Expected a complete local snapshot under one of: "
        + ", ".join(searched)
        + ". Put the decoder files under the repo-local cache; this loader does not download models."
    )
    raise FileNotFoundError(message)


class EmuVisionTokenizer(BaseVisionTokenizer):
    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool,
        image_size: int,
        device: torch.device,
        dtype: torch.dtype,
        cache_dir: Path,
        repo_root: Path,
    ) -> None:
        from transformers import AutoModel

        model_source, resolved_cache = _resolve_local_model_source(model_name, cache_dir, repo_root)
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": True,
        }
        if resolved_cache is not None:
            load_kwargs["cache_dir"] = resolved_cache
        self.model = AutoModel.from_pretrained(
            model_source,
            **load_kwargs,
        )
        self.model.eval()
        self.model.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.image_size = image_size
        self.model_source = model_source
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
    return EmuVisionTokenizer(
        model_name=str(tok_cfg.model_name),
        trust_remote_code=bool(tok_cfg.trust_remote_code),
        image_size=tok_cfg.image_size,
        device=resolved_device,
        dtype=torch_dtype_from_name(tok_cfg.dtype),
        cache_dir=config.paths.cache_dir,
        repo_root=config.repo_root,
    )
