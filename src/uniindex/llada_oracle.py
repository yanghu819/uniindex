from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw, ImageFont

from .tokenizer import (
    LLADA2_UNI_DECODER_ASSET_FILES,
    LLADA2_UNI_REPO_ID,
    _download_siglip_vq_assets,
    _load_llada_decoder_module,
)


_DIGIT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)


@dataclass
class OracleRunConfig:
    model_name: str
    output_dir: str
    cache_dir: str
    prompts: list[str]
    image_h: int
    image_w: int
    token_grid_h: int | None
    token_grid_w: int | None
    generation_steps: int
    block_length: int | None
    temperature: float
    cfg_scale: float
    remasking: str
    decode_steps: int
    decode_mode: str
    resolution_multiplier: int
    device: str
    dtype: str
    local_files_only: bool
    trust_remote_code: bool
    device_map: str | None
    seed: int
    skip_decode: bool


def digit_prompts(prefix: str) -> list[str]:
    return [prefix.format(label=label, digit=i) for i, label in enumerate(_DIGIT_WORDS)]


def normalize_image_token_rows(value: Any) -> list[list[int]]:
    if isinstance(value, dict):
        for key in ("image_tokens", "tokens", "sequences", "output_ids"):
            if key in value:
                return normalize_image_token_rows(value[key])
        raise TypeError(f"could not find image tokens in result keys: {sorted(value)}")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().long()
        if tensor.ndim == 1:
            return [tensor.tolist()]
        if tensor.ndim == 2:
            return [row.tolist() for row in tensor]
        raise TypeError(f"expected 1D/2D token tensor, got shape {tuple(tensor.shape)}")
    if isinstance(value, tuple) and len(value) == 1:
        return normalize_image_token_rows(value[0])
    if isinstance(value, tuple):
        for item in value:
            try:
                return normalize_image_token_rows(item)
            except TypeError:
                continue
        raise TypeError("could not normalize any tuple item as image tokens")
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, int) for item in value):
            return [[int(item) for item in value]]
        rows: list[list[int]] = []
        for item in value:
            rows.extend(normalize_image_token_rows(item))
        return rows
    raise TypeError(f"unsupported image-token result type: {type(value).__name__}")


def infer_grid_shape(token_count: int, requested_h: int | None, requested_w: int | None) -> tuple[int, int]:
    if requested_h is not None and requested_w is not None:
        if requested_h * requested_w != token_count:
            raise ValueError(
                f"requested token grid {requested_h}x{requested_w} does not match {token_count} tokens"
            )
        return requested_h, requested_w
    side = int(round(token_count**0.5))
    if side * side == token_count:
        return side, side
    if requested_h is not None and token_count % requested_h == 0:
        return requested_h, token_count // requested_h
    if requested_w is not None and token_count % requested_w == 0:
        return token_count // requested_w, requested_w
    raise ValueError(f"cannot infer a rectangular token grid from {token_count} tokens")


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _load_official_model(config: OracleRunConfig):
    from transformers import AutoModel, AutoTokenizer

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": _dtype_from_name(config.dtype),
        "local_files_only": config.local_files_only,
    }
    if config.device_map:
        load_kwargs["device_map"] = config.device_map
    model = AutoModel.from_pretrained(config.model_name, **load_kwargs)
    if not config.device_map:
        model = model.to(config.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    return model, tokenizer


def _call_generate_image(model, tokenizer, prompt: str, config: OracleRunConfig):
    if not hasattr(model, "generate_image"):
        raise AttributeError("official model has no generate_image method; check trust_remote_code/model revision")
    method = model.generate_image
    candidates: dict[str, Any] = {
        "tokenizer": tokenizer,
        "prompt": prompt,
        "image_h": config.image_h,
        "image_w": config.image_w,
        "steps": config.generation_steps,
        "num_steps": config.generation_steps,
        "gen_steps": config.generation_steps,
        "generation_steps": config.generation_steps,
        "temperature": config.temperature,
        "cfg_scale": config.cfg_scale,
        "remasking": config.remasking,
    }
    if config.block_length is not None:
        candidates["block_length"] = config.block_length

    signature = inspect.signature(method)
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_var_kwargs:
        kwargs = candidates
    else:
        kwargs = {key: value for key, value in candidates.items() if key in signature.parameters}
    return method(**kwargs), sorted(kwargs)


def _make_grid(images: Iterable[Image.Image], captions: list[str], output_path: Path, cell_size: int = 192) -> None:
    images = [image.convert("RGB").resize((cell_size, cell_size), Image.BICUBIC) for image in images]
    if len(images) != len(captions):
        raise ValueError("images and captions must have the same length")
    cols = min(5, max(1, len(images)))
    rows = (len(images) + cols - 1) // cols
    caption_h = 36
    canvas = Image.new("RGB", (cols * cell_size, rows * (cell_size + caption_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (image, caption) in enumerate(zip(images, captions)):
        row = index // cols
        col = index % cols
        x = col * cell_size
        y = row * (cell_size + caption_h)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y + cell_size, x + cell_size, y + cell_size + caption_h), fill=(246, 246, 246))
        draw.text((x + 5, y + cell_size + 8), caption[:36], fill=(0, 0, 0), font=font)
    canvas.save(output_path)


@torch.inference_mode()
def run_oracle(config: OracleRunConfig) -> dict[str, Any]:
    if config.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_official_model(config)
    decoder_assets = _download_siglip_vq_assets(
        cache_dir=cache_dir,
        model_name=config.model_name,
        allow_patterns=LLADA2_UNI_DECODER_ASSET_FILES,
    )
    decoder = _load_llada_decoder_module(cache_dir)

    records = []
    decoded_images: list[Image.Image] = []
    for index, prompt in enumerate(config.prompts):
        result, generation_kwargs = _call_generate_image(model, tokenizer, prompt, config)
        rows = normalize_image_token_rows(result)
        if not rows:
            raise RuntimeError(f"generate_image returned no image tokens for prompt {prompt!r}")
        token_ids = rows[0]
        grid_h, grid_w = infer_grid_shape(len(token_ids), config.token_grid_h, config.token_grid_w)
        token_path = output_dir / f"sample_{index:02d}_tokens.json"
        with token_path.open("w", encoding="utf-8") as handle:
            json.dump({"prompt": prompt, "tokens": token_ids, "grid_shape": [grid_h, grid_w]}, handle)

        image_path = None
        if not config.skip_decode:
            image = decoder.decode_vq_tokens(
                token_ids,
                h=grid_h,
                w=grid_w,
                model_path=decoder_assets["model_dir"],
                device=torch.device(config.device),
                resolution_multiplier=config.resolution_multiplier,
                num_steps=config.decode_steps,
                decode_mode=config.decode_mode,
            )
            image_path = output_dir / f"sample_{index:02d}.png"
            image.save(image_path)
            decoded_images.append(image)

        records.append(
            {
                "prompt": prompt,
                "token_count": len(token_ids),
                "grid_shape": [grid_h, grid_w],
                "token_path": str(token_path),
                "image_path": str(image_path) if image_path is not None else None,
                "generate_image_kwargs": generation_kwargs,
            }
        )

    grid_path = None
    if decoded_images:
        grid_path = output_dir / "llada_uni_oracle_grid.png"
        _make_grid(decoded_images, [str(i) for i in range(len(decoded_images))], grid_path)

    summary = {
        "status": "ok",
        "config": asdict(config),
        "samples": records,
        "grid_path": str(grid_path) if grid_path is not None else None,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _parse_args() -> OracleRunConfig:
    parser = argparse.ArgumentParser(description="Run the official LLaDA-Uni image-generation oracle.")
    parser.add_argument("--model-name", default=LLADA2_UNI_REPO_ID)
    parser.add_argument("--output-dir", default="output/llada_uni_oracle")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--digit-prompts", action="store_true")
    parser.add_argument("--digit-prefix", default="A clean handwritten digit {label}.")
    parser.add_argument("--image-h", type=int, default=512)
    parser.add_argument("--image-w", type=int, default=512)
    parser.add_argument("--token-grid-h", type=int, default=None)
    parser.add_argument("--token-grid-w", type=int, default=None)
    parser.add_argument("--generation-steps", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--decode-mode", default="decoder-turbo")
    parser.add_argument("--resolution-multiplier", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    prompts = args.prompt or []
    if args.digit_prompts:
        prompts.extend(digit_prompts(args.digit_prefix))
    if not prompts:
        prompts = ["A clean handwritten digit zero."]
    local_files_only = bool(args.local_files_only) and not bool(args.allow_download)
    return OracleRunConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        prompts=prompts,
        image_h=args.image_h,
        image_w=args.image_w,
        token_grid_h=args.token_grid_h,
        token_grid_w=args.token_grid_w,
        generation_steps=args.generation_steps,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        decode_steps=args.decode_steps,
        decode_mode=args.decode_mode,
        resolution_multiplier=args.resolution_multiplier,
        device=args.device,
        dtype=args.dtype,
        local_files_only=local_files_only,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.device_map,
        seed=args.seed,
        skip_decode=bool(args.skip_decode),
    )


def main() -> int:
    config = _parse_args()
    try:
        summary = run_oracle(config)
    except Exception as exc:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {"status": "failed", "config": asdict(config), "error": repr(exc)}
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, indent=2)
        raise
    print(json.dumps({"summary": str(Path(config.output_dir) / "summary.json"), "grid": summary["grid_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
