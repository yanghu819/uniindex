from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .classifier import train_or_load_classifier
from .config import ProjectConfig
from .datasets import build_image_dataset
from .runtime import ensure_project_dirs, resolve_device
from .tokenizer import BaseVisionTokenizer, build_tokenizer


class TokenizedImageDataset(Dataset):
    def __init__(self, path: Path) -> None:
        payload = torch.load(path)
        self.image_tokens = payload["image_tokens"].long()
        self.labels = payload["labels"].long()
        self.grid_shape = tuple(payload["grid_shape"])
        self.image_seq_len = int(payload["image_seq_len"])
        self.codebook_size = int(payload["codebook_size"])

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image_tokens": self.image_tokens[index],
            "label": self.labels[index],
        }


def _build_raw_dataset(dataset_name: str, data_dir: Path, train: bool):
    return build_image_dataset(dataset_name=dataset_name, data_dir=data_dir, train=train)


def _prepare_image(image: Image.Image, image_size: int) -> Image.Image:
    rgb = image.convert("RGB")
    if rgb.size != (image_size, image_size):
        rgb = rgb.resize((image_size, image_size), Image.BILINEAR)
    return rgb


def _slug(value: str | None) -> str:
    raw = value or "none"
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in raw).strip("-")


def _artifact_namespace(config: ProjectConfig) -> str:
    tok = config.tokenizer
    model_slug = _slug(tok.model_name or tok.kind)
    train_limit = "all" if config.dataset.train_limit is None else str(config.dataset.train_limit)
    test_limit = "all" if config.dataset.test_limit is None else str(config.dataset.test_limit)
    compact = "1" if tok.compact_vocab else "0"
    return (
        f"{config.dataset.name}-"
        f"{tok.kind}-{model_slug}-img{tok.image_size}-train{train_limit}-test{test_limit}-compact{compact}"
    )


def tokenizer_state_path(config: ProjectConfig) -> Path:
    return config.paths.artifacts_dir / "tokenized" / _artifact_namespace(config) / "tokenizer_state.pt"


def split_path(config: ProjectConfig, split: str) -> Path:
    return config.paths.artifacts_dir / "tokenized" / _artifact_namespace(config) / f"{config.dataset.name}_{split}.pt"


def _encode_split(
    dataset,
    tokenizer: BaseVisionTokenizer,
    image_size: int,
    limit: int | None,
    out_path: Path,
    codebook_size: int,
) -> tuple[tuple[int, int], int]:
    total = len(dataset) if limit is None else min(limit, len(dataset))
    token_batches = []
    label_batches = []
    grid_shape: tuple[int, int] | None = None
    batch_size = 64

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        images = []
        labels = []
        for idx in range(start, stop):
            image, label = dataset[idx]
            images.append(_prepare_image(image, image_size))
            labels.append(label)
        codes, grid_shape = tokenizer.encode_pil_batch(images)
        token_batches.append(codes.long())
        label_batches.append(torch.tensor(labels, dtype=torch.long))

    image_tokens = torch.cat(token_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    torch.save(
        {
            "image_tokens": image_tokens,
            "labels": labels,
            "grid_shape": grid_shape,
            "image_seq_len": image_tokens.shape[1],
            "codebook_size": codebook_size,
        },
        out_path,
    )
    return grid_shape, int(image_tokens.shape[1])


def _compact_codebook_and_splits(config: ProjectConfig, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    split_payloads = {split: torch.load(split_path(config, split), map_location="cpu") for split in ("train", "test")}
    active = torch.unique(
        torch.cat([payload["image_tokens"].reshape(-1).long() for payload in split_payloads.values()], dim=0),
        sorted=True,
    )
    remap = torch.full((codebook.shape[0],), -1, dtype=torch.long)
    remap[active] = torch.arange(active.numel(), dtype=torch.long)

    for split, payload in split_payloads.items():
        payload["image_tokens"] = remap[payload["image_tokens"].long()]
        payload["codebook_size"] = int(active.numel())
        torch.save(payload, split_path(config, split))

    return codebook[active], active


def prepare_assets(config: ProjectConfig) -> None:
    ensure_project_dirs(config)
    tokenized_dir = tokenizer_state_path(config).parent
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    train_out = split_path(config, "train")
    test_out = split_path(config, "test")

    if tokenizer_state_path(config).exists() and train_out.exists() and test_out.exists():
        train_or_load_classifier(
            data_dir=config.paths.data_dir,
            models_dir=config.paths.models_dir,
            dataset_name=config.dataset.name,
            device=resolve_device(config.train.device, config.train.gpu_index),
            epochs=config.eval.classifier_epochs,
            batch_size=config.eval.classifier_batch_size,
            lr=config.eval.classifier_lr,
        )
        return

    tokenizer_device = resolve_device(config.tokenizer.device, config.train.gpu_index)
    tokenizer = build_tokenizer(config, device=tokenizer_device)
    tokenizer_artifacts = tokenizer.artifacts()

    resolved_grid_shape = None
    image_seq_len = None

    for split, limit in (("train", config.dataset.train_limit), ("test", config.dataset.test_limit)):
        out_path = split_path(config, split)
        should_reencode = config.tokenizer.compact_vocab or not out_path.exists()
        if should_reencode:
            dataset = _build_raw_dataset(config.dataset.name, config.paths.data_dir, train=split == "train")
            resolved_grid_shape, image_seq_len = _encode_split(
                dataset=dataset,
                tokenizer=tokenizer,
                image_size=config.tokenizer.image_size,
                limit=limit,
                out_path=out_path,
                codebook_size=tokenizer_artifacts.codebook_size,
            )
        else:
            payload = torch.load(out_path)
            resolved_grid_shape = tuple(payload["grid_shape"])
            image_seq_len = int(payload["image_seq_len"])

    codebook = tokenizer_artifacts.codebook
    original_token_ids = None
    original_codebook_size = int(tokenizer_artifacts.codebook_size)
    if config.tokenizer.compact_vocab:
        codebook, original_token_ids = _compact_codebook_and_splits(config, tokenizer_artifacts.codebook)

    torch.save(
        {
            "codebook": codebook,
            "codebook_size": int(codebook.shape[0]),
            "embed_dim": int(codebook.shape[1]),
            "grid_shape": resolved_grid_shape,
            "image_seq_len": image_seq_len,
            "image_size": tokenizer_artifacts.image_size,
            "label_values": torch.tensor(config.labels.values, dtype=torch.long),
            "original_token_ids": original_token_ids,
            "original_codebook_size": original_codebook_size,
            "compact_vocab": config.tokenizer.compact_vocab,
        },
        tokenizer_state_path(config),
    )

    train_or_load_classifier(
        data_dir=config.paths.data_dir,
        models_dir=config.paths.models_dir,
        dataset_name=config.dataset.name,
        device=resolve_device(config.train.device, config.train.gpu_index),
        epochs=config.eval.classifier_epochs,
        batch_size=config.eval.classifier_batch_size,
        lr=config.eval.classifier_lr,
    )


def load_tokenizer_state(config: ProjectConfig) -> dict:
    return torch.load(tokenizer_state_path(config), map_location="cpu")


def build_loader(path: Path, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    ds = TokenizedImageDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
