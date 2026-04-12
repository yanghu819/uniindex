from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .classifier import train_or_load_classifier
from .config import ProjectConfig
from .runtime import ensure_project_dirs, resolve_device
from .tokenizer import BaseVisionTokenizer, build_tokenizer


class TokenizedMNISTDataset(Dataset):
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


def _mnist_dataset(data_dir: Path, train: bool) -> datasets.MNIST:
    return datasets.MNIST(root=data_dir, train=train, download=True)


def _prepare_image(image: Image.Image, image_size: int) -> Image.Image:
    rgb = image.convert("RGB")
    if rgb.size != (image_size, image_size):
        rgb = rgb.resize((image_size, image_size), Image.BILINEAR)
    return rgb


def tokenizer_state_path(config: ProjectConfig) -> Path:
    return config.paths.artifacts_dir / "tokenized" / "tokenizer_state.pt"


def split_path(config: ProjectConfig, split: str) -> Path:
    return config.paths.artifacts_dir / "tokenized" / f"mnist_{split}.pt"


def _encode_split(
    dataset: datasets.MNIST,
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


def prepare_assets(config: ProjectConfig) -> None:
    ensure_project_dirs(config)
    tokenized_dir = config.paths.artifacts_dir / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_device = resolve_device(config.tokenizer.device, config.train.gpu_index)
    tokenizer = build_tokenizer(config, device=tokenizer_device)
    tokenizer_artifacts = tokenizer.artifacts()

    resolved_grid_shape = None
    image_seq_len = None

    for split, limit in (("train", config.dataset.train_limit), ("test", config.dataset.test_limit)):
        out_path = split_path(config, split)
        if not out_path.exists():
            dataset = _mnist_dataset(config.paths.data_dir, train=split == "train")
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

    torch.save(
        {
            "codebook": tokenizer_artifacts.codebook,
            "codebook_size": tokenizer_artifacts.codebook_size,
            "embed_dim": tokenizer_artifacts.embed_dim,
            "grid_shape": resolved_grid_shape,
            "image_seq_len": image_seq_len,
            "image_size": tokenizer_artifacts.image_size,
            "label_values": torch.tensor(config.labels.values, dtype=torch.long),
        },
        tokenizer_state_path(config),
    )

    train_or_load_classifier(
        data_dir=config.paths.data_dir,
        models_dir=config.paths.models_dir,
        device=resolve_device(config.train.device, config.train.gpu_index),
        epochs=config.eval.classifier_epochs,
        batch_size=config.eval.classifier_batch_size,
        lr=config.eval.classifier_lr,
    )


def load_tokenizer_state(config: ProjectConfig) -> dict:
    return torch.load(tokenizer_state_path(config), map_location="cpu")


def build_loader(path: Path, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    ds = TokenizedMNISTDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
