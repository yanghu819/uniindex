from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _dataset_num_classes(dataset_name: str) -> int:
    if dataset_name in {"mnist", "cifar10"}:
        return 10
    raise ValueError(f"unsupported dataset {dataset_name}")


def _dataset_input_spec(dataset_name: str) -> tuple[int, int]:
    if dataset_name == "mnist":
        return 1, 28
    if dataset_name == "cifar10":
        return 3, 32
    raise ValueError(f"unsupported dataset {dataset_name}")


def _build_dataset(dataset_name: str, data_dir: Path, train: bool, transform) -> torch.utils.data.Dataset:
    if dataset_name == "mnist":
        return datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    if dataset_name == "cifar10":
        return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    raise ValueError(f"unsupported dataset {dataset_name}")


class SmallImageClassifier(nn.Module):
    def __init__(self, in_channels: int, image_size: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_classes = num_classes
        features = image_size // 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * features * features, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_classifier(dataset_name: str) -> SmallImageClassifier:
    in_channels, image_size = _dataset_input_spec(dataset_name)
    return SmallImageClassifier(
        in_channels=in_channels,
        image_size=image_size,
        num_classes=_dataset_num_classes(dataset_name),
    )


def classifier_path(models_dir: Path, dataset_name: str) -> Path:
    path = models_dir / "eval" / f"{dataset_name}_classifier.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_classifier(path: Path, dataset_name: str, device: torch.device) -> SmallImageClassifier:
    model = _build_classifier(dataset_name).to(device)
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def train_or_load_classifier(
    data_dir: Path,
    models_dir: Path,
    dataset_name: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    force_retrain: bool = False,
) -> Path:
    path = classifier_path(models_dir, dataset_name)
    if path.exists() and not force_retrain:
        return path

    transform = transforms.ToTensor()
    train_ds = _build_dataset(dataset_name, data_dir, train=True, transform=transform)
    test_ds = _build_dataset(dataset_name, data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = _build_classifier(dataset_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images).argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()

    torch.save(
        {
            "model": model.state_dict(),
            "accuracy": correct / max(total, 1),
            "dataset_name": dataset_name,
        },
        path,
    )
    return path


def classify_images(model: SmallImageClassifier, images: torch.Tensor, dataset_name: str) -> torch.Tensor:
    if dataset_name == "mnist":
        prepared = images.mean(dim=1, keepdim=True)
        prepared = F.interpolate(prepared, size=(28, 28), mode="bilinear", align_corners=False)
    elif dataset_name == "cifar10":
        prepared = images
        if prepared.shape[1] == 1:
            prepared = prepared.repeat(1, 3, 1, 1)
        prepared = F.interpolate(prepared, size=(32, 32), mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"unsupported dataset {dataset_name}")
    logits = model(prepared)
    return logits.argmax(dim=-1)
