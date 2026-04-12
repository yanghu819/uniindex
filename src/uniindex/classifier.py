from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.GELU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def classifier_path(models_dir: Path) -> Path:
    path = models_dir / "eval" / "mnist_classifier.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_classifier(path: Path, device: torch.device) -> MNISTClassifier:
    model = MNISTClassifier().to(device)
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def train_or_load_classifier(
    data_dir: Path,
    models_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    force_retrain: bool = False,
) -> Path:
    path = classifier_path(models_dir)
    if path.exists() and not force_retrain:
        return path

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MNISTClassifier().to(device)
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

    torch.save({"model": model.state_dict(), "accuracy": correct / max(total, 1)}, path)
    return path


def classify_images(model: MNISTClassifier, images: torch.Tensor) -> torch.Tensor:
    gray = images.mean(dim=1, keepdim=True)
    gray = F.interpolate(gray, size=(28, 28), mode="bilinear", align_corners=False)
    logits = model(gray)
    return logits.argmax(dim=-1)
