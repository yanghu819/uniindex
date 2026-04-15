from __future__ import annotations

from pathlib import Path
import pickle
import subprocess
import tarfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def ensure_cifar10_downloaded(data_dir: Path) -> None:
    extracted_dir = data_dir / "cifar-10-batches-py"
    if (extracted_dir / "data_batch_1").exists():
        return

    archive_path = data_dir / "cifar-10-python.tar.gz"
    if archive_path.exists() and archive_path.stat().st_size == 0:
        archive_path.unlink()

    if not archive_path.exists():
        urls = [
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
        ]
        last_error: Exception | None = None
        for url in urls:
            try:
                subprocess.run(
                    [
                        "curl",
                        "-L",
                        "--fail",
                        "--retry",
                        "3",
                        "--connect-timeout",
                        "20",
                        "--max-time",
                        "1200",
                        "-o",
                        str(archive_path),
                        url,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                break
            except Exception as exc:
                last_error = exc
                if archive_path.exists():
                    archive_path.unlink()
        else:
            raise RuntimeError("failed to download CIFAR-10 archive") from last_error

    with tarfile.open(archive_path, "r:gz") as handle:
        handle.extractall(path=data_dir)


class CIFAR10PythonDataset(Dataset):
    def __init__(self, root: Path, train: bool, transform=None) -> None:
        ensure_cifar10_downloaded(root)
        extracted_dir = root / "cifar-10-batches-py"
        batch_names = (
            [f"data_batch_{index}" for index in range(1, 6)]
            if train
            else ["test_batch"]
        )
        image_chunks = []
        labels: list[int] = []

        for batch_name in batch_names:
            with (extracted_dir / batch_name).open("rb") as handle:
                payload = pickle.load(handle, encoding="latin1")
            image_chunks.append(payload["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            labels.extend(int(value) for value in payload["labels"])

        self.images = np.concatenate(image_chunks, axis=0)
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index], mode="RGB")
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def build_image_dataset(dataset_name: str, data_dir: Path, train: bool, transform=None):
    if dataset_name == "cifar10":
        return CIFAR10PythonDataset(root=data_dir, train=train, transform=transform)
    if dataset_name == "mnist":
        from torchvision import datasets as tv_datasets

        return tv_datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    raise ValueError(f"unsupported dataset {dataset_name}")
