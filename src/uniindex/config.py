from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    artifacts_dir: Path
    models_dir: Path
    runs_dir: Path
    logs_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class TokenizerConfig:
    kind: str
    model_name: str | None = None
    trust_remote_code: bool | None = None
    image_size: int = 256
    device: str = "cpu"
    dtype: str = "float32"
    codebook_size: int | None = None
    embed_dim: int | None = None
    grid_size: int | None = None
    compact_vocab: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    train_limit: int | None = None
    test_limit: int | None = None


@dataclass(frozen=True)
class TextConfig:
    kind: str = "char"
    strings: list[str] | None = None
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"


@dataclass(frozen=True)
class LabelsConfig:
    values: list[int]


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int
    dropout: float


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: str
    gpu_index: int
    mixed_precision: str | None
    batch_size: int
    eval_batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    grad_clip_norm: float
    stage1_steps: int
    stage2_steps: int
    log_every: int
    save_every: int
    joint_weight: float
    text_weight: float
    stage2_joint_repeats: int = 2
    stage2_text_to_image_repeats: int = 1
    stage2_image_to_text_repeats: int = 1
    image_time_power: float = 1.0
    text_time_power: float = 1.0
    image_to_text_text_time_power: float | None = None
    text_sequence_weight: float = 0.0


@dataclass(frozen=True)
class SamplingConfig:
    steps: int
    temperature: float
    image_time_power: float = 1.0
    text_time_power: float = 1.0
    image_to_text_text_time_power: float | None = None


@dataclass(frozen=True)
class ScheduleConfig:
    kind: str = "power"
    num_points: int = 33
    num_samples: int = 4096
    min_t: float = 0.0


@dataclass(frozen=True)
class EvalConfig:
    num_unconditional_samples: int
    classifier_epochs: int
    classifier_batch_size: int
    classifier_lr: float


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    repo_root: Path
    paths: PathsConfig
    tokenizer: TokenizerConfig
    dataset: DatasetConfig
    text: TextConfig
    labels: LabelsConfig
    model: ModelConfig
    train: TrainConfig
    sampling: SamplingConfig
    schedule: ScheduleConfig
    eval: EvalConfig


def _resolve(base: Path, raw: str) -> Path:
    return (base / raw).resolve()


def _default_text_strings(dataset_name: str, label_values: list[int]) -> list[str]:
    mnist_map = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }
    cifar10_map = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    dataset_map = {
        "mnist": mnist_map,
        "cifar10": cifar10_map,
    }.get(dataset_name, {})
    if all(value in dataset_map for value in label_values):
        return [dataset_map[value] for value in label_values]
    return [str(value) for value in label_values]


def _normalize_text_config(raw: dict[str, Any]) -> dict[str, Any]:
    label_values = list(raw["labels"]["values"])
    text_raw = dict(raw.get("text", {}))
    text_raw.setdefault("kind", "char")
    text_raw.setdefault("pad_token", "<pad>")
    text_raw.setdefault("bos_token", "<bos>")
    text_raw.setdefault("eos_token", "<eos>")
    text_raw.setdefault("strings", _default_text_strings(raw["dataset"]["name"], label_values))
    return text_raw


def _normalize_train_config(raw_train: dict[str, Any]) -> dict[str, Any]:
    return dict(raw_train)


def _normalize_sampling_config(raw_sampling: dict[str, Any]) -> dict[str, Any]:
    return dict(raw_sampling)


def _normalize_schedule_config(raw: dict[str, Any]) -> dict[str, Any]:
    return dict(raw.get("schedule", {}))


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    repo_root = config_path.parent.parent.resolve()
    paths = PathsConfig(
        data_dir=_resolve(repo_root, raw["paths"]["data_dir"]),
        artifacts_dir=_resolve(repo_root, raw["paths"]["artifacts_dir"]),
        models_dir=_resolve(repo_root, raw["paths"]["models_dir"]),
        runs_dir=_resolve(repo_root, raw["paths"]["runs_dir"]),
        logs_dir=_resolve(repo_root, raw["paths"]["logs_dir"]),
        cache_dir=_resolve(repo_root, raw["paths"]["cache_dir"]),
    )
    return ProjectConfig(
        name=raw["project"]["name"],
        repo_root=repo_root,
        paths=paths,
        tokenizer=TokenizerConfig(**raw["tokenizer"]),
        dataset=DatasetConfig(**raw["dataset"]),
        text=TextConfig(**_normalize_text_config(raw)),
        labels=LabelsConfig(**raw["labels"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**_normalize_train_config(raw["train"])),
        sampling=SamplingConfig(**_normalize_sampling_config(raw["sampling"])),
        schedule=ScheduleConfig(**_normalize_schedule_config(raw)),
        eval=EvalConfig(**raw["eval"]),
    )


def as_dict(config: ProjectConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "repo_root": str(config.repo_root),
        "paths": {key: str(value) for key, value in config.paths.__dict__.items()},
        "tokenizer": dict(config.tokenizer.__dict__),
        "dataset": dict(config.dataset.__dict__),
        "text": dict(config.text.__dict__),
        "labels": dict(config.labels.__dict__),
        "model": dict(config.model.__dict__),
        "train": dict(config.train.__dict__),
        "sampling": dict(config.sampling.__dict__),
        "schedule": dict(config.schedule.__dict__),
        "eval": dict(config.eval.__dict__),
    }
