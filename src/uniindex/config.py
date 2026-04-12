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


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    train_limit: int | None = None
    test_limit: int | None = None


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
    label_weight: float


@dataclass(frozen=True)
class SamplingConfig:
    steps: int
    temperature: float


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
    labels: LabelsConfig
    model: ModelConfig
    train: TrainConfig
    sampling: SamplingConfig
    eval: EvalConfig


def _resolve(base: Path, raw: str) -> Path:
    return (base / raw).resolve()


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
        labels=LabelsConfig(**raw["labels"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        sampling=SamplingConfig(**raw["sampling"]),
        eval=EvalConfig(**raw["eval"]),
    )


def as_dict(config: ProjectConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "repo_root": str(config.repo_root),
        "paths": {key: str(value) for key, value in config.paths.__dict__.items()},
        "tokenizer": dict(config.tokenizer.__dict__),
        "dataset": dict(config.dataset.__dict__),
        "labels": dict(config.labels.__dict__),
        "model": dict(config.model.__dict__),
        "train": dict(config.train.__dict__),
        "sampling": dict(config.sampling.__dict__),
        "eval": dict(config.eval.__dict__),
    }
