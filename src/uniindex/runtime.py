from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .config import ProjectConfig, as_dict


def ensure_project_dirs(config: ProjectConfig) -> None:
    for path in (
        config.paths.data_dir,
        config.paths.artifacts_dir,
        config.paths.models_dir,
        config.paths.runs_dir,
        config.paths.logs_dir,
        config.paths.cache_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str, gpu_index: int) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def gpu_type(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


@dataclass
class Metadata:
    commit_sha: str
    run_mode: str
    config: dict
    started_at: str
    finished_at: str | None
    exit_status: str
    cost_estimate: str
    gpu_type: str


class RunContext:
    def __init__(self, config: ProjectConfig, mode: str) -> None:
        self.config = config
        self.mode = mode
        self.run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{mode}"
        self.run_dir = config.paths.runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.run_dir / "metadata.json"
        self.metadata = Metadata(
            commit_sha=git_commit_sha(config.repo_root),
            run_mode=mode,
            config=as_dict(config),
            started_at=utc_now(),
            finished_at=None,
            exit_status="running",
            cost_estimate="not_estimated",
            gpu_type="unknown",
        )
        self.write()

    def set_device(self, device: torch.device) -> None:
        self.metadata.gpu_type = gpu_type(device)
        self.write()

    def update_status(self, status: str) -> None:
        self.metadata.exit_status = status
        self.metadata.finished_at = utc_now()
        self.write()

    def write(self) -> None:
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.metadata), handle, indent=2)

    def log_path(self, name: str) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + os.linesep)
