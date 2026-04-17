from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .data import tokenizer_state_path

DEFAULT_LONG_CONFIG = "configs/flm_joint_work_fullvocab_tsw075.yaml"
DEFAULT_SHORT_CONFIGS = [
    "configs/flm_joint_work_fullvocab_short_i2tp20_tsw075.yaml",
    "configs/flm_joint_work_fullvocab_short_i2tp40_tsw075.yaml",
    "configs/flm_joint_work_fullvocab_short_i2tp60_tsw075.yaml",
]
SUMMARY_ROOT = Path("runs/i2t_power_sweep")
METRIC_ORDER = (
    "image_to_text_exact_match",
    "image_to_text_label_accuracy_constrained",
    "text_to_image_accuracy",
)


@dataclass(frozen=True)
class SweepResult:
    source_config_path: Path
    local_config_path: Path
    metrics_path: Path
    metrics: dict[str, float]
    image_to_text_text_time_power: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _abs_path(root: Path, path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return (root / raw).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, raw: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)


def _runtime_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(root / ".cache/uv"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(root / ".cache/uv-python"))
    env.setdefault("UV_HTTP_TIMEOUT", "600")
    env.setdefault("PIP_CACHE_DIR", str(root / ".cache/pip"))
    env.setdefault("HF_HOME", str(root / ".cache/huggingface"))
    env.setdefault("HF_HUB_CACHE", str(root / ".cache/huggingface/hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(root / ".cache/huggingface/transformers"))
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    env.setdefault("TORCH_HOME", str(root / ".cache/torch"))
    env.setdefault("XDG_CACHE_HOME", str(root / ".cache/xdg"))
    env.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    src_path = str(root / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path
    return env


def _run_cli(root: Path, env: dict[str, str], *args: str) -> None:
    command = [sys.executable, "-m", "uniindex.cli", *args]
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


def _hf_snapshot_dir(cache_dir: Path, model_name: str | None) -> Path | None:
    if not model_name or "/" not in model_name:
        return None
    snapshots_dir = (
        cache_dir
        / "huggingface"
        / "transformers"
        / f"models--{model_name.replace('/', '--')}"
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None
    candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _rewrite_config_for_snapshot(raw: dict[str, Any], snapshot_path: Path) -> dict[str, Any]:
    updated = deepcopy(raw)
    updated["tokenizer"]["model_name"] = str(snapshot_path)
    return updated


def _generated_config_path(root: Path, source_config_path: Path, suffix: str = "local") -> Path:
    return root / "configs_generated" / f"{source_config_path.stem}_{suffix}.yaml"


def _ensure_tokenized_alias(source_config_path: Path, local_config_path: Path) -> None:
    source_config = load_config(source_config_path)
    local_config = load_config(local_config_path)
    source_dir = tokenizer_state_path(source_config).parent
    local_dir = tokenizer_state_path(local_config).parent
    if local_dir == source_dir or local_dir.exists() or not source_dir.exists():
        return
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source_dir, local_dir.parent)
    local_dir.symlink_to(target, target_is_directory=True)


def _latest_metrics_path(config_path: Path) -> Path:
    config = load_config(config_path)
    candidates = sorted(config.paths.runs_dir.glob("*-eval/metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No eval metrics found for {config_path}")
    return candidates[-1]


def _read_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: float(value) for key, value in payload.items()}


def _metric_tuple(metrics: dict[str, float]) -> tuple[float, float, float]:
    return tuple(float(metrics.get(key, float("-inf"))) for key in METRIC_ORDER)


def _sorted_results(results: list[SweepResult]) -> list[SweepResult]:
    return sorted(results, key=lambda result: _metric_tuple(result.metrics), reverse=True)


def _summary_dir(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / SUMMARY_ROOT / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _power_tag(value: float) -> str:
    return f"i2tp{int(round(value * 10)):02d}"


def _baseline_result(results: list[SweepResult], long_config_path: Path) -> SweepResult:
    baseline_power = load_config(long_config_path).train.image_to_text_text_time_power
    matches = [result for result in results if result.image_to_text_text_time_power == baseline_power]
    if len(matches) != 1:
        raise ValueError("Expected exactly one short config matching the long baseline power")
    return matches[0]


def _winner_beats_baseline(winner: SweepResult, baseline: SweepResult) -> bool:
    return _metric_tuple(winner.metrics) > _metric_tuple(baseline.metrics)


def _build_promoted_long_config(
    long_config_path: Path,
    snapshot_path: Path | None,
    winner: SweepResult,
) -> dict[str, Any]:
    raw = _load_yaml(long_config_path)
    power = winner.image_to_text_text_time_power
    tag = _power_tag(power)
    raw["project"]["name"] = f"uniindex-fullvocab-long-tsw075-{tag}"
    raw["paths"]["artifacts_dir"] = "artifacts/fullvocab_short_shared"
    raw["paths"]["models_dir"] = f"models/fullvocab_long_tsw075_{tag}"
    raw["paths"]["runs_dir"] = f"runs/fullvocab_long_tsw075_{tag}"
    raw["paths"]["logs_dir"] = f"logs/fullvocab_long_tsw075_{tag}"
    raw["train"]["image_to_text_text_time_power"] = float(power)
    raw["sampling"]["image_to_text_text_time_power"] = float(power)
    if snapshot_path is not None:
        raw = _rewrite_config_for_snapshot(raw, snapshot_path)
    return raw


def _resolve_short_config_paths(root: Path, short_config_paths: list[str] | None) -> list[Path]:
    raw_paths = short_config_paths or DEFAULT_SHORT_CONFIGS
    return [_abs_path(root, path) for path in raw_paths]


def _localize_config_if_needed(
    root: Path,
    source_config_path: Path,
    snapshot_path: Path | None,
) -> Path:
    if snapshot_path is None:
        return source_config_path
    raw = _rewrite_config_for_snapshot(_load_yaml(source_config_path), snapshot_path)
    local_config_path = _generated_config_path(root, source_config_path)
    _write_yaml(local_config_path, raw)
    _ensure_tokenized_alias(source_config_path, local_config_path)
    return local_config_path


def run_i2t_power_sweep(
    long_config_path: str | Path = DEFAULT_LONG_CONFIG,
    short_config_paths: list[str] | None = None,
    snapshot_model_path: str | None = None,
) -> Path:
    root = _repo_root()
    env = _runtime_env(root)
    long_source_path = _abs_path(root, long_config_path)
    short_source_paths = _resolve_short_config_paths(root, short_config_paths)
    long_source_config = load_config(long_source_path)
    snapshot_path = (
        Path(snapshot_model_path).resolve()
        if snapshot_model_path
        else _hf_snapshot_dir(long_source_config.paths.cache_dir, long_source_config.tokenizer.model_name)
    )
    summary_dir = _summary_dir(root)
    print(f"summary_dir={summary_dir}", flush=True)
    if snapshot_path is not None:
        print(f"snapshot_model_path={snapshot_path}", flush=True)
    else:
        print("snapshot_model_path=none; using source configs directly", flush=True)

    results: list[SweepResult] = []
    for source_path in short_source_paths:
        local_path = _localize_config_if_needed(root, source_path, snapshot_path)
        _run_cli(root, env, "prepare", "--config", str(local_path))
        _run_cli(root, env, "train", "--config", str(local_path), "--stage", "stage1")
        _run_cli(root, env, "train", "--config", str(local_path), "--stage", "stage2")
        _run_cli(root, env, "eval", "--config", str(local_path))
        metrics_path = _latest_metrics_path(local_path)
        metrics = _read_metrics(metrics_path)
        power = float(load_config(local_path).train.image_to_text_text_time_power or 0.0)
        results.append(
            SweepResult(
                source_config_path=source_path,
                local_config_path=local_path,
                metrics_path=metrics_path,
                metrics=metrics,
                image_to_text_text_time_power=power,
            )
        )

    ranked_results = _sorted_results(results)
    winner = ranked_results[0]
    baseline = _baseline_result(results, long_source_path)
    promoted = _winner_beats_baseline(winner, baseline)

    summary: dict[str, Any] = {
        "summary_dir": str(summary_dir),
        "long_source_config_path": str(long_source_path),
        "snapshot_model_path": str(snapshot_path) if snapshot_path is not None else None,
        "baseline_short_config_path": str(baseline.source_config_path),
        "baseline_metrics_path": str(baseline.metrics_path),
        "winner_short_config_path": str(winner.source_config_path),
        "winner_metrics_path": str(winner.metrics_path),
        "winner_image_to_text_text_time_power": winner.image_to_text_text_time_power,
        "promoted_to_long_run": promoted,
        "results": [
            {
                "source_config_path": str(result.source_config_path),
                "local_config_path": str(result.local_config_path),
                "metrics_path": str(result.metrics_path),
                "image_to_text_text_time_power": result.image_to_text_text_time_power,
                "metrics": result.metrics,
            }
            for result in ranked_results
        ],
    }

    if promoted:
        promoted_long_raw = _build_promoted_long_config(long_source_path, snapshot_path, winner)
        promoted_long_path = _generated_config_path(root, long_source_path, suffix=_power_tag(winner.image_to_text_text_time_power))
        _write_yaml(promoted_long_path, promoted_long_raw)
        _run_cli(root, env, "prepare", "--config", str(promoted_long_path))
        _run_cli(root, env, "train", "--config", str(promoted_long_path), "--stage", "stage1")
        _run_cli(root, env, "train", "--config", str(promoted_long_path), "--stage", "stage2")
        _run_cli(root, env, "eval", "--config", str(promoted_long_path))
        promoted_metrics_path = _latest_metrics_path(promoted_long_path)
        summary["promoted_long_config_path"] = str(promoted_long_path)
        summary["promoted_long_metrics_path"] = str(promoted_metrics_path)
        summary["promoted_long_metrics"] = _read_metrics(promoted_metrics_path)

    summary_path = summary_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"summary_path={summary_path}", flush=True)
    return summary_path
