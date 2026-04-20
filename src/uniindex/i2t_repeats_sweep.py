from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .i2t_power_sweep import (
    DEFAULT_LONG_CONFIG,
    _abs_path,
    _generated_config_path,
    _hf_snapshot_dir,
    _latest_metrics_path,
    _load_yaml,
    _localize_config_if_needed,
    _read_metrics,
    _rewrite_config_for_snapshot,
    _run_cli,
    _runtime_env,
    _sorted_results,
    _winner_beats_baseline,
    _write_yaml,
)

DEFAULT_SHORT_BASE_CONFIG = "configs/flm_joint_work_fullvocab_short_i2tp40_tsw075.yaml"
DEFAULT_REPEATS = [4, 6, 8]
SUMMARY_ROOT = Path("runs/i2t_repeats_sweep")


@dataclass(frozen=True)
class RepeatsSweepResult:
    source_config_path: Path
    local_config_path: Path
    metrics_path: Path
    metrics: dict[str, float]
    stage2_image_to_text_repeats: int


def _summary_dir(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / SUMMARY_ROOT / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _repeats_tag(value: int) -> str:
    return f"i2tr{int(value):02d}"


def _set_repeats_paths(raw: dict[str, Any], repeats: int, *, long: bool) -> dict[str, Any]:
    updated = deepcopy(raw)
    tag = _repeats_tag(repeats)
    updated["paths"]["artifacts_dir"] = "artifacts/fullvocab_short_shared"
    if long:
        updated["project"]["name"] = f"uniindex-fullvocab-long-tsw075-{tag}"
        updated["paths"]["models_dir"] = f"models/fullvocab_long_tsw075_{tag}"
        updated["paths"]["runs_dir"] = f"runs/fullvocab_long_tsw075_{tag}"
        updated["paths"]["logs_dir"] = f"logs/fullvocab_long_tsw075_{tag}"
    else:
        updated["project"]["name"] = f"uniindex-fullvocab-short-{tag}-tsw075"
        updated["paths"]["models_dir"] = f"models/fullvocab_short_{tag}_tsw075"
        updated["paths"]["runs_dir"] = f"runs/fullvocab_short_{tag}_tsw075"
        updated["paths"]["logs_dir"] = f"logs/fullvocab_short_{tag}_tsw075"
    updated["train"]["stage2_image_to_text_repeats"] = int(repeats)
    return updated


def _build_short_config(
    short_base_config_path: Path,
    snapshot_path: Path | None,
    repeats: int,
) -> dict[str, Any]:
    raw = _set_repeats_paths(_load_yaml(short_base_config_path), repeats, long=False)
    if snapshot_path is not None:
        raw = _rewrite_config_for_snapshot(raw, snapshot_path)
    return raw


def _build_promoted_long_config(
    long_config_path: Path,
    snapshot_path: Path | None,
    winner: RepeatsSweepResult,
) -> dict[str, Any]:
    raw = _set_repeats_paths(_load_yaml(long_config_path), winner.stage2_image_to_text_repeats, long=True)
    if snapshot_path is not None:
        raw = _rewrite_config_for_snapshot(raw, snapshot_path)
    return raw


def _generated_short_config_paths(
    root: Path,
    short_base_config_path: Path,
    repeats_values: list[int],
) -> list[Path]:
    paths = []
    for repeats in repeats_values:
        path = _generated_config_path(root, short_base_config_path, suffix=_repeats_tag(repeats))
        _write_yaml(path, _build_short_config(short_base_config_path, None, repeats))
        paths.append(path)
    return paths


def _baseline_result(results: list[RepeatsSweepResult], long_config_path: Path) -> RepeatsSweepResult:
    baseline_repeats = load_config(long_config_path).train.stage2_image_to_text_repeats
    matches = [result for result in results if result.stage2_image_to_text_repeats == baseline_repeats]
    if len(matches) != 1:
        raise ValueError("Expected exactly one short config matching the long baseline image-to-text repeats")
    return matches[0]


def run_i2t_repeats_sweep(
    long_config_path: str | Path = DEFAULT_LONG_CONFIG,
    short_base_config_path: str | Path = DEFAULT_SHORT_BASE_CONFIG,
    repeats_values: list[int] | None = None,
    snapshot_model_path: str | None = None,
) -> Path:
    root = Path(__file__).resolve().parents[2]
    env = _runtime_env(root)
    long_source_path = _abs_path(root, long_config_path)
    short_base_source_path = _abs_path(root, short_base_config_path)
    repeats = list(dict.fromkeys(repeats_values or DEFAULT_REPEATS))
    long_source_config = load_config(long_source_path)
    baseline_repeats = long_source_config.train.stage2_image_to_text_repeats
    if baseline_repeats not in repeats:
        raise ValueError(
            f"repeats_values must include the long baseline repeats value {baseline_repeats}"
        )
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

    source_paths = _generated_short_config_paths(root, short_base_source_path, repeats)
    results: list[RepeatsSweepResult] = []
    for source_path in source_paths:
        local_path = _localize_config_if_needed(root, source_path, snapshot_path)
        _run_cli(root, env, "prepare", "--config", str(local_path))
        _run_cli(root, env, "train", "--config", str(local_path), "--stage", "stage1")
        _run_cli(root, env, "train", "--config", str(local_path), "--stage", "stage2")
        _run_cli(root, env, "eval", "--config", str(local_path))
        metrics_path = _latest_metrics_path(local_path)
        metrics = _read_metrics(metrics_path)
        config = load_config(local_path)
        results.append(
            RepeatsSweepResult(
                source_config_path=source_path,
                local_config_path=local_path,
                metrics_path=metrics_path,
                metrics=metrics,
                stage2_image_to_text_repeats=config.train.stage2_image_to_text_repeats,
            )
        )

    ranked_results = _sorted_results(results)
    winner = ranked_results[0]
    baseline = _baseline_result(results, long_source_path)
    promoted = _winner_beats_baseline(winner, baseline)

    summary: dict[str, Any] = {
        "summary_dir": str(summary_dir),
        "long_source_config_path": str(long_source_path),
        "short_base_config_path": str(short_base_source_path),
        "snapshot_model_path": str(snapshot_path) if snapshot_path is not None else None,
        "baseline_short_config_path": str(baseline.source_config_path),
        "baseline_metrics_path": str(baseline.metrics_path),
        "winner_short_config_path": str(winner.source_config_path),
        "winner_metrics_path": str(winner.metrics_path),
        "winner_stage2_image_to_text_repeats": winner.stage2_image_to_text_repeats,
        "promoted_to_long_run": promoted,
        "results": [
            {
                "source_config_path": str(result.source_config_path),
                "local_config_path": str(result.local_config_path),
                "metrics_path": str(result.metrics_path),
                "stage2_image_to_text_repeats": result.stage2_image_to_text_repeats,
                "metrics": result.metrics,
            }
            for result in ranked_results
        ],
    }

    if promoted:
        promoted_long_raw = _build_promoted_long_config(long_source_path, snapshot_path, winner)
        promoted_long_path = _generated_config_path(
            root,
            long_source_path,
            suffix=_repeats_tag(winner.stage2_image_to_text_repeats),
        )
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
