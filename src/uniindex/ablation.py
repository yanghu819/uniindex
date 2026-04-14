from __future__ import annotations

import json
from pathlib import Path

import torch

from .classifier import train_or_load_classifier
from .config import as_dict, load_config
from .data import load_tokenizer_state, prepare_assets, split_path
from .eval import evaluate
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .state import restore_image_tokens
from .train import train_stage


def _run_single(config_path: str | Path, classifier_override_path: Path | None = None) -> dict:
    config = load_config(config_path)
    ensure_project_dirs(config)
    prepare_assets(config)
    stage1_ckpt = train_stage(config, "stage1")
    stage2_ckpt = train_stage(config, "stage2")
    metrics = evaluate(config, classifier_override_path=classifier_override_path)
    return {
        "config_path": str(Path(config_path).resolve()),
        "config": as_dict(config),
        "stage1_checkpoint": str(stage1_ckpt),
        "stage2_checkpoint": str(stage2_ckpt),
        "metrics": metrics,
    }


def _shared_classifier_path(config) -> Path:
    shared_models_dir = config.repo_root / "models" / "ablation_shared"
    shared_models_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.train.seed)
    return train_or_load_classifier(
        data_dir=config.paths.data_dir,
        models_dir=shared_models_dir,
        device=resolve_device(config.train.device, config.train.gpu_index),
        epochs=config.eval.classifier_epochs,
        batch_size=config.eval.classifier_batch_size,
        lr=config.eval.classifier_lr,
    )


def _artifact_equivalence(compact_config_path: str | Path, full_config_path: str | Path) -> dict:
    compact_config = load_config(compact_config_path)
    full_config = load_config(full_config_path)
    compact_state = load_tokenizer_state(compact_config)
    full_state = load_tokenizer_state(full_config)

    result = {
        "compact_codebook_size": int(compact_state["codebook_size"]),
        "full_codebook_size": int(full_state["codebook_size"]),
        "splits": {},
    }
    for split in ("train", "test"):
        compact_payload = torch.load(split_path(compact_config, split), map_location="cpu")
        full_payload = torch.load(split_path(full_config, split), map_location="cpu")
        restored = restore_image_tokens(compact_payload["image_tokens"], compact_state)
        result["splits"][split] = {
            "token_match_after_restore": bool(torch.equal(restored.long(), full_payload["image_tokens"].long())),
            "label_match": bool(torch.equal(compact_payload["labels"].long(), full_payload["labels"].long())),
            "grid_shape_match": tuple(compact_payload["grid_shape"]) == tuple(full_payload["grid_shape"]),
        }
    return result


def run_compact_ablation(compact_config_path: str | Path, full_config_path: str | Path) -> Path:
    base_config = load_config(compact_config_path)
    ensure_project_dirs(base_config)
    run_context = RunContext(base_config, "ablate-compact")
    try:
        shared_classifier = _shared_classifier_path(base_config)
        compact_result = _run_single(compact_config_path, classifier_override_path=shared_classifier)
        full_result = _run_single(full_config_path, classifier_override_path=shared_classifier)
        summary = {
            "shared_classifier_path": str(shared_classifier),
            "artifact_equivalence": _artifact_equivalence(compact_config_path, full_config_path),
            "compact": compact_result,
            "full_vocab": full_result,
            "metric_delta": {
                key: full_result["metrics"][key] - compact_result["metrics"][key]
                for key in compact_result["metrics"].keys()
            },
        }
        summary_path = run_context.log_path("summary.json")
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        run_context.update_status("ok")
        return summary_path
    except Exception:
        run_context.update_status("error")
        raise
