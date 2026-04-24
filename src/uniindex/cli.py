from __future__ import annotations

import argparse
import sys

from .config import load_config
from .data import prepare_assets
from .diagnostics import (
    diagnose_i2t_denoiser,
    diagnose_i2t_image_dependence,
    diagnose_i2t_sampler_trajectory,
    diagnose_i2t_understanding,
)
from .eval import evaluate
from .runtime import RunContext, ensure_project_dirs
from .train import train_stage
from .visualize import export_visualizations
from .ablation import run_compact_ablation
from .i2t_power_sweep import run_i2t_power_sweep
from .i2t_repeats_sweep import run_i2t_repeats_sweep
from .i2t_overfit import run_i2t_overfit_probe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uniindex")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--stage", choices=["stage1", "stage2"], required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--config", required=True)

    diagnose_i2t = subparsers.add_parser("diagnose-i2t")
    diagnose_i2t.add_argument("--config", required=True)
    diagnose_i2t.add_argument("--progress", action="append", type=float, dest="progress_values")

    diagnose_i2t_image_dependence = subparsers.add_parser("diagnose-i2t-image-dependence")
    diagnose_i2t_image_dependence.add_argument("--config", required=True)
    diagnose_i2t_image_dependence.add_argument("--progress", action="append", type=float, dest="progress_values")

    diagnose_i2t_sampler_trajectory = subparsers.add_parser("diagnose-i2t-sampler-trajectory")
    diagnose_i2t_sampler_trajectory.add_argument("--config", required=True)
    diagnose_i2t_sampler_trajectory.add_argument("--progress", action="append", type=float, dest="progress_values")

    diagnose_i2t_understanding = subparsers.add_parser("diagnose-i2t-understanding")
    diagnose_i2t_understanding.add_argument("--config", required=True)
    diagnose_i2t_understanding.add_argument("--sample-count", type=int, default=32)
    diagnose_i2t_understanding.add_argument("--progress", action="append", type=float, dest="progress_values")

    probe_i2t_overfit = subparsers.add_parser("probe-i2t-overfit")
    probe_i2t_overfit.add_argument("--config", required=True)
    probe_i2t_overfit.add_argument("--steps", type=int, default=100)
    probe_i2t_overfit.add_argument("--sample-count", type=int, default=16)
    probe_i2t_overfit.add_argument("--test-sample-count", type=int, default=16)
    probe_i2t_overfit.add_argument("--lr", type=float)
    probe_i2t_overfit.add_argument("--eval-every", type=int, default=25)
    probe_i2t_overfit.add_argument("--progress", action="append", type=float, dest="progress_values")
    probe_i2t_overfit.add_argument("--mismatch-weight", type=float)
    probe_i2t_overfit.add_argument("--label-weight", type=float)
    probe_i2t_overfit.add_argument("--label-text-time", type=float)
    probe_i2t_overfit.add_argument("--save-model", action="store_true")

    visualize = subparsers.add_parser("visualize")
    visualize.add_argument("--config", required=True)

    ablate = subparsers.add_parser("ablate-compact")
    ablate.add_argument("--compact-config", required=True)
    ablate.add_argument("--full-config", required=True)

    sweep = subparsers.add_parser("sweep-i2t-power")
    sweep.add_argument("--long-config", default="configs/flm_joint_work_fullvocab_tsw075.yaml")
    sweep.add_argument("--short-config", action="append", dest="short_configs")
    sweep.add_argument("--snapshot-model-path")

    repeats_sweep = subparsers.add_parser("sweep-i2t-repeats")
    repeats_sweep.add_argument("--long-config", default="configs/flm_joint_work_fullvocab_tsw075.yaml")
    repeats_sweep.add_argument("--short-base-config", default="configs/flm_joint_work_fullvocab_short_i2tp40_tsw075.yaml")
    repeats_sweep.add_argument("--repeats", action="append", type=int, dest="repeats_values")
    repeats_sweep.add_argument("--snapshot-model-path")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", required=True)

    return parser


def _run_prepare(config_path: str) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    prepare_assets(config)
    return 0


def _run_train(config_path: str, stage: str) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    train_stage(config, stage)
    return 0


def _run_eval(config_path: str) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "eval")
    try:
        evaluate(config, run_context=run_context)
        export_visualizations(config, run_context=run_context)
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_diagnose_i2t(config_path: str, progress_values: list[float] | None) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "diagnose-i2t")
    try:
        if progress_values:
            diagnose_i2t_denoiser(config, progress_values=tuple(progress_values), run_context=run_context)
        else:
            diagnose_i2t_denoiser(config, run_context=run_context)
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_diagnose_i2t_image_dependence(config_path: str, progress_values: list[float] | None) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "diagnose-i2t-image-dependence")
    try:
        if progress_values:
            diagnose_i2t_image_dependence(config, progress_values=tuple(progress_values), run_context=run_context)
        else:
            diagnose_i2t_image_dependence(config, run_context=run_context)
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_diagnose_i2t_sampler_trajectory(config_path: str, progress_values: list[float] | None) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "diagnose-i2t-sampler-trajectory")
    try:
        if progress_values:
            diagnose_i2t_sampler_trajectory(config, progress_values=tuple(progress_values), run_context=run_context)
        else:
            diagnose_i2t_sampler_trajectory(config, run_context=run_context)
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_diagnose_i2t_understanding(
    config_path: str,
    sample_count: int,
    progress_values: list[float] | None,
) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "diagnose-i2t-understanding")
    try:
        if progress_values:
            diagnose_i2t_understanding(
                config,
                sample_count=sample_count,
                progress_values=tuple(progress_values),
                run_context=run_context,
            )
        else:
            diagnose_i2t_understanding(config, sample_count=sample_count, run_context=run_context)
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_probe_i2t_overfit(
    *,
    config_path: str,
    steps: int,
    sample_count: int,
    test_sample_count: int,
    lr: float | None,
    eval_every: int,
    progress_values: list[float] | None,
    mismatch_weight: float | None,
    label_weight: float | None,
    label_text_time: float | None,
    save_model: bool,
) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "probe-i2t-overfit")
    try:
        run_i2t_overfit_probe(
            config=config,
            steps=steps,
            sample_count=sample_count,
            test_sample_count=test_sample_count,
            lr=lr,
            eval_every=eval_every,
            progress_values=tuple(progress_values) if progress_values else None,
            mismatch_weight=mismatch_weight,
            label_weight=label_weight,
            label_text_time=label_text_time,
            save_model=save_model,
            run_context=run_context,
        )
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_visualize(config_path: str) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    export_visualizations(config)
    return 0


def _run_compact_ablation(compact_config_path: str, full_config_path: str) -> int:
    compact_config = load_config(compact_config_path)
    ensure_project_dirs(compact_config)
    run_compact_ablation(compact_config_path, full_config_path)
    return 0


def _run_sweep_i2t_power(
    long_config_path: str,
    short_config_paths: list[str] | None,
    snapshot_model_path: str | None,
) -> int:
    long_config = load_config(long_config_path)
    ensure_project_dirs(long_config)
    run_i2t_power_sweep(
        long_config_path=long_config_path,
        short_config_paths=short_config_paths,
        snapshot_model_path=snapshot_model_path,
    )
    return 0


def _run_sweep_i2t_repeats(
    long_config_path: str,
    short_base_config_path: str,
    repeats_values: list[int] | None,
    snapshot_model_path: str | None,
) -> int:
    long_config = load_config(long_config_path)
    ensure_project_dirs(long_config)
    run_i2t_repeats_sweep(
        long_config_path=long_config_path,
        short_base_config_path=short_base_config_path,
        repeats_values=repeats_values,
        snapshot_model_path=snapshot_model_path,
    )
    return 0


def _run_smoke(config_path: str) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    smoke_context = RunContext(config, "smoke")
    try:
        prepare_assets(config)
        train_stage(config, "stage1", run_context=smoke_context)
        train_stage(config, "stage2", run_context=smoke_context)
        evaluate(config, run_context=smoke_context)
        export_visualizations(config, run_context=smoke_context)
        smoke_context.update_status("ok")
    except Exception:
        smoke_context.update_status("error")
        raise
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        return _run_prepare(args.config)
    if args.command == "train":
        return _run_train(args.config, args.stage)
    if args.command == "eval":
        return _run_eval(args.config)
    if args.command == "diagnose-i2t":
        return _run_diagnose_i2t(args.config, args.progress_values)
    if args.command == "diagnose-i2t-image-dependence":
        return _run_diagnose_i2t_image_dependence(args.config, args.progress_values)
    if args.command == "diagnose-i2t-sampler-trajectory":
        return _run_diagnose_i2t_sampler_trajectory(args.config, args.progress_values)
    if args.command == "diagnose-i2t-understanding":
        return _run_diagnose_i2t_understanding(args.config, args.sample_count, args.progress_values)
    if args.command == "probe-i2t-overfit":
        return _run_probe_i2t_overfit(
            config_path=args.config,
            steps=args.steps,
            sample_count=args.sample_count,
            test_sample_count=args.test_sample_count,
            lr=args.lr,
            eval_every=args.eval_every,
            progress_values=args.progress_values,
            mismatch_weight=args.mismatch_weight,
            label_weight=args.label_weight,
            label_text_time=args.label_text_time,
            save_model=args.save_model,
        )
    if args.command == "visualize":
        return _run_visualize(args.config)
    if args.command == "ablate-compact":
        return _run_compact_ablation(args.compact_config, args.full_config)
    if args.command == "sweep-i2t-power":
        return _run_sweep_i2t_power(args.long_config, args.short_configs, args.snapshot_model_path)
    if args.command == "sweep-i2t-repeats":
        return _run_sweep_i2t_repeats(
            args.long_config,
            args.short_base_config,
            args.repeats_values,
            args.snapshot_model_path,
        )
    if args.command == "smoke":
        return _run_smoke(args.config)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
