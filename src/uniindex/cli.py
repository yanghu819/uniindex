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
from .i2t_sampler_state_ft import run_i2t_sampler_state_ft
from .i2t_llm_decoder import download_i2t_llm_assets, run_i2t_llm_decoder_probe
from .label_feature_probe import run_label_feature_probe
from .siglipvq_reconstruction import probe_siglipvq_reconstruction
from .t2i_overfit import run_t2i_overfit_probe
from .t2i_distributional_ft import run_t2i_distributional_ft_probe
from .t2i_token_guard import run_t2i_token_guard
from .tokenizer import download_siglip_vq_assets, download_siglip_vq_decoder_assets
from .vq_text_decoder import run_vq_text_decoder_probe


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

    probe_i2t_sampler_state_ft = subparsers.add_parser("probe-i2t-sampler-state-ft")
    probe_i2t_sampler_state_ft.add_argument("--config", required=True)
    probe_i2t_sampler_state_ft.add_argument("--steps", type=int, default=150)
    probe_i2t_sampler_state_ft.add_argument("--lr", type=float)
    probe_i2t_sampler_state_ft.add_argument("--progress", action="append", type=float, dest="progress_values")
    probe_i2t_sampler_state_ft.add_argument("--contrast-weight", type=float, default=0.0)
    probe_i2t_sampler_state_ft.add_argument("--contrast-margin", type=float, default=1.0)
    probe_i2t_sampler_state_ft.add_argument("--sequence-weight", type=float)
    probe_i2t_sampler_state_ft.add_argument("--anchor-weight", type=float, default=0.0)
    probe_i2t_sampler_state_ft.add_argument("--anchor-task", action="append", dest="anchor_tasks")
    probe_i2t_sampler_state_ft.add_argument("--anchor-temperature", type=float, default=1.0)
    probe_i2t_sampler_state_ft.add_argument(
        "--trainable-scope",
        choices=["all", "head", "last_block", "last_two_blocks"],
        default="all",
    )
    probe_i2t_sampler_state_ft.add_argument("--save-every", type=int)

    probe_i2t_llm_decoder = subparsers.add_parser("probe-i2t-llm-decoder")
    probe_i2t_llm_decoder.add_argument("--config", required=True)
    probe_i2t_llm_decoder.add_argument("--steps", type=int)
    probe_i2t_llm_decoder.add_argument("--eval-every", type=int, default=50)

    download_i2t_llm = subparsers.add_parser("download-i2t-llm")
    download_i2t_llm.add_argument("--config", required=True)

    download_siglip_vq = subparsers.add_parser("download-siglip-vq")
    download_siglip_vq.add_argument("--config", required=True)

    download_siglip_vq_decoder = subparsers.add_parser("download-siglip-vq-decoder")
    download_siglip_vq_decoder.add_argument("--config", required=True)

    probe_label_features = subparsers.add_parser("probe-label-features")
    probe_label_features.add_argument("--config", required=True)
    probe_label_features.add_argument("--steps", type=int, default=200)
    probe_label_features.add_argument("--eval-every", type=int, default=50)
    probe_label_features.add_argument("--vq-only", action="store_true")

    probe_vq_text_decoder = subparsers.add_parser("probe-vq-text-decoder")
    probe_vq_text_decoder.add_argument("--config", required=True)
    probe_vq_text_decoder.add_argument("--steps", type=int, default=200)
    probe_vq_text_decoder.add_argument("--eval-every", type=int, default=50)
    probe_vq_text_decoder.add_argument("--d-model", type=int)
    probe_vq_text_decoder.add_argument("--n-layers", type=int, default=2)
    probe_vq_text_decoder.add_argument("--lr", type=float)
    probe_vq_text_decoder.add_argument("--contrast-weight", type=float, default=0.0)
    probe_vq_text_decoder.add_argument("--contrast-margin", type=float, default=1.0)

    probe_siglipvq_reconstruction = subparsers.add_parser("probe-siglipvq-reconstruction")
    probe_siglipvq_reconstruction.add_argument("--config", required=True)
    probe_siglipvq_reconstruction.add_argument("--sample-count", type=int, default=16)
    probe_siglipvq_reconstruction.add_argument("--split", choices=["train", "test"], default="test")

    probe_t2i_token_guard = subparsers.add_parser("probe-t2i-token-guard")
    probe_t2i_token_guard.add_argument("--config", required=True)
    probe_t2i_token_guard.add_argument("--probe-steps", type=int, default=200)
    probe_t2i_token_guard.add_argument("--eval-every", type=int, default=50)
    probe_t2i_token_guard.add_argument("--samples-per-label", type=int, default=8)
    probe_t2i_token_guard.add_argument("--unconditional-count", type=int, default=32)
    probe_t2i_token_guard.add_argument("--decode-samples", type=int, default=0)
    probe_t2i_token_guard.add_argument("--lr", type=float)

    probe_t2i_overfit = subparsers.add_parser("probe-t2i-overfit")
    probe_t2i_overfit.add_argument("--config", required=True)
    probe_t2i_overfit.add_argument("--steps", type=int, default=200)
    probe_t2i_overfit.add_argument("--sample-count", type=int, default=16)
    probe_t2i_overfit.add_argument("--test-sample-count", type=int, default=16)
    probe_t2i_overfit.add_argument("--lr", type=float)
    probe_t2i_overfit.add_argument("--eval-every", type=int, default=50)
    probe_t2i_overfit.add_argument("--progress", action="append", type=float, dest="progress_values")
    probe_t2i_overfit.add_argument("--token-probe-steps", type=int, default=100)
    probe_t2i_overfit.add_argument("--unconditional-count", type=int, default=8)
    probe_t2i_overfit.add_argument("--save-model", action="store_true")

    probe_t2i_distributional_ft = subparsers.add_parser("probe-t2i-distributional-ft")
    probe_t2i_distributional_ft.add_argument("--config", required=True)
    probe_t2i_distributional_ft.add_argument("--steps", type=int, default=500)
    probe_t2i_distributional_ft.add_argument("--loss-kind", choices=["hard_ce", "set_ce_k16"], default="hard_ce")
    probe_t2i_distributional_ft.add_argument("--state-kind", choices=["simplex"], default="simplex")
    probe_t2i_distributional_ft.add_argument("--set-size", type=int, default=16)
    probe_t2i_distributional_ft.add_argument("--set-temperature", type=float, default=0.25)
    probe_t2i_distributional_ft.add_argument("--endpoint-prob", type=float, default=0.9)
    probe_t2i_distributional_ft.add_argument("--lr", type=float)
    probe_t2i_distributional_ft.add_argument("--eval-step", action="append", type=int, dest="eval_steps")
    probe_t2i_distributional_ft.add_argument("--token-probe-steps", type=int, default=200)
    probe_t2i_distributional_ft.add_argument("--token-probe-eval-every", type=int, default=50)
    probe_t2i_distributional_ft.add_argument("--samples-per-label", type=int, default=4)
    probe_t2i_distributional_ft.add_argument("--unconditional-count", type=int, default=10)

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


def _run_probe_i2t_sampler_state_ft(
    *,
    config_path: str,
    steps: int,
    lr: float | None,
    progress_values: list[float] | None,
    contrast_weight: float,
    contrast_margin: float,
    sequence_weight: float | None,
    anchor_weight: float,
    anchor_tasks: list[str] | None,
    anchor_temperature: float,
    trainable_scope: str,
    save_every: int | None,
) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "probe-i2t-sampler-state-ft")
    try:
        run_i2t_sampler_state_ft(
            config=config,
            steps=steps,
            lr=lr,
            progress_values=tuple(progress_values) if progress_values else None,
            contrast_weight=contrast_weight,
            contrast_margin=contrast_margin,
            sequence_weight=sequence_weight,
            anchor_weight=anchor_weight,
            anchor_tasks=tuple(anchor_tasks) if anchor_tasks else None,
            anchor_temperature=anchor_temperature,
            trainable_scope=trainable_scope,
            save_every=save_every,
            run_context=run_context,
        )
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_probe_i2t_llm_decoder(
    *,
    config_path: str,
    steps: int | None,
    eval_every: int,
) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "probe-i2t-llm-decoder")
    try:
        run_i2t_llm_decoder_probe(
            config=config,
            steps=steps,
            eval_every=eval_every,
            run_context=run_context,
        )
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_download_i2t_llm(config_path: str) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    result = download_i2t_llm_assets(config)
    print(result)
    return 0


def _run_download_siglip_vq(config_path: str) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    result = download_siglip_vq_assets(config)
    print(result)
    return 0


def _run_download_siglip_vq_decoder(config_path: str) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    result = download_siglip_vq_decoder_assets(config)
    print(result)
    return 0


def _run_probe_label_features(*, config_path: str, steps: int, eval_every: int, vq_only: bool) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "probe-label-features")
    try:
        run_label_feature_probe(
            config=config,
            steps=steps,
            eval_every=eval_every,
            vq_only=vq_only,
            run_context=run_context,
        )
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_probe_vq_text_decoder(
    *,
    config_path: str,
    steps: int,
    eval_every: int,
    d_model: int | None,
    n_layers: int,
    lr: float | None,
    contrast_weight: float,
    contrast_margin: float,
) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "probe-vq-text-decoder")
    try:
        run_vq_text_decoder_probe(
            config=config,
            steps=steps,
            eval_every=eval_every,
            d_model=d_model,
            n_layers=n_layers,
            lr=lr,
            contrast_weight=contrast_weight,
            contrast_margin=contrast_margin,
            run_context=run_context,
        )
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_probe_siglipvq_reconstruction(*, config_path: str, sample_count: int, split: str) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "probe-siglipvq-reconstruction")
    try:
        probe_siglipvq_reconstruction(
            config=config,
            sample_count=sample_count,
            split=split,
            run_context=run_context,
        )
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_probe_t2i_token_guard(
    *,
    config_path: str,
    probe_steps: int,
    eval_every: int,
    samples_per_label: int,
    unconditional_count: int,
    decode_samples: int,
    lr: float | None,
) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "probe-t2i-token-guard")
    try:
        run_t2i_token_guard(
            config=config,
            probe_steps=probe_steps,
            eval_every=eval_every,
            samples_per_label=samples_per_label,
            unconditional_count=unconditional_count,
            decode_samples=decode_samples,
            lr=lr,
            run_context=run_context,
        )
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_probe_t2i_overfit(
    *,
    config_path: str,
    steps: int,
    sample_count: int,
    test_sample_count: int,
    lr: float | None,
    eval_every: int,
    progress_values: list[float] | None,
    token_probe_steps: int,
    unconditional_count: int,
    save_model: bool,
) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "probe-t2i-overfit")
    try:
        run_t2i_overfit_probe(
            config=config,
            steps=steps,
            sample_count=sample_count,
            test_sample_count=test_sample_count,
            lr=lr,
            eval_every=eval_every,
            progress_values=tuple(progress_values) if progress_values else None,
            token_probe_steps=token_probe_steps,
            unconditional_count=unconditional_count,
            save_model=save_model,
            run_context=run_context,
        )
        run_context.update_status("ok")
    except Exception:
        run_context.update_status("error")
        raise
    return 0


def _run_probe_t2i_distributional_ft(
    *,
    config_path: str,
    steps: int,
    loss_kind: str,
    state_kind: str,
    set_size: int,
    set_temperature: float,
    endpoint_prob: float,
    lr: float | None,
    eval_steps: list[int] | None,
    token_probe_steps: int,
    token_probe_eval_every: int,
    samples_per_label: int,
    unconditional_count: int,
) -> int:
    config = load_config(config_path)
    ensure_project_dirs(config)
    run_context = RunContext(config, "probe-t2i-distributional-ft")
    try:
        run_t2i_distributional_ft_probe(
            config=config,
            steps=steps,
            loss_kind=loss_kind,
            state_kind=state_kind,
            set_size=set_size,
            set_temperature=set_temperature,
            endpoint_prob=endpoint_prob,
            lr=lr,
            eval_steps=tuple(eval_steps) if eval_steps else None,
            token_probe_steps=token_probe_steps,
            token_probe_eval_every=token_probe_eval_every,
            samples_per_label=samples_per_label,
            unconditional_count=unconditional_count,
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
    if args.command == "probe-i2t-sampler-state-ft":
        return _run_probe_i2t_sampler_state_ft(
            config_path=args.config,
            steps=args.steps,
            lr=args.lr,
            progress_values=args.progress_values,
            contrast_weight=args.contrast_weight,
            contrast_margin=args.contrast_margin,
            sequence_weight=args.sequence_weight,
            anchor_weight=args.anchor_weight,
            anchor_tasks=args.anchor_tasks,
            anchor_temperature=args.anchor_temperature,
            trainable_scope=args.trainable_scope,
            save_every=args.save_every,
        )
    if args.command == "probe-i2t-llm-decoder":
        return _run_probe_i2t_llm_decoder(
            config_path=args.config,
            steps=args.steps,
            eval_every=args.eval_every,
        )
    if args.command == "download-i2t-llm":
        return _run_download_i2t_llm(args.config)
    if args.command == "download-siglip-vq":
        return _run_download_siglip_vq(args.config)
    if args.command == "download-siglip-vq-decoder":
        return _run_download_siglip_vq_decoder(args.config)
    if args.command == "probe-label-features":
        return _run_probe_label_features(
            config_path=args.config,
            steps=args.steps,
            eval_every=args.eval_every,
            vq_only=args.vq_only,
        )
    if args.command == "probe-vq-text-decoder":
        return _run_probe_vq_text_decoder(
            config_path=args.config,
            steps=args.steps,
            eval_every=args.eval_every,
            d_model=args.d_model,
            n_layers=args.n_layers,
            lr=args.lr,
            contrast_weight=args.contrast_weight,
            contrast_margin=args.contrast_margin,
        )
    if args.command == "probe-siglipvq-reconstruction":
        return _run_probe_siglipvq_reconstruction(
            config_path=args.config,
            sample_count=args.sample_count,
            split=args.split,
        )
    if args.command == "probe-t2i-token-guard":
        return _run_probe_t2i_token_guard(
            config_path=args.config,
            probe_steps=args.probe_steps,
            eval_every=args.eval_every,
            samples_per_label=args.samples_per_label,
            unconditional_count=args.unconditional_count,
            decode_samples=args.decode_samples,
            lr=args.lr,
        )
    if args.command == "probe-t2i-overfit":
        return _run_probe_t2i_overfit(
            config_path=args.config,
            steps=args.steps,
            sample_count=args.sample_count,
            test_sample_count=args.test_sample_count,
            lr=args.lr,
            eval_every=args.eval_every,
            progress_values=args.progress_values,
            token_probe_steps=args.token_probe_steps,
            unconditional_count=args.unconditional_count,
            save_model=args.save_model,
        )
    if args.command == "probe-t2i-distributional-ft":
        return _run_probe_t2i_distributional_ft(
            config_path=args.config,
            steps=args.steps,
            loss_kind=args.loss_kind,
            state_kind=args.state_kind,
            set_size=args.set_size,
            set_temperature=args.set_temperature,
            endpoint_prob=args.endpoint_prob,
            lr=args.lr,
            eval_steps=args.eval_steps,
            token_probe_steps=args.token_probe_steps,
            token_probe_eval_every=args.token_probe_eval_every,
            samples_per_label=args.samples_per_label,
            unconditional_count=args.unconditional_count,
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
