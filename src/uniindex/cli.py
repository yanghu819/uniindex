from __future__ import annotations

import argparse
import sys

from .config import load_config
from .data import prepare_assets
from .eval import evaluate
from .runtime import RunContext, ensure_project_dirs
from .train import train_stage


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
    evaluate(config)
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
    if args.command == "smoke":
        return _run_smoke(args.config)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
