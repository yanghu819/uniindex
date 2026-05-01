from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uniindex.config import load_config
from uniindex.data import split_path, tokenizer_state_path
from uniindex.text import build_text_metadata, encode_labels, text_state_dict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a relabeled tokenized dataset without re-encoding images.")
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    return parser


def _copy_split(source_config_path: str, target_config_path: str, split: str) -> None:
    source = load_config(source_config_path)
    target = load_config(target_config_path)
    source_path = split_path(source, split)
    target_path = split_path(target, split)
    if not source_path.exists():
        raise FileNotFoundError(f"missing source split: {source_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(source_path, map_location="cpu")
    text_metadata = build_text_metadata(
        kind=target.text.kind,
        label_values=target.labels.values,
        strings=target.text.strings or [],
        pad_token=target.text.pad_token,
        bos_token=target.text.bos_token,
        eos_token=target.text.eos_token,
    )
    payload["text_tokens"] = encode_labels(payload["labels"], text_metadata).cpu()
    torch.save(payload, target_path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = load_config(args.source_config)
    target = load_config(args.target_config)
    source_state_path = tokenizer_state_path(source)
    target_state_path = tokenizer_state_path(target)
    if not source_state_path.exists():
        raise FileNotFoundError(f"missing source tokenizer state: {source_state_path}")

    _copy_split(args.source_config, args.target_config, "train")
    _copy_split(args.source_config, args.target_config, "test")

    target_state_path.parent.mkdir(parents=True, exist_ok=True)
    state = torch.load(source_state_path, map_location="cpu")
    text_metadata = build_text_metadata(
        kind=target.text.kind,
        label_values=target.labels.values,
        strings=target.text.strings or [],
        pad_token=target.text.pad_token,
        bos_token=target.text.bos_token,
        eos_token=target.text.eos_token,
    )
    state.update(text_state_dict(text_metadata))
    state["label_values"] = torch.tensor(target.labels.values, dtype=torch.long)
    torch.save(state, target_state_path)
    print(f"wrote {target_state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
