#!/usr/bin/env python3
"""Single-file mainline for the SigLIP-VQ unified FLM experiment.

The point of this file is to keep the research contract readable:

1. Encode images as SigLIP-VQ image tokens.
2. Encode class text as ordinary text tokens.
3. Put image and text tokens into one sequence.
4. Train one bidirectional Transformer denoiser with one shared output head.
5. Evaluate both image -> text and text -> image from the same sampler.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parent
MAIN_CONFIG = ROOT / "configs" / "main.yaml"
PYTHON = ROOT / ".venv" / "bin" / "python"


BEST_UNDERSTANDING_RESULT = {
    "branch": "codex/siglipvq-unified-minimal",
    "code_sha": "7ff026bc1a939c2f7e5f377bb72b00012e4807f0",
    "config": "configs/flm_joint_work_siglipvq_generation_labeltoken.yaml",
    "summary": (
        "/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/"
        "runs/minimal_unified_longer/20260428T094756Z-minimal-unified-gaussian-10000/summary.json"
    ),
    "i2t_exact_at_progress_0_5": 0.8671875,
    "i2t_token_at_progress_0_5": 0.93359375,
    "note": "Clean unified FLM run: no extra supervised side channel, no alternate decoder, one shared backbone and head.",
}


BEST_GENERATION_RESULT = {
    "branch": "codex/siglipvq-unified-minimal",
    "code_sha": "7ff026bc1a939c2f7e5f377bb72b00012e4807f0",
    "summary": BEST_UNDERSTANDING_RESULT["summary"],
    "conditioned_token_label_accuracy": 0.07500000298023224,
    "generated_unique_token_count": 769,
    "generated_vs_real_hist_l1": 0.4066070318222046,
    "passes_generation_gate": False,
    "note": "Image-token diversity improved, but text-conditioned generation is still not controlled enough.",
}


ACCEPTANCE = {
    "understanding_i2t_exact_min": 0.85,
    "generation_conditioned_token_label_min": 0.60,
}


MAIN_CONFIG_DATA: dict[str, Any] = {
    "project": {"name": "uniindex-main-siglipvq-clean"},
    "paths": {
        "data_dir": "data",
        "artifacts_dir": "artifacts/main_siglipvq_clean",
        "models_dir": "models/main_siglipvq_clean",
        "runs_dir": "runs/main_siglipvq_clean",
        "logs_dir": "logs/main_siglipvq_clean",
        "cache_dir": ".cache",
    },
    "tokenizer": {
        "kind": "siglip_vq",
        "model_name": "inclusionAI/LLaDA2.0-Uni",
        "trust_remote_code": False,
        "image_size": 512,
        "device": "cuda",
        "dtype": "bfloat16",
        "compact_vocab": False,
    },
    "dataset": {"name": "mnist", "train_limit": 512, "test_limit": 128},
    "text": {"kind": "label", "pad_token": "<pad>"},
    "labels": {"values": list(range(10))},
    "model": {
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 6,
        "mlp_ratio": 4,
        "dropout": 0.0,
    },
    "train": {
        "seed": 42,
        "device": "cuda",
        "gpu_index": 0,
        "mixed_precision": None,
        "batch_size": 8,
        "eval_batch_size": 8,
        "num_workers": 4,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "stage1_steps": 80,
        "stage2_steps": 10000,
        "log_every": 20,
        "save_every": 500,
        "joint_weight": 0.5,
        "text_weight": 1.0,
        "stage2_joint_repeats": 2,
        "stage2_text_to_image_repeats": 2,
        "stage2_image_to_text_repeats": 10,
        "image_time_power": 1.0,
        "text_time_power": 0.25,
        "image_to_text_text_time_power": 4.0,
    },
    "sampling": {
        "steps": 32,
        "temperature": 0.7,
        "image_time_power": 1.0,
        "text_time_power": 0.25,
        "image_to_text_text_time_power": 4.0,
    },
    "schedule": {"kind": "empirical", "num_points": 33, "num_samples": 4096, "min_t": 0.0},
    "eval": {
        "num_unconditional_samples": 16,
        "classifier_epochs": 2,
        "classifier_batch_size": 128,
        "classifier_lr": 0.001,
        "isolate_sampling_rng": True,
        "sampling_seed": 420700,
    },
}


@dataclass(frozen=True)
class Mainline:
    tokenizer: str = "siglip_vq"
    image_size: int = 512
    text: str = "label text tokens"
    backbone: str = "shared bidirectional Transformer denoiser"
    head: str = "one shared output head"
    objective: str = "same FLM denoising objective for image and text positions"
    best_i2t_exact: float = BEST_UNDERSTANDING_RESULT["i2t_exact_at_progress_0_5"]
    t2i_open_metric: float = BEST_GENERATION_RESULT["conditioned_token_label_accuracy"]


def acceptance_report() -> dict[str, Any]:
    understanding_pass = BEST_UNDERSTANDING_RESULT["i2t_exact_at_progress_0_5"] >= ACCEPTANCE["understanding_i2t_exact_min"]
    generation_pass = (
        BEST_GENERATION_RESULT["conditioned_token_label_accuracy"]
        >= ACCEPTANCE["generation_conditioned_token_label_min"]
    )
    return {
        "acceptance": ACCEPTANCE,
        "understanding": {
            "passed": understanding_pass,
            "result": BEST_UNDERSTANDING_RESULT,
        },
        "generation": {
            "passed": generation_pass,
            "result": BEST_GENERATION_RESULT,
        },
        "overall_passed": understanding_pass and generation_pass,
        "conclusion": (
            "Clean SigLIP-VQ unified FLM has a real image-to-text signal; "
            "text-to-image conditioning remains the main open problem."
        ),
    }


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = torch.log(torch.tensor(10000.0, device=t.device)) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
    angles = t.reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.reshape(*t.shape, dim)


class UnifiedFLM(nn.Module):
    """The core model: one sequence, one denoiser, one output head."""

    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.in_proj = nn.Linear(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.modality = nn.Embedding(2, d_model)
        self.time = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model))
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.blocks = nn.TransformerEncoder(block, n_layers, enable_nested_tensor=False)
        except TypeError:
            self.blocks = nn.TransformerEncoder(block, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def features(self, z: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = z.shape
        del bsz
        if seq_len != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {seq_len}")
        pos = torch.arange(seq_len, device=z.device)
        h = self.in_proj(z) + self.pos(pos).unsqueeze(0) + self.modality(modality_ids.to(z.device)).unsqueeze(0)
        te = self.time(time_embedding(t, h.shape[-1]).to(h.dtype))
        h = h + (te.unsqueeze(1) if t.ndim == 1 else te)
        return self.ln(self.blocks(h))

    def forward(self, z: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(z, t, modality_ids))


def write_config(path: Path = MAIN_CONFIG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(MAIN_CONFIG_DATA, handle, sort_keys=False)


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT / "src")
    env.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    env.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface" / "transformers"))
    env.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_uniindex(args: list[str]) -> int:
    cmd = [str(PYTHON if PYTHON.exists() else Path(sys.executable)), "-m", "uniindex.cli", *args]
    return subprocess.call(cmd, cwd=ROOT, env=python_env())


def reproduce_commands(config: Path = MAIN_CONFIG) -> list[list[str]]:
    cfg = str(config.relative_to(ROOT) if config.is_absolute() and config.is_relative_to(ROOT) else config)
    py = str(PYTHON if PYTHON.exists() else Path(sys.executable))
    return [
        [py, "unified.py", "config", "--out", cfg],
        [py, "-m", "uniindex.cli", "prepare", "--config", cfg],
        [py, "-m", "uniindex.cli", "train", "--config", cfg, "--stage", "stage1"],
        [py, "-m", "uniindex.cli", "train", "--config", cfg, "--stage", "stage2"],
        [py, "-m", "uniindex.cli", "eval", "--config", cfg],
        [py, "-m", "uniindex.cli", "visualize", "--config", cfg],
        [py, "-m", "uniindex.cli", "probe-siglipvq-reconstruction", "--config", cfg, "--sample-count", "16"],
    ]


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal SigLIP-VQ unified FLM mainline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("about")
    sub.add_parser("acceptance")
    config_p = sub.add_parser("config")
    config_p.add_argument("--out", default=str(MAIN_CONFIG))
    sub.add_parser("commands")
    sub.add_parser("prepare")
    train_p = sub.add_parser("train")
    train_p.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    sub.add_parser("eval")
    sub.add_parser("visualize")
    args = parser.parse_args(argv)

    if args.cmd == "about":
        print_json(
            {
                "mainline": asdict(Mainline()),
                "best_understanding": BEST_UNDERSTANDING_RESULT,
                "best_generation": BEST_GENERATION_RESULT,
            }
        )
        return 0
    if args.cmd == "acceptance":
        report = acceptance_report()
        print_json(report)
        return 0 if report["overall_passed"] else 1
    if args.cmd == "config":
        write_config(Path(args.out))
        print(Path(args.out))
        return 0
    if args.cmd == "commands":
        for cmd in reproduce_commands():
            print(" ".join(cmd))
        return 0
    if args.cmd == "prepare":
        return run_uniindex(["prepare", "--config", str(MAIN_CONFIG)])
    if args.cmd == "train":
        return run_uniindex(["train", "--config", str(MAIN_CONFIG), "--stage", args.stage])
    if args.cmd == "eval":
        return run_uniindex(["eval", "--config", str(MAIN_CONFIG)])
    if args.cmd == "visualize":
        return run_uniindex(["visualize", "--config", str(MAIN_CONFIG)])
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
