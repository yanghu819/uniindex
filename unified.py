#!/usr/bin/env python3
"""Single-file mainline for the SigLIP-VQ unified FLM experiment.

This file is intentionally small and explicit.  The old package modules are
still kept for historical experiments, but the main branch should be readable
from here:

1. Encode images with SigLIP-VQ tokens.
2. Put image tokens, one label-text token, and one semantic image token into
   one sequence.
3. Train one bidirectional Transformer denoiser with one shared head.
4. Reproduce the best current understanding result with the commands below.
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
    "branch": "codex/resume-schedule-fix-layout-integ",
    "code_sha": "b96d5f195340650e0082725b57af1aa9ad85420c",
    "config": "configs/flm_joint_work_siglipvq_generation_labeltoken_semantic_vqtoken_probe.yaml",
    "summary": (
        "/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/"
        "runs/semantic_vqtoken/20260427T061021Z-semantic-vqtoken-quick/summary.json"
    ),
    "i2t_exact_at_progress_0_5": 0.8828125,
    "i2t_token_at_progress_0_5": 0.94140625,
    "semantic_hidden_label_probe": 0.9609375,
    "note": "SigLIP-VQ token source made the unified FLM semantic token carry label signal.",
}


BEST_GENERATION_RESULTS = {
    "clean_minimal_extra_long": {
        "code_sha": "7ff026bc1a939c2f7e5f377bb72b00012e4807f0",
        "summary": (
            "/fangxueji/Projects/PG/uniindex/worktrees/schedule-fix-layout-integ/"
            "runs/minimal_unified_longer/20260428T094756Z-minimal-unified-gaussian-10000/summary.json"
        ),
        "i2t_exact_at_progress_0_5": 0.8671875,
        "conditioned_token_label_accuracy": 0.07500000298023224,
        "generated_unique_token_count": 769,
        "generated_vs_real_hist_l1": 0.4066070318222046,
        "passes_generation_gate": False,
        "note": "Clean unified FLM scales i2t and token distribution, but text-conditioned t2i control is still chance-like.",
    },
    "label_control_probe_not_mainline": {
        "code_sha": "deca613e21c0c522da2eb17b54d67e4fc804a892",
        "conditioned_token_label_accuracy": 0.8999999761581421,
        "passes_generation_gate": False,
        "note": "Shows t2i can be forced, but it changes the endpoint objective and hurts the clean unified mainline.",
    },
}


ACCEPTANCE = {
    "understanding_i2t_exact_min": 0.85,
    "semantic_hidden_probe_min": 0.90,
    "generation_conditioned_token_label_min": 0.60,
}


MAIN_CONFIG_DATA: dict[str, Any] = {
    "project": {"name": "uniindex-main-siglipvq-semantic-token"},
    "paths": {
        "data_dir": "data",
        "artifacts_dir": "artifacts/main_siglipvq_semantic_token",
        "models_dir": "models/main_siglipvq_semantic_token",
        "runs_dir": "runs/main_siglipvq_semantic_token",
        "logs_dir": "logs/main_siglipvq_semantic_token",
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
        "image_summary_to_text": False,
        "image_semantic_tokens": 1,
        "image_semantic_source": "vq_tokens",
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
        "stage2_steps": 500,
        "log_every": 20,
        "save_every": 100,
        "joint_weight": 0.5,
        "text_weight": 1.0,
        "text_sequence_weight": 0.75,
        "stage2_joint_repeats": 2,
        "stage2_text_to_image_repeats": 2,
        "stage2_image_to_text_repeats": 10,
        "image_time_power": 1.0,
        "text_time_power": 0.25,
        "image_to_text_text_time_power": 4.0,
        "image_to_text_label_weight": 1.0,
        "image_to_text_label_text_time": 0.0,
        "image_to_text_semantic_weight": 5.0,
        "image_to_text_semantic_text_time": 0.0,
        "image_to_text_semantic_pool": "semantic",
    },
    "sampling": {
        "steps": 32,
        "temperature": 0.7,
        "image_time_power": 1.0,
        "text_time_power": 0.25,
        "image_to_text_text_time_power": 4.0,
        "integrator": "legacy_progress_euler",
        "final_decode": "final_model_call",
        "final_model_progress": 0.95,
        "image_to_text_decoder": "sample",
        "image_to_text_projection": "none",
        "image_to_text_projection_progress": 0.5,
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
    "i2t_llm": {
        "enabled": False,
        "model_name": "distilgpt2",
        "cache_dir": ".cache/huggingface",
        "prefix_tokens": 8,
        "adapter_hidden_dim": 512,
        "source_checkpoint": None,
        "train_steps": 200,
        "lr": 0.0003,
        "prompt": "Digit:",
        "feature_progress": 0.5,
        "feature_pool": "semantic",
        "max_new_tokens": 4,
    },
}


@dataclass(frozen=True)
class Mainline:
    tokenizer: str = "siglip_vq"
    image_size: int = 512
    text: str = "single label token"
    backbone: str = "shared bidirectional Transformer denoiser"
    semantic_token_source: str = "vq_tokens"
    best_i2t_exact: float = BEST_UNDERSTANDING_RESULT["i2t_exact_at_progress_0_5"]
    best_semantic_probe: float = BEST_UNDERSTANDING_RESULT["semantic_hidden_label_probe"]


def acceptance_report() -> dict[str, Any]:
    understanding_pass = (
        BEST_UNDERSTANDING_RESULT["i2t_exact_at_progress_0_5"] >= ACCEPTANCE["understanding_i2t_exact_min"]
        and BEST_UNDERSTANDING_RESULT["semantic_hidden_label_probe"] >= ACCEPTANCE["semantic_hidden_probe_min"]
    )
    clean_generation = BEST_GENERATION_RESULTS["clean_minimal_extra_long"]
    generation_pass = (
        clean_generation["conditioned_token_label_accuracy"]
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
            "clean_result": clean_generation,
            "non_mainline_probe": BEST_GENERATION_RESULTS["label_control_probe_not_mainline"],
        },
        "overall_passed": understanding_pass and generation_pass,
        "conclusion": (
            "Understanding is reproduced by the SigLIP-VQ semantic-token route; "
            "clean text-to-image generation is still the open blocker."
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
        image_vocab_size: int,
        seq_len: int,
        image_seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        image_semantic_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.image_vocab_size = image_vocab_size
        self.seq_len = seq_len
        self.image_seq_len = image_seq_len
        self.image_semantic_tokens = image_semantic_tokens
        self.in_proj = nn.Linear(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.modality = nn.Embedding(2, d_model)
        self.time = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model))
        self.vq_code = nn.Embedding(image_vocab_size, d_model)
        self.vq_pos = nn.Embedding(seq_len, d_model)
        self.semantic = nn.Parameter(torch.randn(1, image_semantic_tokens, d_model) * 0.02)
        self.semantic_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
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

    def append_semantic_token(self, h: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        image_probs = z[:, : self.image_seq_len, : self.image_vocab_size].to(h.dtype)
        image_pos = torch.arange(self.image_seq_len, device=z.device)
        image_hidden = image_probs @ self.vq_code.weight.to(h.dtype)
        image_hidden = image_hidden + self.vq_pos(image_pos).to(h.dtype).unsqueeze(0)
        image_summary = image_hidden.mean(dim=1)
        semantic = self.semantic.to(h.dtype) + self.semantic_proj(image_summary).unsqueeze(1)
        if t.ndim == 2:
            gate = (t[:, : self.image_seq_len].mean(1) * (1.0 - t[:, self.image_seq_len :].mean(1))).clamp(0.0, 1.0)
            semantic = semantic * gate.reshape(-1, 1, 1).to(h.dtype)
        return torch.cat([h, semantic.expand(-1, self.image_semantic_tokens, -1)], dim=1)

    def features(self, z: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor, *, include_semantic: bool = False) -> torch.Tensor:
        bsz, seq_len, _ = z.shape
        if seq_len != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {seq_len}")
        pos = torch.arange(seq_len, device=z.device)
        h = self.in_proj(z) + self.pos(pos).unsqueeze(0) + self.modality(modality_ids.to(z.device)).unsqueeze(0)
        te = self.time(time_embedding(t, h.shape[-1]).to(h.dtype))
        h = h + (te.unsqueeze(1) if t.ndim == 1 else te)
        h = self.append_semantic_token(h, z, t)
        h = self.ln(self.blocks(h))
        return h if include_semantic else h[:, :seq_len]

    def forward(self, z: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(z, t, modality_ids))


def write_config(path: Path = MAIN_CONFIG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
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
        [
            py,
            "-m",
            "uniindex.cli",
            "diagnose-i2t",
            "--config",
            cfg,
            "--progress",
            "0.5",
            "--progress",
            "0.9",
            "--progress",
            "0.95",
        ],
        [py, "-m", "uniindex.cli", "probe-label-features", "--config", cfg, "--steps", "200", "--eval-every", "100"],
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
    diag_p = sub.add_parser("diagnose")
    diag_p.add_argument("--progress", action="append", default=["0.5", "0.9", "0.95"])
    sub.add_parser("probe")
    args = parser.parse_args(argv)

    if args.cmd == "about":
        print_json(
            {
                "mainline": asdict(Mainline()),
                "best_understanding": BEST_UNDERSTANDING_RESULT,
                "best_generation": BEST_GENERATION_RESULTS,
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
    if args.cmd == "diagnose":
        cli_args = ["diagnose-i2t", "--config", str(MAIN_CONFIG)]
        for progress in args.progress:
            cli_args.extend(["--progress", str(progress)])
        return run_uniindex(cli_args)
    if args.cmd == "probe":
        return run_uniindex(["probe-label-features", "--config", str(MAIN_CONFIG), "--steps", "200", "--eval-every", "100"])
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
