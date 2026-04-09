#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPO_ROOT
export UV_CACHE_DIR="${REPO_ROOT}/artifacts/cache/uv"
export HF_HOME="${REPO_ROOT}/artifacts/cache/hf"
export HUGGINGFACE_HUB_CACHE="${REPO_ROOT}/artifacts/cache/hf/hub"
export TRANSFORMERS_CACHE="${REPO_ROOT}/artifacts/cache/hf/transformers"
export TORCH_HOME="${REPO_ROOT}/artifacts/cache/torch"
export XDG_CACHE_HOME="${REPO_ROOT}/artifacts/cache/xdg"
export WANDB_DIR="${REPO_ROOT}/artifacts/cache/wandb"
export TMPDIR="${REPO_ROOT}/artifacts/tmp"

mkdir -p \
  "${UV_CACHE_DIR}" \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" \
  "${TMPDIR}" \
  "${REPO_ROOT}/artifacts/datasets" \
  "${REPO_ROOT}/artifacts/models" \
  "${REPO_ROOT}/runs"

