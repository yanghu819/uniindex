#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-smoke}"

export UV_CACHE_DIR="$ROOT/.cache/uv"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_CACHE="$ROOT/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if ! command -v uv >/dev/null 2>&1; then
  export UV_UNMANAGED_INSTALL="$ROOT/.cache/uv-bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$ROOT/.cache/uv-bin:$PATH"
fi


CONFIG="configs/default.yaml"
if [[ "$MODE" == "smoke" ]]; then
  CONFIG="configs/smoke.yaml"
fi

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

case "$MODE" in
  smoke)
    uv run --project "$ROOT" uniindex smoke --config "$CONFIG"
    ;;
  stage1)
    uv run --project "$ROOT" uniindex train --config "$CONFIG" --stage stage1
    ;;
  stage2)
    uv run --project "$ROOT" uniindex train --config "$CONFIG" --stage stage2
    ;;
  eval)
    uv run --project "$ROOT" uniindex eval --config "$CONFIG"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac
