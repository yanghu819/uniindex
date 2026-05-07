#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-smoke}"
shift || true

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_CACHE="$ROOT/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PATH="$ROOT/.cache/uv-bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" && -x "$ROOT/../../.venv/bin/python" ]]; then
  PYTHON="$ROOT/../../.venv/bin/python"
fi

if ! command -v uv >/dev/null 2>&1; then
  export UV_UNMANAGED_INSTALL="$ROOT/.cache/uv-bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$ROOT/.cache/uv-bin:$PATH"
fi

CONFIG="configs/main.yaml"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

run_cli() {
  "$PYTHON" -m uniindex.cli "$@"
}

case "$MODE" in
  prepare)
    run_cli prepare --config "$CONFIG"
    ;;
  smoke)
    if [[ "$CONFIG" == "configs/main.yaml" ]]; then
      CONFIG="configs/smoke.yaml"
    fi
    run_cli smoke --config "$CONFIG"
    ;;
  ablate-compact)
    COMPACT_CONFIG="configs/smoke_compact_clean.yaml"
    FULL_CONFIG="configs/smoke_fullvocab_clean.yaml"
    idx=0
    while [[ $idx -lt ${#ARGS[@]} ]]; do
      case "${ARGS[$idx]}" in
        --compact-config)
          idx=$((idx + 1))
          COMPACT_CONFIG="${ARGS[$idx]}"
          ;;
        --full-config)
          idx=$((idx + 1))
          FULL_CONFIG="${ARGS[$idx]}"
          ;;
      esac
      idx=$((idx + 1))
    done
    run_cli ablate-compact --compact-config "$COMPACT_CONFIG" --full-config "$FULL_CONFIG"
    ;;
  stage1)
    run_cli train --config "$CONFIG" --stage stage1
    ;;
  stage2)
    run_cli train --config "$CONFIG" --stage stage2
    ;;
  eval)
    run_cli eval --config "$CONFIG"
    ;;
  visualize)
    run_cli visualize --config "$CONFIG"
    ;;
  probe-siglipvq-reconstruction)
    run_cli probe-siglipvq-reconstruction --config "$CONFIG" "${ARGS[@]}"
    ;;
  reproduce-main)
    run_cli prepare --config "$CONFIG"
    run_cli train --config "$CONFIG" --stage stage1
    run_cli train --config "$CONFIG" --stage stage2
    run_cli eval --config "$CONFIG"
    run_cli visualize --config "$CONFIG"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac
