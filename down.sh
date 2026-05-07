#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="configs/main.yaml"
DOWNLOAD_I2T_LLM=0
DOWNLOAD_SIGLIP_VQ=0
DOWNLOAD_SIGLIP_VQ_DECODER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --i2t-llm)
      DOWNLOAD_I2T_LLM=1
      shift
      ;;
    --siglip-vq)
      DOWNLOAD_SIGLIP_VQ=1
      shift
      ;;
    --siglip-vq-decoder)
      DOWNLOAD_SIGLIP_VQ_DECODER=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done

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

mkdir -p "$ROOT/.cache" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$ROOT/data" "$ROOT/artifacts" "$ROOT/models" "$ROOT/runs" "$ROOT/logs"

"$PYTHON" -m uniindex.cli prepare --config "$CONFIG"

if [[ "$DOWNLOAD_I2T_LLM" == "1" ]]; then
  "$PYTHON" -m uniindex.cli download-i2t-llm --config "$CONFIG"
fi

if [[ "$DOWNLOAD_SIGLIP_VQ" == "1" ]]; then
  "$PYTHON" -m uniindex.cli download-siglip-vq --config "$CONFIG"
fi

if [[ "$DOWNLOAD_SIGLIP_VQ_DECODER" == "1" ]]; then
  "$PYTHON" -m uniindex.cli download-siglip-vq-decoder --config "$CONFIG"
fi
