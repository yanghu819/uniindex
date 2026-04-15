#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-600}"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_CACHE="$ROOT/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export PATH="$ROOT/.cache/uv-bin:$PATH"

mkdir -p \
  "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$MPLCONFIGDIR"

if ! command -v uv >/dev/null 2>&1; then
  export UV_UNMANAGED_INSTALL="$ROOT/.cache/uv-bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$ROOT/.cache/uv-bin:$PATH"
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  uv venv "$ROOT/.venv"
fi

if uv sync --project "$ROOT" --extra dev --frozen; then
  echo "Environment ready at $ROOT/.venv"
  exit 0
fi

PYTHON_BIN="$ROOT/.venv/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install \
  --timeout "$PIP_DEFAULT_TIMEOUT" \
  --retries 20 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.6.0 \
  torchvision==0.21.0
"$PYTHON_BIN" -m pip install \
  --timeout "$PIP_DEFAULT_TIMEOUT" \
  --retries 20 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  huggingface-hub==0.35.3 \
  numpy==2.2.6 \
  pillow==11.3.0 \
  pyyaml==6.0.3 \
  safetensors==0.6.2 \
  tqdm==4.67.1 \
  transformers==4.57.1 \
  pytest==8.4.2 \
  ruff==0.13.1
"$PYTHON_BIN" -m pip install \
  --timeout "$PIP_DEFAULT_TIMEOUT" \
  --retries 20 \
  -e "$ROOT"

echo "Environment ready at $ROOT/.venv"
