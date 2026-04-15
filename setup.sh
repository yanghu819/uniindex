#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-600}"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export UNIINDEX_WHEELHOUSE_MANIFEST="${UNIINDEX_WHEELHOUSE_MANIFEST:-manifest.sha256}"
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
  "$PIP_CACHE_DIR" \
  "$ROOT/.cache/wheelhouse" \
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

PYTHON_BIN="$ROOT/.venv/bin/python"
WHEELHOUSE_CACHE_DIR="$ROOT/.cache/wheelhouse"
RUNTIME_REQUIREMENTS=(
  huggingface-hub==0.35.3
  numpy==2.2.6
  pillow==11.3.0
  pyyaml==6.0.3
  safetensors==0.6.2
  tqdm==4.67.1
  transformers==4.57.1
)
DEV_REQUIREMENTS=(
  pytest==8.4.2
  ruff==0.13.1
)

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m ensurepip --upgrade
fi

download_wheelhouse() {
  local base_url="$1"
  local manifest_path="$WHEELHOUSE_CACHE_DIR/$UNIINDEX_WHEELHOUSE_MANIFEST"
  mkdir -p "$WHEELHOUSE_CACHE_DIR"
  curl -fL --retry 20 --retry-delay 5 --retry-all-errors \
    -o "$manifest_path" \
    "$base_url/$UNIINDEX_WHEELHOUSE_MANIFEST"
  while read -r sha filename; do
    [[ -z "${sha:-}" || -z "${filename:-}" ]] && continue
    if [[ ! -f "$WHEELHOUSE_CACHE_DIR/$filename" ]]; then
      curl -fL --retry 20 --retry-delay 5 --retry-all-errors \
        -o "$WHEELHOUSE_CACHE_DIR/$filename" \
        "$base_url/$filename"
    fi
  done < "$manifest_path"
  (cd "$WHEELHOUSE_CACHE_DIR" && sha256sum -c "$UNIINDEX_WHEELHOUSE_MANIFEST")
}

install_torch_stack() {
  local find_links="$1"
  "$PYTHON_BIN" -m pip install \
    --timeout "$PIP_DEFAULT_TIMEOUT" \
    --retries 20 \
    --no-index \
    --find-links "$find_links" \
    torch==2.6.0 \
    torchvision==0.21.0
}

install_runtime_stack_offline() {
  local find_links="$1"
  "$PYTHON_BIN" -m pip install \
    --timeout "$PIP_DEFAULT_TIMEOUT" \
    --retries 20 \
    --no-index \
    --find-links "$find_links" \
    "${RUNTIME_REQUIREMENTS[@]}"
}

install_runtime_stack_online() {
  "$PYTHON_BIN" -m pip install \
    --timeout "$PIP_DEFAULT_TIMEOUT" \
    --retries 20 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "${RUNTIME_REQUIREMENTS[@]}"
}

install_dev_stack_online() {
  "$PYTHON_BIN" -m pip install \
    --timeout "$PIP_DEFAULT_TIMEOUT" \
    --retries 20 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "${DEV_REQUIREMENTS[@]}"
}

if [[ "${UNIINDEX_SETUP_USE_PIP:-0}" != "1" ]]; then
  if uv sync --project "$ROOT" --extra dev --frozen; then
    echo "Environment ready at $ROOT/.venv"
    exit 0
  fi
fi

"$PYTHON_BIN" -m pip install --upgrade pip

if [[ -n "${UNIINDEX_WHEELHOUSE_BASE_URL:-}" ]]; then
  download_wheelhouse "$UNIINDEX_WHEELHOUSE_BASE_URL"
  install_torch_stack "$WHEELHOUSE_CACHE_DIR"
  install_runtime_stack_offline "$WHEELHOUSE_CACHE_DIR"
elif [[ -n "${UNIINDEX_WHEELHOUSE_DIR:-}" ]]; then
  install_torch_stack "$UNIINDEX_WHEELHOUSE_DIR"
  install_runtime_stack_offline "$UNIINDEX_WHEELHOUSE_DIR"
else
  "$PYTHON_BIN" -m pip install \
    --timeout "$PIP_DEFAULT_TIMEOUT" \
    --retries 20 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.6.0 \
    torchvision==0.21.0
  install_runtime_stack_online
fi

if [[ "${UNIINDEX_INSTALL_DEV:-0}" == "1" ]]; then
  install_dev_stack_online
fi

"$PYTHON_BIN" -m pip install \
  --timeout "$PIP_DEFAULT_TIMEOUT" \
  --retries 20 \
  -e "$ROOT"

echo "Environment ready at $ROOT/.venv"
