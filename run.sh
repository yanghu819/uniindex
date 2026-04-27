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

if ! command -v uv >/dev/null 2>&1; then
  export UV_UNMANAGED_INSTALL="$ROOT/.cache/uv-bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$ROOT/.cache/uv-bin:$PATH"
fi


run_cli() {
  "$ROOT/.venv/bin/python" -m uniindex.cli "$@"
}


case "$MODE" in
  prepare)
    CONFIG="configs/default.yaml"
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
    run_cli prepare --config "$CONFIG"
    ;;
  smoke)
    CONFIG="configs/smoke.yaml"
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
    run_cli smoke --config "$CONFIG"
    ;;
  ablate-compact)
    COMPACT_CONFIG="configs/smoke_compact_clean.yaml"
    FULL_CONFIG="configs/smoke_fullvocab_clean.yaml"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --compact-config)
          COMPACT_CONFIG="$2"
          shift 2
          ;;
        --full-config)
          FULL_CONFIG="$2"
          shift 2
          ;;
        *)
          shift
          ;;
      esac
    done
    run_cli ablate-compact --compact-config "$COMPACT_CONFIG" --full-config "$FULL_CONFIG"
    ;;
  stage1)
    CONFIG="configs/default.yaml"
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
    run_cli train --config "$CONFIG" --stage stage1
    ;;
  stage2)
    CONFIG="configs/default.yaml"
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
    run_cli train --config "$CONFIG" --stage stage2
    ;;
  eval)
    CONFIG="configs/default.yaml"
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
    run_cli eval --config "$CONFIG"
    ;;
  diagnose-i2t)
    CONFIG="configs/default.yaml"
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
    run_cli diagnose-i2t --config "$CONFIG" "${ARGS[@]}"
    ;;
  diagnose-i2t-image-dependence)
    CONFIG="configs/default.yaml"
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
    run_cli diagnose-i2t-image-dependence --config "$CONFIG" "${ARGS[@]}"
    ;;
  diagnose-i2t-sampler-trajectory)
    CONFIG="configs/default.yaml"
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
    run_cli diagnose-i2t-sampler-trajectory --config "$CONFIG" "${ARGS[@]}"
    ;;
  diagnose-i2t-understanding)
    CONFIG="configs/default.yaml"
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
    run_cli diagnose-i2t-understanding --config "$CONFIG" "${ARGS[@]}"
    ;;
  probe-i2t-overfit)
    CONFIG="configs/default.yaml"
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
    run_cli probe-i2t-overfit --config "$CONFIG" "${ARGS[@]}"
    ;;
  probe-i2t-sampler-state-ft)
    CONFIG="configs/default.yaml"
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
    run_cli probe-i2t-sampler-state-ft --config "$CONFIG" "${ARGS[@]}"
    ;;
  probe-i2t-llm-decoder)
    CONFIG="configs/flm_joint_work_fullvocab_tsw075_i2t_llm_decoder.yaml"
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
    run_cli probe-i2t-llm-decoder --config "$CONFIG" "${ARGS[@]}"
    ;;
  probe-label-features)
    CONFIG="configs/flm_joint_work_fullvocab_tsw075_label_feature_probe.yaml"
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
    run_cli probe-label-features --config "$CONFIG" "${ARGS[@]}"
    ;;
  probe-vq-text-decoder)
    CONFIG="configs/flm_joint_work_siglipvq_text_decoder_probe.yaml"
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
    run_cli probe-vq-text-decoder --config "$CONFIG" "${ARGS[@]}"
    ;;
  probe-siglipvq-reconstruction)
    CONFIG="configs/flm_joint_work_siglipvq_generation_probe.yaml"
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
    run_cli probe-siglipvq-reconstruction --config "$CONFIG" "${ARGS[@]}"
    ;;
  probe-t2i-token-guard)
    CONFIG="configs/flm_joint_work_siglipvq_generation_labeltoken_semantic_vqtoken_probe.yaml"
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
    run_cli probe-t2i-token-guard --config "$CONFIG" "${ARGS[@]}"
    ;;
  probe-t2i-overfit)
    CONFIG="configs/flm_joint_work_siglipvq_generation_labeltoken_semantic_vqtoken_probe.yaml"
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
    run_cli probe-t2i-overfit --config "$CONFIG" "${ARGS[@]}"
    ;;
  visualize)
    CONFIG="configs/default.yaml"
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
    run_cli visualize --config "$CONFIG"
    ;;
  sweep-i2t-power)
    run_cli sweep-i2t-power "$@"
    ;;
  sweep-i2t-repeats)
    run_cli sweep-i2t-repeats "$@"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac
