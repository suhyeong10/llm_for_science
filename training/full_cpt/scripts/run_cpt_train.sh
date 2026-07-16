#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_yaml> [--resume <checkpoint_path>]"
  exit 1
fi

CONFIG_PATH="$1"
shift || true

RESUME_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "[ERROR] --resume requires a checkpoint path"
    exit 1
  fi
  RESUME_ARGS=(--resume_from_checkpoint "$2")
  shift 2 || true
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[ERROR] Python interpreter not found (python3/python)."
    exit 1
  fi
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG_DIR="logs/cpt"
mkdir -p "${RUN_LOG_DIR}"

RUN_LOG="${RUN_LOG_DIR}/train_${TIMESTAMP}.log"

echo "[INFO] Starting CPT training"
echo "[INFO] Config: ${CONFIG_PATH}"
echo "[INFO] Log: ${RUN_LOG}"

"${PYTHON_BIN}" -m training.full_cpt.train \
  --config "${CONFIG_PATH}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "${RUN_LOG}"

echo "[INFO] Training finished."
