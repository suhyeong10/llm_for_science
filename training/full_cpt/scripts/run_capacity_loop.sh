#!/usr/bin/env bash
set -euo pipefail

# Capacity loop for quick resource-fit checks.
# It runs multiple short dry-runs while mutating selected trainer/data knobs.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <base_config_yaml>"
  exit 1
fi

BASE_CONFIG="$1"
OUT_DIR="training/full_cpt/tests/capacity_loop"
LOG_DIR="logs/cpt/capacity_loop"
mkdir -p "$OUT_DIR" "$LOG_DIR"

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

# seq_len, grad_accum, train_bs
CASES=(
  "64,1,1"
  "128,2,1"
  "256,4,1"
)

SUMMARY="$OUT_DIR/summary.tsv"
echo -e "case\tseq_len\tgrad_accum\tbatch\tstatus\tlog" > "$SUMMARY"

i=0
for case in "${CASES[@]}"; do
  i=$((i+1))
  IFS=',' read -r SEQ GA BS <<< "$case"

  CFG="$OUT_DIR/case_${i}.yaml"
  LOG="$LOG_DIR/case_${i}.log"

  "$PYTHON_BIN" - <<PY
import yaml
from copy import deepcopy
cfg = yaml.safe_load(open("$BASE_CONFIG", "r", encoding="utf-8"))
cfg = deepcopy(cfg)
cfg["run"]["output_dir"] = "training/full_cpt/tests/capacity_loop/out_case_${i}"
cfg["data"]["max_seq_length"] = int("$SEQ")
cfg["trainer"]["gradient_accumulation_steps"] = int("$GA")
cfg["trainer"]["per_device_train_batch_size"] = int("$BS")
with open("$CFG", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  status="PASS"
  if ! "$PYTHON_BIN" -m training.full_cpt.train --config "$CFG" --dry_run > "$LOG" 2>&1; then
    status="FAIL"
  fi

  echo -e "case_${i}\t${SEQ}\t${GA}\t${BS}\t${status}\t${LOG}" >> "$SUMMARY"
  echo "[capacity-loop] case_${i}: ${status}"
done

echo "Summary: $SUMMARY"
cat "$SUMMARY"
