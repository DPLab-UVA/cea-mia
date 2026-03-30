#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${CEA_MI_LOG_DIR:-$SCRIPT_DIR/results/logs}"
DB_PATH="${CEA_MI_NANOBOT_DB_PATH:-$HOME/.nanobot/memory/pmc.db}"
DATASET_PATH="${CEA_MI_DATASET:-}"

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

if [[ -z "$DATASET_PATH" ]]; then
  echo "CEA_MI_DATASET must point to benchmark_v2_dataset.json before running natural_attack.py" >&2
  exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset file does not exist: $DATASET_PATH" >&2
  exit 1
fi

echo "=== Starting blackbox v3 ==="
python3 natural_attack.py --access blackbox --num-facts 30 --seed 42 --db "$DB_PATH" --dataset "$DATASET_PATH" 2>&1 | tee "$LOG_DIR/attack_v3_blackbox.log"
echo "=== Blackbox done ==="

echo "=== Starting graybox v3 ==="
python3 natural_attack.py --access graybox --num-facts 30 --seed 42 --db "$DB_PATH" --dataset "$DATASET_PATH" 2>&1 | tee "$LOG_DIR/attack_v3_graybox.log"
echo "=== Graybox done ==="

echo "=== Starting whitebox v3 ==="
python3 natural_attack.py --access whitebox --num-facts 30 --seed 42 --db "$DB_PATH" --dataset "$DATASET_PATH" 2>&1 | tee "$LOG_DIR/attack_v3_whitebox.log"
echo "=== ALL DONE ==="
