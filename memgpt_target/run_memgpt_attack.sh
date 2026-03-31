#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${CEA_MI_LOG_DIR:-$REPO_ROOT/results/logs}"
DATASET_PATH="${CEA_MI_DATASET:-}"
MEMORY_FILE="${CEA_MI_MEMGPT_MEMORY_FILE:-$SCRIPT_DIR/memgpt_memories.json}"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

if [[ -z "$DATASET_PATH" ]]; then
  echo "CEA_MI_DATASET must point to benchmark_v2_dataset.json before running the MemGPT comparison" >&2
  exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset file does not exist: $DATASET_PATH" >&2
  exit 1
fi

echo "=== Setting up embedding memory agent ==="
python3 memgpt_target/setup_memgpt.py standalone --dataset "$DATASET_PATH" --memory-file "$MEMORY_FILE"
echo "=== Setup done ==="

for ACCESS in blackbox graybox whitebox; do
  echo "=== Starting MemGPT ${ACCESS} ==="
  python3 memgpt_target/memgpt_attack.py --access "${ACCESS}" --num-facts 30 --seed 42 \
    --dataset "$DATASET_PATH" \
    --memory-file "$MEMORY_FILE" \
    2>&1 | tee "$LOG_DIR/attack_memgpt_${ACCESS}.log"
  echo "=== ${ACCESS} done ==="
done

echo "=== ALL MEMGPT ATTACKS DONE ==="
