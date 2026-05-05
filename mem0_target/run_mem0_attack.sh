#!/usr/bin/env bash
set -euo pipefail

# CEA-MI attack against the lightweight Mem0-style embedding memory agent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

: "${CEA_MI_API_BASE:?Set CEA_MI_API_BASE to your OpenAI-compatible /v1 endpoint}"
: "${CEA_MI_MODEL:?Set CEA_MI_MODEL to the model name/path served by that endpoint}"

DATASET="${CEA_MI_DATASET:-data/perltqa_seed42.json}"
NUM_FACTS="${CEA_MI_NUM_FACTS:-30}"
SEED="${CEA_MI_SEED:-42}"
LOG_DIR="${CEA_MI_LOG_DIR:-logs}"
MEMORY_FILE="${CEA_MI_MEMORY_FILE:-mem0_memories.json}"

mkdir -p "$LOG_DIR"

echo "=== Setting up Mem0 memory agent ==="
python3 mem0_target/setup_mem0.py standalone --dataset "$DATASET" --memory-file "$MEMORY_FILE"
echo "=== Setup done ==="

for ACCESS in blackbox graybox whitebox; do
    echo "=== Starting Mem0 ${ACCESS} ==="
    python3 mem0_target/mem0_attack.py \
        --dataset "$DATASET" \
        --access "$ACCESS" \
        --num-facts "$NUM_FACTS" \
        --seed "$SEED" \
        --memory-file "$MEMORY_FILE" \
        2>&1 | tee "$LOG_DIR/attack_mem0_${ACCESS}.log"
    echo "=== ${ACCESS} done ==="
done

echo "=== ALL MEM0 ATTACKS DONE ==="
