#!/usr/bin/env bash
set -euo pipefail

# Baseline attack helper for the three supported targets.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

: "${CEA_MI_API_BASE:?Set CEA_MI_API_BASE to your OpenAI-compatible /v1 endpoint}"
: "${CEA_MI_MODEL:?Set CEA_MI_MODEL to the model name/path served by that endpoint}"

DATASET="${CEA_MI_DATASET:-perltqa}"
NUM_FACTS="${CEA_MI_NUM_FACTS:-20}"
SEED="${CEA_MI_SEED:-42}"
CONCURRENCY="${CEA_MI_CONCURRENCY:-40}"
LOG_DIR="${CEA_MI_LOG_DIR:-logs}"
NANOBOT_DB="${CEA_MI_NANOBOT_DB_PATH:-$HOME/.nanobot/memory/pmc.db}"

mkdir -p "$LOG_DIR"

for TARGET in memgpt mem0; do
    if [[ "$TARGET" == "memgpt" ]]; then
        MEMORY_FILE="${CEA_MI_MEMGPT_MEMORY_FILE:-memgpt_memories.json}"
    else
        MEMORY_FILE="${CEA_MI_MEM0_MEMORY_FILE:-mem0_memories.json}"
    fi

    for BASELINE in naive mink reference; do
        echo "=== ${TARGET} baseline=${BASELINE} ==="
        python3 baselines/baseline_attacks.py \
            --target "$TARGET" \
            --dataset "$DATASET" \
            --num-facts "$NUM_FACTS" \
            --seed "$SEED" \
            --concurrency "$CONCURRENCY" \
            --memory-file "$MEMORY_FILE" \
            --baseline "$BASELINE" \
            2>&1 | tee "$LOG_DIR/baselines_${TARGET}_${BASELINE}.log"
    done
done

for BASELINE in naive mink reference; do
    echo "=== nanobot baseline=${BASELINE} ==="
    python3 baselines/baseline_attacks.py \
        --target nanobot \
        --dataset "$DATASET" \
        --num-facts "$NUM_FACTS" \
        --seed "$SEED" \
        --concurrency "$CONCURRENCY" \
        --db "$NANOBOT_DB" \
        --baseline "$BASELINE" \
        2>&1 | tee "$LOG_DIR/baselines_nanobot_${BASELINE}.log"
done

echo "=== ALL BASELINES DONE ==="
