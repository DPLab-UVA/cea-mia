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
MEMORY_BACKEND="${CEA_MI_MEMORY_BACKEND:-light}"
MEMORY_BACKEND="${MEMORY_BACKEND,,}"
if [[ "$MEMORY_BACKEND" == "full" ]]; then
    MEMORY_BACKEND="sdk"
fi
if [[ "$MEMORY_BACKEND" != "light" && "$MEMORY_BACKEND" != "sdk" ]]; then
    echo "Error: CEA_MI_MEMORY_BACKEND must be light, sdk, or full" >&2
    exit 1
fi

normalize_memgpt_query_mode() {
    case "${1,,}" in
        ""|readonly|read-only|read_only|backend) echo "readonly" ;;
        agent|runtime|full-agent|full) echo "agent" ;;
        *)
            echo "Error: CEA_MI_MEMGPT_QUERY_MODE must be readonly or agent" >&2
            exit 1
            ;;
    esac
}

mkdir -p "$LOG_DIR"

for TARGET in memgpt mem0; do
    if [[ "$TARGET" == "memgpt" ]]; then
        MEMORY_FILE="${CEA_MI_MEMGPT_MEMORY_FILE:-memgpt_memories.json}"
        MEMORY_STORE_PATH="${CEA_MI_MEMGPT_MEMORY_STORE_PATH:-${CEA_MI_MEMORY_STORE_PATH:-}}"
        MEMGPT_QUERY_MODE=""
        if [[ "$MEMORY_BACKEND" == "sdk" ]]; then
            MEMGPT_QUERY_MODE="$(normalize_memgpt_query_mode "${CEA_MI_MEMGPT_QUERY_MODE:-readonly}")"
            export CEA_MI_MEMGPT_QUERY_MODE="$MEMGPT_QUERY_MODE"
        fi
    else
        MEMORY_FILE="${CEA_MI_MEM0_MEMORY_FILE:-mem0_memories.json}"
        MEMORY_STORE_PATH="${CEA_MI_MEM0_MEMORY_STORE_PATH:-${CEA_MI_MEMORY_STORE_PATH:-}}"
        MEMGPT_QUERY_MODE=""
    fi

    for BASELINE in naive mink reference; do
        QUERY_LABEL=""
        if [[ -n "$MEMGPT_QUERY_MODE" ]]; then
            QUERY_LABEL="_${MEMGPT_QUERY_MODE}"
        fi
        echo "=== ${TARGET} baseline=${BASELINE} backend=${MEMORY_BACKEND}${QUERY_LABEL} ==="
        ARGS=(
            --target "$TARGET"
            --dataset "$DATASET"
            --num-facts "$NUM_FACTS"
            --seed "$SEED"
            --concurrency "$CONCURRENCY"
            --memory-backend "$MEMORY_BACKEND"
            --baseline "$BASELINE"
        )
        if [[ "$MEMORY_BACKEND" == "light" ]]; then
            ARGS+=(--memory-file "$MEMORY_FILE")
        elif [[ -n "$MEMORY_STORE_PATH" ]]; then
            ARGS+=(--memory-store-path "$MEMORY_STORE_PATH")
        fi
        python3 baselines/baseline_attacks.py "${ARGS[@]}" \
            2>&1 | tee "$LOG_DIR/baselines_${TARGET}_${MEMORY_BACKEND}${QUERY_LABEL}_${BASELINE}.log"
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
