#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Run baseline membership-inference attacks with positional arguments.

Usage:
  ./run_baseline_attacks.sh [api_port] [dataset] [memory_target] [baseline_name] [num_facts]
  ./run_baseline_attacks.sh [api_server] [api_port] [dataset] [memory_target] [baseline_name] [num_facts]

Positional arguments:
  api_server     Optional vLLM/OpenAI API host, e.g. api-host.example.edu.
  api_port       Optional vLLM/OpenAI API port, e.g. 8001. Empty/default uses CEA_MI_API_BASE as-is.
  dataset        Dataset name, or all for perltqa -> locomo -> msc. Default: perltqa.
  memory_target  nanobot, mem0, or memgpt. Default: nanobot.
  baseline_name  naive, loss, mink, reference, multi_contrastive,
                 multi_no_contrastive, multi_direct_no_contrastive,
                 multi_recall_no_reason, multi_judge, or all. Default: all.
  num_facts      N, none, or auto. Default: auto.

Dataset-dependent num_facts when num_facts=auto:
  perltqa/test -> 20
  locomo  -> none
  msc     -> none

Other settings are controlled by environment variables:
  CEA_MI_SEED, CEA_MI_CONCURRENCY, CEA_MI_RESPONSE_SCORER,
  CEA_MI_MAX_USERS, CEA_MI_DIRECT_PROBE_K, CEA_MI_SAVE_PROBE_RESPONSES,
  CEA_MI_NANOBOT_DB_PATH, CEA_MI_LOG_DIR, CEA_MI_RESULTS_DIR,
  CEA_MI_API_BASE, CEA_MI_API_SERVER/CEA_MI_API_HOST, CUDA_VISIBLE_DEVICES.

Examples:
  CEA_MI_API_BASE=http://127.0.0.1:8000/v1 ./run_baseline_attacks.sh perltqa nanobot all
  ./run_baseline_attacks.sh api-host.example.edu 8001 perltqa mem0 mink
  CEA_MI_API_BASE=http://127.0.0.1:8001/v1 ./run_baseline_attacks.sh perltqa mem0 mink
  CEA_MI_API_BASE=http://127.0.0.1:8002/v1 ./run_baseline_attacks.sh msc memgpt reference none
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

is_port_arg() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

is_access_arg() {
  case "${1,,}" in
    black|blackbox|gray|graybox|white|whitebox) return 0 ;;
    *) return 1 ;;
  esac
}

is_target_arg() {
  case "${1,,}" in
    nanobot|mem0|memgpt) return 0 ;;
    *) return 1 ;;
  esac
}

is_dataset_arg() {
  case "${1,,}" in
    all|perltqa|locomo|msc|test|*.json|*/*) return 0 ;;
    *) return 1 ;;
  esac
}

arg_at() {
  local idx="$1"
  echo "${ARGS[$idx]:-}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ARGS=("$@")
API_SERVER="${CEA_MI_API_SERVER:-${CEA_MI_API_HOST:-}}"
API_PORT="${CEA_MI_API_PORT:-}"
ARG_OFFSET=0

if [[ $# -gt 0 ]]; then
  if [[ -z "${ARGS[0]}" ]]; then
    ARG_OFFSET=1
  elif is_port_arg "${ARGS[0]}"; then
    API_PORT="${ARGS[0]}"
    ARG_OFFSET=1
  elif is_access_arg "${ARGS[0]}"; then
    die "access is no longer a positional argument; baselines choose their own run access"
  elif is_dataset_arg "${ARGS[0]}" || { [[ -n "${ARGS[1]:-}" ]] && is_target_arg "${ARGS[1]}"; }; then
    ARG_OFFSET=0
  else
    API_SERVER="${ARGS[0]}"
    if [[ -n "${ARGS[1]:-}" ]] && is_port_arg "${ARGS[1]}"; then
      API_PORT="${ARGS[1]}"
      ARG_OFFSET=2
    else
      ARG_OFFSET=1
    fi
  fi
fi

if [[ -n "$(arg_at "$ARG_OFFSET")" ]] && is_access_arg "$(arg_at "$ARG_OFFSET")"; then
  die "access is no longer accepted; remove the blackbox/graybox/whitebox argument"
fi

DATASET="$(arg_at "$ARG_OFFSET")"
DATASET="${DATASET:-${CEA_MI_DATASET:-perltqa}}"
MEMORY_TARGET="$(arg_at "$((ARG_OFFSET + 1))")"
MEMORY_TARGET="${MEMORY_TARGET:-${CEA_MI_MEMORY_TARGET:-nanobot}}"
BASELINE_NAME="$(arg_at "$((ARG_OFFSET + 2))")"
BASELINE_NAME="${BASELINE_NAME:-${CEA_MI_BASELINE_NAME:-all}}"
NUM_FACTS="$(arg_at "$((ARG_OFFSET + 3))")"
NUM_FACTS="${NUM_FACTS:-${CEA_MI_NUM_FACTS:-auto}}"

SEED="${CEA_MI_SEED:-42}"
CONCURRENCY="${CEA_MI_CONCURRENCY:-40}"
RESPONSE_SCORER="${CEA_MI_RESPONSE_SCORER:-llm}"
SAVE_PROBE_RESPONSES="${CEA_MI_SAVE_PROBE_RESPONSES:-0}"
MAX_USERS="${CEA_MI_MAX_USERS:-}"
DB_PATH="${CEA_MI_NANOBOT_DB_PATH:-}"
LOG_DIR="${CEA_MI_LOG_DIR:-logs}"
RESULTS_DIR="${CEA_MI_RESULTS_DIR:-results}"
API_BASE="${CEA_MI_API_BASE:-}"

DIRECT_PROBE_K="${CEA_MI_DIRECT_PROBE_K:-5}"

BASELINE_ORDER=(naive loss mink reference multi_contrastive multi_no_contrastive multi_direct_no_contrastive multi_recall_no_reason multi_judge)
DATASET_ORDER=(perltqa locomo msc)

normalize_target() {
  case "${1,,}" in
    nanobot|mem0|memgpt) echo "${1,,}" ;;
    *) die "memory_target must be nanobot, mem0, or memgpt" ;;
  esac
}

normalize_baselines() {
  local requested="${1,,}"
  local names=()

  if [[ "$requested" == "all" ]]; then
    names=("${BASELINE_ORDER[@]}")
  else
    IFS=',' read -ra names <<< "$requested"
  fi

  local normalized=()
  local name
  for name in "${names[@]}"; do
    name="${name//[[:space:]]/}"
    case "$name" in
      naive|loss|mink|reference|multi_contrastive|multi_no_contrastive|multi_direct_no_contrastive|multi_recall_no_reason|multi_judge)
        normalized+=("$name")
        ;;
      *)
        die "baseline_name must be naive, loss, mink, reference, multi_contrastive, multi_no_contrastive, multi_direct_no_contrastive, multi_recall_no_reason, multi_judge, or all"
        ;;
    esac
  done

  echo "${normalized[*]}"
}

default_num_facts_for_dataset() {
  local ds
  ds="$(basename "$1")"
  ds="${ds,,}"
  case "$ds" in
    perltqa*|test*) echo "20" ;;
    locomo*|msc*) echo "none" ;;
    *) echo "none" ;;
  esac
}

normalize_datasets() {
  local requested="${1,,}"
  local names=()

  if [[ "$requested" == "all" ]]; then
    names=("${DATASET_ORDER[@]}")
  else
    IFS=',' read -ra names <<< "$1"
  fi

  local normalized=()
  local name
  for name in "${names[@]}"; do
    name="${name//[[:space:]]/}"
    [[ -n "$name" ]] && normalized+=("$name")
  done

  [[ "${#normalized[@]}" -gt 0 ]] || die "dataset must not be empty"
  echo "${normalized[*]}"
}

replace_api_port() {
  local base="$1"
  local port="$2"

  if [[ -z "$base" ]]; then
    die "api_port requires CEA_MI_API_BASE, or pass both api_server and api_port"
  fi

  if [[ "$base" =~ ^(https?://[^/:]+)(:[0-9]+)?(/.*)?$ ]]; then
    local path="${BASH_REMATCH[3]}"
    if [[ -z "$path" ]]; then
      path="/v1"
    fi
    echo "${BASH_REMATCH[1]}:${port}${path}"
    return
  fi

  die "cannot replace port in API base URL: $base"
}

compose_api_base() {
  local server="$1"
  local port="$2"
  local prefix

  if [[ -z "$port" ]]; then
    die "api_server requires api_port, e.g. ./run_baseline_attacks.sh api-host.example.edu 8001 perltqa mem0"
  fi

  if [[ "$server" =~ ^(https?://[^/:/]+)(:[0-9]+)?(/.*)?$ ]]; then
    prefix="${BASH_REMATCH[1]}"
  else
    server="${server#/}"
    server="${server%/}"
    prefix="http://${server}"
  fi

  echo "${prefix}:${port}/v1"
}

MEMORY_TARGET="$(normalize_target "$MEMORY_TARGET")"
BASELINE_LIST="$(normalize_baselines "$BASELINE_NAME")"
DATASET_LIST="$(normalize_datasets "$DATASET")"
REQUESTED_NUM_FACTS="$NUM_FACTS"

if [[ -n "$API_SERVER" ]]; then
  API_BASE="$(compose_api_base "$API_SERVER" "$API_PORT")"
elif [[ -n "$API_PORT" ]]; then
  API_BASE="$(replace_api_port "$API_BASE" "$API_PORT")"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$API_PORT" =~ ^800([0-9])$ ]]; then
  export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
fi

if [[ -n "$API_BASE" ]]; then
  export CEA_MI_API_BASE="$API_BASE"
fi

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

PORT_LABEL=""
if [[ -n "$API_SERVER" && -n "$API_PORT" ]]; then
  SERVER_LABEL="${API_SERVER#http://}"
  SERVER_LABEL="${SERVER_LABEL#https://}"
  SERVER_LABEL="${SERVER_LABEL%%/*}"
  SERVER_LABEL="${SERVER_LABEL//[^A-Za-z0-9_.-]/_}"
  PORT_LABEL="_${SERVER_LABEL}_port${API_PORT}"
elif [[ -n "$API_PORT" ]]; then
  PORT_LABEL="_port${API_PORT}"
fi

echo "=== CEA-MI baseline attack runner ==="
echo "API server:        ${API_SERVER:-<from CEA_MI_API_BASE>}"
echo "Target:            $MEMORY_TARGET"
echo "Dataset request:   $DATASET"
echo "Dataset sequence:  $DATASET_LIST"
echo "Num facts request: $REQUESTED_NUM_FACTS"
echo "Access policy:     loss/mink/reference=graybox; others=whitebox with derived blackbox/graybox/whitebox outputs"
echo "Baseline request:  $BASELINE_NAME"
echo "Baseline sequence: $BASELINE_LIST"
echo "API base:          ${CEA_MI_API_BASE:-<required: CEA_MI_API_BASE or api_server/api_port>}"
echo "CUDA devices:      ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Seed:              $SEED"
echo "Concurrency:       $CONCURRENCY"
echo "Direct probe k:    $DIRECT_PROBE_K"
echo "Scorer:            $RESPONSE_SCORER"
echo "Save responses:    $SAVE_PROBE_RESPONSES"
echo "Results dir:       $RESULTS_DIR"
echo "Log dir:           $LOG_DIR"
echo

for DATASET_ITEM in $DATASET_LIST; do
  DATASET_LABEL="$(basename "$DATASET_ITEM")"
  DATASET_LABEL="${DATASET_LABEL%.*}"
  NUM_FACTS_FOR_DATASET="$REQUESTED_NUM_FACTS"
  if [[ "${REQUESTED_NUM_FACTS,,}" == "auto" ]]; then
    NUM_FACTS_FOR_DATASET="$(default_num_facts_for_dataset "$DATASET_ITEM")"
  fi

  COMMON_ARGS=(
    --target "$MEMORY_TARGET"
    --dataset "$DATASET_ITEM"
    --num-facts "$NUM_FACTS_FOR_DATASET"
    --seed "$SEED"
    --concurrency "$CONCURRENCY"
    --direct-probe-k "$DIRECT_PROBE_K"
    --response-scorer "$RESPONSE_SCORER"
  )

  if [[ -n "$MAX_USERS" ]]; then
    COMMON_ARGS+=(--max-users "$MAX_USERS")
  fi

  if [[ "$MEMORY_TARGET" == "nanobot" && -n "$DB_PATH" ]]; then
    COMMON_ARGS+=(--db "$DB_PATH")
  elif [[ "$MEMORY_TARGET" != "nanobot" && -n "$DB_PATH" ]]; then
    echo "Note: CEA_MI_NANOBOT_DB_PATH is ignored for ${MEMORY_TARGET}." >&2
  fi

  for BASELINE in $BASELINE_LIST; do
    SCORER_SUFFIX=""
    if [[ "$BASELINE" == "multi_contrastive" || "$BASELINE" == "multi_no_contrastive" || "$BASELINE" == "multi_direct_no_contrastive" || "$BASELINE" == "multi_recall_no_reason" ]]; then
      SCORER_SUFFIX="_${RESPONSE_SCORER}"
    fi
    PROBE_SUFFIX=""
    if [[ "$BASELINE" == "multi_direct_no_contrastive" || "$BASELINE" == "multi_recall_no_reason" || "$BASELINE" == "multi_judge" ]]; then
      PROBE_SUFFIX="_k${DIRECT_PROBE_K}"
    fi

    OUTPUT_NAME="${DATASET_LABEL}_${MEMORY_TARGET}_baseline_${BASELINE}${SCORER_SUFFIX}${PROBE_SUFFIX}_seed${SEED}"
    OUTPUT_PATH="$RESULTS_DIR/$OUTPUT_NAME"
    LOG_PATH="$LOG_DIR/${OUTPUT_NAME}${PORT_LABEL}.log"
    RUN_ARGS=("${COMMON_ARGS[@]}")

    if [[ "$MEMORY_TARGET" != "nanobot" ]]; then
      mkdir -p "$OUTPUT_PATH"
      RUN_ARGS+=(--memory-file "$OUTPUT_PATH/${MEMORY_TARGET}_baseline_working_memories.json")
    fi

    case "${SAVE_PROBE_RESPONSES,,}" in
      1|true|yes|y|on) RUN_ARGS+=(--save-probe-responses) ;;
    esac

    echo "=== Starting dataset=${DATASET_ITEM} ${MEMORY_TARGET} baseline=${BASELINE} ==="
    echo "Num facts: ${NUM_FACTS_FOR_DATASET}"
    echo "Output:    ${OUTPUT_PATH}"
    echo "Log:       ${LOG_PATH}"

    python baselines/baseline_attacks.py \
      --baseline "$BASELINE" \
      --output-path "$OUTPUT_PATH" \
      "${RUN_ARGS[@]}" \
      2>&1 | tee "$LOG_PATH"

    echo "=== dataset=${DATASET_ITEM} ${MEMORY_TARGET} baseline=${BASELINE} done ==="
  done
done

echo "=== All requested baseline attacks done ==="
