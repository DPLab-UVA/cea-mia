#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Run CEA-MI natural attacks with positional arguments.

Usage:
  ./run_all_natural_attacks.sh [api_port] [dataset] [memory_target] [num_facts] [memory_backend] [rounds] [defense]
  ./run_all_natural_attacks.sh [api_server] [api_port] [dataset] [memory_target] [num_facts] [memory_backend] [rounds] [defense]

Positional arguments:
  api_server     Optional vLLM/OpenAI API host, e.g. api-host.example.edu.
  api_port       Optional vLLM/OpenAI API port, e.g. 8001. Empty/default uses CEA_MI_API_BASE as-is.
  dataset        Dataset name, or all for perltqa -> locomo -> msc. Default: perltqa.
  memory_target  nanobot, mem0, or memgpt. Default: nanobot.
  num_facts      N, none, or auto. Default: auto.
  memory_backend light, sdk, or full. Default: light. sdk/full is Mem0/Letta SDK.
  rounds         Number of independent dataset rounds. Default: 1.
  defense        none, system_prompt, or strict_system_prompt. Default: none.

Dataset-dependent num_facts when num_facts=auto:
  perltqa/test -> 20
  locomo  -> none
  msc     -> none

Other settings are controlled by environment variables:
  CEA_MI_SEED, CEA_MI_CONCURRENCY, CEA_MI_TARGET_QUERY_CONCURRENCY,
  CEA_MI_RESPONSE_SCORER,
  CEA_MI_MAX_USERS, CEA_MI_DIRECT_PROBE_K, CEA_MI_SAVE_PROBE_RESPONSES,
  CEA_MI_CANDIDATE_PARAPHRASE, CEA_MI_PARAPHRASE_TEMPERATURE,
  CEA_MI_PARAPHRASE_CONCURRENCY, CEA_MI_PARAPHRASE_CACHE,
  CEA_MI_MEMORY_BACKEND, CEA_MI_MEMORY_STORE_PATH, CEA_MI_MEMGPT_QUERY_MODE,
  CEA_MI_MEMGPT_RECALL_BACKEND,
  CEA_MI_NANOBOT_DB_PATH, CEA_MI_LOG_DIR,
  CEA_MI_RESULTS_DIR, CEA_MI_API_BASE, CEA_MI_API_SERVER/CEA_MI_API_HOST,
  CUDA_VISIBLE_DEVICES.

Note:
  For memgpt sdk, CEA_MI_MEMGPT_QUERY_MODE defaults to readonly. Set it to
  agent to use the full Letta agent runtime; in that mode
  CEA_MI_TARGET_QUERY_CONCURRENCY defaults to 1 because local Letta uses
  SQLite by default and concurrent message writes can lock the database.
  In readonly mode, CEA_MI_MEMGPT_RECALL_BACKEND defaults to letta_native,
  which calls Letta's internal embedding-based archival retrieval path.

Examples:
  CEA_MI_API_BASE=http://127.0.0.1:8000/v1 ./run_all_natural_attacks.sh perltqa nanobot
  ./run_all_natural_attacks.sh api-host.example.edu 8001 all mem0
  ./run_all_natural_attacks.sh api-host.example.edu 8001 perltqa mem0
  CEA_MI_API_BASE=http://127.0.0.1:8001/v1 ./run_all_natural_attacks.sh mem0_dataset mem0 none
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

is_memory_backend_arg() {
  case "${1,,}" in
    light|sdk|full) return 0 ;;
    *) return 1 ;;
  esac
}

is_defense_arg() {
  case "${1,,}" in
    none|off|false|0|system_prompt|system-prompt|prompt|privacy_prompt|strict_system_prompt|strict-system-prompt|strict_prompt|strict-prompt|strong_system_prompt|strong-system-prompt) return 0 ;;
    *) return 1 ;;
  esac
}

is_positive_int() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

normalize_defense() {
  local value="${1,,}"
  value="${value//-/_}"
  case "$value" in
    ""|none|off|false|0) echo "none" ;;
    system_prompt|prompt|privacy_prompt) echo "system_prompt" ;;
    strict_system_prompt|strict_prompt|strong_system_prompt|strong_prompt) echo "strict_system_prompt" ;;
    *) die "defense must be none, system_prompt, or strict_system_prompt" ;;
  esac
}

normalize_memgpt_query_mode() {
  case "${1,,}" in
    ""|readonly|read-only|read_only|backend) echo "readonly" ;;
    agent|runtime|full-agent|full) echo "agent" ;;
    *) die "CEA_MI_MEMGPT_QUERY_MODE must be readonly or agent" ;;
  esac
}

normalize_memgpt_recall_backend() {
  local value="${1,,}"
  value="${value//-/_}"
  case "$value" in
    ""|native|letta_native|letta_semantic|semantic) echo "letta_native" ;;
    text|letta_text|passages|sdk_text) echo "letta_text" ;;
    shadow|shadow_embedding|local_shadow) echo "shadow" ;;
    *) die "CEA_MI_MEMGPT_RECALL_BACKEND must be letta_native, letta_text, or shadow" ;;
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
    die "access is no longer a positional argument; natural_attack runs whitebox once and writes blackbox/graybox/whitebox outputs"
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
NUM_FACTS_OR_BACKEND="$(arg_at "$((ARG_OFFSET + 2))")"
MEMORY_BACKEND="$(arg_at "$((ARG_OFFSET + 3))")"
ROUNDS_ARG="$(arg_at "$((ARG_OFFSET + 4))")"
DEFENSE_ARG="$(arg_at "$((ARG_OFFSET + 5))")"
if [[ -n "$NUM_FACTS_OR_BACKEND" && -z "$MEMORY_BACKEND" ]] && is_memory_backend_arg "$NUM_FACTS_OR_BACKEND"; then
  NUM_FACTS="${CEA_MI_NUM_FACTS:-auto}"
  MEMORY_BACKEND="$NUM_FACTS_OR_BACKEND"
elif [[ -n "$NUM_FACTS_OR_BACKEND" && -n "$MEMORY_BACKEND" ]] \
  && is_memory_backend_arg "$NUM_FACTS_OR_BACKEND" && is_positive_int "$MEMORY_BACKEND"; then
  NUM_FACTS="${CEA_MI_NUM_FACTS:-auto}"
  if [[ -n "$ROUNDS_ARG" && -z "$DEFENSE_ARG" ]]; then
    DEFENSE_ARG="$ROUNDS_ARG"
  fi
  ROUNDS_ARG="$MEMORY_BACKEND"
  MEMORY_BACKEND="$NUM_FACTS_OR_BACKEND"
elif [[ -n "$NUM_FACTS_OR_BACKEND" && -n "$MEMORY_BACKEND" ]] \
  && is_memory_backend_arg "$NUM_FACTS_OR_BACKEND" && is_defense_arg "$MEMORY_BACKEND"; then
  NUM_FACTS="${CEA_MI_NUM_FACTS:-auto}"
  DEFENSE_ARG="$MEMORY_BACKEND"
  ROUNDS_ARG=""
  MEMORY_BACKEND="$NUM_FACTS_OR_BACKEND"
else
  NUM_FACTS="${NUM_FACTS_OR_BACKEND:-${CEA_MI_NUM_FACTS:-auto}}"
fi
if [[ -n "$ROUNDS_ARG" && -z "$DEFENSE_ARG" ]] && is_defense_arg "$ROUNDS_ARG"; then
  DEFENSE_ARG="$ROUNDS_ARG"
  ROUNDS_ARG=""
fi
MEMORY_BACKEND="${MEMORY_BACKEND:-${CEA_MI_MEMORY_BACKEND:-light}}"
ROUNDS="${ROUNDS_ARG:-1}"
DEFENSE="$(normalize_defense "${DEFENSE_ARG:-none}")"

SEED="${CEA_MI_SEED:-42}"
CONCURRENCY="${CEA_MI_CONCURRENCY:-40}"
TARGET_QUERY_CONCURRENCY="${CEA_MI_TARGET_QUERY_CONCURRENCY:-}"
RESPONSE_SCORER="${CEA_MI_RESPONSE_SCORER:-llm}"
DIRECT_PROBE_K="${CEA_MI_DIRECT_PROBE_K:-5}"
CANDIDATE_PARAPHRASE="${CEA_MI_CANDIDATE_PARAPHRASE:-none}"
PARAPHRASE_TEMPERATURE="${CEA_MI_PARAPHRASE_TEMPERATURE:-0.3}"
PARAPHRASE_CONCURRENCY="${CEA_MI_PARAPHRASE_CONCURRENCY:-20}"
PARAPHRASE_CACHE="${CEA_MI_PARAPHRASE_CACHE:-}"
SAVE_PROBE_RESPONSES="${CEA_MI_SAVE_PROBE_RESPONSES:-0}"
MAX_USERS="${CEA_MI_MAX_USERS:-}"
DB_PATH="${CEA_MI_NANOBOT_DB_PATH:-}"
MEMORY_STORE_PATH="${CEA_MI_MEMORY_STORE_PATH:-}"
LOG_DIR="${CEA_MI_LOG_DIR:-logs}"
RESULTS_DIR="${CEA_MI_RESULTS_DIR:-results}"
API_BASE="${CEA_MI_API_BASE:-}"

DATASET_ORDER=(perltqa locomo msc)

normalize_target() {
  case "${1,,}" in
    nanobot|mem0|memgpt) echo "${1,,}" ;;
    *) die "memory_target must be nanobot, mem0, or memgpt" ;;
  esac
}

normalize_memory_backend() {
  case "${1,,}" in
    light) echo "light" ;;
    sdk|full) echo "sdk" ;;
    *) die "memory_backend must be light, sdk, or full" ;;
  esac
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
    die "api_server requires api_port, e.g. ./run_all_natural_attacks.sh api-host.example.edu 8001 perltqa mem0"
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
MEMORY_BACKEND="$(normalize_memory_backend "$MEMORY_BACKEND")"
DATASET_LIST="$(normalize_datasets "$DATASET")"
REQUESTED_NUM_FACTS="$NUM_FACTS"
MEMGPT_QUERY_MODE=""
MEMGPT_RECALL_BACKEND=""

is_positive_int "$ROUNDS" || die "rounds must be a positive integer"
DEFENSE="$(normalize_defense "$DEFENSE")"

if [[ "$MEMORY_TARGET" == "memgpt" && "$MEMORY_BACKEND" == "sdk" ]]; then
  MEMGPT_QUERY_MODE="$(normalize_memgpt_query_mode "${CEA_MI_MEMGPT_QUERY_MODE:-readonly}")"
  MEMGPT_RECALL_BACKEND="$(normalize_memgpt_recall_backend "${CEA_MI_MEMGPT_RECALL_BACKEND:-letta_native}")"
  export CEA_MI_MEMGPT_QUERY_MODE="$MEMGPT_QUERY_MODE"
  export CEA_MI_MEMGPT_RECALL_BACKEND="$MEMGPT_RECALL_BACKEND"
fi

if [[ "$MEMORY_TARGET" == "mem0" && "$MEMORY_BACKEND" == "sdk" ]]; then
  export CEA_MI_MEM0_INFER="${CEA_MI_MEM0_INFER:-false}"
fi

if [[ -z "${CEA_MI_TARGET_QUERY_CONCURRENCY:-}" && "$MEMORY_TARGET" == "memgpt" && "$MEMORY_BACKEND" == "sdk" && "$MEMGPT_QUERY_MODE" == "agent" ]]; then
  TARGET_QUERY_CONCURRENCY=1
fi

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

if [[ "$MEMORY_TARGET" == "memgpt" && "$MEMORY_BACKEND" == "sdk" ]]; then
  EMBEDDING_ENDPOINT="${CEA_MI_LETTA_EMBEDDING_ENDPOINT:-http://127.0.0.1:${CEA_MI_EMBEDDING_PORT:-8290}}"
  if [[ "$EMBEDDING_ENDPOINT" =~ ^http://(127\.0\.0\.1|localhost):([0-9]+)$ ]]; then
    if ! python - "$EMBEDDING_ENDPOINT" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen

endpoint = sys.argv[1].rstrip("/")
with urlopen(f"{endpoint}/health", timeout=2) as response:
    if response.status < 500:
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
      die "memgpt sdk requires local embedding server at ${EMBEDDING_ENDPOINT}; start it with ./memgpt_target/start_embedding_server.sh ${API_SERVER:-$(hostname -s)} ${CEA_MI_EMBEDDING_PORT:-8290}"
    fi
  fi
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

echo "=== CEA-MI natural attack runner ==="
echo "API server:      ${API_SERVER:-<from CEA_MI_API_BASE>}"
echo "Target:          $MEMORY_TARGET"
echo "Memory backend:  $MEMORY_BACKEND"
if [[ -n "$MEMGPT_QUERY_MODE" ]]; then
  echo "MemGPT query:    $MEMGPT_QUERY_MODE"
  echo "MemGPT recall:   $MEMGPT_RECALL_BACKEND"
fi
if [[ "$MEMORY_TARGET" == "mem0" && "$MEMORY_BACKEND" == "sdk" ]]; then
  echo "Mem0 infer:      $CEA_MI_MEM0_INFER"
fi
echo "Dataset request: $DATASET"
echo "Dataset sequence: $DATASET_LIST"
echo "Num facts request: $REQUESTED_NUM_FACTS"
echo "Access outputs:   blackbox graybox whitebox"
echo "API base:        ${CEA_MI_API_BASE:-<required: CEA_MI_API_BASE or api_server/api_port>}"
echo "CUDA devices:    ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Seed:            $SEED"
echo "Rounds:          $ROUNDS"
echo "Defense:         $DEFENSE"
echo "Concurrency:     $CONCURRENCY"
echo "Target queries:  ${TARGET_QUERY_CONCURRENCY:-<unlimited>}"
echo "Direct probe k:  $DIRECT_PROBE_K"
echo "Scorer:          $RESPONSE_SCORER"
echo "Candidate para:  $CANDIDATE_PARAPHRASE"
echo "Results dir:     $RESULTS_DIR"
echo "Memory store:    ${MEMORY_STORE_PATH:-<auto>}"
echo "Log dir:         $LOG_DIR"
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
    --rounds "$ROUNDS"
    --concurrency "$CONCURRENCY"
    --direct-probe-k "$DIRECT_PROBE_K"
    --response-scorer "$RESPONSE_SCORER"
    --memory-backend "$MEMORY_BACKEND"
    --defense "$DEFENSE"
    --candidate-paraphrase "$CANDIDATE_PARAPHRASE"
    --paraphrase-temperature "$PARAPHRASE_TEMPERATURE"
    --paraphrase-concurrency "$PARAPHRASE_CONCURRENCY"
    --output-path "$RESULTS_DIR"
  )

  if [[ -n "$MAX_USERS" ]]; then
    COMMON_ARGS+=(--max-users "$MAX_USERS")
  fi

  if [[ -n "$TARGET_QUERY_CONCURRENCY" ]]; then
    COMMON_ARGS+=(--target-query-concurrency "$TARGET_QUERY_CONCURRENCY")
  fi

  case "${SAVE_PROBE_RESPONSES,,}" in
    1|true|yes|y|on) COMMON_ARGS+=(--save-probe-responses) ;;
  esac

  if [[ -n "$PARAPHRASE_CACHE" ]]; then
    COMMON_ARGS+=(--paraphrase-cache "$PARAPHRASE_CACHE")
  fi

  if [[ -n "$MEMORY_STORE_PATH" ]]; then
    COMMON_ARGS+=(--memory-store-path "$MEMORY_STORE_PATH")
  fi

  if [[ "$MEMORY_TARGET" == "nanobot" && -n "$DB_PATH" ]]; then
    COMMON_ARGS+=(--db "$DB_PATH")
  elif [[ "$MEMORY_TARGET" != "nanobot" && -n "$DB_PATH" ]]; then
    echo "Note: CEA_MI_NANOBOT_DB_PATH is ignored for ${MEMORY_TARGET}." >&2
  fi

  PARAPHRASE_SUFFIX=""
  if [[ "$CANDIDATE_PARAPHRASE" != "none" ]]; then
    PARAPHRASE_SUFFIX="_candidate-${CANDIDATE_PARAPHRASE}"
  fi
  ROUND_SUFFIX=""
  if [[ "$ROUNDS" != "1" ]]; then
    ROUND_SUFFIX="_r${ROUNDS}"
  fi
  DEFENSE_SUFFIX=""
  if [[ "$DEFENSE" != "none" ]]; then
    DEFENSE_SUFFIX="_def-${DEFENSE}"
  fi
  BACKEND_SUFFIX=""
  if [[ "$MEMORY_TARGET" != "nanobot" && "$MEMORY_BACKEND" != "light" ]]; then
    if [[ -n "$MEMGPT_QUERY_MODE" ]]; then
      BACKEND_SUFFIX="_${MEMORY_BACKEND}-${MEMGPT_QUERY_MODE}"
    else
      BACKEND_SUFFIX="_${MEMORY_BACKEND}"
    fi
  fi
  OUTPUT_NAME="${DATASET_LABEL}_${MEMORY_TARGET}_${RESPONSE_SCORER}${BACKEND_SUFFIX}_k${DIRECT_PROBE_K}${PARAPHRASE_SUFFIX}_seed${SEED}${DEFENSE_SUFFIX}${ROUND_SUFFIX}"
  OUTPUT_PATH="$RESULTS_DIR/$OUTPUT_NAME"
  LOG_PATH="$LOG_DIR/${OUTPUT_NAME}${PORT_LABEL}.log"

  echo "=== Starting dataset=${DATASET_ITEM} ${MEMORY_TARGET} natural attack ==="
  echo "Num facts: ${NUM_FACTS_FOR_DATASET}"
  echo "Backend:   ${MEMORY_BACKEND}"
  echo "Defense:   ${DEFENSE}"
  echo "Output:    ${OUTPUT_PATH}"
  echo "Log:       ${LOG_PATH}"

  python natural_attack.py \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_PATH"

  echo "=== dataset=${DATASET_ITEM} ${MEMORY_TARGET} natural attack done ==="
done

echo "=== All requested natural attacks done ==="
