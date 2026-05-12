#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Start a local Letta server for CEA-MI MemGPT SDK experiments.

Usage:
  memgpt_target/start_letta_server.sh [server_name] [port]

Arguments:
  server_name  Node/server label used to build LETTA_DIR and log names.
               Default: hostname -s
  port         Letta HTTP port. Default: 8283

Environment overrides:
  CEA_MI_LETTA_BASE_DIR  Base directory for Letta state.
                         Default: results/letta_home
  COMPOSIO_CACHE_DIR     Writable cache directory for Letta's composio import.
                         Default: ${LETTA_LETTA_DIR}_composio
  OPENLLM_AUTH_TYPE      Auth type Letta uses for local/vLLM completions.
                         Default: bearer_token
  OPENLLM_API_KEY        API key Letta sends to vLLM. Default: CEA_MI_API_KEY.
  LETTA_PYTHON           Python executable for the agent environment.
                         Default: /p/pkq2psproject/envs/agent/bin/python

Example:
  memgpt_target/start_letta_server.sh dplab05 8283
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

SERVER_NAME="${1:-$(hostname -s)}"
PORT="${2:-${LETTA_PORT:-8283}}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Error: port must be numeric, got: $PORT" >&2
  exit 1
fi

SAFE_SERVER="${SERVER_NAME//[^A-Za-z0-9_.-]/_}"
BASE_DIR="${CEA_MI_LETTA_BASE_DIR:-$REPO_ROOT/results/letta_home}"
export LETTA_DIR="${BASE_DIR}_${SAFE_SERVER}_port${PORT}"
export LETTA_LETTA_DIR="${LETTA_LETTA_DIR:-$LETTA_DIR}"
export LETTA_HOME="${LETTA_HOME:-$LETTA_LETTA_DIR}"
export COMPOSIO_CACHE_DIR="${COMPOSIO_CACHE_DIR:-${LETTA_LETTA_DIR}_composio}"
export LETTA_HOST="${LETTA_HOST:-0.0.0.0}"
export LETTA_PORT="$PORT"
export OPENLLM_AUTH_TYPE="${OPENLLM_AUTH_TYPE:-bearer_token}"
export OPENLLM_API_KEY="${OPENLLM_API_KEY:-${CEA_MI_API_KEY:-token-vllm}}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

PYTHON_BIN="${LETTA_PYTHON:-/p/pkq2psproject/envs/agent/bin/python}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR" "$LETTA_DIR" "$LETTA_LETTA_DIR" "$LETTA_HOME" "$COMPOSIO_CACHE_DIR"

LOG_PATH="$LOG_DIR/letta_server_${SAFE_SERVER}_port${PORT}.log"
PID_PATH="$LOG_DIR/letta_server_${SAFE_SERVER}_port${PORT}.pid"
ENV_PATH="$LOG_DIR/letta_server_${SAFE_SERVER}_port${PORT}.env"

write_env_file() {
  cat > "$ENV_PATH" <<EOF
export LETTA_BASE_URL=http://127.0.0.1:${PORT}
export LETTA_PORT=${PORT}
export LETTA_DIR=${LETTA_DIR}
export LETTA_LETTA_DIR=${LETTA_LETTA_DIR}
export LETTA_HOME=${LETTA_HOME}
export HOME=${LETTA_HOME}
export COMPOSIO_CACHE_DIR=${COMPOSIO_CACHE_DIR}
export CEA_MI_LETTA_SERVER_NAME=${SAFE_SERVER}
EOF
}

if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
  write_env_file
  echo "Letta server already appears to be running: pid=$(cat "$PID_PATH")"
  echo "PID file: $PID_PATH"
  echo "Log file: $LOG_PATH"
  echo "Experiment env: source $ENV_PATH"
  exit 0
fi

echo "Starting Letta server"
echo "  server_name: $SERVER_NAME"
echo "  host:        $LETTA_HOST"
echo "  port:        $LETTA_PORT"
echo "  LETTA_DIR:   $LETTA_DIR"
echo "  LETTA_LETTA_DIR: $LETTA_LETTA_DIR"
echo "  HOME:        $LETTA_HOME"
echo "  COMPOSIO_CACHE_DIR: $COMPOSIO_CACHE_DIR"
echo "  OPENLLM_AUTH_TYPE:  $OPENLLM_AUTH_TYPE"
echo "  python:      $PYTHON_BIN"
echo "  log:         $LOG_PATH"

write_env_file

rm -f "$LOG_PATH"
nohup env \
  HOME="$LETTA_HOME" \
  LETTA_HOME="$LETTA_HOME" \
  LETTA_DIR="$LETTA_DIR" \
  LETTA_LETTA_DIR="$LETTA_LETTA_DIR" \
  COMPOSIO_CACHE_DIR="$COMPOSIO_CACHE_DIR" \
  LETTA_HOST="$LETTA_HOST" \
  LETTA_PORT="$LETTA_PORT" \
  OPENLLM_AUTH_TYPE="$OPENLLM_AUTH_TYPE" \
  OPENLLM_API_KEY="$OPENLLM_API_KEY" \
  TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  RAYON_NUM_THREADS="$RAYON_NUM_THREADS" \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
  NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  "$PYTHON_BIN" memgpt_target/setup_memgpt.py serve \
  > "$LOG_PATH" 2>&1 &

echo $! > "$PID_PATH"
echo "Started Letta server pid=$(cat "$PID_PATH")"

STARTUP_TIMEOUT="${LETTA_STARTUP_TIMEOUT:-120}"
READY=0
for _ in $(seq 1 "$STARTUP_TIMEOUT"); do
  if ! kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "Letta server exited during startup. Last log lines:" >&2
    tail -80 "$LOG_PATH" >&2 || true
    exit 1
  fi

  if grep -q "Uvicorn running on" "$LOG_PATH" 2>/dev/null; then
    READY=1
    break
  fi

  if "$PYTHON_BIN" - "$LETTA_PORT" <<'PY' >/dev/null 2>&1
import sys
import socket

port = sys.argv[1]
with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
    raise SystemExit(0)
PY
  then
    READY=1
    break
  fi

  sleep 1
done

if [[ "$READY" != "1" ]]; then
  echo "Letta server did not become reachable within ${STARTUP_TIMEOUT} seconds. Last log lines:" >&2
  tail -80 "$LOG_PATH" >&2 || true
  exit 1
fi

echo "Letta server is healthy: http://127.0.0.1:$LETTA_PORT/v1/health/"
echo "Experiment env: source $ENV_PATH"
echo "Tail logs with: tail -f $LOG_PATH"
