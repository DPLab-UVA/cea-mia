#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SERVER_NAME="${1:-$(hostname -s)}"
PORT="${2:-${CEA_MI_EMBEDDING_PORT:-8290}}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Error: port must be numeric, got: $PORT" >&2
  exit 1
fi

SAFE_SERVER="${SERVER_NAME//[^A-Za-z0-9_.-]/_}"
export CEA_MI_EMBEDDING_HOST="${CEA_MI_EMBEDDING_HOST:-0.0.0.0}"
export CEA_MI_EMBEDDING_PORT="$PORT"
export CEA_MI_EMBEDDING_MODEL="${CEA_MI_EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"
export CEA_MI_EMBEDDING_NORMALIZE="${CEA_MI_EMBEDDING_NORMALIZE:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

PYTHON_BIN="${EMBEDDING_PYTHON:-/p/pkq2psproject/envs/agent/bin/python}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

LOG_PATH="$LOG_DIR/embedding_server_${SAFE_SERVER}_port${PORT}.log"
PID_PATH="$LOG_DIR/embedding_server_${SAFE_SERVER}_port${PORT}.pid"

if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
  echo "Embedding server already appears to be running: pid=$(cat "$PID_PATH")"
  echo "PID file: $PID_PATH"
  echo "Log file: $LOG_PATH"
  exit 0
fi

echo "Starting local embedding server"
echo "  server_name: $SERVER_NAME"
echo "  host:        $CEA_MI_EMBEDDING_HOST"
echo "  port:        $CEA_MI_EMBEDDING_PORT"
echo "  model:       $CEA_MI_EMBEDDING_MODEL"
echo "  python:      $PYTHON_BIN"
echo "  log:         $LOG_PATH"

rm -f "$LOG_PATH"
nohup env \
  CEA_MI_EMBEDDING_HOST="$CEA_MI_EMBEDDING_HOST" \
  CEA_MI_EMBEDDING_PORT="$CEA_MI_EMBEDDING_PORT" \
  CEA_MI_EMBEDDING_MODEL="$CEA_MI_EMBEDDING_MODEL" \
  CEA_MI_EMBEDDING_NORMALIZE="$CEA_MI_EMBEDDING_NORMALIZE" \
  TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  RAYON_NUM_THREADS="$RAYON_NUM_THREADS" \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
  NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  "$PYTHON_BIN" memgpt_target/local_embedding_server.py \
  --host "$CEA_MI_EMBEDDING_HOST" \
  --port "$CEA_MI_EMBEDDING_PORT" \
  > "$LOG_PATH" 2>&1 &

echo $! > "$PID_PATH"
echo "Started embedding server pid=$(cat "$PID_PATH")"

STARTUP_TIMEOUT="${EMBEDDING_STARTUP_TIMEOUT:-300}"
READY=0
for _ in $(seq 1 "$STARTUP_TIMEOUT"); do
  if ! kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "Embedding server exited during startup. Last log lines:" >&2
    tail -80 "$LOG_PATH" >&2 || true
    exit 1
  fi

  if "$PYTHON_BIN" - "$CEA_MI_EMBEDDING_PORT" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen

port = sys.argv[1]
with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
    if response.status < 500:
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    READY=1
    break
  fi

  sleep 1
done

if [[ "$READY" != "1" ]]; then
  echo "Embedding server did not become healthy within ${STARTUP_TIMEOUT} seconds. Last log lines:" >&2
  tail -80 "$LOG_PATH" >&2 || true
  exit 1
fi

echo "Embedding server is healthy: http://127.0.0.1:$CEA_MI_EMBEDDING_PORT/health"
echo "Tail logs with: tail -f $LOG_PATH"
