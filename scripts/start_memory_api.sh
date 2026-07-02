#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" && -x /opt/conda/bin/conda ]]; then
  CONDA_BIN=/opt/conda/bin/conda
fi
if [[ -z "$CONDA_BIN" ]]; then
  echo "conda not found; set CONDA_EXE or add conda to PATH" >&2
  exit 1
fi
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "${CONDA_ENV:-${CONDA_ENV_NAME:-memory}}"

mkdir -p runtime data/sessions
PID_FILE=runtime/memory_api.pid
LOG_FILE=runtime/memory_api.log
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MEMORY_ROOT="${MEMORY_ROOT:-$ROOT}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-vllm}"
export EMBEDDING_SERVICE_URL="${EMBEDDING_SERVICE_URL:-http://127.0.0.1:18083/v1}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"
export EMBEDDING_TIMEOUT_SECONDS="${EMBEDDING_TIMEOUT_SECONDS:-60}"

HOST="$(python -c 'from memory_system.config import settings; print(settings.memory_api_host)')"
PORT="$(python -c 'from memory_system.config import settings; print(settings.memory_api_port)')"

if curl -fsS "http://${HOST}:${PORT}/openapi.json" 2>/dev/null | grep -q '"title":"Memory Wiki API"'; then
  echo "Memory API already available at http://${HOST}:${PORT}"
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Memory API already running: PID $(cat "$PID_FILE")"
  exit 0
fi

nohup python -m memory_system.api >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
sleep 1
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Memory API failed to start on http://${HOST}:${PORT}. Log follows:"
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi
echo "Memory API started at http://${HOST}:${PORT}: PID $(cat "$PID_FILE"), log $LOG_FILE"
