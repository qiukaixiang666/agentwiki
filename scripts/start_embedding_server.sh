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

mkdir -p runtime data
PID_FILE=runtime/embedding_server.pid
LOG_FILE=runtime/embedding_server.log
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MEMORY_ROOT="${MEMORY_ROOT:-$ROOT}"
export BGE_DEVICE="${BGE_DEVICE:-cuda:1}"

HOST="$(python -c 'from memory_system.config import settings; print(settings.embedding_api_host)')"
PORT="$(python -c 'from memory_system.config import settings; print(settings.embedding_api_port)')"

if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "Embedding service already available at http://${HOST}:${PORT}"
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Embedding service already running: PID $(cat "$PID_FILE")"
  exit 0
fi

nohup python -m memory_system.embedding_server >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
sleep 1
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Embedding service failed to start. Log follows:"
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi
echo "Embedding service started at http://${HOST}:${PORT}: PID $(cat "$PID_FILE"), log $LOG_FILE"
