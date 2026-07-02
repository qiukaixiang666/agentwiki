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

mkdir -p runtime

HOST="${VLLM_EMBEDDING_HOST:-127.0.0.1}"
PORT="${VLLM_EMBEDDING_PORT:-18083}"
MODEL="${EMBEDDING_MODEL:-$ROOT/bge-m3}"
SERVED_MODEL_NAME="${VLLM_EMBEDDING_SERVED_MODEL_NAME:-bge-m3}"
GPU="${VLLM_EMBEDDING_GPU:-1}"
GPU_MEMORY_UTILIZATION="${VLLM_EMBEDDING_GPU_MEMORY_UTILIZATION:-0.18}"
PID_FILE="${VLLM_EMBEDDING_PID_FILE:-runtime/vllm_embedding_server.pid}"
LOG_FILE="${VLLM_EMBEDDING_LOG_FILE:-runtime/vllm_embedding_server.log}"

if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM embedding service already available at http://${HOST}:${PORT}/v1"
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "vLLM embedding service already running: PID $(cat "$PID_FILE")"
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" nohup python -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --runner pooling \
  --convert embed \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  >"$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
sleep 2
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "vLLM embedding service failed to start. Log follows:"
  tail -n 100 "$LOG_FILE" || true
  exit 1
fi

echo "vLLM embedding service starting at http://${HOST}:${PORT}/v1"
echo "PID $(cat "$PID_FILE"), log $LOG_FILE"
