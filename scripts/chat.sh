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

MEMORY_ROOT_ARG=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --memory-root)
      if [[ $# -lt 2 ]]; then
        echo "--memory-root requires a value" >&2
        exit 1
      fi
      MEMORY_ROOT_ARG="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --memory-root=*)
      MEMORY_ROOT_ARG="${1#*=}"
      ARGS+=("$1")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$MEMORY_ROOT_ARG" ]]; then
  if [[ "$MEMORY_ROOT_ARG" = /* ]]; then
    export MEMORY_ROOT="$MEMORY_ROOT_ARG"
  else
    export MEMORY_ROOT="$ROOT/$MEMORY_ROOT_ARG"
  fi
else
  export MEMORY_ROOT="${MEMORY_ROOT:-$ROOT}"
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-vllm}"
export EMBEDDING_SERVICE_URL="${EMBEDDING_SERVICE_URL:-http://127.0.0.1:18083/v1}"
export EMBEDDING_TIMEOUT_SECONDS="${EMBEDDING_TIMEOUT_SECONDS:-60}"

bash scripts/start_vllm_embedding_server.sh
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"
bash scripts/start_memory_api.sh

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "${CONDA_ENV:-${CONDA_ENV_NAME:-memory}}"

python -m memory_system.chat_cli "${ARGS[@]}"
