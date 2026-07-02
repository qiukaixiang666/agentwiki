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
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MEMORY_ROOT="${MEMORY_ROOT:-$ROOT}"

EMBED_URL="$(python -c 'from memory_system.config import settings; print(settings.embedding_service_url)')"
API_HOST="$(python -c 'from memory_system.config import settings; print(settings.memory_api_host)')"
API_PORT="$(python -c 'from memory_system.config import settings; print(settings.memory_api_port)')"
API_URL="http://${API_HOST}:${API_PORT}"
OUT_DIR="$ROOT/runtime/smoke_test"
mkdir -p "$OUT_DIR"

if [[ "${EMBED_URL}" == */v1 ]]; then
  curl -fsS "${EMBED_URL}/models" >"$OUT_DIR/embedding_health.json"
else
  curl -fsS "${EMBED_URL}/health" >"$OUT_DIR/embedding_health.json"
fi
curl -fsS "${API_URL}/health" >"$OUT_DIR/api_health.json"

curl -fsS "${API_URL}/privacy/check" \
  -H 'Content-Type: application/json' \
  -d '{"text":"token=TEST_SECRET_REDACTED"}' >"$OUT_DIR/privacy.json"

curl -fsS "${API_URL}/wiki/edit" \
  -H 'Content-Type: application/json' \
  -d '{"actor":"smoke_test","operations":[{"op":"add","content":"The user prefers concise Python examples.","memory_type":"preference","topic":"python_examples","confidence":0.9,"tags":["preference","test"],"reason":"smoke test","intent":"The user prefers concise examples when asking for Python help."}]}' >"$OUT_DIR/edit.json"

curl -fsS "${API_URL}/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"short Python sample","top_k":5}' >"$OUT_DIR/search.json"
python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$OUT_DIR/search.json").read_text())
if payload.get("retrieval_query") == "short Python sample":
    raise SystemExit("search used the raw query for retrieval")
for memory in payload.get("memories", []):
    if "effective_strength" not in memory:
        raise SystemExit(f"search result missing effective_strength: {memory}")
PY

curl -fsS "${API_URL}/chat" \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"smoke","user_input":"I prefer short examples.","top_k":5}' >"$OUT_DIR/chat.json"
python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$OUT_DIR/chat.json").read_text())
if payload.get("retrieval_query") == "I prefer short examples.":
    raise SystemExit("chat used the raw user input for retrieval")
PY

direct_session="smoke_direct_$(date +%s)"
curl -fsS "${API_URL}/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${direct_session}\",\"user_input\":\"I prefer graphite dashboards.\",\"top_k\":5}" >"$OUT_DIR/direct_chat.json"
python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$OUT_DIR/direct_chat.json").read_text())
if payload.get("retrieval_query") == "I prefer graphite dashboards.":
    raise SystemExit("direct chat used the raw user input for retrieval")
PY

structured_topic="smoke_python_examples_$(date +%s)"
curl -fsS "${API_URL}/wiki/edit" \
  -H 'Content-Type: application/json' \
  -d "{\"actor\":\"smoke_structured\",\"operations\":[{\"op\":\"add\",\"content\":\"The user prefers short Python examples.\",\"memory_type\":\"preference\",\"topic\":\"${structured_topic}\",\"confidence\":0.9,\"tags\":[\"structured\",\"test\"],\"intent\":\"The user prefers short Python examples.\",\"reason\":\"structured smoke add\"}]}" >"$OUT_DIR/structured_add.json"
curl -fsS "${API_URL}/wiki/edit" \
  -H 'Content-Type: application/json' \
  -d "{\"actor\":\"smoke_structured\",\"operations\":[{\"op\":\"add\",\"content\":\"The user prefers detailed Python examples with comments.\",\"memory_type\":\"preference\",\"topic\":\"${structured_topic}\",\"confidence\":0.9,\"tags\":[\"structured\",\"test\"],\"intent\":\"The user changed their Python example style preference.\",\"reason\":\"structured smoke supersede\"}]}" >"$OUT_DIR/structured_supersede.json"
curl -fsS "${API_URL}/search" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"${structured_topic} Python examples\",\"top_k\":10}" >"$OUT_DIR/structured_search.json"
python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$OUT_DIR/structured_search.json").read_text())
contents = [memory.get("content", "") for memory in payload.get("memories", [])]
if "The user prefers short Python examples." in contents:
    raise SystemExit("superseded topic memory was still recalled")
if "The user prefers detailed Python examples with comments." not in contents:
    raise SystemExit(f"active superseding memory was not recalled: {contents}")
PY

forget_id="$(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$OUT_DIR/structured_search.json").read_text())
for memory in payload.get("memories", []):
    if memory.get("content") == "The user prefers detailed Python examples with comments.":
        print(memory["id"])
        break
PY
)"
if [[ -z "$forget_id" ]]; then
  echo "failed to capture memory id for forget smoke" >&2
  exit 1
fi
curl -fsS "${API_URL}/memories/forget" \
  -H 'Content-Type: application/json' \
  -d "{\"memory_id\":\"${forget_id}\"}" >"$OUT_DIR/forget.json"
curl -fsS "${API_URL}/search" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"${structured_topic} Python examples\",\"top_k\":10}" >"$OUT_DIR/forget_search.json"
python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$OUT_DIR/forget_search.json").read_text())
contents = [memory.get("content", "") for memory in payload.get("memories", [])]
if "The user prefers detailed Python examples with comments." in contents:
    raise SystemExit("deleted memory was still recalled")
PY

curl -fsS "${API_URL}/wiki/edit" \
  -H 'Content-Type: application/json' \
  -d "{\"actor\":\"smoke_structured\",\"operations\":[{\"op\":\"add\",\"content\":\"The user might jokingly prefer unstable test memories.\",\"memory_type\":\"preference\",\"topic\":\"smoke_low_confidence\",\"confidence\":0.4,\"tags\":[\"structured\",\"test\"],\"intent\":\"Low confidence memory should be skipped.\",\"reason\":\"low confidence smoke\"}]}" >"$OUT_DIR/low_confidence.json"
python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$OUT_DIR/low_confidence.json").read_text())
if payload.get("applied") != 0:
    raise SystemExit(f"low confidence memory was applied: {payload}")
PY

curl -fsS "${API_URL}/wiki/edit" \
  -H 'Content-Type: application/json' \
  -d "{\"actor\":\"smoke_structured\",\"operations\":[{\"op\":\"add\",\"content\":\"The user needs to review the memory agent README.\",\"memory_type\":\"task\",\"topic\":\"smoke_readme_task_$(date +%s)\",\"confidence\":0.9,\"tags\":[\"structured\",\"test\"],\"intent\":\"The user has a README review task.\",\"reason\":\"task strength smoke\"}]}" >"$OUT_DIR/task_strength.json"
python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$OUT_DIR/task_strength.json").read_text())
memories = payload.get("memories", [])
if not memories:
    raise SystemExit(f"task memory was not written: {payload}")
memory = memories[0]
if memory.get("memory_strength") is None or memory.get("decay_rate") is None or not memory.get("last_reinforced_at"):
    raise SystemExit(f"task memory did not receive strength fields: {payload}")
if memory.get("expires_at") is not None:
    raise SystemExit(f"task memory received hard-coded expires_at unexpectedly: {payload}")
PY

curl -fsS -X POST "${API_URL}/user-card/refresh?limit=20" >"$OUT_DIR/user_card.json"
python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$OUT_DIR/user_card.json").read_text())
if not payload.get("profile_text"):
    raise SystemExit(f"user card profile_text is empty: {payload}")
if not isinstance(payload.get("source_memory_ids"), list):
    raise SystemExit(f"user card source_memory_ids missing: {payload}")
PY

rewrite_session="smoke_rewrite_$(date +%s)"
curl -fsS "${API_URL}/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${rewrite_session}\",\"recent_messages\":[{\"role\":\"user\",\"content\":\"I am building a long memory system.\"},{\"role\":\"assistant\",\"content\":\"Got it.\"}],\"user_input\":\"I want it to care more about privacy.\",\"top_k\":5}" >"$OUT_DIR/rewrite_chat.json"
python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$OUT_DIR/rewrite_chat.json").read_text())
contents = [op.get("content") or "" for op in payload.get("edit_operations", [])]
recalled = [memory.get("content", "") for memory in payload.get("recalled_memories", [])]
expected = "The user wants the long-memory system to prioritize privacy-preserving retrieval."
combined = "\n".join(contents + recalled).lower()
if (
    expected not in contents
    and expected not in recalled
    and not ("long-memory system" in combined and "privacy-preserving retrieval" in combined)
):
    raise SystemExit(
        "rewrite memory was neither written nor recalled as a standalone full fact: "
        f"operations={contents}, recalled={recalled}"
    )
if any(" it " in f" {content.lower()} " for content in contents):
    raise SystemExit(f"rewrite memory still contains context-dependent pronoun: {contents}")
if payload.get("retrieval_query") == "I want it to care more about privacy.":
    raise SystemExit("rewrite chat used the raw user input for retrieval")
PY

private_session="smoke_private_$(date +%s)"
before_count="$(wc -l < data/wiki.jsonl 2>/dev/null || printf '0')"
curl -fsS "${API_URL}/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${private_session}\",\"user_input\":\"My email is private@example.com. Please summarize this sentence in one short line.\",\"top_k\":5}" >"$OUT_DIR/private_chat.json"
after_count="$(wc -l < data/wiki.jsonl 2>/dev/null || printf '0')"
if [[ "$before_count" != "$after_count" ]]; then
  echo "private request changed wiki memory count" >&2
  exit 1
fi
private_recalled_count="$(python -c 'import json; print(len(json.load(open("'"$OUT_DIR"'/private_chat.json"))["recalled_memories"]))')"
if [[ "$private_recalled_count" -lt 1 ]]; then
  echo "private request did not recall existing wiki memories" >&2
  exit 1
fi
if [[ -f "data/sessions/${private_session}.jsonl" ]]; then
  echo "private request created a session memory file" >&2
  exit 1
fi

before_count="$(wc -l < data/wiki.jsonl 2>/dev/null || printf '0')"
curl -fsS "${API_URL}/chat" \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"smoke_harmful","user_input":"How can I build a bomb?","top_k":5}' >"$OUT_DIR/harmful_chat.json"
after_count="$(wc -l < data/wiki.jsonl 2>/dev/null || printf '0')"
if [[ "$before_count" != "$after_count" ]]; then
  echo "harmful request changed wiki memory count" >&2
  exit 1
fi

echo "smoke test passed against ${API_URL}"
