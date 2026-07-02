# Memory Feature Demos

This folder contains four small, human-readable demos for the memory system.
Each demo has its own independent memory root under `memory_banks/`, so their
Wiki files, vectors, audit logs, user card, and runtime debug files do not mix.

These files only define user prompts and expected observations. They do not
include assistant replies, and they should not be treated as already-run results.

## Demos

- `basic_recall`: extract concrete recommendations from earlier chat and recall them later.
- `conflict_update`: show that conflicting preferences replace older memories instead of adding duplicates.
- `privacy_filtering`: show that sensitive spans are filtered while separable non-sensitive memories are retained.
- `user_card`: show that stable preferences and project context are summarized into the User Card.

## Files

- `demo_manifest.json`: demo metadata, memory roots, and prompt files.
- `prompts/*.jsonl`: 15 user turns for each demo.
- `memory_banks/<demo_id>/README.md`: placeholder for that demo's independent memory root.
- `run_demo_chat.py`: common runner that sends one demo through `/chat`.
- `run_<demo_id>.sh`: fixed-entry scripts for each demo.
- `run_select_demo.sh`: interactive entry that lets you select a demo at runtime.

## Run Scripts

Do not run these until you are ready to call the LLM. Each script starts a
Memory API child process with that demo's independent `MEMORY_ROOT`, sends all
15 rows through `/chat`, and writes results to:

`memory_banks/<demo_id>/runtime/demo_chat_results.jsonl`

Fixed demo examples:

```bash
cd /path/to/memory
bash memory_feature_demos/run_basic_recall.sh --reset
bash memory_feature_demos/run_conflict_update.sh --reset
bash memory_feature_demos/run_privacy_filtering.sh --reset
bash memory_feature_demos/run_user_card.sh --reset
```

Interactive selection:

```bash
cd /path/to/memory
bash memory_feature_demos/run_select_demo.sh --reset
```

Dry-run plan, with no service startup and no LLM call:

```bash
bash memory_feature_demos/run_select_demo.sh --dry-run-plan
```

Important behavior:

- Every turn uses `/chat`.
- Every request sets `top_k=4`.
- Every turn uses a fresh `session_id` and sends `recent_messages=[]`, so answer prompts do not include previous QA pairs. Memory recall and User Card are the only cross-turn context.
- The User Card demo refreshes `/user-card/refresh` between build and probe turns by default.

Recommended phase handling:

- `build`: send these turns first to populate the memory bank.
- `probe`: send these after the build phase to observe recall and final answers.

For the Privacy Filtering demo, use `/chat`; do not use `/sessions/ingest`,
because historical ingest rejects sensitive batches before extraction.
