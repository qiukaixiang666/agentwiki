# LongMemEval Memory Pipeline

This module standardizes the two-step LongMemEval test flow used for the current memory system.

## Flow

1. Build one isolated memory bank for each LongMemEval question.
   - Input supports a cleaned LongMemEval JSON list such as `LongMemEval/longmemeval_s_cleaned.json`, a JSONL file, a single wrapped error-example JSON file, or a directory of JSON/JSONL files.
   - Each item is unwrapped from `raw_dataset_item` when present.
   - Historical `haystack_sessions` are ingested through `/sessions/ingest` using true QA pairs by default.
   - Memory data is written under `runtime/longmemeval_memory_pipeline/memory_banks/<question_type>__<question_id>/`.

2. Answer each question using raw-question retrieval.
   - The retrieval query is exactly the original `question` field, not the LongMemEval answer prompt.
   - The top 6 recalled memories are inserted into `memory_system.prompts.build_chat_prompt`.
   - Qwen answers the final LongMemEval answer prompt.

## Stage 1: Build Memory Banks

```bash
cd /path/to/memory
conda run -n memory python -m longmemeval_memory_pipeline.build_banks \
  --input LongMemEval/longmemeval_s_cleaned.json \
  --output-root runtime/longmemeval_memory_pipeline/memory_banks \
  --provider deepseek \
  --model deepseek-v4-flash \
  --embedding-url http://127.0.0.1:18083/v1 \
  --embedding-provider vllm \
  --embedding-model bge-m3
```

Useful options:

- `--start N --limit M` selects a slice.
- `--question-id ID` can be repeated.
- `--question-type TYPE` can be repeated.
- `--resume` skips items already marked `built` in `build_index.jsonl`.
- `--overwrite` rebuilds existing memory-bank folders.
- `--dry-run-plan` prints the planned items and does not start the Memory API.

## Stage 2: Raw Top-6 Recall + Qwen Answer

```bash
cd /path/to/memory
conda run -n memory python -m longmemeval_memory_pipeline.answer_raw_top6 \
  --input LongMemEval/longmemeval_s_cleaned.json \
  --memory-bank-root runtime/longmemeval_memory_pipeline/memory_banks \
  --output runtime/longmemeval_memory_pipeline/raw_top6_qwen_predictions.jsonl \
  --top-k 6 \
  --embedding-url http://127.0.0.1:18083/v1 \
  --embedding-provider vllm \
  --embedding-model bge-m3
```

The script also writes companion files:

- `raw_top6_qwen_predictions.json`
- `raw_top6_qwen_predictions.md`

## One Command Wrapper

```bash
cd /path/to/memory
conda run -n memory python -m longmemeval_memory_pipeline.run_two_stage \
  --input LongMemEval/longmemeval_s_cleaned.json \
  --run-root runtime/longmemeval_memory_pipeline
```

The wrapper simply runs Stage 1 and Stage 2 in sequence. Use the explicit two commands above when you want tighter control.
