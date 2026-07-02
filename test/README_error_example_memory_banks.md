# Qwen Error Example Memory Banks

This folder contains a builder for creating one isolated Memory Wiki data directory per Qwen error example.

Default input:

`result/qwen_error_examples_by_type/examples`

Default output:

`runtime/qwen_error_example_memory_banks/<question_type>__<question_id>`

The builder unwraps each error-analysis JSON object, reads `raw_dataset_item`, starts an isolated Memory API with `MEMORY_ROOT` pointing at that question's output directory, ingests the LongMemEval haystack through `/sessions/ingest`, refreshes `/user-card`, and writes `manifest.json`.

It intentionally does not call `/chat` for the final question, so the memory bank contains only historical memories extracted from the haystack.

## Plan Only

```bash
test/build_qwen_error_example_memory_banks.sh --dry-run-plan
```

## Build All Banks

Start an embedding service first, then run:

```bash
test/build_qwen_error_example_memory_banks.sh \
  --provider qwen \
  --model qwen3.5-27b \
  --embedding-url http://127.0.0.1:18083/v1 \
  --embedding-provider vllm \
  --embedding-model bge-m3 \
  --resume
```

Use `--question-id <id>` or `--limit 1` for a small run. Use `--overwrite` only when you want to rebuild an existing output directory.
