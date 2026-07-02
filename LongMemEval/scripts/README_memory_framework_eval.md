# LongMemEval Memory Framework Evaluation

`memory_framework_eval.py` evaluates the actual memory framework rather than the direct-context baseline.

It uses:

`LongMemEval/longmemeval_s_sampled_100.json`

## Pipeline

For each LongMemEval item, the script:

1. Starts an isolated Memory API process with a fresh `MEMORY_ROOT`.
2. Sends historical user/assistant turns to `/sessions/ingest`, passing prior dialog as context.
3. Lets the framework run safety, privacy checks, retrieval rewriting, memory extraction, conflict handling, decay/reinforcement bookkeeping, and User Card refresh.
4. Optionally adds one session-summary extraction turn per haystack session.
5. Refreshes `/user-card`.
6. Asks the LongMemEval question through `/chat`.
7. Calls `/search` for retrieval diagnostics.
8. Writes one JSONL row with prediction, recalled memories, User Card, and framework stats.

The default provider is Aliyun Bailian Qwen:

```bash
conda run -n memory python LongMemEval/scripts/memory_framework_eval.py --dry-run-plan --limit 1
```

Run one example with Qwen 3.5 27B:

```bash
scripts/start_vllm_embedding_server.sh

conda run -n memory python LongMemEval/scripts/memory_framework_eval.py \
  --provider qwen \
  --model qwen3.5-27b \
  --embedding-url http://127.0.0.1:18083/v1 \
  --embedding-provider vllm \
  --embedding-model bge-m3 \
  --limit 1 \
  --keep-item-memory
```

Run all 100 sampled examples:

```bash
conda run -n memory python LongMemEval/scripts/memory_framework_eval.py \
  --provider qwen \
  --model qwen3.5-27b \
  --embedding-url http://127.0.0.1:18083/v1 \
  --embedding-provider vllm \
  --embedding-model bge-m3 \
  --resume
```

Run the fuller LongMemEval-oriented mode with session summaries:

```bash
conda run -n memory python LongMemEval/scripts/memory_framework_eval.py \
  --provider qwen \
  --model qwen3.5-27b \
  --embedding-url http://127.0.0.1:18083/v1 \
  --embedding-provider vllm \
  --embedding-model bge-m3 \
  --enable-session-summary \
  --resume
```

Use DeepSeek instead:

```bash
conda run -n memory python LongMemEval/scripts/memory_framework_eval.py \
  --provider deepseek \
  --model deepseek-v4-flash \
  --resume
```

## Notes

- This script is intentionally slower than the direct-context baseline because it exercises the whole memory framework.
- It starts one isolated Memory API process per dataset item to avoid cross-item contamination.
- It reuses the configured embedding service through `EMBEDDING_SERVICE_URL`.
- vLLM embedding service uses OpenAI-compatible `/v1/embeddings`; start it with `scripts/start_vllm_embedding_server.sh` and pass `--embedding-url http://127.0.0.1:18083/v1 --embedding-provider vllm --embedding-model bge-m3`.
- Historical ingestion does not ask the LLM to generate a replacement assistant reply; it only extracts and writes memories from the existing LongMemEval dialog.
- The LLM generates a new answer only for the final LongMemEval question.
- Qwen thinking mode is disabled by setting `LLM_EXTRA_BODY_JSON` to include `enable_thinking=false`.
- Default ingestion uses `--ingest-turns user_with_context`: user turns trigger memory writes, while prior dialog is passed as context.
- Add `--enable-session-summary` to capture assistant-origin evidence such as prior recommendations or concrete conclusions.
- Progress bars are enabled by default. Add `--no-progress` for plain logs.
- Default output is `LongMemEval/results/memory_framework_predictions.jsonl`.
