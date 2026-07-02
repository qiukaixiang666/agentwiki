# LongMemEval Direct-Context Baseline

This folder contains a lightweight baseline and evaluation script for:

`LongMemEval/longmemeval_s_sampled_100.json`

The baseline directly puts the haystack conversation history into the DeepSeek-compatible chat API and asks the question. It does not use the memory framework.

## Scripts

- `direct_context_baseline.py`
  - Input: LongMemEval JSON list.
  - Output: JSONL predictions.
  - Uses the existing `memory_system.llm_client` and `.env` settings.
  - Supports DeepSeek by default and Aliyun DashScope Qwen with `--provider aliyun`.
  - Supports `--resume` for interrupted runs.
  - Supports `--dry-run` for prompt construction checks without API calls.
  - Supports `--concurrency N` for concurrent DeepSeek API calls.

- `evaluate_predictions.py`
  - Input: prediction JSONL.
  - Output: JSON evaluation report.
  - Default `--judge heuristic` is offline.
  - Optional `--judge llm` uses the configured DeepSeek API as a semantic judge.
  - Supports `--concurrency N` for concurrent LLM judge calls.

## Suggested Commands

Run these from the project root:

```bash
cd /path/to/memory
```

Prompt-construction check only, no API call:

```bash
conda run -n memory python LongMemEval/scripts/direct_context_baseline.py --dry-run --limit 2
```

Run the direct-context baseline on all 100 sampled examples:

```bash
conda run -n memory python LongMemEval/scripts/direct_context_baseline.py --resume
```

Run the direct-context baseline with Aliyun Qwen:

```bash
export ALIYUN_API_KEY='<your-aliyun-api-key>'
conda run -n memory python LongMemEval/scripts/direct_context_baseline.py --provider aliyun --model qwen3.5-27b --resume
```

Aliyun Qwen thinking mode is disabled by default with `enable_thinking=false`.

Run the baseline with 4 concurrent API calls:

```bash
conda run -n memory python LongMemEval/scripts/direct_context_baseline.py --resume --concurrency 4
```

Run Aliyun Qwen with 4 concurrent API calls:

```bash
conda run -n memory python LongMemEval/scripts/direct_context_baseline.py --provider aliyun --model qwen3.5-27b --resume --concurrency 4
```

If the model context window cannot fit the full haystack, use a character budget. The script keeps the newest sessions under that budget:

```bash
conda run -n memory python LongMemEval/scripts/direct_context_baseline.py --resume --max-context-chars 120000
```

Offline heuristic evaluation:

```bash
conda run -n memory python LongMemEval/scripts/evaluate_predictions.py
```

DeepSeek semantic-judge evaluation:

```bash
conda run -n memory python LongMemEval/scripts/evaluate_predictions.py --judge llm --resume
```

DeepSeek semantic-judge evaluation with 8 concurrent judge calls:

```bash
conda run -n memory python LongMemEval/scripts/evaluate_predictions.py --judge llm --resume --concurrency 8
```

## Outputs

Default prediction output:

`LongMemEval/results/direct_context_predictions.jsonl`

Each row includes:

- `question_id`
- `question_type`
- `question`
- `gold_answer`
- `prediction`
- `error`
- prompt/context size metadata

Default evaluation output:

`LongMemEval/results/direct_context_eval.json`

The summary includes:

- overall accuracy
- average score
- prediction error count
- metrics grouped by `question_type`

## Notes

- `--max-context-chars 0` is the default and means full direct context.
- `--concurrency 1` is the default for both scripts.
- Aliyun Qwen uses the DashScope OpenAI-compatible endpoint by default:
  `https://dashscope.aliyuncs.com/compatible-mode/v1`
- For Aliyun, the script reads `ALIYUN_API_KEY`, `DASHSCOPE_API_KEY`, or `QWEN_API_KEY`.
- Add `--aliyun-enable-thinking` only if you want to enable Qwen thinking mode.
- The sampled examples are roughly 460k-510k characters each before prompt overhead.
- No API key is printed by either script.
