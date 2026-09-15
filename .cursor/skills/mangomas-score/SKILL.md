---
name: mangomas-score
description: SCoRe-SFT collect — student rollout, teacher review prompt, resume from verified prefix. Use when asked for SCoRe-SFT or preference pairs. Defer GRPO/SCoRe-RL/SDAR.
cli: scripts/harness/collect_score.py
inputs:
  input: Prompts JSONL (CLI --input)
  output: Corrected traces JSONL (CLI --output)
  harness_id: Harness YAML id (CLI --harness-id)
  prefs: Optional preference-pair JSONL (CLI --prefs)
  strict: Fail on malformed JSONL rows instead of skipping (CLI --strict)
---

# Collect SCoRe-SFT corrected traces

Run **after** teacher behavioral cloning. Student explores; the teacher reviews the full chain and replaces the first **semantic** miss (wrong/missing final answer), not a recovered `DispatchError`. Resume from the verified prefix. `--prefs` dumps chosen/rejected pairs on the same prefix (DPO/GRPO later, not trained here).

```bash
python scripts/harness/collect_score.py \
  --input "$INPUT_JSONL" \
  --output "$OUTPUT_JSONL" \
  --prefs "$PREFS_JSONL"
```

Pass `--strict` to fail on bad JSON or empty prompts. Empty teacher `generate` still skips that row. Source `expected` is copied onto output traces.

Do not vendor SCoRe runtimes. Do not add launcher GRPO keys.
