---
name: mangomas-collect
description: Collect trusted JSONL trajectories through the local harness. Use when asked to collect traces, rollouts, or trajectory SFT data. Not SageMaker predict_fn.
cli: scripts/harness/collect_trajectories.py
inputs:
  input: Source JSONL with a prompt field (CLI --input)
  output: Legacy prompt/completion JSONL (CLI --output)
  harness_id: Harness YAML id (CLI --harness-id)
  strict: Fail-fast on the first bad or empty row (CLI --strict)
---

# Collect harness trajectories

Input JSONL is **trusted operator data**. The CLI turns injection detection off (`strict_injection=False`) so coding prompts with `#`, quotes, or SQL-like text are collected. Empty prompts are skipped with a warning; pass `--strict` to fail-fast.

Do not point this CLI at untrusted user tasks — use `scripts/harness/run_agent.py` (injection on) for interactive runs.

```bash
python scripts/harness/collect_trajectories.py \
  --input "$INPUT_JSONL" \
  --output "$OUTPUT_JSONL"
```

Omit `--harness-id` unless it is non-empty (the CLI defaults to `MANGOMAS_HARNESS_ID` or `base_react`). Empty `"${MANGOMAS_HARNESS_ID}"` from `.env.example` would fail.

Raw traces are written next to `--output` as `*.raw.jsonl` (gitignored). Trajectory SFT is local: `python scripts/training/train_distilled_adapter.py --trajectory_mode True`.
