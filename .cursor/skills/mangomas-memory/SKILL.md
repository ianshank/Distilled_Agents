---
name: mangomas-memory
description: Build AMD-lite workflow/function memory from successful teacher traces. Use when asked for hierarchical memory, workflow prefix, or tool_error retrieval. Opposite of tailor drop-tool.
cli: scripts/harness/build_memory.py
inputs:
  traces: JSONL of Trajectory objects (CLI --traces)
  output: Memory bank JSON path (CLI --output)
---

# Build AMD-lite teacher memory

Keeps successful traces (has `final_answer`, no `loop`). Inject via YAML `memory.bank_path` or `MANGOMAS_HARNESS_MEMORY_BANK`. Students get a workflow prefix (keyword overlap) and function hints on `tool_error`. Do not reuse `HarnessTailor` drop-tool.

```bash
python scripts/harness/build_memory.py \
  --traces "$TRACES_JSONL" \
  --output "$BANK_JSON"
```
