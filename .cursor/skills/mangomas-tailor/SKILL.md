---
name: mangomas-tailor
description: Propose deterministic harness YAML patches from traces. Use when asked to tailor, evolve, or patch a harness. APPLY_PATCHES defaults false.
cli: scripts/harness/tailor_harness.py
inputs:
  harness_id: Harness YAML id (CLI --harness-id)
  traces: JSONL of Trajectory objects (CLI --traces)
  archive_dir: Archive directory (CLI --archive-dir)
---

# Tailor a harness (rule-based)

```bash
python scripts/harness/tailor_harness.py \
  --harness-id "$HARNESS_ID" \
  --traces "$TRACES" \
  --archive-dir "$ARCHIVE"
```

Does not rewrite Python. `--apply` is off unless operators pass it or set `MANGOMAS_HARNESS_APPLY_PATCHES=true`.
