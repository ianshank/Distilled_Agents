# Design — golden-passk-aqa

## Decisions

1. **pass@k definition:** For each golden row, run the harness eval loop `k`
   independent trials (temperature / seed / `sag_samples` as configured).
   `pass@1` = fraction of rows with ≥1 success in trial 1.
   `pass@k` = fraction of rows with ≥1 success among k trials.
   Success for **hard** slice = `answers_match` (exact) AND no security_failed.
   Success for **core** slice may include semantic_match if flagged
   `allow_semantic: true` on the row (default false for hard).

2. **SAG vs pass@k:** Do not equate `harness_sag_samples` with pass@k.
   pass@k is an eval harness concern; SAG remains parse-majority at step time.

3. **CI:** Add a job (or step) that runs
   `make aqa-gate` / `run_aqa_gate.py` with `--scripted` mocks only.
   Fail if JSON summary lacks `pass_at_1` / `pass_at_k` keys after this change.

4. **prompt_render guard:** CI or unit test asserts byte-equality of function
   bodies (or full file after normalizing the cross-reference docstring) for
   `enhanced_system/harness/prompt_render.py` and
   `scripts/training/distill/prompt_render.py`.

5. **Rule matrix:** `configs/rule_traceability/matrix.yaml` lists
   `rule_id, source_trace, harness_id, golden_id, solver_status`.
   Until P3, `solver_status: pending` is allowed; every `slice: hard` golden
   row MUST appear.

## Settings

New optional knobs (defaults conservative):

- `MANGOMAS_EVAL_PASS_K` (int, default 5) — eval only
- `MANGOMAS_EVAL_PASS_K_TEMPERATURE` (float, default 0.7) when k>1

No call-site literals for these values.
