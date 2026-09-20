# Design — golden-passk-aqa

## Decisions

1. **pass@k definition:** Use the Chen et al. unbiased estimator (arXiv:2107.03374)
   over $n$ samples per problem ($n \ge k$):
   $$\text{pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$
   Reject the biased estimator $1 - (1 - \hat{p})^k$.
   Defaults: CI runs $n=5$ samples per problem, gates on $k=3$, and reports
   both `pass_at_1` and `pass_at_k` (with $k=3$). Temperature guidance: $T=0.2$
   for pass@1 reporting, $T=0.8$ for pass@3 sampling when neural runs exist.
   Success for **hard** and **ood** slices requires exact `answers_match` only
   and `security_failed: false`. `semantic_match` MUST NOT count as success on
   hard/ood slices. Unlabeled non-truncated rows MUST NOT count as success on
   hard/ood.
   Success for **core** slice may include `semantic_match` only if flagged
   `allow_semantic: true` on the row (default false for hard/ood).

2. **Dataset layout and composition:** Require new golden set file
   `configs/golden_sets/sqe_hard_ood.jsonl` with $\ge 24$ rows across hard and
   OOD buckets:
   - Hard dependency DAG satisfy ($\ge 6$ rows): lexicographically least valid
     topological order, space-separated node ids.
   - Hard condition tree satisfy ($\ge 6$ rows): nested AND/OR/NOT assignment
     JSON with sorted keys.
   - Hard mixed graph + predicates ($\ge 4$ rows).
   - OOD cycle ($\ge 2$ rows): expected refusal token `BLOCKED:CYCLE_DETECTED`.
   - OOD unsat predicates ($\ge 2$ rows): expected refusal token `BLOCKED:UNSAT`.
   - OOD schema/syntax ($\ge 2$ rows): expected refusal token `BLOCKED:SCHEMA_VIOLATION`
     or `BLOCKED:SYNTAX_INVALID`.
   - OOD unknown theory ($\ge 2$ rows): expected refusal token `BLOCKED:UNSUPPORTED_THEORY`.
   Every hard and OOD row must specify `grader: "exact"` (`allow_semantic: false`).
   Keep `configs/golden_sets/core_sdlc.jsonl` as easy regression.

3. **Synthetic-success prohibition for OOD:** For every row with `ood: true`,
   the expected target is canonical refusal token `BLOCKED:<CODE>`. Any sample
   emitting a non-empty, non-matching `final_answer` is an OOD synthetic-success
   violation and causes CI gate failure.

4. **CLI and AQA gate:** Prefer dedicated CLI `scripts/harness/run_pass_at_k.py`
   or extend `run_aqa_gate.py` with `--pass-at-k`, `--samples`, `--slice`.
   Emits summary JSON containing `pass_at_k`, `exact_only: true`,
   `semantic_counted: false`, and `ood_synthetic_success_violations`.
   Exit nonzero if `pass_at_k["3"]` is below threshold or if any OOD
   synthetic-success violation occurs.
   Phase 0 CI uses scripted fixtures under `tests/fixtures/mock_responses_sqe_passk.json`
   with threshold 1.0; neural threshold is deferred until student model exists.
   Makefile target `aqa-gate-passk` invokes this gate.
   Correct `scripts/harness/run_aqa_gate.py` docstring (single-pass today).

5. **SAG vs pass@k:** Do not equate `harness_sag_samples` with pass@k.
   pass@k is an eval harness concern; SAG remains parse-majority at step time.

6. **prompt_render guard:** CI or unit test asserts byte-equality of function
   bodies (or full file after normalizing the cross-reference docstring) for
   `enhanced_system/harness/prompt_render.py` and
   `scripts/training/distill/prompt_render.py`.

7. **Rule matrix:** `configs/rule_traceability/matrix.yaml` lists
   `rule_id, source_trace, harness_id, golden_id, solver_status`.
   Until P3, `solver_status: pending` is allowed; every `slice: hard` golden
   row MUST appear.

## Settings

New optional knobs (defaults conservative):

- `MANGOMAS_EVAL_PASS_K` (int, default 3) — eval only
- `MANGOMAS_EVAL_PASS_N` (int, default 5) — samples per problem for Chen estimator
- `MANGOMAS_EVAL_PASS_K_TEMPERATURE` (float, default 0.8) when k>1

No call-site literals for these values.
