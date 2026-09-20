# MangoMAS agent distillation

Papers (Kang, SCoRe, DualDistill, TRL-on-traces) assume the **student is trained on the same token string the agent loop feeds at test time**. Until that is true, adopting those recipes is theater.

This document is the source of truth for that constraint. ADR 0006 is the harness–policy decision; this file is how distillation maps onto it.

## Format triangle

Three surfaces used to speak three different strings:

| Surface | Entry | String |
| --- | --- | --- |
| Harness generate | `TransformersBackend._render_prompt` | `system:` / `user:` / `assistant:` turns, then a trailing `assistant:` (optional FTP prefix) |
| Trajectory collator | `scripts/training/distill/trajectory_collator.py` | **Same role-tagged body** via `prompt_render.supervised_spans` (SageMaker copy under `scripts/training/distill/prompt_render.py`) |
| SageMaker `predict_fn` | `scripts/inference.py` | Raw `input_data["prompt"]`, sampling decode, **no tools** |

Harness generate and masked SFT now share one renderer:

- `enhanced_system/harness/prompt_render.py` (runtime)
- `scripts/training/distill/prompt_render.py` (SageMaker `source_dir`; **no** `enhanced_system` import)

Keep those two files in sync. Labels sit on assistant thought+action spans only. User turns, observations, and `parse_error:` / `tool_error:` messages are unlabeled. The collator encodes that rendered body **once** (same string as `TransformersBackend`) and masks assistant spans from character/token offsets so BPE/SentencePiece context matches inference. Trajectory SFT needs a Hugging Face **fast** tokenizer (`offset_mapping`); a slow tokenizer raises instead of dropping every row. `predict_fn` stays single-shot (ADR 0006): a CodeAct student on the endpoint still does not see the harness string.

```text
system: {planning.instruction}          # unlabeled (Kang I_agent)
user: {task}                            # unlabeled
assistant: {thought} {action}           # labeled
user: {observation | fault: message}    # unlabeled
...
```

Thought vs action is split with `split_thought_action` (text before the parsed JSON/Call). Teacher-only FTP still seeds `steps[0].thought`. Student thought is empty unless the generation contains a remainder before the action.

## Two training stacks

Do not mix them.

### 1. SageMaker prompt/completion (role LoRAs)

`scripts/training/train_agent_skill.py` → `create_job_spec` → estimator `transformers_version=4.26.0`, `pytorch_version=1.13.1`, `peft==0.4.0`. Checked-in `data/training/*.jsonl` is Distilled_Agents **prompt/completion** (FastAPI snippets, not traces). Classic KD (`distillation_alpha` > 0) is **unsafe** on the default pair (Mistral teacher vs DialoGPT student: vocab mismatch; unshifted KL on all positions). DialoGPT is a unit-test stub; LoRA `q_proj,...gate_proj` does not match `c_attn`.

Do **not** add `trajectory_mode` to `create_job_spec` until argparse and the launcher change together. That image cannot run Qwen2.5-Instruct trajectory SFT.

### 2. Local trajectory BC (Kang spine)

```bash
python scripts/harness/collect_trajectories.py --input prompts.jsonl --output traces.jsonl
python scripts/training/train_distilled_adapter.py --trajectory_mode True
```

Collect copies `expected` from the prompt JSONL onto each trace when present. `compose_dualdistill.py` skips unlabeled rows, so collect → DualDistill is a dead pipeline without that field.

`--trajectory_mode` unloads the teacher. `distillation_alpha` comes from `MANGOMAS_TRAJECTORY_DISTILL_ALPHA` (default `0.0`). Collect JSONL is trusted (`strict_injection=False`). Interactive `scripts/harness/run_agent.py` keeps injection on.

Local training rule: occasional runs, ≤10k traces, privacy-sensitive data → **local** SFT, not the 4.26/1.13 SageMaker image. Real students: Qwen2.5-Instruct 0.5B–3B.

**512-token trap:** collator `max_length=512`, backend `max_input_length=512`, `max_new_tokens=128`. The prompt is encoded first and unlabeled; a long prompt zeros out labels and `has_supervised_tokens` **drops the row**.

## What `planning.style` does (and does not)

Schema allows `react` / `codeact` / `plan_and_solve`. `AgentRuntime.run` does **not** read `spec.planning.style`. Every harness is one loop + different `tool_ids`. `style: codeact` is not CodeAct (no interpreter, no `exec`). DualDistill cannot be “collect codeact vs plan_and_solve.”

Kang `I_agent` is `planning.instruction` (YAML + schema; `additionalProperties: false` means both `configs/harnesses/` and `enhanced_system/config/harnesses/` must change together).

## Paper false-friends (do not vendor)

| Recipe | Honest mapping here | Not this |
| --- | --- | --- |
| Kang / Nardien | Shared renderer + instruction + outcome-filtered teacher traces + masked SFT | smolagents; execution SAG; subprocess CodeAct sandbox |
| EasyDistill operators | Collect `--input`; exact-match on `final_answer`; optional prefs dump | LangGraph `agentkd`; `trust_remote_code: true`; live web search |
| DualDistill | `compose_dualdistill.py`: same `x`, two teachers, grader `G(y,a)`, `y1 ⊕ t ⊕ y2`, drop `(0,0)` | Concatenating architect vs SWE JSONL (that is multi-task SFT) |
| SCoRe-SFT | `collect_score.py`: student rollout, teacher **review prompt** on the full chain, resume prefix, prefs byproduct | Treating `step.fault` as the task failure; GRPO / SCoRe-RL |
| AMD-lite | `build_memory.py`: student workflow prefix + function memory on `tool_error` | `HarnessTailor` dropping tools after two errors |
| SDAR / AgentArk PAD / TRL GKD | Deferred. SDPO with observations as `privileged_context` is the honest TRL mapping | Launcher hyperparams; gated OPSD on GRPO |
| AgentDistill MCP | Frozen `TOOL_REGISTRY`; human-reviewed tool ids only | Autoload / `exec` |

SAG is parse/schema majority vote. Defaults `harness_sag_samples=1`, `harness_sag_temperature=0.0`. It is **not** execute-and-vote-on-observation.

Tools are checklists (`pytest_runner` does not run pytest). Kang’s “small models retrieve/code instead of memorizing” does not apply until a sandbox exists.

`evaluate_agent_skill` counts pre-filled `passed` / `actual==expected` in a JSON file. It does **not** run a model. Harness quality is `scripts/harness/eval_harness.py` (`AgentRuntime`, Echo in tests): exact-match `final_answer`, tool-id validity, truncation, faults.

`DataCurator` is prompt/completion quality heuristics, not traces.

## Collect, filter, eval

```bash
# Teacher traces (default). Outcome filter only when the row has expected.
python scripts/harness/collect_trajectories.py \
  --input prompts.jsonl --output traces.jsonl --harness-id swe_codeact --teacher

# Student rollouts (same loop, student weights).
python scripts/harness/collect_trajectories.py \
  --input prompts.jsonl --output student.jsonl --student

# Runtime eval (not skill-fixture pass rate)
python scripts/harness/eval_harness.py --input prompts.jsonl --harness-id base_react
```

Outcome filter skips a row when `expected` is present **and** `final_answer` misses it. Intermediate `parse_error` / `tool_error` are **kept** when the outcome matches — recovery is the point. No `expected` ⇒ no outcome skip. Kept rows still carry `expected` so compose can grade.

## Trace Critic Cascade (Phase P2)

Trace criticism provides deterministic validation and explicit reject telemetry during trajectory collection and DualDistill composition.

### 1. Reusable Pure Critic Helpers (`enhanced_system/harness/critic.py`)

- **Allowlist Validity (`check_tool_allowlist`)**: Validates that all tool invocations and parsed actions in a trajectory belong to the harness's declared allowlist (including `final_answer`). Any unauthorized or unknown tool immediately fails with `SCHEMA_VIOLATION`.
- **Expected Tools Check (`check_expected_tools`)**: Confirms that required tools were executed (supports both set inclusion and strict ordering). Fails with `SCHEMA_VIOLATION` if missing.
- **Outcome Grading (`check_outcome`)**: Exact match grading against reference `expected` answer (or semantic match if `allow_semantic: true`). Misses fail with `OUTCOME_MISMATCH`.
- **Recovery Trace Preservation (`is_recovery_trace`)**: Intermediate `parse_error` or `tool_error` steps are retained if the final answer matches `expected`. Increments the `critic_kept_recovery` counter.

### 2. Standard Reject Codes (`openspec/changes/_shared/blocked-reject-codes.md`)

Structured reject records and telemetry log canonical codes from the shared contract:
- Deterministic solver / teacher-rule codes: `CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`.
- Upstream pipeline codes: `OUTCOME_MISMATCH`, `DUALDISTILL_DROP_0_0`.

### 3. Collection and Telemetry Sink (`collect_trajectories.py` & `compose_dualdistill.py`)

- Gated by `MANGOMAS_CRITIC_ENABLED` (via `get_settings().critic_enabled`, default `True`; override with `--critic` / `--no-critic`).
- Drops are appended as structured JSONL records to `artifacts/critic_rejects.jsonl` (or `--reject-log` / `MANGOMAS_CRITIC_REJECT_LOG`):
  ```json
  {"critic_reject_code": "OUTCOME_MISMATCH", "prompt": "...", "metadata": {"line_no": 1, "expected": "...", "final_answer": "..."}}
  ```
- DualDistill `compose_pair` drops `(0, 0)` with `critic_reject_code: DUALDISTILL_DROP_0_0` without inventing a second drop rule.
- Counters emitted: `critic_rejected_outcome_mismatch`, `critic_rejected_allowlist`, `critic_rejected_expected_tools`, `critic_rejected_dualdistill_0_0`, and `critic_kept_recovery`.


## Golden Sets and Pass@K Multi-Trial Eval (Phase P1)

Golden evaluation rows follow `GoldenRow`:
- `prompt`: string (required)
- `expected`: string (required for labeled rows)
- `id`: stable identifier (required for `slice: hard`)
- `harness_id`: optional harness YAML id
- `expected_tools`: optional list of tool names
- `slice`: `"core"` | `"hard"` (default `"core"`)
- `allow_semantic`: bool (default `false`)

Evaluating golden sets with multi-trial pass@k:

```bash
python scripts/harness/eval_harness.py \
  --input configs/golden_sets/core_sdlc.jsonl \
  --harness-id base_react \
  --pass-k 5
```

Evaluation rules:
- `k` trials are run independently per row (sampling temperature from `MANGOMAS_EVAL_PASS_K_TEMPERATURE`, default `0.8` when `k>1`, `0.0` for `k=1`).
- `pass@1`: fraction of rows succeeding on trial 1.
- `pass@k`: fraction of rows succeeding on at least 1 of the `k` trials (or Chen unbiased estimate across $n$ samples).
- **Hard and OOD slices**: success strictly requires exact `answers_match` and no security failures (`semantic_match` alone is a failure; unlabeled non-truncated rows do not count). OOD rows must produce canonical `BLOCKED:<CODE>` refusal tokens.
- **Core slice**: `semantic_match` counts as success only when the row explicitly specifies `allow_semantic: true`.

AQA Regression Gate runs both core and hard golden sets in CI with deterministic mock fixtures:

```bash
python scripts/harness/run_aqa_gate.py \
  --golden-set configs/golden_sets/core_sdlc.jsonl \
  --threshold 75.0 \
  --scripted tests/fixtures/mock_responses.json

python scripts/harness/run_aqa_gate.py \
  --golden-set configs/golden_sets/hard_sdlc.jsonl \
  --threshold 75.0 \
  --hard-threshold 75.0 \
  --require-hard \
  --scripted tests/fixtures/mock_responses.json
```

Traceability rule matrix (`configs/rule_traceability/matrix.yaml`) verifies every hard golden row is backed by a tracked rule and source trace:

```bash
python scripts/harness/check_rule_matrix.py \
  --matrix configs/rule_traceability/matrix.yaml \
  --golden configs/golden_sets/hard_sdlc.jsonl
```

## AMD-lite (after format + eval)

```bash
python scripts/harness/build_memory.py --traces traces.jsonl --output memory.json
# YAML memory.bank_path or MANGOMAS_HARNESS_MEMORY_BANK
```

Successful teacher traces (has `final_answer`, no `loop`) contribute a workflow hint (keyword overlap on the task) and per-tool call guides. On student `tool_error`, a function-memory hint is appended to the observation. This is the **opposite** of tailor drop-tool. Checklist tools will show small lift; do not claim AMD paper numbers.

## DualDistill compose (labeled same-task suite only)

Needs the same prompt `x`, reference `a`, two heterogeneous teachers, second solution **conditioned on the first**, grader `G(y,a)`, compose `y1 ⊕ t ⊕ y2`, drop `(0,0)`. Mixing harness JSONL by role is multi-task SFT.

```bash
python scripts/harness/compose_dualdistill.py \
  --first teacher_a.jsonl --second teacher_b.jsonl --output composed.jsonl
```

Rows without `expected` are skipped. Collect must preserve `expected` on traces (`trajectory_to_legacy(..., expected=...)`). When both teachers fail the reference grade (0,0), the pair is dropped and emits `critic_reject_code: "DUALDISTILL_DROP_0_0"` into `artifacts/critic_rejects.jsonl`. Their agentic teacher is OpenHands+interpreter; checklist tools cannot play `\pi_A`.

## Critic rejection telemetry (Phase 0 I2)

Config-gated via `MANGOMAS_CRITIC_ENABLED` (or `--critic`), the critic cascade records explicit reason codes into `artifacts/critic_rejects.jsonl`:
- `OUTCOME_MISMATCH`: Emitted by `collect_trajectories.py` outcome filter when `final_answer` fails `answers_match`.
- `DUALDISTILL_DROP_0_0`: Emitted by `compose_dualdistill.py` when both teachers score 0 on the task.
- Recovery traces (intermediate `parse_error` / `tool_error` but matching final answer) are kept, incrementing `critic_kept_recovery`.
- Standard teacher-rule reject codes follow `openspec/changes/_shared/blocked-reject-codes.md`: `CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`.

## SCoRe-SFT collect (after BC, not instead of it)

Cold-start BC on successful teacher traces first. Then:

```bash
python scripts/harness/collect_score.py \
  --input prompts.jsonl --output corrected.jsonl --prefs prefs.jsonl
```

Student explores; teacher `generate`s a review of the **full chain** and the corrected action is injected at the first **semantic** miss (wrong/missing final answer), not `DispatchError` recovered on the way. Resume from the verified prefix. Preference pairs (`σ_k` vs `σ'_k`) are a byproduct for later DPO/GRPO — not a separate EasyDistill job. Defer SCoRe-RL, GRPO, SDAR.

## Still deferred

Execution-consistent SAG; CodeAct subprocess sandbox; SageMaker `trajectory_mode`; TRL/peft pin + SDPO/GKD; live search; MCP autotools; `InputValidator` regex edits; turning on `distillation_alpha` in trajectory mode before KL is label-masked and vocab-aligned.
