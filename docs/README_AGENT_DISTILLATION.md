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

Keep those two files in sync. Labels sit on assistant thought+action spans only. User turns, observations, and `parse_error:` / `tool_error:` messages are unlabeled. `predict_fn` stays single-shot (ADR 0006): a CodeAct student on the endpoint still does not see the harness string.

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

Rows without `expected` are skipped. Collect must preserve `expected` on traces (`trajectory_to_legacy(..., expected=...)`). Their agentic teacher is OpenHands+interpreter; checklist tools cannot play `\pi_A`.

## SCoRe-SFT collect (after BC, not instead of it)

Cold-start BC on successful teacher traces first. Then:

```bash
python scripts/harness/collect_score.py \
  --input prompts.jsonl --output corrected.jsonl --prefs prefs.jsonl
```

Student explores; teacher `generate`s a review of the **full chain** and the corrected action is injected at the first **semantic** miss (wrong/missing final answer), not `DispatchError` recovered on the way. Resume from the verified prefix. Preference pairs (`σ_k` vs `σ'_k`) are a byproduct for later DPO/GRPO — not a separate EasyDistill job. Defer SCoRe-RL, GRPO, SDAR.

## Still deferred

Execution-consistent SAG; CodeAct subprocess sandbox; SageMaker `trajectory_mode`; TRL/peft pin + SDPO/GKD; live search; MCP autotools; `InputValidator` regex edits; turning on `distillation_alpha` in trajectory mode before KL is label-masked and vocab-aligned.
