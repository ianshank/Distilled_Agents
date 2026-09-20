# Opportunity: Symbolic knowledge distillation & disposition layer

**Repo:** [ianshank/Distilled_Agents](https://github.com/ianshank/Distilled_Agents)  
**Stage:** discover / intake  
**Status:** proposed (HOLD / fail-closed for Spec Writer until eval, telemetry, and disposition gates exist as artifacts)  
**Related:** ADR 0006 (harness–policy pair: H disposes, R_δ proposes); `docs/README_AGENT_DISTILLATION.md`; `make aqa-gate` / `scripts/harness/run_aqa_gate.py`

## Problem

Distilled_Agents trains small LoRA student adapters on teacher trajectories (collect → DualDistill / SCoRe-SFT → masked trajectory SFT / DPO) and exercises them with harness `H` (`AgentRuntime`). Today, the regression gating CLI (`aqa-gate` / `scripts/harness/run_aqa_gate.py`) is a **single-pass pass_rate on a ~4-row golden set** (`configs/golden_sets/core_sdlc.jsonl` evaluated via `scripts/harness/eval_harness.py`). True Pass@K evaluation on a dedicated hard or out-of-distribution (OOD) slice is a **gate to build**, not a present capability.

Crucially, the current runtime harness is **neural-only at disposition time**:
- ADR 0006 authorizes harness `H` to dispose and policy `R_δ` to propose; it does **not** provide a symbolic solver or sound decision layer today.
- Today's disposition is strictly allowlist checking (`_ensure_allowed`) plus parse/schema AST validation (`parse_action` / `JsonSchemaTool`), not mathematical or logical soundness.
- Existing tools in `TOOL_REGISTRY` are static checklists (`pytest_runner` validates non-empty string lists and does not execute pytest; `sqe_checklist` only checks for the substring `"assert"`).
- SAG (Self-Consistency / Self-Agreement) in `enhanced_system/harness/policies/sag.py` is parse/schema majority voting on strings, not execution-and-vote-on-observation.
- Execution-consistent sandboxing and TRL GKD / SDPO remain explicitly deferred (`docs/README_AGENT_DISTILLATION.md`, `docs/NEXT_STEPS.md`).

On hard, multi-step, or OOD tasks, pure pattern distillation (behavior cloning on teacher traces) risks capability collapse on distribution tails. Without a **deterministic symbolic solver/verifier disposition path**, a small student (<4B) cannot reliably satisfy domain constraints, and OOD failure tends toward confabulation or loop truncation rather than safe, fail-closed refusal.

The defensible product move is **not** to "train a neurosymbolic neural net" (which would violate repo architecture and introduce untracked model paradigms). It is **symbolic knowledge distillation**:
1. An open-weight local teacher emits rules, constraints, or verified rationale traces.
2. A deterministic critic/verifier filters and logs rejection codes before dataset composition.
3. A small student (proposer) plus a deterministic symbolic disposition layer executes under harness `H`, preserving the ADR 0006 contract where the cognitive policy proposes and the harness disposes.

## User & Scope Fence

**Scope fence:**
- **Distilled_Agents only.** No claims of delivering Mango INV-16 or Edge-AI VII capabilities. Edge/offline inference refers strictly to local trajectory behavior cloning (BC) on Qwen2.5-Instruct 0.5B–3B already documented in-repo (`docs/README_AGENT_DISTILLATION.md`), not a separate product, runtime, or hardware offering.

**Target users:**
- **Primary:** Operators and ML maintainers of Distilled_Agents / MangoMAS who need **smaller, license-clean students** (0.5B–3B) that execute predictably under the frozen harness (`TOOL_REGISTRY`, AST/JSON dispatch, no `exec`, strict schema enforcement).
- **Secondary:** Local/offline practitioners who require bounded latency and deterministic fail-closed behavior on out-of-distribution inputs.

## Success Metrics

All metrics must be measurable in CI or via reproducible CLIs before this opportunity is cleared for Spec Writer:

1. **Hard/OOD Pass@K gate (gate to build):** An extended golden set containing hard and OOD tasks evaluated with true Pass@K ($k \ge 3$, exact match required on the hard slice; semantic match is optional and does not count as success on the hard slice). CI fails if Pass@K regresses even if Pass@1 is maintained. (Contrasted with current `run_aqa_gate.py` which only computes single-pass pass rate on a 4-row set).
2. **Deterministic constraint satisfaction rate = 100% (gate to build):** On a scoped vertical task with an explicit symbolic constraint, measured against a named disposition tool:
   - **Named Vertical:** Software Quality / Acceptance Specification (validating dependency graphs and condition trees).
   - **Named Dispose Tool ID:** `sqe_constraint_solver` (a deterministic solver tool extending `TOOL_REGISTRY`, distinct from the checklist `sqe_checklist`).
   - *Status:* Spec Writer is **BLOCKED** until `sqe_constraint_solver` specification and contract tests are formally written.
   - The neural-only baseline must be evaluated on the identical suite to demonstrate empirical lift.
3. **Critic traceability and rejection logging:** Every retained teacher-emitted rule or verified trace maps to a row plugging into existing harness evidence (exact match grader `answers_match`, DualDistill drop `(0,0)`, outcome-filtered collect). Critic rejection rate and explicit rejection reason codes (e.g., `UNSAT`, `SYNTAX_INVALID`, `CYCLE_DETECTED`, `SCHEMA_VIOLATION`) are logged in collect/compose pipelines.
4. **Student envelope:** Task-specialized student in the **0.5B–3B** band (e.g., Qwen2.5-Instruct) already targeted by local trajectory BC, operating in conjunction with the symbolic disposition layer. Unbounded multi-tool students $\ge 7\text{B}$ are explicitly out of scope.
5. **Fail-closed OOD disposition:** When the symbolic layer cannot verify or satisfy constraints, the trajectory terminates deterministically as `BLOCKED` / `tool_error` with zero synthetic success, preserving ADR 0006 "no `exec`" security posture.

## Kill Criteria (Spec Writer Blockers)

Spec Writer is **BLOCKED** and this opportunity shall be killed or placed on indefinite HOLD if any of the following artifacts fail to materialize:

1. **No CI Pass@K on Hard Slice:** Spec Writer is blocked until an automated CI gate measures Pass@K on an extended hard/OOD evaluation set with strict exact-match grading (`answers_match`). Semantic matching (`semantic_match`) cannot count as success on this slice.
2. **Missing Critic Rejection Telemetry:** Spec Writer is blocked until `scripts/harness/collect_trajectories.py` and `scripts/harness/compose_dualdistill.py` log explicit critic rejection reason codes when filtering teacher-generated traces and candidate rules.
3. **Synthetic Success on OOD:** Spec Writer is blocked if any OOD, unsatisfiable, or unparseable input produces an unverified or optimistic `final_answer`. The disposition path must terminate as `BLOCKED` / `tool_error`.

## Non-goals

To prevent scope creep and align with `docs/README_AGENT_DISTILLATION.md`:
- **No neurosymbolic neural network:** We are not training a hybrid neural net architecture with embedded fuzzy-logic neurosymbolic weights. The model is a standard causal LM LoRA adapter; the symbolic layer is an external deterministic solver in harness `H`.
- **No execution sandbox / No `exec`:** Tools remain deterministic, safe Python code in `TOOL_REGISTRY`. No subprocess CodeAct sandbox or arbitrary code execution (`exec`/`eval`). Kang paper assumptions requiring code execution are explicitly excluded.
- **No paper false-friends:**
  - DualDistill is strictly same-task $x$, reference answer $a$, two heterogeneous teachers, transition stitching $y_1 \oplus t \oplus y_2$, and dropping $(0,0)$ via `compose_dualdistill.py`. It is not concatenated role JSONL (which is multi-task SFT).
  - SCoRe-SFT is student exploration with teacher full-chain review and prefix resumption (`collect_score.py`). It is not GRPO, SCoRe-RL, or SDAR.
  - AMD-lite is workflow prefix and function hints on `tool_error` (`build_memory.py`), not `HarnessTailor` tool-dropping.
  - SAG is parse/schema majority voting, not execute-and-vote on observations.
- **No deferred TRL GKD / SDPO in this phase:** Weight training remains local trajectory BC via `scripts/training/train_distilled_adapter.py --trajectory_mode True`. Generalized Knowledge Distillation (GKD) and SDPO remain deferred until a separate TRL/peft version pinning PR lands.
- **No closed-frontier teacher APIs:** All teacher traces must originate from local open-weight models (MIT / Apache-2.0, e.g., Qwen / DeepSeek-R1 distills) to preserve repository license hygiene.
- **No SageMaker tool loop expansion:** SageMaker `predict_fn` stays single-shot per ADR 0006. Tool loops execute locally via `scripts/harness/run_agent.py`.
- **No unbounded tool registries or MCP autotools:** Tool registry remains frozen and human-reviewed; no dynamic MCP autotool loading.

## Why Yes (Grounded Repo Alignment)

| Existing Repo Component | Grounded Symbolic KD Fit |
| --- | --- |
| Local trajectory BC (`train_distilled_adapter.py`) | Student remains in 0.5B–3B parameter regime; symbolic layer compensates for parameter reduction |
| Harness `H` / Policy $R_\delta$ (ADR 0006) | Policy proposes candidate solution/rules; harness disposes via deterministic solver tool |
| Outcome filter & DualDistill (`compose_dualdistill.py`) | Grounded on reference `expected` and `answers_match`; critic rejection codes hook directly into trajectory ingest |
| Single-pass AQA Gate (`run_aqa_gate.py`) | Provides existing baseline CLI infrastructure to be upgraded into a true Pass@K hard-slice gate |
| Frozen `TOOL_REGISTRY` (`enhanced_system/harness/tools/`) | Plug point for `sqe_constraint_solver` without introducing subprocess execution or security vulnerabilities |

## Phased Execution Plan

1. **Phase 0 (Prerequisite Gate):** Build hard/OOD evaluation set and Pass@K gate CLI (`run_pass_at_k.py` or enhanced `run_aqa_gate.py`). Log critic rejection reason codes during collect.
2. **Phase 1 (Harness Disposition Tool):** Specify and implement `sqe_constraint_solver` in `TOOL_REGISTRY` with unit tests for sound constraint satisfaction and fail-closed OOD rejection.
3. **Phase 2 (Trajectory Collect & Compose):** Collect teacher traces with verified constraint rationale, filter through the critic logging reject codes, and compose dual-teacher datasets.
4. **Phase 3 (Student Adaptation):** Train 0.5B–3B LoRA student on verified trajectories using local trajectory BC (`--trajectory_mode True`). Evaluate under harness $H$ against the Pass@K gate.

## Hand-off

- **Board Verdict:** **HOLD** (Fail closed for Spec Writer until Kill Criteria 1, 2, and 3 are delivered as tangible artifacts).
- **Next Step:** Researcher / Eval Owner to draft the Pass@K hard slice dataset specification and `sqe_constraint_solver` tool interface contract.
