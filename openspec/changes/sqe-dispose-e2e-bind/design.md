# Design: sqe-dispose-e2e-bind

Cites: `docs/plans/symbolic-kd/architecture.md` Section 11 and Section 11.7 (Architect PR https://github.com/ianshank/Distilled_Agents/pull/30).

## 1. System Overview

```
Golden row (configs/golden_sets/sqe_hard_ood.jsonl)
  -> scripts/harness/run_pass_at_k.py (sole Chen Pass@K CLI)
       -> AgentRuntime (H) initialized with spec "sqe_dispose"
            -> ModelBackend / scripted mock proposes action
            -> parse_action enforces allowlist [sqe_constraint_solver, final_answer]
            -> Step 1: TOOL_REGISTRY["sqe_constraint_solver"].run(payload)
                 returns SAT (topological order / assignment) or UNSAT / reject_code
            -> Step 2: TOOL_REGISTRY["final_answer"].run(answer)
                 returns exact expected answer or canonical BLOCKED:<CODE>
       -> Gate asserts tool-call compliance:
            - sqe_constraint_solver MUST precede final_answer
            - Trajectory must satisfy expected_tools
            - Fails gate if final_answer reached without prior solver call
       -> answers_match (exact) on hard/OOD
       -> Chen unbiased pass@k summary emitted to aqa-passk-summary.json
```

## 2. Normative Contracts

### 2.1 Golden Dataset Contract (`configs/golden_sets/sqe_hard_ood.jsonl`)

Every row in `configs/golden_sets/sqe_hard_ood.jsonl` MUST declare:
- `harness_id: "sqe_dispose"`
- `expected_tools: ["sqe_constraint_solver", "final_answer"]` (strictly requiring `sqe_constraint_solver` before `final_answer`)
- `grader: "exact"` and `allow_semantic: false`
- Disjunctive OOD classification: row is OOD iff `slice == "ood"` OR `ood == true`
- OOD `expected`: canonical `BLOCKED:<CODE>` token conforming to `openspec/changes/_shared/blocked-reject-codes.md`
- Bucket minima preserved: DAG >= 6, condition tree >= 6, mixed >= 4, cycle >= 2, unsat >= 2, schema >= 2, theory >= 2 (total >= 24)

### 2.2 Scripted Fixtures Contract (`tests/fixtures/mock_responses_sqe_passk.json`)

For each problem ID in `sqe_hard_ood.jsonl`:
1. The mock response sequence must first invoke `sqe_constraint_solver` with valid arguments matching the task prompt.
2. Following solver execution, the mock response invokes `final_answer` with text matching `expected`.
3. For OOD rows, `final_answer` text must match the canonical refusal token `BLOCKED:<CODE>`. Non-matching non-empty answers increment `ood_synthetic_success_violations`.
4. Fixtures that only call `final_answer` are non-compliant and rejected.

### 2.3 Vacuity Falsifier

Any fixture or trajectory that reaches `final_answer` without a prior invocation of `sqe_constraint_solver` MUST fail `make aqa-gate-passk`, even if `answers_match` evaluates to true and `ood_synthetic_success_violations` equals 0. This enforces the architectural invariant that the cognitive policy proposes and the disposition harness disposes.

### 2.4 Harness Configuration Allowlist

`configs/harnesses/sqe_dispose.yaml` and its packaged duplicate `enhanced_system/config/harnesses/sqe_dispose.yaml` MUST restrict `action.tool_ids` to exactly:
```yaml
action:
  tool_ids:
    - sqe_constraint_solver
    - final_answer
```
No other tools may be allowlisted in this harness.

### 2.5 Rule Traceability Matrix Binding (`configs/rule_traceability/matrix.yaml`)

Every row ID from `configs/golden_sets/sqe_hard_ood.jsonl` must be tracked in `configs/rule_traceability/matrix.yaml` with:
- `harness_id: "sqe_dispose"`
- `solver_status` in {`fixture`, `active`} (cannot be `none` or `pending` at E1 sign-off)

### 2.6 Sole Pass@K CLI

`scripts/harness/run_pass_at_k.py` and `make aqa-gate-passk` remain the sole Pass@K gate entrypoints. Bolting Pass@K evaluation onto `scripts/harness/run_aqa_gate.py` is forbidden.

## 3. E2E Sequencing (E0-E5)

- **E0 (This OpenSpec change):** Spec Writer pins `sqe_dispose` harness binding, solver-before-answer fixture contract, and vacuity falsifiers.
- **E1 (Pass@K x dispose):** Implementer annotates `sqe_hard_ood.jsonl`, updates mock fixtures to solver->final_answer, verifies matrix status in {fixture, active}, and achieves green `make aqa-gate-passk`. Closes `symbolic-disposition` task 8.
- **E2 (Collect / compose):** Collect/compose with critic enabled against `sqe_dispose` traces; sinks `OUTCOME_MISMATCH` and `DUALDISTILL_DROP_0_0` to `artifacts/critic_rejects.jsonl`.
- **E3 (BC smoke):** Small behavior-cloning / trajectory smoke on dispose harness. GKD enable remains locked until E3 is green. Job and corpus naming deferred.
- **E4 (CI harden):** CI assertions ensuring golden rows and fixtures cannot bypass solver.
- **E5 (Archive):** Archive change packages upon full E2E verification.

## 4. Byte-Stable Identifiers

- Tool ID: `sqe_constraint_solver`
- Harness ID: `sqe_dispose`
- Artifact name: `aqa-passk-summary.json`
- Shared contract path: `openspec/changes/_shared/blocked-reject-codes.md`
