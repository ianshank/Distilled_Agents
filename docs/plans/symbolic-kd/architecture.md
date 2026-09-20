# Architecture - Phase 0 gate build (symbolic KD)

**Repo:** [ianshank/Distilled_Agents](https://github.com/ianshank/Distilled_Agents)  
**Plan:** `docs/plans/symbolic-kd/PLAN.md`  
**Baseline:** `main` after intake PR #18 and OpenSpec PR #20 (Critic PASS on specify)  
**Audience:** Implementer (build), Tester (gates), Conductor (unlock decisions)  
**Status:** Phase 0 kill criteria CLEARED on main (run 35518614849). Phase 2-3 (training / GKD / Serve / Edge-AI / INV-16) stay locked. E2E Pass@K x sqe_dispose addendum defined in Section 11.

---

## 0. Precursor PR status and Implementer sequencing

### PR #21 - non-compliant precursor (do not merge as Phase 0 gate)

[PR #21](https://github.com/ianshank/Distilled_Agents/pull/21) (`cursor/golden-passk-aqa-45f9`) is an early Pass@K attempt. It is **CONFLICTING** with `main` and **does not** satisfy Critic-cleared OpenSpec on `main` (PR #20). Treat it as a precursor to **rebase-and-rewrite or replace**, not as the Phase 0 gate.

| OpenSpec pin (PR #20 / `_shared`) | PR #21 actual | Verdict |
| --- | --- | --- |
| Sole Pass@K CLI `scripts/harness/run_pass_at_k.py` + `make aqa-gate-passk` | Extends `scripts/harness/run_aqa_gate.py` / `eval_harness.py` with `--pass-k` | Non-compliant |
| Chen unbiased estimator, gate `n=5`, `k=3`, report k=1 and k=3 | Multi-trial rates via `MANGOMAS_EVAL_PASS_K` default 5; no dedicated Chen CLI | Non-compliant |
| Golden `configs/golden_sets/sqe_hard_ood.jsonl` with bucket minima (DAG>=6 / tree>=6 / mixed>=4 / OOD cycle>=2 / unsat>=2 / schema-syntax>=2 / unknown-theory>=2, total>=24) | `configs/golden_sets/hard_sdlc.jsonl` (4 hard rows, no OOD buckets) | Non-compliant |
| OOD iff `slice=="ood"` OR `ood==true`; OOD success = exact `BLOCKED:<CODE>` only | Hard slice only (`core`\|`hard`); no OOD / `BLOCKED` synthetic-success gate | Non-compliant |
| `critic_reject_code` + `artifacts/critic_rejects.jsonl` (`OUTCOME_MISMATCH`, `DUALDISTILL_DROP_0_0`) | Absent | Missing |
| Tool `sqe_constraint_solver` + harness `configs/harnesses/sqe_dispose.yaml` | Absent | Missing |
| Shared codes path `openspec/changes/_shared/blocked-reject-codes.md` | Not wired | Missing |

**Disposition:** close #21 after a compliant Implementer PR lands, or strip #21 to reusable scraps (EchoBackend `reset()`, prompt_render sync test, golden schema typing) and re-land under the contracts below. Do **not** merge #21 onto `main` as-is.

### Implementer PR sequence (Phase 0 only)

Land in order. Each PR must stay within Phase 0; no training / GKD / Serve / Edge-AI / INV-16.

| Order | Working title | OpenSpec package(s) | Must land | Clears kill artifact |
| --- | --- | --- | --- | --- |
| **I1** | Pass@K hard/OOD gate | `golden-passk-aqa` | `scripts/harness/run_pass_at_k.py`, `make aqa-gate-passk`, CI job, `configs/golden_sets/sqe_hard_ood.jsonl` (+ bucket validator), docstring-only fix to `run_aqa_gate.py`, scripted fixtures `tests/fixtures/mock_responses_sqe_passk.json`, summary JSON with `pass_at_k`, `exact_only`, `semantic_counted`, `ood_synthetic_success_violations` | Kill #1: upload `aqa-passk-summary.json` from green `aqa-gate-passk` |
| **I2** | Critic reject telemetry | `critic-cascade` | `critic_reject_code` on collect/compose drops; sink `artifacts/critic_rejects.jsonl`; codes `OUTCOME_MISMATCH` + existing DualDistill `(0,0)` only as `DUALDISTILL_DROP_0_0` (no second drop rule); `MANGOMAS_CRITIC_ENABLED` | Kill #2: upload `critic_rejects.jsonl` proving both codes |
| **I3** | Symbolic dispose tool | `symbolic-disposition` | Register `sqe_constraint_solver` (pure-Python DAG topo + boolean trees; no Z3/clingo/`exec`/`eval`/`subprocess`); `configs/harnesses/sqe_dispose.yaml` allowlisting solver + `final_answer`; map rejects to `BLOCKED:<CODE>` per `_shared/blocked-reject-codes.md` | Kill #3: solver unit report + OOD `ood_synthetic_success_violations==0` on same green gate run |

**Hard dependencies**

1. I3 must not be marked done until I1 hard-slice Pass@K job exists and is green on scripted fixtures.
2. I2 may land after I1 metrics exist; it must not wait on I3.
3. Reuse from #21 only after re-homing onto I1 contracts (no Pass@K inside `run_aqa_gate.py`).
4. Kill criteria stay **HOLD** until I1-I3 CI artifacts are green on an Implementer PR (docs-only merges do not clear them). Now CLEARED on `main` via run `35518614849`.
5. Phase 2-3 remain locked regardless of I1-I3 green.

---

## 1. Scope fence

### In scope (Phase 0 gate build)

Implement the three OpenSpec changes pinned on PR #20, only as far as CI can prove the intake kill criteria:

| Change id | Phase 0 deliverable |
| --- | --- |
| `golden-passk-aqa` | Hard/OOD golden set, Chen Pass@K CLI, Makefile + CI job, exact-only hard/OOD success, OOD synthetic-success prohibition |
| `critic-cascade` | `critic_reject_code` telemetry + `artifacts/critic_rejects.jsonl` on collect/compose drops (no second DualDistill drop rule) |
| `symbolic-disposition` | Frozen tool `sqe_constraint_solver`, harness `configs/harnesses/sqe_dispose.yaml`, fail-closed `BLOCKED:<CODE>` mapping; **not done** until hard-slice Pass@K gate is green on scripted fixtures |

Shared contract (path byte-stable; do not rename or move):

- `openspec/changes/_shared/blocked-reject-codes.md`

### Out of scope (do not unlock)

- TRL / GKD / on-policy KD (`trl-gkd-usage` / PLAN Phase 4)
- Student weight training, teacher Hub pulls, SageMaker trajectory jobs
- Serve / endpoint tool loops, Edge-AI, INV-16, Mango cross-repo claims
- Z3 / clingo / `exec` / `eval` / `subprocess` solver backends in default CI
- Extending `scripts/harness/run_aqa_gate.py` into a Pass@K dual-CLI

### Compatibility rules

- Additive registry and harness YAML only; keep `sqe_validate` checklist-only.
- Settings via env / settings objects already used by the repo (`MANGOMAS_*`); no new call-site literals for Pass@K knobs.
- No hardcoded secrets, model IDs, or Hub revision pins in Phase 0 gate code paths.
- Prefer reuse of `enhanced_system.harness.score.answers_match`, existing `TOOL_REGISTRY`, and ADR 0006 harness-policy pair.

---

## 2. Runtime component map

```
Golden row (configs/golden_sets/sqe_hard_ood.jsonl)
  -> scripts/harness/run_pass_at_k.py   # sole Pass@K entrypoint
       -> AgentRuntime (H) + scripted/mock backend (CI)
            -> parse_action(allowlist from harness YAML)
            -> TOOL_REGISTRY[tool_id].run(args)
                 sqe_constraint_solver  # dispose (symbolic-disposition)
                 final_answer           # terminal
            -> observation / fault token
       -> answers_match (exact) on hard/OOD
       -> Chen unbiased pass@k summary JSON + exit code

Collect / compose (critic-cascade; config-gated)
  collect_trajectories.py
    -> outcome filter -> critic_reject_code=OUTCOME_MISMATCH
  compose_dualdistill.py / compose_pair
    -> existing (0,0) drop only -> critic_reject_code=DUALDISTILL_DROP_0_0
  -> append artifacts/critic_rejects.jsonl
```

ADR posture: student / teacher / mock backend **proposes**; harness H **disposes** via frozen tools. Symbolic solver is disposition, not a training target in Phase 0.

---

## 3. Delivery-board agent / tool map

| Agent | Phase 0 role | Allowed surfaces | Forbidden |
| --- | --- | --- | --- |
| Architect (this doc) | Contracts, maps, CI artifact definitions | `docs/plans/symbolic-kd/architecture.md`, OpenSpec cross-links | Unlocking Phase 2-3; inventing second drop rules |
| Implementer | Code + fixtures per OpenSpec tasks | `enhanced_system/harness/**`, `scripts/harness/run_pass_at_k.py`, golden JSONL, harness YAML, unit tests, Makefile, `.github/workflows/ci.yml`, ADR 0007 (or 0006 addendum) at land | GKD trainers; Z3/clingo in default deps; editing shared codes path; Pass@K inside `run_aqa_gate.py` |
| Tester | Prove gates and kill-artifact schemas | pytest, `make aqa-gate-passk`, CI logs/artifacts | Lowering thresholds to greenwash; counting `semantic_match` on hard/OOD |
| Critic | Spec/design review only (already PASS on #20) | OpenSpec diffs | Implementer work |
| Conductor | Unlock when kill artifacts clear | PR checks + artifact URLs | Treating docs-only PR as kill clear |
| Experimenter / Releaser / Edge agents | Idle for Phase 0 | - | Training, Serve, Edge-AI, INV-16 |

Implementer command allowlist (local/CI): `pytest`, `ruff`, `mypy`, `make aqa-gate-passk`, `make aqa-gate` (single-pass regression only), `make validate` subset already on trunk. No network model calls in the Pass@K CI job.

---

## 4. Tool and harness allowlists

### `TOOL_REGISTRY` (Phase 0 delta)

| Tool id | Status | Engine constraints |
| --- | --- | --- |
| `sqe_constraint_solver` | **Add** (primary dispose id; freeze spelling) | Pure-Python DAG topo order + boolean condition trees only. No Z3, clingo, network, shell, `exec`, `eval`, or `subprocess`. Out-of-fragment ops -> `UNSUPPORTED_THEORY`. |
| `final_answer` | Existing | Terminal answer tool |
| `sqe_checklist` | Existing | Remains validate-only; do not conflate with solver |

Optional Z3/clingo extras (if ever added later) must stay non-default and emit `UNSUPPORTED_THEORY` when absent. Phase 0 CI must not require them.

### Harness YAML

| File | Allowlisted tool ids | Notes |
| --- | --- | --- |
| `configs/harnesses/sqe_dispose.yaml` (+ packaged copy under `enhanced_system/config/harnesses/`) | `sqe_constraint_solver`, `final_answer` | New; Phase 0 |
| `configs/harnesses/sqe_validate.yaml` | `sqe_checklist`, `final_answer` | Unchanged checklist-only |

`dispatch.parse_action` continues to enforce the harness allowlist. Unknown tool ids remain reject / KeyError paths already covered by harness tests.

### Solver I/O (normative summary)

- Input: `mode` (`check` \| `solve`), `graph`, `constraints`, optional `assignment`, `limits` (`max_nodes` 64, `max_edges` 256, `max_constraints` 128).
- Output JSON string: `ok`, `status` (`SAT` \| `UNSAT`), `order`, `assignment`, `reject_code`, `details`.
- SAT: lexicographically least valid topological order; `reject_code` null.
- Cycle / unsat predicates: return `ok: false` (never raise) with `CYCLE_DETECTED` or `UNSAT`.
- Schema / syntax / unsupported / resource: raise `ValueError("<CODE>: ...")` -> runtime `tool_error` -> policy `final_answer` = `BLOCKED:<CODE>`.

---

## 5. Eval and data contracts

### Golden set

- Path: `configs/golden_sets/sqe_hard_ood.jsonl` (total rows >= 24).
- Per-bucket minima: hard DAG >= 6; hard condition tree >= 6; hard mixed >= 4; OOD cycle >= 2; OOD unsat >= 2; OOD schema/syntax >= 2; OOD unknown-theory >= 2.
- Hard/OOD rows: `grader: "exact"`, `allow_semantic: false`.
- Keep `configs/golden_sets/core_sdlc.jsonl` for easy single-pass regression via existing `make aqa-gate`.

### OOD detector (single rule)

A row is OOD **iff** `slice == "ood"` **OR** `ood == true`. Tests and gates must treat both fields as equivalent.

### Success rules

| Slice | Success predicate |
| --- | --- |
| hard / ood | Exact `answers_match` only; `security_failed` / security fault must be false. No `semantic_match`. No unlabeled non-truncated success. |
| OOD refusal rows | `final_answer` must equal canonical `BLOCKED:<CODE>` from the shared table. Any non-empty non-matching answer increments `ood_synthetic_success_violations` and fails the gate. |
| core (existing gate) | Unchanged single-pass semantics; semantic only if explicitly allowed on the row. |

### Chen Pass@K

- Estimator: Chen et al. unbiased pass@k (reject biased `1-(1-p)^k`).
- Defaults: `n=5` samples/problem; gate on `k=3`; report `pass_at_1` and `pass_at_k` for `k=3`.
- Env knobs (eval only, no call-site literals): `MANGOMAS_EVAL_PASS_K` (default 3), `MANGOMAS_EVAL_PASS_N` (default 5), `MANGOMAS_EVAL_PASS_K_TEMPERATURE` (default 0.8 when k>1).
- Sole CLI: `scripts/harness/run_pass_at_k.py`.
- Makefile: `aqa-gate-passk` -> that CLI only.
- `run_aqa_gate.py`: docstring correction only (single-pass `pass_rate`); no Pass@K mode.

### Critic telemetry

- Field: `critic_reject_code` (explicit per drop, not counters alone).
- Sink: `artifacts/critic_rejects.jsonl` (or `--reject-log`).
- Codes for Phase 0 wiring:
  - `OUTCOME_MISMATCH` - collect outcome filter when `final_answer` fails `answers_match`.
  - `DUALDISTILL_DROP_0_0` - existing DualDistill (0,0) drop only; **do not** invent a second drop rule.
  - Teacher-rule codes (same enum as solver): `CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`.
- Gate: `MANGOMAS_CRITIC_ENABLED` (default false until unit tests land; then true in CI paths that assert telemetry).

### Shared reject / refusal table

Authoritative rows live only in `openspec/changes/_shared/blocked-reject-codes.md`:

| Reject code | Tool behavior | Canonical refusal |
| --- | --- | --- |
| `CYCLE_DETECTED` | Return UNSAT | `BLOCKED:CYCLE_DETECTED` |
| `UNSAT` | Return UNSAT | `BLOCKED:UNSAT` |
| `SCHEMA_VIOLATION` | Raise `ValueError` | `BLOCKED:SCHEMA_VIOLATION` |
| `SYNTAX_INVALID` | Raise `ValueError` | `BLOCKED:SYNTAX_INVALID` |
| `UNSUPPORTED_THEORY` | Raise `ValueError` | `BLOCKED:UNSUPPORTED_THEORY` |
| `RESOURCE_LIMIT` | Raise `ValueError` | `BLOCKED:RESOURCE_LIMIT` |

---

## 6. CI gate wiring

### Jobs / targets

| Surface | Command / job | Must fail when |
| --- | --- | --- |
| Makefile | `make aqa-gate-passk` | `pass_at_k["3"]` below threshold **or** `ood_synthetic_success_violations > 0` |
| Makefile | `make aqa-gate` | Existing single-pass regression (unchanged role) |
| GitHub Actions | New/extended job on PR/push invoking `aqa-gate-passk` with scripted fixtures | Same as Makefile; must be deterministic (no Hub/network) |
| Unit tests | OOD detector equivalence; Chen aggregator; solver cycle/unsat/schema; critic code emission; prompt_render sync | Spec falsifiers |
| Matrix check | `configs/rule_traceability/matrix.yaml` covers every hard golden id | Missing hard row |

Phase 0 Pass@K threshold: **1.0** against `tests/fixtures/mock_responses_sqe_passk.json` (name may match OpenSpec `tests/fixtures/mock_responses_sqe_passk.json`). Neural threshold deferred until a student exists (out of Phase 0 unlock).

### Summary JSON (Pass@K CLI)

Emit at least:

- `pass_at_k` with keys `"1"` and `"3"`
- `exact_only: true`
- `semantic_counted: false`
- `ood_synthetic_success_violations` (int)
- `n`, `k`, golden path, harness id

Upload this JSON (and raw logs) as a CI artifact named e.g. `aqa-passk-summary.json`.

### Sequencing inside Phase 0

1. Land golden schema + `sqe_hard_ood.jsonl` + Chen CLI + CI job (gate may start red on incomplete fixtures, then green on scripted mocks).
2. Land critic reject telemetry (does not wait on solver).
3. Land `sqe_constraint_solver` + `sqe_dispose` harness **only after** hard-slice Pass@K job exists; mark solver tasks done only when scripted hard-slice gate is green.

---

## 7. Failure modes

| Failure mode | Detection | Disposition |
| --- | --- | --- |
| Biased pass@k or single-pass labeled as Pass@K | Review + unit tests against Chen formula; docstring lint on `run_aqa_gate.py` | Block merge |
| Dual CLI / Pass@K bolted onto `run_aqa_gate.py` | Diff allowlist; architecture forbid | Block merge |
| Semantic or unlabeled success on hard/OOD | Gate counters / unit tests | Fail `aqa-gate-passk` |
| OOD synthetic success (non-empty non-`BLOCKED` answer) | `ood_synthetic_success_violations` | Fail gate (nonzero exit) |
| OOD detector ignores `slice` or `ood` | Equivalence tests | Fail tests |
| Solver returns SAT / fabricated `order` on cycle or unsat | Unit tests | Fail tests |
| Z3/clingo/`exec`/`eval`/`subprocess` in default solver path | Bandit + review + import tests | Fail security / unit |
| Second DualDistill drop rule | Code review + test that only (0,0) logs `DUALDISTILL_DROP_0_0` | Block merge |
| Missing `critic_reject_code` on drop | Unit tests | Fail tests |
| Shared codes file moved/renamed | Path byte-stability check in CI or review checklist | Block merge |
| Solver marked done without hard Pass@K gate | Conductor checklist + OpenSpec task gate | Keep HOLD on solver done |
| Vacuous gate (cannot fail) | Fixtures include deliberate failing OOD synthetic case in unit tests | Fail tests if vacuity detected |
| prompt_render drift | Byte-equality test between harness and distill copies | Fail CI |
| Scope creep into GKD/Serve/Edge | PR labels + Conductor fence | Reject from Phase 0 PR |

Runtime fault token: add `blocked` on trajectories that terminate in canonical refusal (additive alongside existing `parse_error` / `tool_error` / `loop`; do not remove existing fault enums).

---

## 8. Kill criteria - CLEARED on main

Intake kill criteria were **HOLD** until Implementer CI artifacts landed, and are now **CLEARED** on `main` for I1-I3 citing GitHub Actions run `35518614849` and artifacts `aqa-passk-summary.json` + `critic-rejects`. Docs-only merges do **not** clear gates. **E2E Pass@K x dispose proof is a separate bar** - see Section 11; green `aqa-gate-passk` on Echo/`final_answer`-only fixtures does not satisfy Section 11.

| Kill criterion (intake) | Artifact that clears it | Clear condition | Status |
| --- | --- | --- | --- |
| 1. CI Pass@K on extended hard/OOD with exact match | CI artifact `aqa-passk-summary.json` + green `aqa-gate-passk` job on the implementing PR (and main after merge) | Job uses `scripts/harness/run_pass_at_k.py`; `exact_only: true`; `semantic_counted: false`; `pass_at_k["3"]` meets Phase 0 threshold on scripted fixtures; golden file meets per-bucket minima; workflow is mandatory on PR/push | CLEARED (run 35518614849, `aqa-passk-summary.json`) |
| 2. Critic rejection telemetry with explicit reason codes | CI/test artifact `critic_rejects.jsonl` (or uploaded copy) from unit/integration run with critic enabled | Contains at least one `OUTCOME_MISMATCH` and one `DUALDISTILL_DROP_0_0` record with field `critic_reject_code`; no alternate (0,0) drop path | CLEARED (run 35518614849, `critic-rejects`) |
| 3. Deterministic fail-closed OOD (BLOCKED / tool_error, zero synthetic success) | Pass@K summary + solver unit-test report on same PR | `ood_synthetic_success_violations == 0` on the green gate run; tests prove cycle/unsat/schema paths yield `BLOCKED:<CODE>` or `tool_error` and never SAT confabulation; `sqe_dispose` allowlist proven | CLEARED (run 35518614849) |

Conductor unlock rule:

- Phase 0 I1-I3 kill criteria are CLEARED on `main` citing GitHub Actions run `35518614849` and artifacts `aqa-passk-summary.json` + `critic-rejects`.
- **E2E unlock (Section 11):** Pass@K must prove `H(sqe_dispose)` + `sqe_constraint_solver` before claiming dispose-path eval; follow E0-E5.
- GKD enable waits on Section 11 E3 BC smoke. Serve / Edge-AI / INV-16 stay locked.

---

## 9. Implementer handoff checklist

Follow **I1 -> I2 -> I3** in Section 0. Do not merge PR #21 as the gate.

1. **I1:** Add `configs/golden_sets/sqe_hard_ood.jsonl` + bucket-minima validator; add `scripts/harness/run_pass_at_k.py` + `make aqa-gate-passk` + CI job + mock fixtures; fix `run_aqa_gate.py` docstring only (no Pass@K mode).
2. **I2:** Wire `critic_reject_code` + `artifacts/critic_rejects.jsonl` in collect/compose (`OUTCOME_MISMATCH`, `DUALDISTILL_DROP_0_0` only for existing (0,0) drop).
3. **I3:** Register `sqe_constraint_solver`; add `configs/harnesses/sqe_dispose.yaml`; map rejects to `BLOCKED:<CODE>` per shared table; mark solver done only when I1 hard-slice gate is green.
4. Upload kill artifacts (`aqa-passk-summary.json`, `critic_rejects.jsonl`, solver/OOD evidence); leave kill criteria HOLD until all three are green. Cleared on main.
5. Land canonical ADR 0007 (`docs/adr/0007-symbolic-dispose-tools.md`) describing dispose tools and fail-closed OOD; claim number at land.
6. Leave `trl-gkd-usage`, Serve, Edge-AI, and INV-16 untouched.

---

## 10. References

- PR #20: OpenSpec Phase 0 kill-contract pins (Critic PASS specify)
- `openspec/changes/archive/golden-passk-aqa/`
- `openspec/changes/archive/critic-cascade/`
- `openspec/changes/archive/symbolic-disposition/`
- `openspec/changes/_shared/blocked-reject-codes.md`
- `docs/plans/symbolic-kd/PLAN.md`
- ADR 0006: runtime harness-policy pair
- ADR 0007: canonical `docs/adr/0007-symbolic-dispose-tools.md`

---

## 11. E2E addendum - Pass@K x sqe_dispose (PR2)

**Status:** Phase 0 kill artifacts are CLEARED on `main`, but Pass@K still does **not** prove the dispose path. Current `configs/golden_sets/sqe_hard_ood.jsonl` rows omit `harness_id` / `expected_tools`, and `tests/fixtures/mock_responses_sqe_passk.json` emits `final_answer` only (Echo shortcut). That is insufficient for `... -> H(sqe_dispose) -> sqe_constraint_solver -> aqa-gate-passk`.

**PR #28** (`trl-gkd-usage` library) stays off the E2E critical path until BC smoke is green. No Edge-AI / INV-16 in this track.

### 11.1 Golden contract

Every row in `configs/golden_sets/sqe_hard_ood.jsonl` MUST set:

| Field | Required value |
| --- | --- |
| `harness_id` | `sqe_dispose` |
| `expected_tools` | includes `sqe_constraint_solver` then `final_answer` (solver call before terminal answer) |
| `grader` / `allow_semantic` | exact-only on hard/OOD (`allow_semantic: false`) |
| `slice` / `ood` | OOD iff `slice == "ood"` OR `ood == true` |
| `expected` (OOD) | canonical `BLOCKED:<CODE>` from `openspec/changes/_shared/blocked-reject-codes.md` |

Bucket minima from Section 5 remain normative (DAG>=6 / tree>=6 / mixed>=4 / OOD cycle>=2 / unsat>=2 / schema-syntax>=2 / unknown-theory>=2; total>=24).

`configs/golden_sets/hard_sdlc.jsonl` and `base_react` / `qc_constraints` rows are **out of band** for `aqa-gate-passk`. They must not be the Chen gate corpus.

### 11.2 Scripted fixtures

`tests/fixtures/mock_responses_sqe_passk.json` (or successor keyed by golden `id`) MUST, for each `sqe_hard_ood` row:

1. Emit at least one `sqe_constraint_solver` tool call with args matching the prompt fragment (graph / constraints).
2. Then emit `final_answer` whose text equals the golden `expected`.
3. On OOD rows, `final_answer` MUST be exactly `BLOCKED:<CODE>` from the shared table (`CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`). Non-matching non-empty answers remain `ood_synthetic_success_violations`.

Fixtures that only call `final_answer` are non-compliant for E2E Pass@K x dispose even if exact-match scores stay 1.0.

**Falsifier (MUST fail `aqa-gate-passk`):** any scripted fixture or collected trace that reaches `final_answer` for a `sqe_hard_ood` row **without** a prior `sqe_constraint_solver` invocation is a gate failure (dispose-path vacuity), even when `answers_match` would pass and even when `ood_synthetic_success_violations == 0`. Implementer SHALL add an explicit unit/CI check for this vacuity case.

**OpenSpec close-out:** completing E1 (green Chen gate on solver-before-answer fixtures bound to `sqe_hard_ood`) is what closes `openspec/changes/archive/symbolic-disposition/tasks.md` item **8** (solver gated against hard-slice Pass@K). Do not mark task 8 done on Echo/`final_answer`-only fixtures.

### 11.3 Matrix and Makefile binding

| Surface | Binding |
| --- | --- |
| Makefile `aqa-gate-passk` | Remains the **sole** Chen Pass@K gate; must invoke `scripts/harness/run_pass_at_k.py` on `configs/golden_sets/sqe_hard_ood.jsonl` with scripted dispose fixtures |
| `run_aqa_gate.py` / `make aqa-gate` | Single-pass regression only; no Pass@K mode |
| `configs/rule_traceability/matrix.yaml` | Every `sqe_hard_ood` hard/OOD `id` MUST appear with `harness_id: sqe_dispose` and `solver_status` in {`fixture`,`active`} (not `none`) |
| CI job | Mandatory `make aqa-gate-passk` uploading `aqa-passk-summary.json` with `exact_only: true`, `semantic_counted: false`, `ood_synthetic_success_violations: 0` |

### 11.4 E2E PR sequence

Land in order. Docs-only does not clear E2E proof.

| Order | PR theme | Deliverable | Gate |
| --- | --- | --- | --- |
| **E0** | Docs / OpenSpec delta | This addendum + Spec Writer change package (`openspec/changes/sqe-dispose-e2e-bind/`) pinning harness_id/fixture contracts | Review only |
| **E1** | Pass@K x dispose | Annotate `sqe_hard_ood.jsonl`; rewrite mock fixtures to solver->`final_answer`; matrix rows for all hard/OOD ids; keep `aqa-gate-passk` sole Chen entry | Green `aqa-gate-passk` proving solver tool calls on scripted path |
| **E2** | Collect / compose | Critic-on collect/compose against `sqe_dispose` traces; reject sink still `artifacts/critic_rejects.jsonl` | Unit/integration with `OUTCOME_MISMATCH` / `DUALDISTILL_DROP_0_0` |
| **E3** | BC smoke | Small behavior-cloning / trajectory smoke on dispose harness (no GKD enable) | Smoke job green |
| **E4** | CI harden | Fail CI if golden lacks `harness_id: sqe_dispose` or fixtures skip solver | Required checks |
| **E5** | OpenSpec archive | Archive completed change folders after E1-E4 green | Archive hygiene |

**Locks**

- Do **not** enable GKD / on-policy KD (`trl-gkd-usage` / PR #28 path) until **E3 BC smoke** is green.
- Do **not** open Edge-AI / INV-16 / Serve tool-loop work under this addendum.
- `qc_constraints` / `constraint_check` remain a separate vertical; they do not substitute for `sqe_constraint_solver` on the Chen gate.

### 11.5 Failure modes (E2E-specific)

| Failure | Detection | Disposition |
| --- | --- | --- |
| Golden row missing `harness_id: sqe_dispose` | Schema / matrix check | Fail E1/E4 |
| Fixture `final_answer`-only (no solver call) | Fixture linter or trajectory assert on `expected_tools` | Fail E1 |
| OOD non-`BLOCKED` answer | `ood_synthetic_success_violations` | Fail `aqa-gate-passk` |
| Chen gate pointed at `hard_sdlc` / `base_react` | Makefile / CI review | Block merge |
| GKD wired before BC smoke | PR scope / Conductor fence | Reject |
| Edge / INV-16 scope creep | PR labels | Reject |

### 11.6 Implementer note

Prefer amending `sqe_hard_ood.jsonl` + `mock_responses_sqe_passk.json` in place over adding a parallel golden. Keep tool id spelling `sqe_constraint_solver` and harness id `sqe_dispose` byte-stable with ADR 0007 / OpenSpec `symbolic-disposition`.

Coordinate OpenSpec deltas with Spec Writer under existing packages (`golden-passk-aqa`, `symbolic-disposition`) or a thin follow-on change; do not invent a second Pass@K CLI.

### 11.7 Pins Spec Writer / Implementer must cite

| Pin | Canonical value |
| --- | --- |
| Architecture section | `docs/plans/symbolic-kd/architecture.md` Section 11 (this addendum); PR https://github.com/ianshank/Distilled_Agents/pull/30 |
| OpenSpec delta folder | `openspec/changes/sqe-dispose-e2e-bind/` (Spec Writer) |
| Golden path | `configs/golden_sets/sqe_hard_ood.jsonl` (not `hard_sdlc.jsonl`) |
| Field name | `harness_id: sqe_dispose` (not a parallel alias) |
| Harness allowlist | `configs/harnesses/sqe_dispose.yaml` (+ packaged copy): tool ids **only** `sqe_constraint_solver`, `final_answer` |
| Shared refusal table | `openspec/changes/_shared/blocked-reject-codes.md` (path byte-stable) |
| Sole Chen CLI / Make | `scripts/harness/run_pass_at_k.py` / `make aqa-gate-passk` |
| Matrix | `configs/rule_traceability/matrix.yaml` - every hard/OOD golden id -> `harness_id: sqe_dispose` |
| ADR 0007 | Canonical `docs/adr/0007-symbolic-dispose-tools.md` (supersedes `0007-symbolic-disposition-fail-closed.md`). Do not invent ADR 0008 for dispose. |
| Task close | `openspec/changes/archive/symbolic-disposition/tasks.md` Section 8 closes only when Section 11 falsifier + `aqa-gate-passk` are green on dispose-bound fixtures |
| Out of scope | GKD enable before E3 BC smoke; Edge-AI; INV-16; Pass@K via `run_aqa_gate.py` |
