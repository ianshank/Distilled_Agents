# Architecture — Phase 0 gate build (symbolic KD)

**Repo:** [ianshank/Distilled_Agents](https://github.com/ianshank/Distilled_Agents)
**Plan:** `docs/plans/symbolic-kd/PLAN.md`
**Baseline:** `main` after intake PR #18 and OpenSpec PR #20 (Critic PASS on specify)
**Audience:** Implementer (build), Tester (gates), Conductor (unlock decisions)
**Status:** Phase 0 design/build may proceed. Phase 2–3 (training / GKD / Serve / Edge-AI / INV-16) stay locked.


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
- Prefer reuse of `enhanced_system.harness.score.answers_match`, existing `TOOL_REGISTRY`, and ADR 0006 harness–policy pair.

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
| Architect (this doc) | Contracts, maps, CI artifact definitions | `docs/plans/symbolic-kd/architecture.md`, OpenSpec cross-links | Unlocking Phase 2–3; inventing second drop rules |
| Implementer | Code + fixtures per OpenSpec tasks | `enhanced_system/harness/**`, `scripts/harness/run_pass_at_k.py`, golden JSONL, harness YAML, unit tests, Makefile, `.github/workflows/ci.yml`, ADR 0007 (or 0006 addendum) at land | GKD trainers; Z3/clingo in default deps; editing shared codes path; Pass@K inside `run_aqa_gate.py` |
| Tester | Prove gates and kill-artifact schemas | pytest, `make aqa-gate-passk`, CI logs/artifacts | Lowering thresholds to greenwash; counting `semantic_match` on hard/OOD |
| Critic | Spec/design review only (already PASS on #20) | OpenSpec diffs | Implementer work |
| Conductor | Unlock when kill artifacts clear | PR checks + artifact URLs | Treating docs-only PR as kill clear |
| Experimenter / Releaser / Edge agents | Idle for Phase 0 | — | Training, Serve, Edge-AI, INV-16 |

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
  - `OUTCOME_MISMATCH` — collect outcome filter when `final_answer` fails `answers_match`.
  - `DUALDISTILL_DROP_0_0` — existing DualDistill (0,0) drop only; **do not** invent a second drop rule.
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

## 8. Kill criteria — HOLD until CI artifacts

Intake kill criteria remain **HOLD** until Implementer produces the artifacts below. Docs-only or OpenSpec-only merges do **not** clear them.

| Kill criterion (intake) | Artifact that clears it | Clear condition |
| --- | --- | --- |
| 1. CI Pass@K on extended hard/OOD with exact match | CI artifact `aqa-passk-summary.json` + green `aqa-gate-passk` job on the implementing PR (and main after merge) | Job uses `scripts/harness/run_pass_at_k.py`; `exact_only: true`; `semantic_counted: false`; `pass_at_k["3"]` meets Phase 0 threshold on scripted fixtures; golden file meets per-bucket minima; workflow is mandatory on PR/push |
| 2. Critic rejection telemetry with explicit reason codes | CI/test artifact `critic_rejects.jsonl` (or uploaded copy) from unit/integration run with critic enabled | Contains at least one `OUTCOME_MISMATCH` and one `DUALDISTILL_DROP_0_0` record with field `critic_reject_code`; no alternate (0,0) drop path |
| 3. Deterministic fail-closed OOD (BLOCKED / tool_error, zero synthetic success) | Pass@K summary + solver unit-test report on same PR | `ood_synthetic_success_violations == 0` on the green gate run; tests prove cycle/unsat/schema paths yield `BLOCKED:<CODE>` or `tool_error` and never SAT confabulation; `sqe_dispose` allowlist proven |

Conductor unlock rule:

- Phase 0 **build** may proceed now under this architecture.
- Kill criteria flip HOLD -> CLEARED only when all three artifact columns are present and green on an Implementer PR.
- Phase 2–3 / GKD / Serve / Edge-AI / INV-16 stay locked regardless of Phase 0 green.

---

## 9. Implementer handoff checklist

1. Add `configs/golden_sets/sqe_hard_ood.jsonl` + bucket-minima validator test.
2. Add `scripts/harness/run_pass_at_k.py` + `make aqa-gate-passk` + CI job + mock fixtures; fix `run_aqa_gate.py` docstring only.
3. Wire `critic_reject_code` + `artifacts/critic_rejects.jsonl` in collect/compose.
4. Register `sqe_constraint_solver`; add `configs/harnesses/sqe_dispose.yaml`; map rejects to `BLOCKED:<CODE>` per shared table.
5. Upload kill artifacts; do not claim solver “done” until hard-slice Pass@K is green.
6. Land ADR 0007 (or 0006 addendum) describing dispose tools and fail-closed OOD; claim number at land.
7. Leave `trl-gkd-usage`, Serve, Edge-AI, and INV-16 untouched.

---

## 10. References

- PR #20: OpenSpec Phase 0 kill-contract pins (Critic PASS specify)
- `openspec/changes/golden-passk-aqa/`
- `openspec/changes/critic-cascade/`
- `openspec/changes/symbolic-disposition/`
- `openspec/changes/_shared/blocked-reject-codes.md`
- `docs/plans/symbolic-kd/PLAN.md`
- ADR 0006: runtime harness–policy pair
