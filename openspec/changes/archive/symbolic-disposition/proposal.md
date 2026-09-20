# Proposal: symbolic-disposition

**Status:** archived  
**Date:** 2026-09-20  
**Depends on:** `golden-passk-aqa` hard-slice gate exists (scripted CI green); `critic-cascade` metrics preferred
**Plan:** Phase 3  
**Why:** Consensus reframes "neurosymbolic student" as symbolic KD + small
proposer. Repo disposition is schema-only; tools are checklist stubs.

## Scope

- Add primary frozen tool id `sqe_constraint_solver` (distinct from `sqe_checklist`).
  Phase default engine is pure-Python DAG topological ordering plus boolean constraint
  trees only. No Z3 or clingo required for CI. Raise `UNSUPPORTED_THEORY` for out-of-fragment
  operations; no `exec`, `eval`, or subprocess execution.
- Dedicated harness config `configs/harnesses/sqe_dispose.yaml` allowlisting
  `sqe_constraint_solver` and `final_answer`. Keep `sqe_validate.yaml` checklist-only.
- Standardized reject codes: `CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`,
  `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT` conforming to
  `openspec/changes/_shared/blocked-reject-codes.md`, with fail-closed mapping to
  `BLOCKED:<CODE>` or runtime `tool_error`.
- ADR 0006 addendum: cognitive proposes; tool body disposes; OOD -> fail-closed
  refusal via `final_answer` policy (`BLOCKED:<CODE>`), never confabulate SAT.
- Build-time rule extract stub CLI (teacher -> matrix row) with mend hooks
  deferred to follow-on if needed.
- Teacher license allowlist documented (Apache-2.0 / MIT).

## Sequencing and Gate Dependency

- Depends on: `golden-passk-aqa` hard-slice gate exists. Solver MUST NOT be marked
  done without the hard-slice gate in place.
- Soften optional z3 language: extras only behind `UNSUPPORTED_THEORY` or separate
  follow-on; default CI stays pure Python.

## Non-goals

- General multi-turn tool calling below BFCL floor for unbounded tools.
- Native SMT / clingo in CI; mandatory external solver libraries.
- Ctrl-G HMM student weights.
- `exec`/`eval` CodeAct sandbox (separate NEXT_STEPS item).
- Serve-side tool loop.
