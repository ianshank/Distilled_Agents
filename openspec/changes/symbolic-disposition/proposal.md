# Proposal: symbolic-disposition

**Status:** proposed  
**Date:** 2026-09-20  
**Depends on:** `golden-passk-aqa` CI green on scripted hard slice; `critic-cascade` merged  
**Plan:** Phase 3  
**Why:** Consensus reframes “neurosymbolic student” as symbolic KD + small
proposer. Repo disposition is schema-only; tools are checklist stubs.

## Scope

- Add frozen tool class(es) e.g. `constraint_check` (and optionally `asp_solve`
  behind an extra/optional dependency).
- New harness YAML with a **new id** (do not overload `swe_codeact`).
- ADR 0006 addendum: cognitive proposes; tool body disposes; OOD → ValueError
  → `tool_error` / refuse via `final_answer` policy documented in harness
  instruction.
- Build-time rule extract stub CLI (teacher → matrix row) with mend hooks
  deferred to follow-on if needed.
- Teacher license allowlist documented (Apache-2.0 / MIT).

## Non-goals

- General multi-turn tool calling below BFCL floor for unbounded tools.
- Ctrl-G HMM student weights.
- `exec`/`eval` CodeAct sandbox (separate NEXT_STEPS item).
- Serve-side tool loop.
