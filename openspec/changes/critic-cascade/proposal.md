# Proposal: critic-cascade

**Status:** proposed  
**Date:** 2026-09-20  
**Depends on:** `golden-passk-aqa` (metrics exist)
**Plan:** Phase 2  
**Why:** All three models require a filter/critic stage; repo only has outcome
exact-match skip + DualDistill (0,0) drop + optional Bandit on code args.

## Scope

Config-gated cascade on collect and DualDistill compose: schema/tool allowlist
validity, outcome grade, explicit reject telemetry with `critic_reject_code`,
structured logging of drops (including `DUALDISTILL_DROP_0_0` for existing (0,0)
drops and `OUTCOME_MISMATCH` for collect outcome filter drops) to JSONL sink
`artifacts/critic_rejects.jsonl`. Standard reject codes conform to
`openspec/changes/_shared/blocked-reject-codes.md`: `CYCLE_DETECTED`, `UNSAT`,
`SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`.
Preserve recovery traces when final outcome matches (existing collect philosophy).

## Sequencing note

Lands after `golden-passk-aqa` metrics exist. Does not wait on `symbolic-disposition`.

## Non-goals

GRPO; replacing DualDistill; inventing a second DualDistill drop rule;
enabling `distillation_alpha`; solver tools (P3).
