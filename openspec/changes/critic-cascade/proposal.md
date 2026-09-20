# Proposal: critic-cascade

**Status:** proposed  
**Date:** 2026-09-20  
**Depends on:** `golden-passk-aqa` (matrix + metrics exist)  
**Plan:** Phase 2  
**Why:** All three models require a filter/critic stage; repo only has outcome
exact-match skip + DualDistill (0,0) drop + optional Bandit on code args.

## Scope

Config-gated cascade on collect and DualDistill compose: schema/tool allowlist
validity, outcome grade, optional teacher self-check hook (interface only if
model call is expensive), structured reject/keep logs. Preserve recovery traces
when final outcome matches (existing collect philosophy).

## Non-goals

GRPO; replacing DualDistill; enabling `distillation_alpha`; solver tools (P3).
