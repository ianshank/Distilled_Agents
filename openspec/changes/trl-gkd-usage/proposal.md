# Proposal: trl-gkd-usage

**Status:** implemented  
**Date:** 2026-09-20  
**Depends on:** P1–P2 green; P3 recommended before claiming “neurosymbolic”  
**Plan:** Phase 4  
**Why:** Consensus notes TRL `GKDTrainer` / on-policy distill as implementable
today; NEXT_STEPS correctly defers *usage*. `pyproject` already has a TRL
version *range* — that is not a usage ADR.

## Scope

- Pin an exact TRL minor (still within policy).
- Optional path in local trajectory training to use TRL on-policy/GKD APIs
  with label-masked, vocab-aligned invariants tested.
- Keep default `trajectory_distill_alpha=0.0` until tests prove safety.

## Non-goals

Enabling GKD on SageMaker 4.26/1.13 image; GRPO; claiming edge readiness (P5).
