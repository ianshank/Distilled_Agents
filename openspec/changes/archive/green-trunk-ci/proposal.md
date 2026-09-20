# Proposal: green-trunk-ci

**Status:** archived  
**Date:** 2026-09-20  
**Base:** `94e6c6a`  
**Plan:** `docs/plans/symbolic-kd/PLAN.md` Phase 0  
**Why:** Main CI after #16 fails security/lint/types while tests pass. Feature
work on a red trunk is unsafe and blocks Dependabot re-checks.

## Problem

Workflow run `35438606162`: lint (`ruff format --check`, ~10 files), types
(`data_governance.py:75` `no-any-return`), security (Bandit B404/B603/B615).

## Scope

Hygiene-only fixes. No behavior change to harness loop, trainers, or golden
semantics.

## Non-goals

Refactor Presidio; disable Bandit globally; merge Dependabot majors.

## Compile-down

- Code: listed files under `enhanced_system/harness/`, `scripts/`
- CI: existing jobs turn green
- Docs: one CHANGELOG line
