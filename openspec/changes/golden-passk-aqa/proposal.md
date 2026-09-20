# Proposal: golden-passk-aqa

**Status:** proposed  
**Date:** 2026-09-20  
**Depends on:** `green-trunk-ci` merged  
**Plan:** Phase 1  
**Why:** Consensus and REVIEW require eval-before-weights. #16 shipped a
vacuous “Pass@K” label on a 4-row pass_rate wrapper not wired into CI.

## Problem

- Golden schema too thin (`prompt`/`expected` only).
- `run_aqa_gate.py` claims Pass@K; implements pass_rate only.
- `eval_harness` treats semantic_match as success.
- AQA absent from `.github/workflows/ci.yml`.
- No rule traceability artifact; no prompt_render sync guard.

## Scope

Golden corpus + eval metrics + CI scripted AQA + matrix stub + render hash
check. No solver tools. No trainer changes.

## Non-goals

Live model CI; expanding to hundreds of unlabeled prompts; Serve tool loop.
