# Proposal: golden-passk-aqa

**Status:** proposed  
**Date:** 2026-09-20  
**Depends on:** `green-trunk-ci` merged  
**Plan:** Phase 1  
**Why:** Consensus and REVIEW require eval-before-weights. #16 shipped a
vacuous "Pass@K" label on a 4-row pass_rate wrapper not wired into CI.

## Problem

- Golden schema too thin (`prompt`/`expected` only); no hard/OOD slice with >=24 rows.
- `run_aqa_gate.py` claims Pass@K; implements single-pass pass_rate only.
- `eval_harness` treats semantic_match as success and credits unlabeled non-truncated rows.
- Pass@K formula requires Chen et al. unbiased estimator, not biased or single-pass proxies.
- AQA absent from `.github/workflows/ci.yml`.
- No rule traceability artifact; no prompt_render sync guard.

## Scope

Golden corpus (`configs/golden_sets/sqe_hard_ood.jsonl`, >=24 rows meeting per-bucket
minima) + Chen unbiased Pass@K eval metrics ($n=5$, gate $k=3$, report $k=1,3$) via
single CLI `scripts/harness/run_pass_at_k.py` (Makefile `aqa-gate-passk`) + disjunctive
OOD row detector (`slice: "ood"` OR `ood: true`) + OOD synthetic-success prohibition
(canonical `BLOCKED:<CODE>` from `openspec/changes/_shared/blocked-reject-codes.md`) +
matrix stub + render hash check. Correct `run_aqa_gate.py` docstring only (single-pass).
No solver tools. No trainer changes.

## Non-goals

Live model CI; expanding to hundreds of unlabeled prompts; Serve tool loop.
