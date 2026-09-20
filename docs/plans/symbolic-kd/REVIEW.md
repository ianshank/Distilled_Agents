# Peer review — Symbolic KD / disposition-first programme

**ID:** REVIEW-2026-09-20-symbolic-kd  
**Date:** 2026-09-20 (America/New_York)  
**Base tip reviewed:** `94e6c6a` on `main` (post PR #16 `feature/distillation-expansion`)  
**Inputs:** multi-model consensus brief (GPT / Claude / Nemotron); live repo scan;
prior plan lock; Architect / SDLC / QA / Product personas (Mango `openspec-peer-review`).

## Verdict

**Conditional go.** Proceed with inverted order: **green trunk → honest eval
(pass@k) → critic cascade → symbolic dispose tools → weights (TRL GKD usage)**.
Do **not** train a “neurosymbolic student network.” Do **not** treat #16’s AQA
docstring as pass@k delivery.

**Overall sign-off:** Architect ✅ · SDLC ✅ (after P0) · QA ⚠️ (P1 must land
before P3) · Product ✅ (narrow task only).

---

## Thesis / counter / rebuttal

**Thesis.** The defensible product is *symbolic knowledge distillation*: a local
open-weight teacher emits rules/traces; a critic filters; a deterministic solver
(or checklist→solver tool) **disposes**; a small LoRA student only proposes
structured actions. That matches ADR 0006’s H + R_δ split and INV-style
fail-closed better than a bigger LoRA.

**Counter.** PR #16 already shipped golden sets, AQA gate, DPO, semantic match,
Blue/Green adapters, and Bandit-on-trajectory — so “eval first” looks done and
GKD/Jetson/ASP look like the next sexy steps. `trl` is already in
`pyproject.toml` (`>=0.14,<0.16`).

**Rebuttal.** Evidence against “eval done”:

1. `configs/golden_sets/core_sdlc.jsonl` has **4** rows (`prompt`/`expected` only).
2. `run_aqa_gate.py` documents “Pass@K” but enforces a single **pass_rate**
   threshold via one `eval_harness` run (`sag_samples` default **1**).
3. `eval_harness` counts **semantic_match** as success — dilutes hard-slice
   signal (RLVR caveat: pass@1 can rise while pass@k on hard items falls).
4. AQA is in `Makefile validate`, **absent** from `.github/workflows/ci.yml`.
5. `TOOL_REGISTRY` tools are **checklist stubs** (no Z3/ASP/Scallop); disposition
   today is allowlist + parse, not a sound solver.
6. Main CI: **test green**; **lint/types/security red** (hygiene from #16) —
   trunk is not merge-healthy for feature work.

---

## Persona matrix

### Architect

| Finding | Severity | Disposition |
|---|---|---|
| A1. Correct boundary: add dispose tools inside `tools/registry.py` + harness YAML allowlists; do not put solvers in `predict_fn` | High (positive) | Keep |
| A2. Format-triangle drift risk: two `prompt_render.py` copies (docstring-only diff today). Need hash guard before Serve tools | Med | P1 task |
| A3. `planning.style: codeact` is decorative — naming a solver harness `swe_codeact` will confuse operators | Med | New harness id e.g. `qc_constraints` |
| A4. Ctrl-G / HMM constraint layer is a different architecture (weights as constraint model) — park unless Jetson latency falsifies P3 | Low | Park |

### SDLC / CI Lead

| Finding | Severity | Disposition |
|---|---|---|
| S1. P0 reds are hygiene-only: ruff format (~10 files); `data_governance.py:75` `no-any-return`; Bandit B404/B603 (AQA/security subprocess) + B615 (unpinned `from_pretrained` / `load_dataset`) | High | P0 hotfix first |
| S2. Dependabot Actions majors (#4–#6) still frozen per NEXT_STEPS until harness coverage re-checked | Med | Keep freeze |
| S3. OpenSpec without F-ID proofs can go stale (Agents spike lesson) — archive discipline + PLAN ticks required | Med | `openspec/project.md` reversibility |
| S4. Wiring AQA into CI touches `.github/**` — keep scripted Echo/mocks only | High | P1b |

### QA Director

| Finding | Severity | Disposition |
|---|---|---|
| Q1. pass@k not implemented; SAG≠multi-trial eval | High | P1b falsifier |
| Q2. Semantic-as-success hides exact regressions | High | Separate `exact_pass_rate` / hard-slice gate |
| Q3. Golden rows lack `harness_id`, `expected_tools`, `slice` | Med | P1a schema |
| Q4. Critic cascade absent beyond outcome filter + DualDistill drop (0,0) | Med | P2 |
| Q5. No regression matrix for rules (Nemotron mend-loop precursor) | Med | P1c |

### Product

| Finding | Severity | Disposition |
|---|---|---|
| P1. Unbounded agentic tool surface is out of scope (Sol “prompt+solver” wins there) | High | Charter non-goal |
| P2. Narrow first vertical: QC/metadata or structured checklist→constraint (Edge-DIT-like), not general SWE | High | P3 scope lock |
| P3. Legal path needs Apache-2.0/MIT teachers (Qwen3 / R1-distills); avoid Llama MAU/AUP flow-down for NBC playbooks | Med | Teacher allowlist in design |
| P4. Phase 6 SageMaker registry/A/B must not leapfrog fail-closed harness quality | Med | NEXT_STEPS deferral note |

---

## Evidence ledger (tip `94e6c6a`)

| Claim | Evidence |
|---|---|
| Test job green; lint/types/security red | Actions run `35438606162` |
| Lint = format only (`ruff check` passed) | Job `105885426694` |
| Types = `data_governance.py:75` no-any-return | Job `105885426714` |
| Security = B404/B603/B615 | Job `105885426690` |
| Golden = 4 lines | `configs/golden_sets/core_sdlc.jsonl` |
| AQA ≠ pass@k | `scripts/harness/run_aqa_gate.py` |
| Tools = 6 checklist ids | `enhanced_system/harness/tools/registry.py` |
| SAG = parse majority | `enhanced_system/harness/policies/sag.py` |
| Custom KD trainer, not GKD | `scripts/training/distill/trainer.py` |
| `trl` range present, unused for GKD | `pyproject.toml` |
| prompt_render sync (docstring only) | md5 differ; unified diff = header comment |

---

## Own-goals corrected this review

1. Earlier plan said “build golden from scratch” — **wrong**; harden #16 scaffolding.
2. “Defer TRL pin” — **partially stale**; pin *range* exists; missing is **usage ADR + exact minor pin + tests**.
3. “Fail-closed like INV-16” — **overclaim**; today fail-closed is schema allowlist, not solver soundness.

---

## Knowledge gaps (do not block P0–P1)

- KG1: Exact Jetson Orin Nano + Hailo-8 latency/energy budgets for P5.
- KG2: Which NBC vertical is the first constraint domain (confirm with Ian).
- KG3: Whether Presidio/`[security]` extra is required in CI for PII scrub tests.

---

## Sign-off condition

Implementers may start **P0** immediately. **P3+** requires P1b pass@k CI green
on a hard slice and P2 critic metrics logged. Product confirms narrow task id
before solver tool merge.
