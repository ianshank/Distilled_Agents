# OpenSpec — project context (Distilled_Agents / MangoMAS)

OpenSpec is a **thin coordination/authoring layer** for in-flight changes.
It is **not** a second source of truth.

## Authoritative back-end (this repo)

| Concern | Source of truth |
|---|---|
| Runtime harness–policy | ADR 0006 (`docs/adr/0006-runtime-harness-policy.md`) |
| Distillation format triangle | `docs/README_AGENT_DISTILLATION.md` |
| Operator follow-ups | `docs/NEXT_STEPS.md` |
| Settings / knobs | `enhanced_system/ops/settings.py` (`MANGOMAS_*`) |
| Frozen tools | `enhanced_system/harness/tools/registry.py` |
| CI gates | `.github/workflows/ci.yml` + `Makefile` |
| Decisions | `docs/adr/NNNN-*.md` (claimed **at land**, never reserved) |

This directory never restates those docs; it only coordinates proposals and
maps them onto ADRs, CLIs, tests, and CI jobs.

## Why OpenSpec here

Distilled_Agents has no `features.yaml` / F-ID proof system (unlike Agents).
OpenSpec supplies a shared fleet vocabulary (proposal → design → spec delta →
tasks → review → archive) while CI + ADRs remain the enforcement surface.

## Reversibility

Deleting `openspec/` must leave ADR 0006, the distill README, harness code,
and CI intact. No Python module may import from `openspec/`. No CI job may
*require* OpenSpec files until an explicit ADR says otherwise (index/guard
optional later).

## Conventions

- No hard-coded AWS accounts, Hub revisions as unset Hub loads, or tool ids
  outside `TOOL_REGISTRY`.
- Numeric knobs stay on `MangoMASSettings` / env (`MANGOMAS_*`).
- Paper recipes map through `docs/README_AGENT_DISTILLATION.md` false-friends
  table before any trainer change.
- Falsifiers accompany every acceptance criterion (what must go red if the
  invariant is broken).
