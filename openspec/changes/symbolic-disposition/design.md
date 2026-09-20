# Design — symbolic-disposition

## Architecture

```
User task
  → AgentRuntime (H)
      → ModelBackend proposes thought + JSON/Call action  (R_δ)
      → dispatch.parse_action (allowlist)
      → TOOL_REGISTRY[tool].run(args)   ## deterministic dispose
      → observation back to loop
  → final_answer only when solver/checklist OK or explicit refuse
```

## Tool contract

- Input: JSON-literal args only (existing dispatch).
- Success: return string observation (JSON ok).
- Failure: `raise ValueError` → runtime `tool_error` (fail-closed for that step).
- No network, no shell, no `exec`. Optional clingo/z3 imported inside tool
  module and skipped with clear ImportError message if extra not installed;
  CI tests use a pure-Python constraint fixture tool so default CI stays
  dependency-light.

## Narrow vertical (default assumption)

Structured validation / QC-style constraints expressible as required keys +
simple predicates. Product may rename the harness id; do not start with
open-ended SWE.

## ADR

Land ADR 0007 (or amend 0006 if maintainers prefer) at merge time describing
dispose tools and OOD refusal. Number claimed at land.
