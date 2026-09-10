# ADR 0003: Coverage ratchet in CI

## Status

Accepted

## Context

Tox previously failed under 80% coverage while tests did not match APIs and CI did not exist. After enterprise hardening, measured coverage on `enhanced_system` (omit tests/examples) was approximately **65.75%** with 103 tests passing. The validation pack measured **67.45%** with 132 tests.

## Decision

GitHub Actions reports coverage on every PR. `tool.coverage.report.fail_under` is an **absolute floor** plus a **monotonic ratchet**:

| Date | Floor | Measured (approx.) |
| --- | --- | --- |
| Hardening PR | 60 | 65.75 |
| Validation pack | **65** | 67.45 |

Do not jump to 80% in one change. Raise the floor only after a measured increase. A later job should cover `scripts/` at a lower gate so ops/CLIs are not omitted forever.

`pytest -m "unit or integration or regression"` is the AQA suite (`make test-aqa`). Markers `e2e`, `slow`, and `benchmark` stay unused until real suites exist.

## Consequences

CI stays green while coverage increases honestly. Gaming coverage by omitting modules is avoided by adding ops tests now and a scripts gate later.
