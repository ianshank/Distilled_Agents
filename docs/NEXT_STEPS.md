# Next steps (out of this validation pack)

Tracked follow-ups so this pack stays reviewable without mixing large refactors.

## Remaining module splits

Keep facades; split implementations in a dedicated PR when a file stays above ~400 lines:

- `enhanced_system/core/adaptive_router.py`
- `enhanced_system/core/confidence_calibrator.py`
- `enhanced_system/evaluation/skill_evaluator.py`
- `enhanced_system/core/monitoring.py`
- `enhanced_system/core/consensus_inference.py`
- `enhanced_system/training/data_curator.py`

## Coverage ratchet toward 80%

ADR 0003: floor is **65** (measured ~65.75% on `enhanced_system` after hardening). Raise `fail_under` monotonically; do not jump to 80 in one change. Add a second coverage job for `scripts/` at a lower gate so ops/CLIs are not omitted forever.

## Dead architecture surface

Wire or delete unused factories for router/batch/calibrator/consensus/monitor, and `BaseProcessor` if it still has no implementer.

## Dependencies

Revisit the `peft==0.4.0` training-image pin when SageMaker images move forward.

## Types

Incremental mypy covers `enhanced_system/ops` and `enhanced_system/core/cache` only. Strict mypy on all of `enhanced_system` is later.

## Tests

Do not add live AWS e2e or empty `e2e` markers. Keep `e2e` / `slow` / `benchmark` unused until a real harness exists.
