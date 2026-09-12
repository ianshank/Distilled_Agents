# Next steps (out of this harness + operator pack)

Tracked follow-ups so this change stays reviewable without mixing large refactors.

## Remaining module splits

Keep facades; split implementations in a dedicated PR when a file stays above ~300–400 lines. Do not split these here:

- `enhanced_system/core/adaptive_router.py`
- `enhanced_system/core/input_validator.py`
- `enhanced_system/evaluation/skill_evaluator.py`
- `enhanced_system/core/confidence_calibrator.py`
- `enhanced_system/core/monitoring.py`
- `enhanced_system/core/consensus_inference.py`
- `enhanced_system/training/data_curator.py`

Do not edit `InputValidator` regexes to “allow coding prompts”; serving tests require `SELECT … FROM` to fail. Collect vs `run_agent` already forks `enable_injection_detection`.

## Coverage ratchet toward 80% on **core**

ADR 0003: global `fail_under` stays **60**. Dual-marked harness tests inflate tree coverage; do not raise 60→65 on that harness-inflated number. Raise monotonically toward 80% on core modules, with a later second job for `scripts/` at a lower gate.

Harness package gate stays `.coveragerc.harness` `fail_under=90`. Do not omit `protocols.py` from the *global* omit list.

## Harness follow-ups

- Wire an endpoint handler if distilled CodeAct students should run tools in SageMaker (`predict_fn` stays single-shot until then).
- Add `trajectory_mode` to `create_job_spec` only together with launcher argparse (SageMaker `source_dir` cannot import `enhanced_system`).
- CodeAct sandbox (no `eval`/`exec` in-process).
- Multiprocess collect needs `fcntl` (or equivalent) around `JsonlTraceStore`; in-process `threading.Lock` is enough for unit tests.
- SAG is parse/schema majority vote, not execution-consistent Kang SAG.

## Dead architecture surface

Wire or delete unused factories for router/batch/calibrator/consensus/monitor, and `BaseProcessor` if it still has no implementer.

## Docker / compose

`.dockerignore` exists at repo root. Compose `context: ../..` from `enhanced_system/infrastructure/docker/` still does not match a root-context Dockerfile layout; fix that in a dedicated image PR. Do not bump the image Python 3.9 pin here.

## Dependencies

Revisit the `peft==0.4.0` training-image pin when SageMaker images move forward. DialoGPT LoRA `target_modules` vs Mistral, and SageMaker `transformers_version=4.26.0` vs requirements `>=4.30`, stay known product limits.

## Types

Incremental mypy covers `enhanced_system/ops`, `enhanced_system/core/cache`, and `enhanced_system/harness` (`python_version = 3.11`). Strict mypy on all of `enhanced_system` is later. Do not use `--follow-imports=skip` to hide harness errors.

## Tests

Do not add live AWS e2e. Keep `e2e` / `slow` / `benchmark` unused for **live AWS**. Local Echo harness pipelines stay `integration` (and `harness`). Contract tests must keep boto3 monkeypatched.

## Dependabot

Leave GitHub Actions major bumps (#4/#5/#6, checkout/setup-python/upload-artifact v7) unmerged until the harness coverage job is re-checked against those runners. Pip PRs (#7–#11) stay separate.
