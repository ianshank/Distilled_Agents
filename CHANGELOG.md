# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Enterprise validation pack: Makefile (`make validate`), Gitleaks, incremental mypy on `enhanced_system/ops` and `enhanced_system/core/cache`, composite GitHub Action, Cursor skills, C4 docs, and a deterministic skill/CLI harness.
- `enhanced_system.ops.archive.safe_extract_tar` for path-traversal-safe model archives.
- `SkillTrainingConfig` as the training-system dataclass (launcher keeps `AgentTrainingConfig`).
- Coverage floor raised to 65% with a regression marker for CLI/import contracts.

### Changed

- `scripts/deployment/run_sagemaker_training.py` imports `MangoMASSageMakerLauncher` from `enhanced_system.ops` and reads region from `get_settings()`.
- Distill/inference bind host, `trust_remote_code`, and model revision go through `get_settings()` with env fallback for SageMaker images.
- Docker runtime image uses Python 3.11; false `:8080/health` HEALTHCHECK removed.

### Deprecated

- `training_system.AgentTrainingConfig` alias (one release; emits `DeprecationWarning`). Use `SkillTrainingConfig`.

## [1.0.0] - 2026-09-10

### Added

- Installable `mangomas` package, root `pyproject.toml`, GitHub Actions CI.
- Shared SageMaker launcher and `MangoMASSettings`.
- Cache/errors/distill facades; JSON cache serialization (no pickle).
- ADRs 0001–0003 (JSON cache, unified launcher, coverage ratchet).
