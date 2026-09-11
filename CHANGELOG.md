# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Native `enhanced_system.harness` runtime (YAML specs, frozen tool-id registry, AST/JSON dispatch, Echo/Transformers backends, rule-based tailor). Local CLIs: `scripts/harness/run_agent.py`, `collect_trajectories.py`, `tailor_harness.py`.
- Trajectory SFT collator in `scripts/training/distill/trajectory_collator.py` and `--trajectory_mode` on `train_distilled_adapter.py` (alpha from `MANGOMAS_TRAJECTORY_DISTILL_ALPHA`, default `0.0`). SageMaker `create_job_spec` and `predict_fn` are unchanged.
- Dual coverage gates: global `fail_under=60` and harness `.coveragerc.harness` `fail_under=90`.
- Operator pack: `Makefile` (`make validate`), `.gitleaks.toml`, incremental mypy on `ops` / `core/cache` / `harness` (`python_version = 3.11`), tox `harness` env, composite `.github/actions/mangomas-validate` for skills (does not replace CI jobs).
- Cursor skills `mangomas-collect` and `mangomas-validate`; scan skill documents `make validate` while `cli:` stays `security_scan.py`.
- C4 L1–L3 (`docs/architecture/`) distinguishing Distill, Serve `predict_fn`, and local harness H.

### Changed

- Collect path disables injection detection and keeps original task text (PII off) for the trajectory **and** backend/prefix. `run_agent` keeps injection on. `InputValidator` patterns are not edited.
- Gitleaks allowlists only checked-in training JSONL fixtures. Collect writes a distinct `*.raw.traces.jsonl` when `--output` already ends in `.raw.jsonl`. Trajectory alpha is validated in `[0.0, 1.0]`.
- FTP teacher prefix is cleared in `finally` after the first generate attempt.
- CI adds a `types` job and gitleaks on `security` beside existing lint / unit+integration@60 / harness@90. Action pins stay `checkout@v4` / `setup-python@v5` / `upload-artifact@v4`.
- Nested `enhanced_system/.pre-commit-config.yaml` is a stub; root hooks add gitleaks v8.21.2 and scoped mypy.
- Dockerfile false `:8080/health` HEALTHCHECK removed (`CMD` is still `basic_inference`). Image Python stays 3.9.

## [1.0.0] - 2026-09-10

### Added

- Installable `mangomas` package, root `pyproject.toml`, GitHub Actions CI.
- Shared SageMaker launcher and `MangoMASSettings`.
- Cache/errors/distill facades; JSON cache serialization (no pickle).
- ADRs 0001–0003 (JSON cache, unified launcher, coverage ratchet).
