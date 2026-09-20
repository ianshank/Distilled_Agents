# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Phase P3 `symbolic-disposition`:** Pure-Python constraint checking, symbolic disposition tools, fail-closed refusal on OOD, and hard golden coverage.
- Added `ConstraintCheckTool` (`constraint_check`) and `SqeConstraintSolverTool` (`sqe_constraint_solver`) in `enhanced_system/harness/tools/constraint.py` and exported through `TOOL_REGISTRY`.
- Supported pure-Python DAG topological sorting (lexicographically least order via Kahn's algorithm with a min-heap) and boolean condition tree solving without external native solver binaries.
- Enforced standard reject codes conforming to `openspec/changes/_shared/blocked-reject-codes.md` (`CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`).
- Added dedicated harness YAML specifications in `configs/harnesses/qc_constraints.yaml`, `enhanced_system/config/harnesses/qc_constraints.yaml`, `configs/harnesses/sqe_dispose.yaml`, and `enhanced_system/config/harnesses/sqe_dispose.yaml`.
- Enforced fail-closed refusal mapping (`BLOCKED:<CODE>`) on out-of-distribution (OOD) tasks, preventing confabulation of false SAT results.
- Expanded `configs/golden_sets/hard_sdlc.jsonl` with hard and OOD rows for `qc_constraints`, backed by deterministic fixtures in `tests/fixtures/mock_responses.json`.
- Updated rule traceability matrix `configs/rule_traceability/matrix.yaml` with active and fixture rules for all new hard golden items.
- Added `scripts/harness/extract_rules.py` stub script extracting rule candidates from teacher traces without auto-promotion.
- Added comprehensive unit tests in `enhanced_system/tests/unit/test_harness_tools.py` including falsifier tests guaranteeing cyclic and unsat inputs never report SAT.
- Documented ADR 0007 (`docs/adr/0007-symbolic-dispose-tools.md`).

### Added
- **Phase P2 `critic-cascade`:** Pure critic helpers, allowlist cascade filtering, reject telemetry, and recovery trace retention.
- Added reusable critic helpers in `enhanced_system/harness/critic.py`: `check_tool_allowlist`, `check_outcome`, `check_expected_tools`, `is_recovery_trace`, and composite `evaluate_trace` without external Hub dependencies.
- Added canonical `CriticRejectCode` enum and standard reject code validation conforming to `openspec/changes/_shared/blocked-reject-codes.md` (`CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`, `OUTCOME_MISMATCH`, `DUALDISTILL_DROP_0_0`).
- Wired trace critic cascade into `scripts/harness/collect_trajectories.py` behind `MANGOMAS_CRITIC_ENABLED` (via `get_settings().critic_enabled`, default `True`), supporting `--critic`, `--no-critic`, and `--reject-log` CLI options.
- Structured reject records appended to JSONL sink `artifacts/critic_rejects.jsonl` (or `--reject-log` override) containing `critic_reject_code`, `prompt`, and `metadata`.
- DualDistill compose path in `enhanced_system/harness/dualdistill.py` and `scripts/harness/compose_dualdistill.py` now logs `critic_reject_code: DUALDISTILL_DROP_0_0` and appends to reject sink when dropping `(0, 0)` pairs without inventing a second drop rule.
- Preserved recovery traces when final outcome matches `expected` despite intermediate `parse_error` or `tool_error`, incrementing `critic_kept_recovery`.
- Emitted aggregated and per-event structured counters: `critic_rejected_outcome_mismatch`, `critic_rejected_allowlist`, `critic_rejected_expected_tools`, `critic_rejected_dualdistill_0_0`, and `critic_kept_recovery`.
- Added comprehensive unit and harness tests in `enhanced_system/tests/unit/test_harness_critic.py` covering all spec falsifiers and cascade stages.
- Exported critic primitives in `enhanced_system/harness/__init__.py`.

### Added
- Phase 0 I1: Pass@K hard/OOD gate via dedicated CLI `scripts/harness/run_pass_at_k.py` with Chen et al. unbiased estimator (default n=5, gate k=3).
- Hard/OOD golden dataset `configs/golden_sets/sqe_hard_ood.jsonl` (>=24 rows meeting DAG, condition tree, mixed, cycle, unsat, schema/syntax, and unknown-theory bucket minima).
- Bucket minima validator in `enhanced_system/harness/golden.py`.
- Scripted fixtures for Phase 0 deterministic testing in `tests/fixtures/mock_responses_sqe_passk.json`.
- Rule traceability matrix in `configs/rule_traceability/matrix.yaml` and checker `scripts/harness/check_rule_matrix.py`.
- Makefile target `aqa-gate-passk` and CI job `aqa-passk` uploading `aqa-passk-summary.json`.
- Synchronization test `tests/unit/test_prompt_render_sync.py`.
- OpenSpec coordination layer and symbolic-KD programme docs (`openspec/`, `docs/plans/symbolic-kd/{PLAN,REVIEW}.md`) locking P0–P4 disposition-first plan after tip `94e6c6a` peer review.

### Changed
- Clarified docstring in `scripts/harness/run_aqa_gate.py` to reflect single-pass pass_rate regression gate, pointing to `run_pass_at_k.py` for multi-sample Pass@K.


### Added
- **Distillation Expansion Phase 2:** DevSecOps and Data Governance.
- Added `SecurityScanner` (Bandit API) validating agent trajectory AST safety.
- Added `PIIScrubber` (Presidio + Spacy) enforcing data governance on agent outputs.
- **Distillation Expansion Phase 3:** Governed ML Pipeline & DPO.
- Added `scripts/training/distill/dpo_collator.py` enabling Direct Preference Optimization formatting.
- Added `train_dpo_adapter.py` orchestrating `trl.DPOTrainer` alongside PEFT.
- **Distillation Expansion Phase 4:** AQA & Golden Set Gating.
- Added Pass@K Regression gating logic to CI via `aqa-gate` (`run_aqa_gate.py`).
- Added semantic matching alongside exact matching in `eval_harness.py`.
- Added Tool Sequence Accuracy validation to trajectory evaluations.
- **Distillation Expansion Phase 5:** Observability & Serving.
- Added Prometheus `/metrics` endpoint to the local inference container tracking latency, errors, and throughput.
- Implemented Multi-Adapter Blue/Green Hot-swapping via `/adapter/load`, `/adapter/switch`, and `/adapter/unload` endpoints.
- Addressed `Bandit B108` and resolved Pydantic v2 `dict()` deprecations in basic inference E2E journeys.
- Regression test `test_rca_trajectory_filter_truncation_boundary` validating supervised token truncation boundary in `has_supervised_tokens`.
- Regression test `test_rca_gpu_device_fallback` validating dynamic CUDA/CPU device selection.
- `.dockerignore` for cleaner container builds.

### Changed
- **BREAKING:** Upgraded `peft==0.4.0` → `peft>=0.14.0` (DoRA, QLoRA, modern architecture support).
- Upgraded `transformers>=4.30.0` → `transformers>=4.45.0` (Qwen2.5, Llama3, fast tokenizers).
- Upgraded `accelerate>=0.22.0` → `accelerate>=0.34.0`.
- Upgraded `torch>=2.0.0` → `torch>=2.2.0` (CUDA 12.x support).
- SageMaker estimator updated from `transformers 4.26.0 / PyTorch 1.13.1 / py39` to `transformers 4.45.0 / PyTorch 2.2.0 / py311`.
- `--lora_target_modules` now defaults to empty (auto-detected) instead of hardcoded LLaMA module names.
- Reorganized repository structure for clean separation of concerns.
- Refactored `production_deployment.py` to use `MangoMASSageMakerLauncher`.
- Moved core components to `enhanced_system/core/`.
- Moved operational scripts to `scripts/`.
- Reorganized tests into unit, integration, and security suites.
- Consolidated documentation into `docs/`.
