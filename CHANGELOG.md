# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **SDLC Hardening & Pre-PR Verification Pack:**
  - Automated Quality Assurance (AQA) Gate deterministic validations for agents and skills (`make aqa-gate`, `make aqa-gate-passk`).
  - Expanded golden set `configs/golden_sets/core_sdlc.jsonl` from 4 to 12 items, backed by deterministic mock fixtures in `tests/fixtures/mock_responses.json`.
  - Updated rule traceability matrix in `configs/rule_traceability/matrix.yaml` with active status for covered SDLC rules.
  - Dedicated regression test suite in `tests/test_regression_suite.py` and test environment fixtures in `tests/conftest.py`.
  - Scripted trajectory collection support for input JSON files via `--scripted` in `scripts/harness/collect_trajectories.py`.
  - Standardized configuration prefix with `MANGOMAS_MODEL_DIR` and `MANGOMAS_PORT` in `scripts/inference.py` and `scripts/training/train_dpo_adapter.py`.
  - Added SEC-001 constant-time token comparison with `hmac.compare_digest` for PEFT adapter hot-swapping auth.
  - Added SEC-002 exception re-raising in `predict_fn` to ensure unhandled errors surface with appropriate HTTP status codes.
  - Added SEC-003 prompt payload size bounds checking (`MANGOMAS_MAX_PROMPT_BYTES`) to prevent unbounded tokenization DoS.
  - Added CQ-001 graceful degradation in `SecurityScanner` when Bandit JSON parsing fails due to stderr warnings.
  - Added CQ-002 graceful degradation in `PIIScrubber` when Presidio NLP libraries are not installed.
  - Added CQ-004 dynamic AWS account ID resolution via `MANGOMAS_AWS_ACCOUNT_ID` or STS identity, eliminating hardcoded dummy account IDs.
  - Added TRN-001 fail-fast `ValueError` check in `AgentDistillationTrainer` when a trajectory dataset filters to 0 rows.
  - Added narrow-critic hook and Tier A/B/C verification documentation (`/sdlc-orchestrator` workflow).

- **Local GPU E2E Suite (`run_e2e_gpu.py`, `tests/e2e/test_local_gpu_e2e.py`):**
  - Added standalone CLI `scripts/harness/run_e2e_gpu.py` exercising EchoBackend, EvalHarness, and live CUDA TransformersBackend inference on GPU hardware (`NVIDIA GeForce RTX 5060 Ti`), with automatic VRAM garbage collection and structured JSON reporting.
  - Added automated Pytest suite `tests/e2e/test_local_gpu_e2e.py` with `@pytest.mark.e2e` and `@pytest.mark.gpu` decorators covering device detection, CUDA tensor placement, agent runtime multi-turn tool loops, and CLI runner execution.
  - Added Makefile targets `test-e2e-gpu` and `run-e2e-gpu`.
  - Generated validation evidence in `artifacts/e2e_results.json` confirming 7 PASS, 0 FAIL, 0 ERROR.

- **Antigravity Gated SDLC Agent Pack Integration:**
  - Integrated gated workflow contract (`config/workflow-contract.json`) and agent pack (`.agents/agents/`, `.agents/skills/`, `.agents/rules/`, `.agents/hooks/`, `hooks.json`, `plugin.json`, `GEMINI.md`).
  - Added stage-aware path gating in `.agents/hooks/common.py` allowing source tree edits during `build`, `evaluate`, and `ship` stages while enforcing artifact immutability during intake and specification.
  - Added `scripts/infrastructure/agent_pack_smoke_test.py` smoke test validating agent pack hooks, schemas, and contract invariants.

- **Phase P4 `trl-gkd-usage`:** Pinned exact TRL version (`trl==0.15.2`), implemented config-driven GKD / on-policy distillation adapter module, label-masked generalized JSD and KL divergence, safe vocab mismatch handling, default `trajectory_distill_alpha=0.0`, and ADR 0008.
  - Pinned exact `trl==0.15.2` in `pyproject.toml` (under `alignment` and `all`), documenting version rationale (stable GKDTrainer/GKDConfig, SFT caching fixes, transformers>=4.45.0 compatibility).
  - Added GKD operational settings in `enhanced_system/ops/settings.py` (`gkd_enabled`, `gkd_lmbda`, `gkd_beta`, `gkd_temperature`, `gkd_max_new_tokens`, `gkd_seq_kd`, `gkd_loss_type`), preserving fail-closed defaults (`gkd_enabled: false`, `trajectory_distill_alpha: 0.0`).
  - Implemented `enhanced_system/training/gkd_adapter.py` and `scripts/training/distill/gkd_adapter.py` providing `AgentGKDTrainer`, `compute_label_masked_gkd_loss`, `validate_vocab_alignment`, `verify_trl_version_pin`, and `GKDTrainingConfig`.
  - Enforced strict label-masking for divergence loss: KL/JSD divergence is computed only on supervised tokens (`labels != -100`), ensuring user turns, observations, and error frames are not contaminated.
  - Enforced fail-safe vocabulary compatibility: mismatched tokenizers or logits vocabularies are strictly refused or safely fall back to pure task loss with runtime warnings; cross-vocab KL is never computed.
  - Added standalone CLI `scripts/training/train_gkd_adapter.py` for trajectory on-policy training with LoRA.
  - Added `--use_gkd`, `--gkd_beta`, and `--gkd_lmbda` flags to `scripts/training/train_distilled_adapter.py`.
  - Updated `docs/README_AGENT_DISTILLATION.md` paper false-friends table updating GKD from Deferred to Optional behind ADR 0008.
  - Documented ADR 0008 (`docs/adr/0008-trl-gkd-usage.md`).

- **Phase P3 / I3 `symbolic-disposition`:** Pure-Python constraint checking, symbolic disposition tools, fail-closed refusal on OOD, and hard golden coverage.
  - Added `ConstraintCheckTool` (`constraint_check`) and `SqeConstraintSolverTool` (`sqe_constraint_solver`) in `enhanced_system/harness/tools/solver.py` and exported through `TOOL_REGISTRY`.
  - Supported pure-Python DAG topological sorting (lexicographically least order via Kahn's algorithm with a min-heap) and boolean condition tree solving without external native solver binaries.
  - Enforced standard reject codes conforming to `openspec/changes/_shared/blocked-reject-codes.md` (`CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`).
  - Added dedicated harness YAML specifications in `configs/harnesses/qc_constraints.yaml`, `enhanced_system/config/harnesses/qc_constraints.yaml`, `configs/harnesses/sqe_dispose.yaml`, and `enhanced_system/config/harnesses/sqe_dispose.yaml`.
  - Enforced fail-closed refusal mapping (`BLOCKED:<CODE>`) on out-of-distribution (OOD) tasks, preventing confabulation of false SAT results.
  - Added runtime fault token `"blocked"` in `enhanced_system/harness/runtime.py` when trajectories terminate with canonical refusal `BLOCKED:<CODE>`.
  - Expanded `configs/golden_sets/hard_sdlc.jsonl` with hard and OOD rows for `qc_constraints` and `sqe_dispose`, backed by deterministic fixtures in `tests/fixtures/mock_responses.json`.
  - Updated rule traceability matrix `configs/rule_traceability/matrix.yaml` with active and fixture rules for all new hard golden items.
  - Added `scripts/harness/extract_rules.py` stub script extracting rule candidates from teacher traces without auto-promotion.
  - Added comprehensive unit tests in `enhanced_system/tests/unit/test_harness_tools.py` and `enhanced_system/tests/unit/test_harness_solver.py` including falsifier tests guaranteeing cyclic and unsat inputs never report SAT, proof of all 6 reject codes, and absence of forbidden execution constructs.
  - Documented ADR 0007 (`docs/adr/0007-symbolic-dispose-tools.md`).

- **Phase P2 `critic-cascade`:** Pure critic helpers, cascade trajectory filtering, structured reject sink, and DualDistill (0,0) drop telemetry.
  - Added pure functional critic module in `enhanced_system/harness/critic.py` providing `CriticCascade`, `CriticRejectSink`, and individual stage helpers (`check_allowlist`, `check_expected_tools`, `check_outcome`).
  - Wired trace critic cascade into `scripts/harness/collect_trajectories.py` behind `MANGOMAS_CRITIC_ENABLED` (via `get_settings().critic_enabled`, default `True`), supporting `--critic`, `--no-critic`, and `--reject-log` CLI options.
  - Structured reject records appended to JSONL sink `artifacts/critic_rejects.jsonl` (or `--reject-log` override) containing `critic_reject_code`, `prompt`, and `metadata`.
  - DualDistill compose path in `enhanced_system/harness/dualdistill.py` and `scripts/harness/compose_dualdistill.py` now logs `critic_reject_code: DUALDISTILL_DROP_0_0` and appends to reject sink when dropping `(0, 0)` pairs without inventing a second drop rule.
  - Preserved recovery traces when final outcome matches `expected` despite intermediate `parse_error` or `tool_error`, incrementing `critic_kept_recovery`.
  - Emitted aggregated and per-event structured counters: `critic_rejected_outcome_mismatch`, `critic_rejected_allowlist`, `critic_rejected_expected_tools`, `critic_rejected_dualdistill_0_0`, and `critic_kept_recovery`.
  - Added comprehensive unit and harness tests in `enhanced_system/tests/unit/test_harness_critic.py` covering all spec falsifiers and cascade stages.
  - Exported critic primitives in `enhanced_system/harness/__init__.py`.

- **Phase 0 I1 `passk-hard-ood-gate`:** Pass@K hard/OOD gate via dedicated CLI `scripts/harness/run_pass_at_k.py` with Chen et al. unbiased estimator (default n=5, gate k=3).
  - Hard/OOD golden dataset `configs/golden_sets/sqe_hard_ood.jsonl` (>=24 rows meeting DAG, condition tree, mixed, cycle, unsat, schema/syntax, and unknown-theory bucket minima).
  - Bucket minima validator in `enhanced_system/harness/golden.py`.
  - Scripted fixtures for Phase 0 deterministic testing in `tests/fixtures/mock_responses_sqe_passk.json`.
  - Rule traceability matrix in `configs/rule_traceability/matrix.yaml` and checker `scripts/harness/check_rule_matrix.py`.
  - Makefile target `aqa-gate-passk` and CI job `aqa-passk` uploading `aqa-passk-summary.json`.
  - Synchronization test `tests/unit/test_prompt_render_sync.py`.
  - OpenSpec coordination layer and symbolic-KD programme docs (`openspec/`, `docs/plans/symbolic-kd/{PLAN,REVIEW}.md`) locking P0–P4 disposition-first plan after tip `94e6c6a` peer review.

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

### Fixed
- Re-raised predict exceptions instead of returning silent 500 error dicts in `scripts/inference.py` (SEC-002).
- Fixed AWS account ID resolution via `MANGOMAS_AWS_ACCOUNT_ID` or STS caller identity (CQ-004).
- Fixed lazy loading of `presidio` for PII scanning to degrade gracefully when optional dependency is absent (CQ-002).
- Improved `bandit` JSON parse handling to avoid crashes on stderr contamination (CQ-001).
- Replaced `evaluation_strategy` with `eval_strategy` in `train_distilled_adapter.py` to fix Transformers v4.41+ compat.
- Added `**kwargs` to `compute_loss` in `DistillationTrainer` to fix Transformers v4.46+ compat.
- Trunk CI hygiene (Phase P0 `green-trunk-ci`): formatted files across `enhanced_system`, `scripts`, and `tests` with `ruff format`; resolved mypy `no-any-return` on `PIIScrubber.redact_text` in `enhanced_system/harness/data_governance.py` and `_case_passed` in `enhanced_system/ops/training_system.py`; aligned `Makefile` typecheck target with CI scoped mypy (`enhanced_system/ops enhanced_system/core/cache enhanced_system/harness`); added revision parameter and pin to `resolve_target_modules_for_model` in `scripts/training/distill/trainer.py`; added targeted nosec annotations for controlled subprocess execution and local JSON dataset loading across `enhanced_system/harness/security.py`, `scripts/harness/run_aqa_gate.py`, and `scripts/training/train_dpo_adapter.py`.
- `test_trajectory_mode_filters_unsupervised_rows` failed because `max_length=16` truncated supervised tokens beyond char position 19 in a 20-char rendered body. Increased to `max_length=32`.
- `test_trajectory_eval_keeps_raw_rows` had latent truncation bug producing 0-row datasets silently. Added `max_length=32` and row-count assertion.
- Inline `# nosec B615` on 5 reviewed `from_pretrained()` calls that already pass `revision=` dynamically (false positives after global B615 skip removal).

### Changed
- Clarified docstring in `scripts/harness/run_aqa_gate.py` to reflect single-pass pass_rate regression gate, pointing to `run_pass_at_k.py` for multi-sample Pass@K.
- Collect path disables injection detection and keeps original task text (PII off) for the trajectory **and** backend/prefix. `run_agent` keeps injection on. `InputValidator` patterns are not edited. Collect and SCoRe-SFT score copy source `expected` onto legacy JSONL so DualDistill compose can pair. FTP is skipped when `resume_steps is not None` or `inject_action` is set, and the seed uses `planning.instruction`. `eval_harness` applies the threshold whenever `total > 0` (labeled exact-match combined with unlabeled completion) and skips per-row `ValueError`. Trajectory collator encodes the full rendered body once and requires offset mapping (fast tokenizer) when token/char counts differ. DualDistill `completion` is the merged thought/action sequence.
- Distillation format triangle, decorative `planning.style`, two training stacks, and paper false-friends: [docs/README_AGENT_DISTILLATION.md](docs/README_AGENT_DISTILLATION.md). Train and harness share one role-tagged string; SageMaker `predict_fn` stays raw-prompt single-shot.
- Gitleaks allowlists only checked-in training JSONL fixtures. Collect writes a distinct `*.raw.traces.jsonl` when `--output` already ends in `.raw.jsonl`. Trajectory alpha is validated in `[0.0, 1.0]`.
- FTP teacher prefix is cleared in `finally` after the first generate attempt.
- CI adds a `types` job and gitleaks on `security` beside existing lint / unit+integration@60 / harness@90. Action pins stay `checkout@v4` / `setup-python@v5` / `upload-artifact@v4`.
- Nested `enhanced_system/.pre-commit-config.yaml` is a stub; root hooks add gitleaks v8.21.2 and scoped mypy.
- Dockerfile false `:8080/health` HEALTHCHECK removed (`CMD` is still `basic_inference`). Image Python stays 3.9.
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

## [1.0.0] - 2026-09-10

### Added
- Installable `mangomas` package, root `pyproject.toml`, GitHub Actions CI.
- Shared SageMaker launcher and `MangoMASSettings`.
- Cache/errors/distill facades; JSON cache serialization (no pickle).
- ADRs 0001–0003 (JSON cache, unified launcher, coverage ratchet).
