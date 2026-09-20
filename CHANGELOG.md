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
- **Phase P1 `golden-passk-aqa`:** Honest multi-trial evaluation, golden sets expansion, and CI-load-bearing AQA gate.
- Expanded golden JSONL schema (`prompt`, `expected`, optional `id`, `harness_id`, `expected_tools`, `slice` [`core`|`hard`], `allow_semantic`) with backwards-compatible Pydantic model `GoldenRow`.
- Expanded `configs/golden_sets/` with `core_sdlc.jsonl` and `hard_sdlc.jsonl`, backed by deterministic mock responses in `tests/fixtures/mock_responses.json`.
- Implemented genuine multi-trial `pass_at_1` and `pass_at_k` evaluation in `scripts/harness/eval_harness.py` configured via `MANGOMAS_EVAL_PASS_K` (default 3), `MANGOMAS_EVAL_PASS_N` (default 5), and `MANGOMAS_EVAL_PASS_K_TEMPERATURE` (default 0.8 for k>1).
- Enforced hard-slice exact match rule: hard rows require exact `answers_match` and no security faults (`semantic_match` alone fails); core slice rows require explicit `allow_semantic: true` for semantic matches.
- Updated `scripts/harness/run_aqa_gate.py` with honest single-pass docs/help text, directing to dedicated multi-sample Pass@K evaluation.
- Added rule traceability matrix `configs/rule_traceability/matrix.yaml` and validation CLI `scripts/harness/check_rule_matrix.py` ensuring all hard golden row IDs are mapped.
- Added prompt_render synchronization test (`tests/unit/test_prompt_render_sync.py`) guaranteeing parity between `enhanced_system/harness/prompt_render.py` and `scripts/training/distill/prompt_render.py`.
- Added first-class `aqa-gate` job in `.github/workflows/ci.yml` and wired into `Makefile` (`aqa-gate`, `validate`).
- OpenSpec coordination layer and symbolic-KD programme docs (`openspec/`, `docs/plans/symbolic-kd/{PLAN,REVIEW}.md`) locking P0–P4 disposition-first plan after tip `94e6c6a` peer review.


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

### Fixed
- Trunk CI hygiene (Phase P0 `green-trunk-ci`): formatted files across `enhanced_system`, `scripts`, and `tests` with `ruff format`; resolved mypy `no-any-return` on `PIIScrubber.redact_text` in `enhanced_system/harness/data_governance.py` and `_case_passed` in `enhanced_system/ops/training_system.py`; aligned `Makefile` typecheck target with CI scoped mypy (`enhanced_system/ops enhanced_system/core/cache enhanced_system/harness`); added revision parameter and pin to `resolve_target_modules_for_model` in `scripts/training/distill/trainer.py`; added targeted nosec annotations for controlled subprocess execution and local JSON dataset loading across `enhanced_system/harness/security.py`, `scripts/harness/run_aqa_gate.py`, and `scripts/training/train_dpo_adapter.py`.
- `test_trajectory_mode_filters_unsupervised_rows` failed because `max_length=16` truncated supervised tokens beyond char position 19 in a 20-char rendered body. Increased to `max_length=32`.
- `test_trajectory_eval_keeps_raw_rows` had latent truncation bug producing 0-row datasets silently. Added `max_length=32` and row-count assertion.
- Inline `# nosec B615` on 5 reviewed `from_pretrained()` calls that already pass `revision=` dynamically (false positives after global B615 skip removal).

- Native `enhanced_system.harness` runtime (YAML specs, frozen tool-id registry, AST/JSON dispatch, Echo/Transformers backends, rule-based tailor). Local CLIs: `scripts/harness/run_agent.py`, `collect_trajectories.py`, `eval_harness.py`, `build_memory.py`, `compose_dualdistill.py`, `collect_score.py`, `tailor_harness.py`.
- Shared `prompt_render` for harness generate and masked trajectory SFT (SageMaker-safe copy under `scripts/training/distill/`). Optional `planning.instruction` (Kang `I_agent`). `split_thought_action` remainder is the thought channel. Shared JSONL reader `enhanced_system.harness.jsonl` (skip vs `--strict`) for collect / eval / score / compose / memory.
- Collect `--teacher` / `--student` with outcome filter only when JSONL has `expected` (recovery faults kept). AMD-lite memory bank; DualDistill compose (drop `(0,0)`); SCoRe-SFT collect with teacher review prompt and preference-pair dump. Eval `--threshold/--student/--strict`; score `--prefs/--strict`.
- Trajectory SFT collator in `scripts/training/distill/trajectory_collator.py` and `--trajectory_mode` on `train_distilled_adapter.py` (alpha from `MANGOMAS_TRAJECTORY_DISTILL_ALPHA`, default `0.0`). SageMaker `create_job_spec` and `predict_fn` are unchanged.
- Dual coverage gates: global `fail_under=60` and harness `.coveragerc.harness` `fail_under=90`.
- Operator pack: `Makefile` (`make validate`), `.gitleaks.toml`, incremental mypy on `ops` / `core/cache` / `harness` (`python_version = 3.11`), tox `harness` env, composite `.github/actions/mangomas-validate` for skills (does not replace CI jobs).
- Cursor skills `mangomas-collect`, `mangomas-eval-harness`, `mangomas-memory`, `mangomas-dualdistill`, `mangomas-score`, and `mangomas-validate`; scan skill documents `make validate` while `cli:` stays `security_scan.py`.
- C4 L1–L3 (`docs/architecture/`) distinguishing Distill, Serve `predict_fn`, and local harness H.

### Changed

- Collect path disables injection detection and keeps original task text (PII off) for the trajectory **and** backend/prefix. `run_agent` keeps injection on. `InputValidator` patterns are not edited. Collect and SCoRe-SFT score copy source `expected` onto legacy JSONL so DualDistill compose can pair. FTP is skipped when `resume_steps is not None` or `inject_action` is set, and the seed uses `planning.instruction`. `eval_harness` applies the threshold whenever `total > 0` (labeled exact-match combined with unlabeled completion) and skips per-row `ValueError`. Trajectory collator encodes the full rendered body once and requires offset mapping (fast tokenizer) when token/char counts differ. DualDistill `completion` is the merged thought/action sequence.
- Distillation format triangle, decorative `planning.style`, two training stacks, and paper false-friends: [docs/README_AGENT_DISTILLATION.md](docs/README_AGENT_DISTILLATION.md). Train and harness share one role-tagged string; SageMaker `predict_fn` stays raw-prompt single-shot.
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
