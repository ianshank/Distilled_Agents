# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
