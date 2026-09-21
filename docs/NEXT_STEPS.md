# Next steps (out of this harness + operator pack)

## Programme of record - Symbolic KD (2026-09-20)

Disposition-first symbolic knowledge distillation is tracked here:

- Plan: [docs/plans/symbolic-kd/PLAN.md](plans/symbolic-kd/PLAN.md)
- Peer review: [docs/plans/symbolic-kd/REVIEW.md](plans/symbolic-kd/REVIEW.md)
- OpenSpec index: [openspec/README.md](../openspec/README.md)

**Order:** P0 green trunk -> P1 golden + real pass@k AQA in CI (implemented in Phase 0 I1) -> P2 critic cascade -> P3 solver dispose tools -> P4 TRL GKD *usage* (gated). Phase 6 SageMaker registry / A/B MUST NOT leapfrog P0-P3. Do not treat #16's AQA docstring as pass@k delivery.

*Note (Phase P1 completed):* P1 `golden-passk-aqa` has implemented genuine multi-trial pass@1 and pass@k, expanded core and hard golden corpora, rule traceability verification, and CI-load-bearing scripted AQA gating. Unlocks Phase P2 (`critic-cascade`). Archived under `openspec/changes/archive/golden-passk-aqa/`.

*Note (Phase P2 completed):* P2 `critic-cascade` has implemented pure critic helpers (`enhanced_system/harness/critic.py`), allowlist and outcome cascade filtering on `collect_trajectories.py`, DualDistill (0,0) drop telemetry, recovery trace retention (`critic_kept_recovery`), structured JSONL reject sink (`artifacts/critic_rejects.jsonl`), and standard reject codes per `_shared/blocked-reject-codes.md`. Unlocks Phase P3 (`symbolic-disposition`). Archived under `openspec/changes/archive/critic-cascade/`.

*Note (Phase P3 / I3 completed; Phase 0 kill CLEARED):* P3 / I3 `symbolic-disposition` has implemented pure-Python constraint checking (`constraint_check`) and DAG topological sort + boolean constraint solving (`sqe_constraint_solver`), dedicated harness YAML specs (`qc_constraints`, `sqe_dispose`), fail-closed OOD refusal (`BLOCKED:<CODE>`), hard golden coverage, rule traceability mapping, candidate rule extraction (`scripts/harness/extract_rules.py`), unit proofs, and canonical ADR 0007 (`docs/adr/0007-symbolic-dispose-tools.md`). Phase 0 intake kill criteria are CLEARED on main citing GitHub Actions run `35518614849` and artifacts `aqa-passk-summary.json` + `critic-rejects`. Phase 2-3 / Serve / Edge-AI / INV-16 remain locked. GKD remains optional behind ADR 0008 (`trl-gkd-usage`). Next step is E2E functional proof (Pass@K x sqe_dispose; tracked in `openspec/changes/sqe-dispose-e2e-bind/`). Archived under `openspec/changes/archive/symbolic-disposition/`.

*Note (Phase P4 completed):* P4 `trl-gkd-usage` has pinned `trl==0.15.2`, implemented the GKD/on-policy adapter (`enhanced_system/training/gkd_adapter.py`, `scripts/training/distill/gkd_adapter.py`, `scripts/training/train_gkd_adapter.py`), enforced label-masked divergence and fail-safe vocab mismatch checks, preserved default alpha=0.0 and fail-closed GKD gating, and documented ADR 0008. GKD usage remains optional behind ADR 0008 and gated on E2E BC smoke.

*Note (Local GPU E2E & Antigravity Agent Pack completed):* Implemented checked-in local GPU E2E verification suite (`scripts/harness/run_e2e_gpu.py`, `tests/e2e/test_local_gpu_e2e.py`) executing live CUDA inference and multi-turn agent loops on workstation GPU hardware (NVIDIA GeForce RTX 5060 Ti, 15.93 GB VRAM). Integrated Antigravity gated SDLC agent pack (`.agents/`, `config/workflow-contract.json`, `GEMINI.md`) with stage-aware pre-tool path gates, smoke tested and verified clean across full pre-PR test suite.

---

Tracked follow-ups so this change stays reviewable without mixing large refactors.

## Remaining module splits

Keep facades; split implementations in a dedicated PR when a file stays above ~300-400 lines. Do not split these here:

- `enhanced_system/core/adaptive_router.py`
- `enhanced_system/core/input_validator.py`
- `enhanced_system/evaluation/skill_evaluator.py`
- `enhanced_system/core/confidence_calibrator.py`
- `enhanced_system/core/monitoring.py`
- `enhanced_system/core/consensus_inference.py`
- `enhanced_system/training/data_curator.py`

Do not edit `InputValidator` regexes to "allow coding prompts"; serving tests require `SELECT ... FROM` to fail. Collect vs `run_agent` already forks `enable_injection_detection`.

## Coverage ratchet toward 80% on **core**

ADR 0003: global `fail_under` stays **60**. Dual-marked harness tests inflate tree coverage; do not raise 60->65 on that harness-inflated number. Raise monotonically toward 80% on core modules, with a later second job for `scripts/` at a lower gate.

Harness package gate stays `.coveragerc.harness` `fail_under=90`. Do not omit `protocols.py` from the *global* omit list.

## Harness follow-ups

Train and harness now share `prompt_render` (see [README_AGENT_DISTILLATION.md](README_AGENT_DISTILLATION.md)). JSONL skip/strict helper, expected-preserve on collect/score, and FTP resume/inject skip landed. Defer splitting `test_harness_runtime.py`. Remaining:

- Wire an endpoint handler if distilled CodeAct students should run tools in SageMaker (`predict_fn` stays single-shot until then). That is the last vertex of the format triangle.
- Add `trajectory_mode` to `create_job_spec` only together with launcher argparse (SageMaker `source_dir` cannot import `enhanced_system`; the 4.26/1.13 image is the wrong stack for Qwen trajectory SFT).
- CodeAct sandbox (no `eval`/`exec` in-process). Execution-consistent SAG (vote on observations, not parse majority).
- Multiprocess collect needs `fcntl` (or equivalent) around `JsonlTraceStore`; in-process `threading.Lock` is enough for unit tests.
- SAG is parse/schema majority vote (`N=1` default), not Kang execute-and-vote.
- TRL GKD / on-policy distillation is optionally available behind ADR 0008 (pinned `trl==0.15.2`). Defer GRPO, SCoRe-RL, SDAR, AgentArk PAD, MCP autotools.
- Trajectory mode KL is now label-masked and vocab-aligned with default `alpha=0.0`; enable only when operator explicitly configures `MANGOMAS_TRAJECTORY_DISTILL_ALPHA > 0` on compatible model pairs.

## Dead architecture surface

Wire or delete unused factories for router/batch/calibrator/consensus/monitor, and `BaseProcessor` if it still has no implementer.

## Docker / compose

`.dockerignore` exists at repo root. Compose `context: ../..` from `enhanced_system/infrastructure/docker/` still does not match a root-context Dockerfile layout; fix that in a dedicated image PR. Do not bump the image Python 3.9 pin here.

## Phase 6: Deployment & A/B Testing

With Phases 1-5 closed out on `feature/distillation-expansion`, the immediate next steps are:
1. **Model Registration**: Wire up the completed SageMaker LoRA training outputs to the SageMaker Model Registry.
2. **Endpoint Provisioning**: Expand `simple_launch_sagemaker.py` to support real-time endpoint deployment of the compiled DPO weights.
3. **A/B Testing Framework**: Extend the Prometheus `/metrics` pipeline we just added to include cohort routing telemetry, allowing us to mathematically compare base Agent implementations against Distilled versions via the `predict_fn`.

*Note (Phase 6 SDLC Dry Run completed):* The `train_distilled_adapter.py` pipeline was updated to resolve breaking API changes in Transformers v4.41+ and v4.46+. A local Distillation Dry Run, backed by deterministic `aqa-gate` thresholding on `core_sdlc.jsonl` and `hard_sdlc.jsonl`, passed entirely. The CI pipeline now mandates these AQA checks and `/narrow-critic` style/security scanning before human merge.

## Dependencies

The `peft==0.4.0` training-image pin has successfully been upgraded to `peft>=0.14.0,<0.15.0`. SageMaker estimators now run `transformers 4.45.0` natively. The next step is evaluating Qwen2.5 integration fully on SageMaker instances.

## Types

Strict type-checking (`disallow_untyped_defs = true`) has been successfully established for `enhanced_system.core`, `enhanced_system.evaluation`, `enhanced_system.training`, `enhanced_system.config`, `enhanced_system.harness`, and `enhanced_system.ops`. The remaining frontier is `scripts/`.

## Tests

Do not add live AWS e2e. Keep `e2e` / `slow` / `benchmark` unused for **live AWS**. Local Echo harness pipelines stay `integration` (and `harness`). Contract tests must keep boto3 monkeypatched. Harness code coverage is now enforced at 80% via `--cov-fail-under=80`. Dedicated regression suite (`tests/test_regression_suite.py`) enforces SEC-001..003, CQ-001..004, CFG-001..002, and TRN-001. Deterministic gating runs across `core_sdlc.jsonl`, `hard_sdlc.jsonl`, and `sqe_hard_ood.jsonl` (Pass@K Chen estimator). Local GPU E2E validation is checked in under `scripts/harness/run_e2e_gpu.py` and `tests/e2e/test_local_gpu_e2e.py` (marked `@pytest.mark.e2e` and `@pytest.mark.gpu`), validating local CUDA execution with zero live AWS dependencies.

## Dependabot

Leave GitHub Actions major bumps (#4/#5/#6, checkout/setup-python/upload-artifact v7) unmerged until the harness coverage job is re-checked against those runners. Pip PRs (#7-11) stay separate.
