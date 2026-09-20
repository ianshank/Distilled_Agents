# Next steps (out of this harness + operator pack)

## Programme of record — Symbolic KD (2026-09-20)

Disposition-first symbolic knowledge distillation is tracked here:

- Plan: [docs/plans/symbolic-kd/PLAN.md](plans/symbolic-kd/PLAN.md)
- Peer review: [docs/plans/symbolic-kd/REVIEW.md](plans/symbolic-kd/REVIEW.md)
- OpenSpec index: [openspec/README.md](../openspec/README.md)

**Order:** P0 green trunk → P1 golden + real pass@k AQA in CI → P2 critic cascade → P3 solver dispose tools → P4 TRL GKD *usage* (gated). Phase 6 SageMaker registry / A/B MUST NOT leapfrog P0–P3. Do not treat #16’s AQA docstring as pass@k delivery.

*Note (Phase P1 completed):* P1 `golden-passk-aqa` has implemented genuine multi-trial pass@1 and pass@k, expanded core and hard golden corpora, rule traceability verification, and CI-load-bearing scripted AQA gating. Unlocks Phase P2 (`critic-cascade`).

*Note (Phase P2 completed):* P2 `critic-cascade` has implemented pure critic helpers (`enhanced_system/harness/critic.py`), allowlist and outcome cascade filtering on `collect_trajectories.py`, DualDistill (0,0) drop telemetry, recovery trace retention (`critic_kept_recovery`), structured JSONL reject sink (`artifacts/critic_rejects.jsonl`), and standard reject codes per `_shared/blocked-reject-codes.md`. Unlocks Phase P3 (`symbolic-disposition`).

---

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

Train and harness now share `prompt_render` (see [README_AGENT_DISTILLATION.md](README_AGENT_DISTILLATION.md)). JSONL skip/strict helper, expected-preserve on collect/score, and FTP resume/inject skip landed. Defer splitting `test_harness_runtime.py`. Remaining:

- Wire an endpoint handler if distilled CodeAct students should run tools in SageMaker (`predict_fn` stays single-shot until then). That is the last vertex of the format triangle.
- Add `trajectory_mode` to `create_job_spec` only together with launcher argparse (SageMaker `source_dir` cannot import `enhanced_system`; the 4.26/1.13 image is the wrong stack for Qwen trajectory SFT).
- CodeAct sandbox (no `eval`/`exec` in-process). Execution-consistent SAG (vote on observations, not parse majority).
- Multiprocess collect needs `fcntl` (or equivalent) around `JsonlTraceStore`; in-process `threading.Lock` is enough for unit tests.
- SAG is parse/schema majority vote (`N=1` default), not Kang execute-and-vote.
- Local SDPO/GKD only after an explicit TRL/peft pin PR. Defer GRPO, SCoRe-RL, SDAR, AgentArk PAD, MCP autotools.
- DualDistill compose is same-task + expected + two teachers. Do not treat mixed-role JSONL concat as DualDistill.
- Do not turn on `distillation_alpha` in trajectory mode before KL is label-masked and vocab-aligned.

## Dead architecture surface

Wire or delete unused factories for router/batch/calibrator/consensus/monitor, and `BaseProcessor` if it still has no implementer.

## Docker / compose

`.dockerignore` exists at repo root. Compose `context: ../..` from `enhanced_system/infrastructure/docker/` still does not match a root-context Dockerfile layout; fix that in a dedicated image PR. Do not bump the image Python 3.9 pin here.

## Phase 6: Deployment & A/B Testing

With Phases 1-5 closed out on `feature/distillation-expansion`, the immediate next steps are:
1. **Model Registration**: Wire up the completed SageMaker LoRA training outputs to the SageMaker Model Registry.
2. **Endpoint Provisioning**: Expand `simple_launch_sagemaker.py` to support real-time endpoint deployment of the compiled DPO weights.
3. **A/B Testing Framework**: Extend the Prometheus `/metrics` pipeline we just added to include cohort routing telemetry, allowing us to mathematically compare base Agent implementations against Distilled versions via the `predict_fn`.

## Dependencies

The `peft==0.4.0` training-image pin has successfully been upgraded to `peft>=0.14.0,<0.15.0`. SageMaker estimators now run `transformers 4.45.0` natively. The next step is evaluating Qwen2.5 integration fully on SageMaker instances.

## Types

Strict type-checking (`disallow_untyped_defs = true`) has been successfully established for `enhanced_system.core`, `enhanced_system.evaluation`, `enhanced_system.training`, `enhanced_system.config`, `enhanced_system.harness`, and `enhanced_system.ops`. The remaining frontier is `scripts/`.

## Tests

Do not add live AWS e2e. Keep `e2e` / `slow` / `benchmark` unused for **live AWS**. Local Echo harness pipelines stay `integration` (and `harness`). Contract tests must keep boto3 monkeypatched. Harness code coverage is now enforced at 80% via `--cov-fail-under=80`.

## Dependabot

Leave GitHub Actions major bumps (#4/#5/#6, checkout/setup-python/upload-artifact v7) unmerged until the harness coverage job is re-checked against those runners. Pip PRs (#7–#11) stay separate.
