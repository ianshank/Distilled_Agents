# Agent guide (MangoMAS)

This repository trains, evaluates, and launches distilled LoRA agents. Prefer checked-in CLIs and Cursor skills over ad-hoc scripts.

Shared chat format: `enhanced_system/harness/prompt_render.py` and `scripts/training/distill/prompt_render.py` must stay in sync. Paper mappings and false-friends: `docs/README_AGENT_DISTILLATION.md`.

## Skills

| Task | Skill | CLI |
| --- | --- | --- |
| Train a role adapter | `.cursor/skills/mangomas-train/SKILL.md` | `scripts/training/train_agent_skill.py` |
| Evaluate skill fixtures | `.cursor/skills/mangomas-evaluate/SKILL.md` | `scripts/evaluation/evaluate_agent_skill.py` |
| Evaluate harness H | `.cursor/skills/mangomas-eval-harness/SKILL.md` | `scripts/harness/eval_harness.py` |
| Launch SageMaker jobs | `.cursor/skills/mangomas-launch/SKILL.md` | `scripts/deployment/simple_launch_sagemaker.py` |
| Lint / bandit | `.cursor/skills/mangomas-scan/SKILL.md` | `scripts/infrastructure/security_scan.py` |
| Pre-PR validate | `.cursor/skills/mangomas-validate/SKILL.md` | `Makefile` |
| Run the local harness | `.cursor/skills/mangomas-harness/SKILL.md` | `scripts/harness/run_agent.py` |
| Collect trajectories | `.cursor/skills/mangomas-collect/SKILL.md` | `scripts/harness/collect_trajectories.py` |
| Build AMD-lite memory | `.cursor/skills/mangomas-memory/SKILL.md` | `scripts/harness/build_memory.py` |
| DualDistill compose | `.cursor/skills/mangomas-dualdistill/SKILL.md` | `scripts/harness/compose_dualdistill.py` |
| SCoRe-SFT collect | `.cursor/skills/mangomas-score/SKILL.md` | `scripts/harness/collect_score.py` |
| Tailor harness YAML | `.cursor/skills/mangomas-tailor/SKILL.md` | `scripts/harness/tailor_harness.py` |
| Deterministic AQA gate | `.cursor/skills/mangomas-validate/SKILL.md` | `scripts/harness/run_aqa_gate.py` |
| Pass@K hard/OOD gate | `.cursor/skills/mangomas-validate/SKILL.md` | `scripts/harness/run_pass_at_k.py` |
| Train DPO preference adapter | `.cursor/skills/mangomas-train/SKILL.md` | `scripts/training/train_dpo_adapter.py` |

## Rules

- Read settings from `get_settings()` (`MANGOMAS_` prefix). Never hardcode AWS account IDs, access keys, or production bucket names. Use `MANGOMAS_AWS_ACCOUNT_ID` in headless environments without AWS credentials.
- Import `MangoMASSageMakerLauncher` from `enhanced_system.ops`.
- Import `HarnessFactory` / `AgentRuntime` from `enhanced_system.harness`, not from `enhanced_system.core`.
- Keep `trust_remote_code` false unless an operator sets `MANGOMAS_TRUST_REMOTE_CODE=true`.
- Bind local Flask inference to `MANGOMAS_BIND_HOST` (default `127.0.0.1`) and `MANGOMAS_PORT` (default `8080`). Standardize config on `MANGOMAS_MODEL_DIR`.
- Protect inference endpoints against DoS via `MANGOMAS_MAX_PROMPT_BYTES` (default 1MB).
- Adapter auth tokens must be compared using constant-time `hmac.compare_digest`.
- SageMaker `predict_fn` stays single-shot. Tool loops run locally via `scripts/harness/run_agent.py`.
- Collect JSONL is trusted: `collect_trajectories.py` disables injection detection. Interactive `run_agent.py` keeps it on.
- Trajectory SFT is local: `python scripts/training/train_distilled_adapter.py --trajectory_mode True` (alpha from `MANGOMAS_TRAJECTORY_DISTILL_ALPHA`, default `0.0`). Do not add launcher hyperparameters until argparse and `create_job_spec` change together. Filtered datasets must retain >=1 row or raise `ValueError`.
- `evaluate_agent_skill` is a JSON fixture counter. Agent-loop quality is `eval_harness.py`.
- DualDistill is same-task + expected + two teachers (`compose_dualdistill.py`), not concatenated role JSONL. `planning.style` is unused by the loop.
- AMD-lite memory is workflow/function hints, not `HarnessTailor` drop-tool.
- Do not add live AWS e2e tests. Contract tests must monkeypatch boto3.

## Antigravity & SDLC Agents

This workspace integrates the gated Product + SDLC + ML agent pack (`.agents/`, `config/workflow-contract.json`, `GEMINI.md`):

1. **Custom Agents** (`.agents/agents/`):
   - `product-intake`: Opportunity framing and requirements intake.
   - `researcher`: Read-only literature and codebase exploration.
   - `spec-writer`: OpenSpec proposals, design docs, and tasks (`openspec/changes/`).
   - `architect` / `ml-framer`: Architecture, component modeling, and experiment plans.
   - `delivery`: Main implementation agent (Planning Mode, test reports).
   - `critic`: Read-only dialectic review, falsifier verification, and schema checks.
   - `releaser`: Release notes and shipping gates.
2. **Workflow Contract & Invariants** (`config/workflow-contract.json`, `.agents/rules/workflow-invariants.md`):
   - Strict stages: `discover` -> `specify` -> `design` -> `build` -> `evaluate` -> `ship` -> `learn`.
   - Current stage tracked in `artifacts/stage.json`.
   - Ship requires `artifacts/eval_decision.json` with decision `ship`.
3. **Automated Gates & Hooks** (`.agents/hooks.json`):
   - `PreToolUse`: Verifies allowed write prefixes and command allowlists via `.agents/hooks/pre_tool_use.py`.
   - `Stop`: Verifies stage artifact completeness before allowing turns to conclude via `.agents/hooks/stop_gate.py`.

## Pre-PR

```bash
make validate
pytest -m "unit or integration or regression or harness" --cov=enhanced_system --cov-report=term --cov-fail-under=60
pytest -m harness --cov-config=.coveragerc.harness --cov=enhanced_system.harness --cov-report=term --cov-fail-under=80
```

