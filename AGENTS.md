# Agent guide (MangoMAS)

This repository trains, evaluates, and launches distilled LoRA agents. Prefer checked-in CLIs and Cursor skills over ad-hoc scripts.

## Skills

| Task | Skill | CLI |
| --- | --- | --- |
| Train a role adapter | `.cursor/skills/mangomas-train/SKILL.md` | `scripts/training/train_agent_skill.py` |
| Evaluate pass rate | `.cursor/skills/mangomas-evaluate/SKILL.md` | `scripts/evaluation/evaluate_agent_skill.py` |
| Launch SageMaker jobs | `.cursor/skills/mangomas-launch/SKILL.md` | `scripts/deployment/simple_launch_sagemaker.py` |
| Lint / bandit | `.cursor/skills/mangomas-scan/SKILL.md` | `scripts/infrastructure/security_scan.py` |
| Run the local harness | `.cursor/skills/mangomas-harness/SKILL.md` | `scripts/harness/run_agent.py` |
| Tailor harness YAML | `.cursor/skills/mangomas-tailor/SKILL.md` | `scripts/harness/tailor_harness.py` |

## Rules

- Read settings from `get_settings()` (`MANGOMAS_` prefix). Never hardcode AWS account IDs, access keys, or production bucket names.
- Import `MangoMASSageMakerLauncher` from `enhanced_system.ops`.
- Import `HarnessFactory` / `AgentRuntime` from `enhanced_system.harness`, not from `enhanced_system.core`.
- Keep `trust_remote_code` false unless an operator sets `MANGOMAS_TRUST_REMOTE_CODE=true`.
- Bind local Flask inference to `MANGOMAS_BIND_HOST` (default `127.0.0.1`).
- SageMaker `predict_fn` stays single-shot. Tool loops run locally via `scripts/harness/run_agent.py`.
- Trajectory SFT is local: `python scripts/training/train_distilled_adapter.py --trajectory_mode True` (forces `distillation_alpha=0`). Do not add launcher hyperparameters until argparse and `create_job_spec` change together.
- Do not add live AWS e2e tests. Contract tests must monkeypatch boto3.

## Pre-PR

```bash
pytest -m "unit or integration"
pytest -m harness --cov-config=.coveragerc.harness
```
