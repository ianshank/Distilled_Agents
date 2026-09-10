# Agent guide (MangoMAS)

This repository trains, evaluates, and launches distilled LoRA agents. Prefer checked-in CLIs and Cursor skills over ad-hoc scripts.

## Skills

| Task | Skill | CLI |
| --- | --- | --- |
| Train a role adapter | `.cursor/skills/mangomas-train/SKILL.md` | `scripts/training/train_agent_skill.py` |
| Evaluate pass rate | `.cursor/skills/mangomas-evaluate/SKILL.md` | `scripts/evaluation/evaluate_agent_skill.py` |
| Launch SageMaker jobs | `.cursor/skills/mangomas-launch/SKILL.md` | `scripts/deployment/simple_launch_sagemaker.py` |
| Lint / secrets / bandit | `.cursor/skills/mangomas-scan/SKILL.md` | `make validate` |

## Rules

- Read settings from `get_settings()` (`MANGOMAS_` prefix). Never hardcode AWS account IDs, access keys, or production bucket names.
- Import `MangoMASSageMakerLauncher` from `enhanced_system.ops`, not from the thin `launch_all_agents_sagemaker` CLI.
- Keep `trust_remote_code` false unless an operator sets `MANGOMAS_TRUST_REMOTE_CODE=true`.
- Bind local Flask inference to `MANGOMAS_BIND_HOST` (default `127.0.0.1`).
- Do not add live AWS e2e tests. Contract tests must monkeypatch boto3.
- Do not split remaining 400-line core modules in the same change as harness/docs work.

## Pre-PR

```bash
make validate
```

Equivalent composite action: `.github/actions/mangomas-validate`.
