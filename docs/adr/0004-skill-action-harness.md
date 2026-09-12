# ADR 0004: Skill and composite-action harness

## Status

Accepted

## Context

Operators and coding agents need a stable way to train, evaluate, launch, and scan without copying one-off commands or calling live AWS from CI.

## Decision

Version Cursor skills under `.cursor/skills/mangomas-*/SKILL.md` with YAML frontmatter (`name`, `description`, `cli`, `inputs`). Skills call existing CLIs only.

Deterministic tests in `tests/harness/test_skills_contract.py`:

- Frontmatter keys present and `cli` path exists
- Public CLI `--help` flag allowlists (including collect `--input/--output/--harness-id/--strict`)
- AGENTS.md skill paths and CLI column match `.cursor/skills/*` names and frontmatter `cli`
- `mangomas-validate` points at `Makefile` (`make validate`); scan `cli:` stays `security_scan.py`
- `get_settings()` defaults (`trust_remote_code is False`, `bind_host == "127.0.0.1"`)
- boto3 is monkeypatched (no network)

## Consequences

Flag drift fails CI. Skills cannot silently point at removed scripts. Live SageMaker e2e remains out of scope. Makefile/gitleaks live in ADR 0005; this ADR only requires skill contracts.
