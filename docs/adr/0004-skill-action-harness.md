# ADR 0004: Skill and composite-action harness

## Status

Accepted

## Context

Operators and coding agents need a stable way to train, evaluate, launch, and scan without copying one-off commands or calling live AWS from CI.

## Decision

Version Cursor skills under `.cursor/skills/mangomas-*/SKILL.md` with YAML frontmatter (`name`, `description`, `cli`, `inputs`). Skills call existing CLIs only. GitHub composite action `.github/actions/mangomas-validate` wraps `make validate`.

Deterministic tests in `tests/harness/test_skills_contract.py`:

- Frontmatter keys present and `cli` path exists
- Public CLI `--help` flag allowlists
- `get_settings()` defaults (`trust_remote_code is False`, `bind_host == "127.0.0.1"`)
- boto3 is monkeypatched (no network)

## Consequences

Flag drift fails CI. Skills cannot silently point at removed scripts. Live SageMaker e2e remains out of scope.
