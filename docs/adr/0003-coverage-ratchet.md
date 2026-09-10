# ADR 0003: Coverage ratchet in CI

## Status

Accepted

## Context

Tox previously failed under 80% coverage while tests did not match APIs and CI did not exist.

## Decision

GitHub Actions reports coverage on every PR. `fail_under` is introduced only after a measured baseline and is raised toward 80% on `enhanced_system` core modules. Lint format may be non-blocking until the tree is ruff-formatted.

## Consequences

CI can stay green while coverage increases honestly. Gaming coverage by omitting modules is avoided by adding ops tests with a separate (lower) gate later.
