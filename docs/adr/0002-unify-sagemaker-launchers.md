# ADR 0002: Unify SageMaker launchers

## Status

Accepted

## Context

Multiple SageMaker CLIs duplicated `MangoMASSageMakerLauncher` (GPU vs CPU, simple vs full).

## Decision

Keep one implementation in `enhanced_system.ops.sagemaker_launcher.MangoMASSageMakerLauncher`. Scripts under `scripts/deployment/` are thin CLIs with stable flags.

## Consequences

Behavior stays compatible for operators; instance type is a parameter (`--cpu` or settings) instead of a forked file.
