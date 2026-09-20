# Spec delta — on-policy-kd

## ADDED Requirements

### Requirement: Exact TRL pin before GKD usage

GKD/on-policy training code MUST depend on an exact TRL version pin, not only
a open range.

### Requirement: Default trajectory alpha remains zero

Unless an operator sets `MANGOMAS_TRAJECTORY_DISTILL_ALPHA` > 0, training MUST
not apply teacher KL.

### Requirement: Vocab and label-mask safety

When alpha > 0, KL MUST be label-masked and MUST refuse unsafe vocab mismatch
(no silent wrong KL). Existing DistillationTrainer invariants remain the floor.

### Requirement: Gate on harness quality

Merging a PR that enables GKD by default MUST be blocked while hard-slice
pass@k CI is failing.

## Falsifier

A test with mismatched vocabs and alpha>0 MUST NOT compute cross-vocab KL.
