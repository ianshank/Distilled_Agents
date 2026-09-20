# ADR 0008: TRL GKD / on-policy distillation usage and safety invariants

## Status

Accepted

## Context

Phase P4 of the Symbolic KD programme (`openspec/changes/trl-gkd-usage/`) addresses on-policy knowledge distillation for distilled agent policies. Following Generalized Knowledge Distillation (GKD; Agarwal et al., 2023) and on-policy trajectory distillation, student agents can be trained against teacher output distributions on rollout states.

Prior to this ADR:
1. `pyproject.toml` declared an open version range `trl>=0.14.0,<0.16.0` without an exact pin or usage contract.
2. Trajectory distillation alpha defaulted to `0.0` (`MANGOMAS_TRAJECTORY_DISTILL_ALPHA`), and GKD was deferred in `docs/README_AGENT_DISTILLATION.md`.
3. Naive adoption of GKD on agent loops presents significant risks:
   - **Cross-vocabulary divergence**: Computing KL divergence across disparate token spaces (e.g. 32K Mistral teacher vs 50K DialoGPT student) crashes tensor shapes or computes mathematically invalid divergence.
   - **Unmasked divergence**: Computing divergence across entire prompt sequences (system prompts, user instructions, environment observations, and error frames) violates the format triangle and degrades agent reasoning.
   - **Default creep**: Enabling GKD by default without passing hard-slice pass@k AQA gates risks masking agent harness regressions.

## Decision

1. **Exact TRL Version Pin (`trl==0.15.2`):**
   - Pin `trl==0.15.2` in `pyproject.toml` under `[project.optional-dependencies]` `alignment` and `all`.
   - *Rationale:* Version 0.15.2 is within prior policy (`>=0.14.0,<0.16.0`), provides stable `GKDTrainer` and `GKDConfig`, fixes SFT caching and LigerKernel compatibility, and natively aligns with `transformers>=4.45.0` and `accelerate>=0.34.0`.

2. **Fail-Closed Default Invariants:**
   - Default `trajectory_distill_alpha` and `MANGOMAS_TRAJECTORY_DISTILL_ALPHA` remain strictly **0.0**.
   - Default `gkd_enabled` and `MANGOMAS_GKD_ENABLED` remain strictly **False**.
   - Training runs pure supervised fine-tuning (SFT) on trajectory labels unless an operator explicitly opts into non-zero alpha and GKD.

3. **Label-Masked Divergence Loss:**
   - In `compute_label_masked_gkd_loss`, divergence (Generalized JSD, forward KL, or reverse KL) is computed **strictly on supervised token positions** where `labels != -100` (assistant thought and action spans).
   - Unlabeled positions (user turns, environment observations, system prompts, error frames) are excluded from the divergence calculation. If all labels are masked, zero loss is returned.

4. **Vocab Safety & Refusal:**
   - `validate_vocab_alignment` checks student and teacher vocabulary sizes. Mismatched vocabularies raise `ValueError` in strict mode.
   - In `compute_label_masked_gkd_loss` and `AgentGKDTrainer.compute_loss`, runtime logit vocabulary mismatch emits a `RuntimeWarning` and falls back safely to pure task cross-entropy loss. Under no circumstance is cross-vocab KL divergence computed.

5. **Config-Driven Adapter Module & CLIs:**
   - Implement `enhanced_system/training/gkd_adapter.py`, `scripts/training/distill/gkd_adapter.py`, and CLI `scripts/training/train_gkd_adapter.py`.
   - Settings are wired via `MangoMASSettings` / `get_settings()` with `MANGOMAS_*` prefix (`MANGOMAS_GKD_ENABLED`, `MANGOMAS_GKD_LMBDA`, `MANGOMAS_GKD_BETA`, `MANGOMAS_GKD_TEMPERATURE`, `MANGOMAS_GKD_MAX_NEW_TOKENS`, `MANGOMAS_GKD_SEQ_KD`).
   - No hardcoded Hub tokens, AWS account IDs, or magic model paths exist in call sites.

6. **When GKD May Be Enabled:**
   An operator may enable GKD (`MANGOMAS_GKD_ENABLED=true` and `MANGOMAS_TRAJECTORY_DISTILL_ALPHA > 0.0`) **only when all** of the following conditions hold:
   - Student and teacher models share identical vocabulary and tokenization (e.g. Qwen2.5-7B teacher to Qwen2.5-0.5B student).
   - Trajectory data is collated with `TrajectoryDataCollator` ensuring assistant span labeling.
   - Local training rule conditions are met (occasional runs, $\le$10k traces, local compute; not the SageMaker 4.26/1.13 image).
   - Hard-slice pass@k AQA gate in CI remains green.

## Consequences

- Distillation engineers have a sound, tested on-policy distillation path without risking cross-vocab divergence or silent training failures.
- Unit testing environments without `trl` installed remain supported via clean import guards.
- SageMaker estimator images (`transformers 4.26.0 / PyTorch 1.13.1`) and GRPO/SDAR remain explicit non-goals.
- Opens the path to Phase 5 edge profiling and evaluation.
