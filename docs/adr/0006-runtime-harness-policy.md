# ADR 0006: Runtime harness–policy pair

## Status

Accepted

## Context

Distilled_Agents distilled role LoRAs as prompt/completion pairs with no reason-act-observe loop. Kang et al. (2025) train students on agent trajectories with observation masking. HarnessForge (2026) treats an agent as `G = (H, R_δ)`. SageMaker `source_dir` is `scripts/training` and cannot import `enhanced_system`. HuggingFace `DataCollatorForLanguageModeling` overwrites labels.

## Decision

1. Runtime `H` lives in `enhanced_system.harness` (YAML specs, frozen tool-id registry, AST/JSON dispatch, no `exec`).
2. Policy `R_δ` remains the existing LoRA student. Trajectory collection is a local CLI. Masked collator lives in `scripts/training/distill/trajectory_collator.py`. `--trajectory_mode` on `train_distilled_adapter.py` sets `distillation_alpha=0`. `create_job_spec` is unchanged.
3. Operator skills (ADR 0004) are a separate layer from the runtime harness.
4. SageMaker `predict_fn` stays single-shot; local `scripts/harness/run_agent.py` is the tool loop.

## Consequences

A CodeAct-distilled student on today's endpoint will not run tools until a later endpoint handler is added. YAML omitted fields inherit `MangoMASSettings`. Rule-based YAML tailor archives patches unless `MANGOMAS_HARNESS_APPLY_PATCHES` is true.
