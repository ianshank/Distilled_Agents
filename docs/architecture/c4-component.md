# C4 — Component (L3)

Core + ops remain the inference library. Harness H is a sibling package: it may call `get_settings()` and `InputValidator`, but Serve `predict_fn` does not import the tool loop.

```mermaid
C4Component
    title Distill vs Serve vs Harness H
    Container_Boundary(core, "enhanced_system.core") {
        Component(router, "Adaptive router", "adaptive_router.py\n(Tier A Strict Typed)")
        Component(cache, "Cache facade", "cache_manager.py + cache/")
        Component(validator, "Input validator", "input_validator.py")
    }
    Container_Boundary(ops, "enhanced_system.ops") {
        Component(settings, "Settings", "settings.py\n(Tier A Strict Typed)")
        Component(launcher, "SageMaker launcher", "sagemaker_launcher.py")
    }
    Container_Boundary(harness, "enhanced_system.harness") {
        Component(runtime, "AgentRuntime", "runtime.py")
        Component(tools, "TOOL_REGISTRY", "tools/registry.py")
        Component(dispatch, "AST/JSON dispatch", "dispatch.py")
        Component(renderer, "prompt_render", "prompt_render.py")
        Component(memory, "memory_bank", "memory_bank.py")
        Component(dual, "dualdistill", "dualdistill.py")
        Component(score, "score", "score.py")
        Component(security, "SecurityScanner", "security.py")
        Component(pii, "PIIScrubber", "data_governance.py")
        Component(critic, "CriticCascade", "critic.py\n(Phase P2 filter & telemetry)")
        Component(solver, "Symbolic Tools", "tools/solver.py\n(sqe_constraint_solver, constraint_check)")
    }
    Container_Boundary(serve, "Serve") {
        Component(predict, "predict_fn", "scripts/inference.py\n(Single-shot + Prometheus + Blue/Green)")
    }
    Container_Boundary(train, "Training") {
        Component(dpo, "DPO Collator", "scripts/training/distill/dpo_collator.py")
        Component(dpo_script, "train_dpo", "scripts/training/train_dpo_adapter.py")
        Component(gkd, "GKD Adapter", "enhanced_system/training/gkd_adapter.py")
    }
    Container_Boundary(sdlc, "SDLC Automation & Gating") {
        Component(narrow_critic, "Narrow Critic", "Security/Style PR hook")
        Component(aqa_gate, "AQA Gate", "Deterministic Skill Validation (run_aqa_gate.py)")
        Component(passk_gate, "Pass@K Gate", "Chen Unbiased Estimator (run_pass_at_k.py)")
        Component(gpu_e2e, "GPU E2E Suite", "Live CUDA Loop (run_e2e_gpu.py)")
        Component(agent_pack, "Agent Pack", "Gated SDLC Pack (.agents/ + contract)")
        Component(tier_c, "Human Merge Gate", "Objective-gate verification")
    }
    Rel(runtime, settings, "get_settings(); not Config.harness limits")
    Rel(runtime, validator, "user-task only; collect turns injection off")
    Rel(runtime, tools, "frozen tool ids (including solver)")
    Rel(runtime, dispatch, "no eval/exec")
    Rel(runtime, renderer, "role-tagged turns; I_agent")
    Rel(runtime, memory, "workflow prefix; tool_error hints")
    Rel(runtime, security, "bandit scans on trajectory steps")
    Rel(runtime, pii, "presidio scrubbing on final answers")
    Rel(runtime, critic, "filters traces; emits structured reject codes")
    Rel(predict, tools, "not wired v1")
    Rel(launcher, settings, "region, models, trust_remote_code")
    Rel(dpo_script, dpo, "formats chosen/rejected via Chat Template")
    Rel(aqa_gate, tools, "runs deterministic validations")
    Rel(passk_gate, tools, "runs multi-trial Pass@K validation")
    Rel(narrow_critic, serve, "scans code for style/security")
```

JSON serialization is used for L2/L3 (`docs/adr/0001-json-cache-serialization.md`). All SageMaker CLIs share `MangoMASSageMakerLauncher` (`docs/adr/0002-unify-sagemaker-launchers.md`). Runtime harness–policy pair: `docs/adr/0006-runtime-harness-policy.md`. SageMaker collator uses `scripts/training/distill/prompt_render.py` (twin of `enhanced_system.harness.prompt_render`; no `enhanced_system` import).
Additionally, Tier A/B/C SDLC verification enforces `make aqa-gate` (deterministic behavior) and `narrow-critic` (security/style) before human merge (`Tier C`).
