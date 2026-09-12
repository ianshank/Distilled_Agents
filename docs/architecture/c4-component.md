# C4 — Component (L3)

Core + ops remain the inference library. Harness H is a sibling package: it may call `get_settings()` and `InputValidator`, but Serve `predict_fn` does not import the tool loop.

```mermaid
C4Component
    title Distill vs Serve vs Harness H
    Container_Boundary(core, "enhanced_system.core") {
        Component(router, "Adaptive router", "adaptive_router.py")
        Component(cache, "Cache facade", "cache_manager.py + cache/")
        Component(validator, "Input validator", "input_validator.py")
    }
    Container_Boundary(ops, "enhanced_system.ops") {
        Component(settings, "Settings", "settings.py")
        Component(launcher, "SageMaker launcher", "sagemaker_launcher.py")
    }
    Container_Boundary(harness, "enhanced_system.harness") {
        Component(runtime, "AgentRuntime", "runtime.py")
        Component(tools, "TOOL_REGISTRY", "tools/registry.py")
        Component(dispatch, "AST/JSON dispatch", "dispatch.py")
    }
    Container_Boundary(serve, "Serve") {
        Component(predict, "predict_fn", "scripts/inference.py")
    }
    Rel(runtime, settings, "get_settings(); not Config.harness limits")
    Rel(runtime, validator, "user-task only; collect turns injection off")
    Rel(runtime, tools, "frozen tool ids")
    Rel(runtime, dispatch, "no eval/exec")
    Rel(predict, tools, "not wired v1")
    Rel(launcher, settings, "region, models, trust_remote_code")
```

JSON serialization is used for L2/L3 (`docs/adr/0001-json-cache-serialization.md`). All SageMaker CLIs share `MangoMASSageMakerLauncher` (`docs/adr/0002-unify-sagemaker-launchers.md`). Runtime harness–policy pair: `docs/adr/0006-runtime-harness-policy.md`.
