# C4 — Component (L3, core + ops only)

Component view is limited to `enhanced_system/core` and `enhanced_system/ops`. Code-level diagrams are omitted.

```mermaid
C4Component
    title enhanced_system core and ops
    Container_Boundary(core, "enhanced_system.core") {
        Component(router, "Adaptive router", "adaptive_router.py")
        Component(cache, "Cache facade", "cache_manager.py + cache/")
        Component(errors, "Error handler", "errors/")
        Component(consensus, "Consensus inference", "consensus_inference.py")
        Component(calibrator, "Confidence calibrator", "confidence_calibrator.py")
    }
    Container_Boundary(ops, "enhanced_system.ops") {
        Component(settings, "Settings", "settings.py")
        Component(launcher, "SageMaker launcher", "sagemaker_launcher.py")
        Component(training, "Training system", "training_system.py")
        Component(archive, "Safe tar extract", "archive.py")
    }
    Rel(router, cache, "lookup / store")
    Rel(router, errors, "retries / fallback")
    Rel(consensus, calibrator, "agreement + confidence")
    Rel(launcher, settings, "region, models, trust_remote_code")
    Rel(training, settings, "bucket, table, mock path")
    Rel(launcher, archive, "download extract")
```

JSON serialization is used for L2/L3 (`docs/adr/0001-json-cache-serialization.md`). All SageMaker CLIs share `MangoMASSageMakerLauncher` (`docs/adr/0002-unify-sagemaker-launchers.md`).
