# Ops configuration

Canonical YAML and agent profiles for local/dev/production.

`enhanced_system.config.load_config()` searches, in order:

1. `MANGOMAS_CONFIG_DIR` / `ENHANCED_SYSTEM_CONFIG_DIR`
2. this directory
3. `enhanced_system/config/` (installable package fallback)

Override AWS/model/instance defaults with `MANGOMAS_*` environment variables (see `.env.example`) rather than committing secrets.

Harness YAML lives in `configs/harnesses/` (search order: `MANGOMAS_HARNESS_DIR` → this tree → `enhanced_system/config/harnesses/`). Keep both trees byte-identical; a unit test asserts they match. Omitted numeric fields inherit `get_settings()`, not `${MANGOMAS_*}` interpolation in YAML.
