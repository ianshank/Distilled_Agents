# Ops configuration

Canonical YAML and agent profiles for local/dev/production.

`enhanced_system.config.load_config()` searches, in order:

1. `MANGOMAS_CONFIG_DIR` / `ENHANCED_SYSTEM_CONFIG_DIR`
2. this directory
3. `enhanced_system/config/` (installable package fallback)

Override AWS/model/instance defaults with `MANGOMAS_*` environment variables (see `.env.example`) rather than committing secrets.
