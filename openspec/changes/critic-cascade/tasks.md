# Tasks — critic-cascade

- [ ] 1. Add `enhanced_system/harness/critic.py` (or similar) with pure
  functions: `check_tool_allowlist`, `check_outcome`, `check_expected_tools`
  — no Hub calls in unit tests.
- [ ] 2. Wire cascade into `collect_trajectories.py` behind
  `MANGOMAS_CRITIC_ENABLED` (default false → true once tests land).
- [ ] 3. Wire grade+filter metrics into `compose_dualdistill.py` logging.
- [ ] 4. Keep rows with intermediate parse/tool faults when outcome matches.
- [ ] 5. Emit counters: `critic_rejected_*`, `critic_kept_recovery`.
- [ ] 6. Unit tests for each filter; integration with Echo backend.
- [ ] 7. Docs: distill README + NEXT_STEPS; CHANGELOG.
