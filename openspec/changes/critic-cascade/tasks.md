# Tasks: critic-cascade

- [x] 1. Add `enhanced_system/harness/critic.py` (or similar) with pure
  functions: `check_tool_allowlist`, `check_outcome`, `check_expected_tools`
  - no Hub calls in unit tests.
- [x] 2. Wire cascade and reject logging into `collect_trajectories.py` behind
  `MANGOMAS_CRITIC_ENABLED` (default false -> true once tests land); log at least
  `critic_reject_code: OUTCOME_MISMATCH` when outcome filter drops a row.
- [x] 3. Wire telemetry into `compose_dualdistill.py` / `compose_pair`: when existing
  (0,0) drop returns None, log `critic_reject_code: DUALDISTILL_DROP_0_0` (do not
  invent a second drop rule).
- [x] 4. Append structured reject records to JSONL sink `artifacts/critic_rejects.jsonl`
  (or `--reject-log` override).
- [x] 5. Support reject codes for later teacher-rule critic conforming to
  `openspec/changes/_shared/blocked-reject-codes.md`: `CYCLE_DETECTED`, `UNSAT`,
  `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`.
- [x] 6. Keep rows with intermediate parse/tool faults when outcome matches.
- [x] 7. Emit counters: `critic_rejected_*`, `critic_kept_recovery` alongside
  per-row `critic_reject_code` log entries.
- [x] 8. Unit tests for each filter and reject code emission; integration with Echo backend.
- [x] 9. Docs: distill README + NEXT_STEPS; CHANGELOG.
