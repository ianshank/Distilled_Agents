# Tasks — green-trunk-ci

- [ ] 1. `ruff format` the files failing `ruff format --check` (incl. at least:
  `data_governance.py`, `score.py`, `security.py`, `eval_harness.py`,
  `run_aqa_gate.py`, `inference.py`, `train_dpo_adapter.py`,
  `tests/test_dpo_collator.py`, and any others reported by CI).
- [ ] 2. Fix `enhanced_system/harness/data_governance.py` `redact_text` return
  type (`str(...)` / cast) so mypy `no-any-return` is clean.
- [ ] 3. Pin `revision=` (or documented local-only path) for Hub
  `from_pretrained` in `scripts/training/distill/trainer.py`; justify
  `# nosec B615` only where Hub is not contacted (e.g. local JSON
  `load_dataset`).
- [ ] 4. For AQA/security subprocess: keep `shell=False` list args; add
  targeted `# nosec B404/B603` with comment pointing at this change id, **or**
  extend `[tool.bandit] skips` only if project policy prefers skips — prefer
  nosec at call site for auditability.
- [ ] 5. Confirm `make validate` locally equivalent gates: lint, typecheck
  subset CI uses, bandit.
- [ ] 6. CHANGELOG entry under Fixed.
- [ ] 7. Merge only when Actions lint + types + security + test are green on
  the PR tip.
