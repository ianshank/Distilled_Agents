PYTHON ?= python3
PKG_PATHS := enhanced_system scripts tests
GITLEAKS ?= gitleaks
# Nested enhanced_system/pytest.ini testpaths would miss tests/harness/.
PYTEST := $(PYTHON) -m pytest --rootdir=$(CURDIR)

.PHONY: install lint fmt typecheck test test-harness test-aqa test-regression security gitleaks aqa-gate aqa-gate-passk validate collect-sqe-dispose compose-dualdistill-sqe test-critic-sqe-dispose train-dpo agent-pack-smoke test-e2e-gpu run-e2e-gpu

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check $(PKG_PATHS)
	$(PYTHON) -m ruff format --check $(PKG_PATHS)

fmt:
	$(PYTHON) -m ruff check --fix $(PKG_PATHS)
	$(PYTHON) -m ruff format $(PKG_PATHS)

typecheck:
	$(PYTHON) -m mypy enhanced_system/ops enhanced_system/core/cache enhanced_system/harness

test:
	$(PYTEST) -m "unit or integration" --cov=enhanced_system --cov-report=term --cov-fail-under=60

test-harness:
	$(PYTEST) -m harness --cov-config=.coveragerc.harness --cov=enhanced_system.harness --cov-report=term --cov-fail-under=80

test-aqa:
	$(PYTEST) -m "unit or integration or regression or harness" --cov=enhanced_system --cov-report=term --cov-fail-under=60
	$(PYTEST) -m harness --cov-config=.coveragerc.harness --cov=enhanced_system.harness --cov-report=term --cov-fail-under=80

test-regression:
	$(PYTEST) -m regression -v

security:
	$(PYTHON) -m bandit -r enhanced_system scripts -x tests,enhanced_system/tests,enhanced_system/examples -c pyproject.toml -q

gitleaks:
	$(GITLEAKS) detect --source . --verbose --redact --config .gitleaks.toml

aqa-gate:
	$(PYTHON) scripts/harness/run_aqa_gate.py --golden-set configs/golden_sets/core_sdlc.jsonl --threshold 75.0 --scripted tests/fixtures/mock_responses.json
	$(PYTHON) scripts/harness/run_aqa_gate.py --golden-set configs/golden_sets/hard_sdlc.jsonl --threshold 75.0 --scripted tests/fixtures/mock_responses.json
	$(PYTHON) scripts/harness/check_rule_matrix.py --matrix configs/rule_traceability/matrix.yaml --golden configs/golden_sets/hard_sdlc.jsonl

aqa-gate-passk:
	$(PYTHON) scripts/harness/run_pass_at_k.py --golden-set configs/golden_sets/sqe_hard_ood.jsonl --threshold 1.0 --scripted tests/fixtures/mock_responses_sqe_passk.json --output aqa-passk-summary.json

collect-sqe-dispose:
	mkdir -p artifacts/trajectories
	$(PYTHON) scripts/harness/collect_trajectories.py --input configs/golden_sets/sqe_hard_ood.jsonl --output artifacts/trajectories/sqe_dispose_teacher_a.jsonl --harness-id sqe_dispose --scripted tests/fixtures/mock_responses_sqe_passk.json --critic --reject-log artifacts/critic_rejects.jsonl
	cp -f artifacts/trajectories/sqe_dispose_teacher_a.jsonl artifacts/trajectories/sqe_dispose_teacher_b.jsonl

compose-dualdistill-sqe:
	mkdir -p artifacts/trajectories
	@if [ ! -f artifacts/trajectories/sqe_dispose_teacher_a.jsonl ] || [ ! -f artifacts/trajectories/sqe_dispose_teacher_b.jsonl ]; then \
		$(MAKE) collect-sqe-dispose; \
	fi
	$(PYTHON) scripts/harness/compose_dualdistill.py --first artifacts/trajectories/sqe_dispose_teacher_a.jsonl --second artifacts/trajectories/sqe_dispose_teacher_b.jsonl --output artifacts/trajectories/sqe_dispose_dualdistill.jsonl --reject-log artifacts/critic_rejects.jsonl

test-critic-sqe-dispose:
	$(PYTEST) -m "harness and e2_sqe"

train-dpo:
	$(PYTHON) scripts/training/train_dpo_adapter.py --model_name_or_path "gpt2" --dataset_path "tests/fixtures/dpo_preferences.jsonl" --epochs 1 --batch_size 1 --output_dir "./dpo_adapter_test"

agent-pack-smoke:
	$(PYTHON) scripts/infrastructure/agent_pack_smoke_test.py

test-e2e-gpu:
	$(PYTEST) tests/e2e/ -v -m "e2e and gpu"

run-e2e-gpu:
	$(PYTHON) scripts/harness/run_e2e_gpu.py

validate: lint typecheck test-aqa test-regression aqa-gate aqa-gate-passk agent-pack-smoke security gitleaks
