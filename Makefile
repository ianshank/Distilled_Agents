PYTHON ?= python3
PKG_PATHS := enhanced_system scripts tests
GITLEAKS ?= gitleaks
# Nested enhanced_system/pytest.ini testpaths would miss tests/harness/.
PYTEST := $(PYTHON) -m pytest --rootdir=$(CURDIR)

.PHONY: install lint fmt typecheck test test-harness test-aqa security gitleaks validate

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check $(PKG_PATHS)
	$(PYTHON) -m ruff format --check $(PKG_PATHS)

fmt:
	$(PYTHON) -m ruff check --fix $(PKG_PATHS)
	$(PYTHON) -m ruff format $(PKG_PATHS)

typecheck:
	$(PYTHON) -m mypy enhanced_system/ops enhanced_system/harness enhanced_system/core enhanced_system/evaluation enhanced_system/training enhanced_system/config

test:
	$(PYTEST) -m "unit or integration" --cov=enhanced_system --cov-report=term --cov-fail-under=60

test-harness:
	$(PYTEST) -m harness --cov-config=.coveragerc.harness --cov=enhanced_system.harness --cov-report=term

test-aqa:
	$(PYTEST) -m "unit or integration or regression or harness" --cov=enhanced_system --cov-report=term --cov-fail-under=60
	$(PYTEST) -m harness --cov-config=.coveragerc.harness --cov=enhanced_system.harness --cov-report=term

security:
	$(PYTHON) -m bandit -r enhanced_system scripts -x tests,enhanced_system/tests,enhanced_system/examples -c pyproject.toml -q

gitleaks:
	$(GITLEAKS) detect --source . --verbose --redact --config .gitleaks.toml

aqa-gate:
	$(PYTHON) scripts/harness/run_aqa_gate.py --golden-set configs/golden_sets/core_sdlc.jsonl --threshold 0.75 --scripted tests/fixtures/mock_responses.json

train-dpo:
	$(PYTHON) scripts/training/train_dpo_adapter.py --model_name_or_path "gpt2" --dataset_path "configs/golden_sets/core_sdlc.jsonl" --epochs 1 --batch_size 1 --output_dir "./dpo_adapter_test"

validate: lint typecheck test-aqa security gitleaks

