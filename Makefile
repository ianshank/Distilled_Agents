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
	$(PYTHON) -m mypy enhanced_system/ops enhanced_system/core/cache enhanced_system/harness

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

validate: lint typecheck test-aqa security gitleaks
