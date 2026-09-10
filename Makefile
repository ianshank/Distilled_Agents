PYTHON ?= python3
PKG_PATHS := enhanced_system scripts tests
GITLEAKS ?= gitleaks
GITLEAKS_VERSION ?= 8.21.2

.PHONY: install lint fmt typecheck test test-aqa security gitleaks validate

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check $(PKG_PATHS)
	$(PYTHON) -m ruff format --check $(PKG_PATHS)

fmt:
	$(PYTHON) -m ruff check --fix $(PKG_PATHS)
	$(PYTHON) -m ruff format $(PKG_PATHS)

typecheck:
	$(PYTHON) -m mypy enhanced_system/ops enhanced_system/core/cache

test:
	$(PYTHON) -m pytest -m "unit or integration" --cov=enhanced_system --cov-report=term --cov-fail-under=65

test-aqa:
	$(PYTHON) -m pytest -m "unit or integration or regression" --cov=enhanced_system --cov-report=term --cov-fail-under=65

security:
	$(PYTHON) -m bandit -r enhanced_system scripts -x tests,enhanced_system/tests,enhanced_system/examples -c pyproject.toml -q

gitleaks:
	$(GITLEAKS) detect --source . --verbose --redact --config .gitleaks.toml

validate: lint typecheck test-aqa security gitleaks
