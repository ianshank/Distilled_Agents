# Testing Guide

Comprehensive guide for running tests in the Enhanced Agent System.

## Overview

The test suite is organized into:
- **Unit Tests**: Test individual components in isolation (80%+ coverage target)
- **Integration Tests**: Test component interactions (70%+ coverage target)
- **E2E Tests**: Test complete system workflows (60%+ coverage target)
- **Performance Tests**: Benchmark critical paths

## Prerequisites

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Quick Start

Run all tests:

```bash
cd enhanced_system
pytest
```

## Running Specific Test Categories

### Unit Tests Only

```bash
pytest -m unit
```

### Integration Tests Only

```bash
pytest -m integration
```

### End-to-End Tests Only

```bash
pytest -m e2e
```

### Exclude Slow Tests

```bash
pytest -m "not slow"
```

## Coverage Reports

### Generate HTML Coverage Report

```bash
pytest --cov=enhanced_system --cov-report=html
```

View the report:

```bash
# Open htmlcov/index.html in your browser
```

### Generate Terminal Coverage Report

```bash
pytest --cov=enhanced_system --cov-report=term
```

### Generate XML Coverage Report (for CI)

```bash
pytest --cov=enhanced_system --cov-report=xml
```

### Enforce Minimum Coverage

```bash
pytest --cov=enhanced_system --cov-report=term --cov-fail-under=60
```

## Running Tests for Specific Modules

### Cache Manager Tests

```bash
pytest tests/unit/test_cache_manager.py -v
```

### Error Handler Tests

```bash
pytest tests/unit/test_error_handler.py -v
```

### Consensus Tests

```bash
pytest tests/unit/test_consensus_inference.py -v
```

### Full Inference Pipeline (Integration)

```bash
pytest tests/integration/test_full_inference_pipeline.py -v
```

## Using Tox for Multi-Environment Testing

Test across multiple Python versions:

```bash
tox
```

Test specific environment:

```bash
# Python 3.9
tox -e py39

# Python 3.10
tox -e py310

# Python 3.11
tox -e py311
```

Run coverage with tox:

```bash
tox -e coverage
```

Run linting:

```bash
tox -e lint
```

Format code:

```bash
tox -e format
```

## Performance Benchmarking

Run performance benchmarks:

```bash
pytest -m benchmark --benchmark-only
```

Compare benchmarks:

```bash
# Run and save baseline
pytest -m benchmark --benchmark-save=baseline

# Run again and compare
pytest -m benchmark --benchmark-compare=baseline
```

## Debugging Failed Tests

### Verbose Output

```bash
pytest -vv
```

### Show Local Variables on Failure

```bash
pytest -l
```

### Stop on First Failure

```bash
pytest -x
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Run Specific Test Function

```bash
pytest tests/unit/test_cache_manager.py::TestIntelligentCacheManager::test_l1_cache_get_set -v
```

## Continuous Integration

### GitHub Actions

The test suite runs automatically on:
- Push to main/develop
- Pull requests

### Pre-commit Hooks

Install pre-commit hooks:

```bash
cd enhanced_system
pre-commit install
```

Run pre-commit manually:

```bash
pre-commit run --all-files
```

## Test Organization

```
enhanced_system/tests/
├── conftest.py                 # Shared fixtures
├── fixtures/
│   ├── factories.py           # Test factories
│   └── mocks.py               # Mock objects
├── unit/
│   ├── test_input_validator.py
│   ├── test_cache_manager.py
│   ├── test_error_handler.py
│   ├── test_confidence_calibrator.py
│   ├── test_consensus_inference.py
│   ├── test_adaptive_router.py
│   ├── test_streaming_inference.py
│   ├── test_batch_processor.py
│   └── test_monitoring.py
├── integration/
│   ├── test_full_inference_pipeline.py
│   ├── test_training_pipeline.py
│   └── test_ab_testing_flow.py
└── e2e/
    └── test_system_e2e.py
```

## Writing New Tests

### Test Structure

```python
import pytest
from enhanced_system.core.module import Component

@pytest.mark.unit
class TestComponent:
    """Test Component class."""
    
    def test_basic_functionality(self, test_config):
        """Test basic component behavior."""
        component = Component(test_config)
        result = component.process("input")
        assert result is not None
    
    def test_error_handling(self, test_config):
        """Test error handling."""
        component = Component(test_config)
        with pytest.raises(ValueError):
            component.process(None)
```

### Using Fixtures

```python
@pytest.mark.asyncio
async def test_with_mock_agent(test_config, mock_agent):
    """Test using mock agent fixture."""
    result = await mock_agent("test task")
    assert result['response'] is not None
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("valid input", True),
    ("", False),
    (None, False),
])
def test_validation(input, expected):
    """Test validation with multiple inputs."""
    result = validate(input)
    assert result == expected
```

## Coverage Goals

| Test Type    | Coverage Target |
|--------------|-----------------|
| Unit Tests   | 80%+            |
| Integration  | 70%+            |
| E2E          | 60%+            |
| Overall      | 80%+            |

## Current Coverage Status

Generate current coverage report:

```bash
pytest --cov=enhanced_system --cov-report=term-missing
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Use Fixtures**: Leverage pytest fixtures for setup/teardown
3. **Mock External Dependencies**: Use mocks for Redis, S3, APIs
4. **Test Edge Cases**: Include boundary conditions and error cases
5. **Clear Naming**: Use descriptive test function names
6. **Documentation**: Add docstrings explaining test purpose
7. **Async Tests**: Use `@pytest.mark.asyncio` for async functions
8. **Fast Tests**: Keep unit tests fast (<1s each)

## Troubleshooting

### Tests Hanging

Check for:
- Missing `await` in async tests
- Infinite loops
- Deadlocks in concurrent code

### Import Errors

Ensure you're running from the correct directory:

```bash
cd enhanced_system
pytest
```

### Fixture Not Found

Check that fixtures are defined in `conftest.py` or test file.

### Redis/S3 Connection Errors

Unit tests should mock external services. If integration tests fail:

```bash
# Disable Redis/S3 in test config
export ENHANCED_SYSTEM_ENV=development
pytest
```

## Continuous Improvement

### Identify Untested Code

```bash
pytest --cov=enhanced_system --cov-report=term-missing
```

Look for lines marked with `!!!!`.

### Add Tests for Bug Fixes

When fixing a bug:
1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify the test passes

### Review Coverage Trends

```bash
# Generate coverage over time
pytest --cov=enhanced_system --cov-report=xml
# Track coverage.xml in CI
```

## Questions?

For questions or issues with testing:
1. Check this guide
2. Review example tests
3. Open an issue

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- [tox documentation](https://tox.wiki/)

