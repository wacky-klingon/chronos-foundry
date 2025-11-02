# Chronos Trainer Tests

Comprehensive test suite for the chronos-foundry package.

## Test Structure

```
tests/
├── __init__.py                   # Test package initialization
├── test_basic.py                 # Basic integration tests (12 tests)
├── test_checkpoint_manager.py    # CheckpointManager tests (6 tests)
├── test_resumable_loader.py      # ResumableDataLoader tests (6 tests)
├── test_incremental_trainer.py   # IncrementalTrainer tests (3 tests)
└── README.md                     # This file (if exists)
```

## Test Categories

### Unit Tests
- **TestImports**: Verify all package imports work
- **TestDataGeneration**: Test synthetic data generation utilities
- **TestDataValidation**: Test data quality checks

### Integration Tests
- **TestBasicTraining**: Test training workflow with real API calls
- **TestCheckpointing**: Test checkpoint functionality
- **TestModelVersioning**: Test model versioning system

### Component Tests (NEW)
- **TestCheckpointManager**: Test checkpoint save/load, progress tracking (test_checkpoint_manager.py)
- **TestResumableDataLoader**: Test file discovery, loading, checkpoint integration (test_resumable_loader.py)
- **TestIncrementalTrainer**: Test incremental training workflows, resume functionality (test_incremental_trainer.py)

## Running Tests

### All Fast Tests (Recommended for CI/CD)
```bash
poetry run pytest tests/ -v
```

### Including Slow Tests (Model Training)
```bash
poetry run pytest tests/ -v -m slow
```

### Specific Test Class
```bash
poetry run pytest tests/test_basic.py::TestImports -v
```

### With Coverage Report
```bash
poetry run pytest tests/ -v --cov=chronos_trainer --cov-report=html
open htmlcov/index.html
```

### Fast Feedback Loop (Failed Tests First)
```bash
poetry run pytest tests/ -v --failed-first
```

## Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.slow` - Tests that train models (requires GPU/compute)
- `@pytest.mark.integration` - Integration tests requiring external resources
- `@pytest.mark.unit` - Fast unit tests

### Skip Slow Tests
```bash
poetry run pytest tests/ -v -m "not slow"
```

### Run Only Integration Tests
```bash
poetry run pytest tests/ -v -m integration
```

## Test Data

Tests use synthetic time series data generated with realistic patterns:

- **Trend**: Linear growth component
- **Seasonality**: Daily and weekly patterns
- **Noise**: Gaussian random noise
- **Covariates**: Correlated features

Example:
```python
from tests.test_basic import TestDataGeneration

# Generate 1000 hours of synthetic data
df = TestDataGeneration.generate_synthetic_timeseries(
    start_date="2020-01-01",
    periods=1000,
    freq="1H",
    n_covariates=3
)
```

## Test Configuration

Configuration is managed via `pytest.ini` in the project root:

- Verbose output by default
- Short traceback format
- Strict marker enforcement
- Test discovery patterns

## Writing New Tests

### Test Structure
```python
import pytest
from chronos_trainer import CovariateTrainer

class TestMyFeature:
    """Test suite for my new feature"""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing test data"""
        return TestDataGeneration.generate_synthetic_timeseries()

    def test_basic_functionality(self, sample_data):
        """Test basic feature functionality"""
        # Arrange
        trainer = CovariateTrainer(config)

        # Act
        result = trainer.some_method(sample_data)

        # Assert
        assert result is not None
```

### Best Practices

1. **Fixtures**: Use pytest fixtures for reusable test data
2. **Markers**: Mark slow tests with `@pytest.mark.slow`
3. **Descriptive Names**: Use clear, descriptive test names
4. **Arrange-Act-Assert**: Follow AAA pattern
5. **Isolation**: Tests should not depend on each other
6. **Cleanup**: Use `temp_dir` fixture for file operations

## Continuous Integration

### Recommended CI Pipeline

```yaml
# .github/workflows/test.yml
- name: Run fast tests
  run: poetry run pytest tests/ -v -m "not slow"

- name: Run with coverage
  run: poetry run pytest tests/ -v --cov=chronos_trainer --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Test Coverage Goals

- **Core modules**: > 90% coverage
- **Training modules**: > 80% coverage
- **Utilities**: > 95% coverage
- **Overall**: > 85% coverage

Check current coverage:
```bash
poetry run pytest tests/ --cov=chronos_trainer --cov-report=term-missing
```

## Debugging Tests

### Run with pdb on failure
```bash
poetry run pytest tests/ -v --pdb
```

### Show print statements
```bash
poetry run pytest tests/ -v -s
```

### Verbose error output
```bash
poetry run pytest tests/ -vv --tb=long
```

## Performance Testing

For performance benchmarks, use pytest-benchmark:

```python
def test_training_performance(benchmark):
    result = benchmark(trainer.train, data)
    assert result is not None
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

