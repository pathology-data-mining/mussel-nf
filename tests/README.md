# Tests for mussel-nf

This directory contains unit tests for the mussel-nf pipeline.

## Running Tests

### Prerequisites

Install the required testing dependencies:

```bash
pip install pytest pytest-cov pandas pyyaml
```

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=scripts --cov-report=term-missing
```

### Run Specific Test File

```bash
pytest tests/test_create_manifest.py
```

### Run Specific Test

```bash
pytest tests/test_create_manifest.py::TestCreateManifest::test_manifest_generation_basic
```

### Run Tests with Verbose Output

```bash
pytest -v
```

## Test Structure

- `test_create_manifest.py` - Unit tests for the `create_manifest.py` script
- `fixtures/` - Test data and fixtures

## Coverage Reports

After running tests with coverage, an HTML report will be generated in `htmlcov/`. 
Open `htmlcov/index.html` in a browser to view detailed coverage information.

## Adding New Tests

When adding new functionality to the pipeline:

1. Create a new test file following the naming convention `test_<module_name>.py`
2. Use descriptive test names that clearly indicate what is being tested
3. Include docstrings explaining the purpose of each test
4. Use pytest fixtures to set up test data and clean up after tests
5. Ensure tests are independent and can run in any order
