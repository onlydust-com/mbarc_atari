# Testing Infrastructure for MBARC

This document describes the Docker-based testing infrastructure for the MBARC Atari project.

## Overview

The testing infrastructure uses:
- **Docker** for reproducible, isolated test environments
- **uv** for fast, reliable Python dependency management
- **pytest** for test execution and organization
- **Python 3.12** with modern dependencies

## Quick Start

### Build the Docker Test Image

```bash
docker build -f Dockerfile.test -t mbarc-test .
```

**First build time:** ~5-8 minutes (downloads dependencies)
**Subsequent builds:** ~30 seconds (uses Docker cache)

### Run All Tests

```bash
docker run --rm mbarc-test
```

### Run Specific Test Categories

```bash
# Run only fast tests (exclude slow tests)
docker run --rm mbarc-test pytest tests/ -m "not slow" -v

# Run only integration tests
docker run --rm mbarc-test pytest tests/ -m "integration" -v

# Run a specific test file
docker run --rm mbarc-test pytest tests/integration/test_training_poc.py -v

# Run a specific test function
docker run --rm mbarc-test pytest tests/integration/test_training_poc.py::test_pytorch_installation -v
```

### Interactive Development

```bash
# Start an interactive shell in the container
docker run --rm -it mbarc-test /bin/bash

# Inside the container, you can run:
pytest tests/ -v                    # Run tests
pytest tests/ -v -s                 # Run tests with print output
pytest tests/ -k "pytorch"          # Run tests matching pattern
pytest tests/ --lf                  # Run only last failed tests
python -m mbarc --help              # Run the application
```

### Mount Local Code for Development

If you want to test local changes without rebuilding the Docker image:

```bash
docker run --rm -v $(pwd):/app mbarc-test pytest tests/ -v
```

**Note:** This mounts your local code into the container, so changes are immediately reflected.

## Project Structure

```
mbarc_atari/
├── Dockerfile.test           # Docker image for testing
├── pyproject.toml           # Project metadata and dependencies (uv format)
├── pytest.ini               # Pytest configuration
├── .dockerignore            # Files to exclude from Docker build
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared pytest fixtures
│   └── integration/
│       ├── __init__.py
│       └── test_training_poc.py  # Integration tests
├── mbarc/                   # Source code
└── atari_utils/             # Utilities
```

## Available Tests

### Integration Tests (`tests/integration/test_training_poc.py`)

These tests validate that all dependencies work correctly together:

- ✅ `test_pytorch_installation` - PyTorch installation and basic operations
- ✅ `test_gymnasium_installation` - Gymnasium with Atari support
- ✅ `test_create_simple_environment` - CartPole environment creation
- ✅ `test_create_atari_environment` - Atari (Pong) environment creation
- ✅ `test_opencv_installation` - OpenCV operations
- ✅ `test_core_imports` - MBARC module imports
- ✅ `test_torch_and_numpy_compatibility` - PyTorch/NumPy interop
- ✅ `test_simple_environment_rollout` - Complete environment rollout
- ✅ `test_torch_basic_neural_network` - Neural network creation
- ✅ `test_dependencies_versions` - Dependency version checks

## Test Configuration

### Fixtures (`tests/conftest.py`)

Available pytest fixtures:

- **`test_device`** - Returns CPU or CUDA device for testing
- **`minimal_config`** - Minimal configuration for MBARC components
- **`temp_model_dir`** - Temporary directory for model saving/loading
- **`set_random_seeds`** - Auto-applied fixture for reproducible tests

### Pytest Markers

Use markers to categorize and select tests:

```bash
# Available markers
pytest -m "integration"     # Run integration tests only
pytest -m "slow"            # Run slow tests only
pytest -m "not slow"        # Skip slow tests
```

## Advanced Usage

### Run Tests with Coverage

```bash
docker run --rm mbarc-test pytest tests/ --cov=mbarc --cov=atari_utils --cov-report=html
```

Coverage report will be generated in `htmlcov/` directory.

### Debug Test Failures

```bash
# Show local variables in traceback
docker run --rm mbarc-test pytest tests/ -v --tb=long -l

# Drop into debugger on failure
docker run --rm -it mbarc-test pytest tests/ --pdb

# Show print statements
docker run --rm mbarc-test pytest tests/ -v -s
```

### Run Tests with Different Timeouts

```bash
# Override default timeout (600s)
docker run --rm mbarc-test pytest tests/ --timeout=300

# Disable timeout for debugging
docker run --rm -it mbarc-test pytest tests/ --timeout=0
```

### Custom Environment Variables

```bash
# Use CUDA device (if available in Docker)
docker run --rm --gpus all -e DEVICE=cuda mbarc-test

# Set custom environment variables
docker run --rm -e CUSTOM_VAR=value mbarc-test pytest tests/ -v
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Build test image
        run: docker build -f Dockerfile.test -t mbarc-test .

      - name: Run tests
        run: docker run --rm mbarc-test pytest tests/ -v --tb=short

      - name: Run tests with coverage
        run: |
          docker run --rm -v $(pwd)/coverage:/app/coverage mbarc-test \
            pytest tests/ --cov=mbarc --cov=atari_utils --cov-report=xml:/app/coverage/coverage.xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage/coverage.xml
```

## Important Notes

### Dependency Modernization

⚠️ **The original MBARC codebase uses outdated dependencies:**
- `gym==0.15.7` → Now using `gymnasium>=0.29.0`
- `torch==1.7.1` → Now using `torch>=2.1.0`
- `numpy==1.16.4` → Now using `numpy>=1.24.0`
- Python 3.7 → Now using Python 3.12

**This means:**
- The existing MBARC code **will not run** without modifications to support the new gymnasium API
- The integration tests validate the infrastructure (Docker, uv, pytest, dependencies)
- To run the actual training code, you'll need to update the codebase to use gymnasium's API

### Migration Path

To make the existing code work with modern dependencies:

1. Replace `gym` imports with `gymnasium`
2. Update environment creation: `gym.make("PongNoFrameskip-v4")` → `gym.make("ALE/Pong-v5")`
3. Update step API: `obs, reward, done, info` → `obs, reward, terminated, truncated, info`
4. Update reset API: `obs = env.reset()` → `obs, info = env.reset()`

## Local Development Without Docker

If you prefer to develop without Docker:

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### Install Dependencies

```bash
# Install all dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Run tests
pytest tests/ -v
```

## Troubleshooting

### Docker Build Fails

```bash
# Clean Docker cache and rebuild
docker system prune -a
docker build --no-cache -f Dockerfile.test -t mbarc-test .
```

### Tests Timeout

```bash
# Increase timeout or run without slow tests
docker run --rm mbarc-test pytest tests/ -m "not slow" --timeout=1200
```

### Atari ROMs Not Found

The Dockerfile automatically installs Atari ROMs via `gymnasium[atari,accept-rom-license]`. If you see ROM errors:

```bash
# Rebuild the image to ensure ROMs are installed
docker build --no-cache -f Dockerfile.test -t mbarc-test .
```

### Import Errors

If you see import errors for `mbarc` or `atari_utils`:

```bash
# Make sure the project is installed in editable mode
docker run --rm -it mbarc-test /bin/bash
uv pip install -e .
```

## Performance

**Typical runtimes** (on modern hardware):

- Docker build (first time): ~5-8 minutes
- Docker build (cached): ~30 seconds
- Full test suite: ~2-3 minutes
- Fast tests only (`-m "not slow"`): ~30 seconds

## Next Steps

1. **Add more tests** - Expand test coverage for specific components
2. **Update codebase** - Migrate code to gymnasium API
3. **Add unit tests** - Create `tests/unit/` for individual component testing
4. **CI/CD integration** - Set up automated testing on pull requests

## Resources

- [uv documentation](https://docs.astral.sh/uv/)
- [pytest documentation](https://docs.pytest.org/)
- [Gymnasium documentation](https://gymnasium.farama.org/)
- [Docker documentation](https://docs.docker.com/)
