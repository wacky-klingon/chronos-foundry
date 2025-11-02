# Installation Guide

This guide covers installing Chronos Foundry using Poetry (recommended) or pip.

## Prerequisites

- Python 3.10 or higher
- Poetry 1.5+ (recommended) or pip

## Installation

### Using Poetry (Recommended)

```bash
# From PyPI (when published)
poetry add chronos-foundry

# From source (development)
git clone https://github.com/yourusername/chronos-foundry.git
cd chronos-foundry
poetry install

# For development with all tools
poetry install --with dev
```

### Using pip (Alternative)

```bash
# From PyPI (when published)
pip install chronos-foundry

# From source
git clone https://github.com/yourusername/chronos-foundry.git
cd chronos-foundry
pip install -e .
```

## Verify Installation

```python
# Test import
python -c "from chronos_trainer import CovariateTrainer; print('Installation successful!')"
```

## Dependencies

Chronos Foundry requires:
- PyTorch (for Chronos models)
- Pandas (for data handling)
- Pydantic (for configuration validation)
- Other dependencies are listed in `pyproject.toml`

## Next Steps

- [Quick Example](quick-example.md) - Run your first training
- [Complete Usage Guide](../user-guides/usage-guide.md) - Detailed usage instructions
- [AWS Quickstart](aws-quickstart.md) - Deploy on AWS

