# Copilot Instructions for trends.earth-algorithms

## Repository Overview

**trends.earth-algorithms** is a Python package providing algorithms for analyzing remotely-sensed datasets to monitor land degradation. It serves as the core algorithmic library for the Trends.Earth ecosystem - an open-source tool used for restoration monitoring, urbanization tracking, and UN reporting on land degradation.

**Repository Stats:**
- Size: ~1.7MB, 35 Python files
- Python 3.8-3.13 support
- Modern packaging (pyproject.toml)
- MIT licensed

## Build & Development Setup

### Prerequisites
```bash
# System dependencies (Ubuntu/Debian)
sudo apt-add-repository ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev

# Python dependencies
python -m pip install --upgrade pip
```

### Installation & Development Setup
**WARNING:** Full installation requires network access to GitHub for te_schemas dependency. For basic development:

```bash
# Install core dependencies first (works offline)
pip install openpyxl backoff marshmallow defusedxml marshmallow-dataclass

# Install dev tools
pip install ruff pytest invoke sphinx sphinx_rtd_theme

# Install GDAL to match system version
pip install GDAL==`gdal-config --version`

# For full installation (requires network):
pip install -e .
```

**Network Issues:** If `pip install -e .` fails due to te_schemas git dependency timeout, install dependencies individually first.

### Build Commands

**Always run these commands in this order:**

1. **Linting** (matches CI): `ruff check --output-format=github .`
2. **Testing**: `pytest -v` (requires te_schemas for full test suite)
3. **Documentation**: `sphinx-build -b html docs/source docs/build` (from docs/ directory)

### Pre-commit Validation
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Manual validation
pre-commit run --all-files
```

## Project Architecture

### Directory Structure
```
te_algorithms/
├── api/           # API utilities and interfaces
├── common/        # Common algorithms (soil organic carbon)
├── gdal/          # GDAL-based raster processing
│   └── land_deg/  # Land degradation algorithms
├── gee/           # Google Earth Engine algorithms
└── data/          # Static data files
```

### Key Configuration Files
- `pyproject.toml` - Modern Python packaging configuration
- `.pre-commit-config.yaml` - Code quality hooks (Ruff linting)
- `tasks.py` - Invoke task definitions (version management)
- `version.txt` - Current version number
- `.github/workflows/` - CI/CD pipelines

### Dependencies
- **Core:** marshmallow (serialization), backoff (retry logic), openpyxl (Excel)
- **Optional:** GDAL (raster processing), earthengine-api (GEE), boto3 (AWS), numba (performance)
- **External:** te_schemas (git dependency for data schemas)

## Continuous Integration

### GitHub Actions Workflows
1. **ruff.yaml** - Code linting on every push/PR
2. **test.yaml** - Test suite across Python 3.9-3.13 on Ubuntu

### Known CI Issues & Workarounds
- Tests require GDAL system packages (automatically installed in CI)
- Some tests depend on te_schemas git dependency
- Ruff currently reports ~10 linting issues (unused variables, bare except clauses)

## Common Development Tasks

### Making Code Changes
1. **Always** install dependencies first: `pip install openpyxl backoff marshmallow defusedxml marshmallow-dataclass`
2. **Always** run linting before commits: `ruff check .` 
3. Fix linting issues with: `ruff check --fix .`
4. Run tests: `pytest` (may fail without te_schemas)
5. Check specific modules: `python -c "import te_algorithms; print(te_algorithms.__version__)"`

### Version Management
```bash
# Set new version
invoke set-version --v=2.1.18

# Create git tag
invoke set-tag
```

### Documentation Updates
```bash
# Build docs locally (expect warnings for missing dependencies)
cd docs/
pip install sphinx sphinx_rtd_theme
sphinx-build -b html source build
# Note: Will show warnings for missing ee (earthengine-api) and numpy dependencies
```

## Common Issues & Solutions

### Installation Problems
- **te_schemas timeout:** Install core dependencies individually first
- **GDAL errors:** Ensure system GDAL packages installed before pip install
- **Import errors:** Check te_algorithms is importable: `python -c "import te_algorithms"`

### Testing Issues
- **ModuleNotFoundError:** Install missing test dependencies: `pip install marshmallow-dataclass`
- **Schema errors:** Tests require te_schemas - may need network access

### Linting Issues (Current State)
The codebase has 11 Ruff violations:
- 9 F841: Unused variables (timing variables, debug variables)
- 2 E722: Bare except clauses in error handling

**Do not fix these unless specifically asked** - they may be intentional.

## Quick Reference

### Module Import Paths
```python
# Core package (always works)
import te_algorithms

# Basic modules
from te_algorithms.common import soc

# GDAL modules (require GDAL system packages)
# from te_algorithms.api import util as api_util  # needs GDAL
# from te_algorithms.gdal import util as gdal_util  # needs GDAL

# GEE modules (require earthengine-api)
# from te_algorithms.gee import util as gee_util  # needs ee

# Version info
print(te_algorithms.__version__)  # Current: 2.1.17
```

### File Layout Summary
- **Root files:** pyproject.toml (config), tasks.py (invoke), version.txt
- **Main code:** te_algorithms/ package with api/, common/, gdal/, gee/ modules  
- **Tests:** tests/ (minimal test suite, requires te_schemas)
- **Docs:** docs/source/ (Sphinx), .readthedocs.yaml (RTD config)
- **CI:** .github/workflows/ (ruff.yaml, test.yaml)

## Trust These Instructions

**IMPORTANT:** Trust these instructions and only perform repository exploration if:
1. The information here is incomplete for your specific task
2. You find errors in these instructions
3. You need to understand code beyond what's documented here

The commands and information above have been validated and reflect the current state of the repository.