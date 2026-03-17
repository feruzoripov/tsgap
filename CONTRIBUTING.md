# Contributing to TSGap

Thank you for your interest in contributing to TSGap.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/tsgap.git`
3. Create a virtual environment: `python -m venv .env && source .env/bin/activate`
4. Install in development mode: `pip install -e ".[dev]"`

## Running Tests

```bash
pytest tsgap/tests/ -v
```

All tests must pass before submitting a pull request.

## Adding a New Pattern

1. Implement your pattern function in `tsgap/patterns.py` following the existing signature:
   ```python
   def apply_my_pattern(mask, shape, rng=None, **kwargs):
       ...
       return modified_mask
   ```
2. Register it in the `PATTERNS` dictionary at the bottom of the file.
3. Add tests in `tsgap/tests/test_missingness.py`.
4. Update the docstring in `tsgap/core.py` and `README.md`.

## Adding a New Mechanism

1. Implement in `tsgap/mechanisms.py` following the existing signature.
2. Register it in the `MECHANISMS` dictionary.
3. Add tests and update documentation.

## Code Style

- Use type hints (`from __future__ import annotations`)
- Include docstrings with Parameters/Returns sections (NumPy style)
- Keep functions focused and composable

## Reporting Issues

Open an issue on GitHub with:
- A minimal reproducible example
- Expected vs. actual behavior
- Python and NumPy versions

## Pull Requests

- One feature per PR
- Include tests for new functionality
- Update documentation as needed
- Ensure all existing tests still pass
