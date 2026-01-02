# Lockfiles

This directory contains optional exact dependency snapshots for reference and debugging.
They are **NOT required** for installation.

## Files

Lockfiles are named by platform and Python version:
- `macos-arm64-py310.txt` - Developer's Mac (M1 Pro, macOS, Python 3.10)
- `linux-x86_64-cuda121-py310.txt` - Cloud GPU (Ubuntu, CUDA 12.1, Python 3.10)

## To Generate a Lockfile

```bash
pip freeze > locks/$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)-py$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")').txt
```

## Important Note

These lockfiles include platform-specific wheels and should **NOT** be used
for cross-platform installation. Use `pyproject.toml` and `requirements/torch-*.txt` instead.

