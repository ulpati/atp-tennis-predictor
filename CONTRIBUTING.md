# Contributing

Thank you for helping. This page explains how to set up the project, the
rules every change must follow and how to submit it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

For a reproducible environment on Python 3.12 or newer, install the
hash-pinned lock file first. It includes the build backend, so the last
command downloads nothing:

```bash
pip install --require-hashes -r requirements-dev.txt
pip install --no-deps --no-build-isolation -e .
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
bandit -c pyproject.toml -r atp_core
pytest --cov
```

CI runs the same checks and fails below 85% coverage.

## Rules for every change

- **No leakage.** A feature may only use information available before the
  match starts. Add features to `LeagueState.snapshot` or `LeagueState.pair`
  so training and prediction share them, and keep `test_features.py` green.
- **Decisions on validation only.** Never use the test period to choose
  parameters, models or thresholds.
- **No unsafe deserialisation or code execution.** Do not use `pickle`,
  `joblib`, `eval`, `exec`, `shell=True` or `np.load(allow_pickle=True)`.
- **Validate inputs** at the boundary: arguments, files, environment
  variables and network responses. Add a test for each new check.
- **Report honestly.** Metrics in the README must come from a run that anyone
  can reproduce with the documented commands.
- **Dependencies** need a reason. Update `pyproject.toml`, then regenerate
  both lock files with [uv](https://docs.astral.sh/uv/):

  ```bash
  uv pip compile pyproject.toml requirements-build.in --universal --generate-hashes --no-header -o requirements.txt
  uv pip compile pyproject.toml requirements-build.in --extra dev --universal --generate-hashes --no-header -o requirements-dev.txt
  ```

## Reporting bugs and security issues

Open an issue with the command you ran, the output and your environment.
Report security problems privately, as described in [SECURITY.md](SECURITY.md).
