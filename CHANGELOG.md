# Changelog

## 1.0.0 (2026-10-04)

First public release.

- `atp` command with eleven subcommands: `download`, `train`, `evaluate`,
  `predict`, `batch`, `list`, `player`, `h2h`, `importance`, `shap` and `runs`.
- Leakage-free features read from each player's pre-match state: overall and
  surface Elo with dynamic K, ranking, recent form, serve and return
  statistics, workload, age, height, handedness and head-to-head.
- Chronological train, validation and test periods. Every choice is made on
  the validation period, and the final model is then retrained on every match.
- Optional random search, expanding-window cross-validation, per-surface
  models, isotonic calibration and CUDA training.
- Strict player-name matching that ignores case, accents and hyphens and never
  guesses silently.
- Pickle-free runs (XGBoost UBJSON, JSON and `.npz`) verified against a
  SHA-256 manifest before loading.
- Data download from a pinned commit over HTTPS, checked against known
  SHA-256 values.
- Typed, range-checked input, size limits, CSV formula-injection protection,
  control characters kept out of the terminal and one-line error messages.
- Hash-pinned lock files and CI with Ruff, Bandit, pip-audit, CodeQL and
  Dependabot.
