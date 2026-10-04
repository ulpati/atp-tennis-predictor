# 🎾 ATP Tennis Match Predictor

[![CI](https://github.com/ulpati/atp-tennis-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ulpati/atp-tennis-predictor/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ulpati/atp-tennis-predictor/actions/workflows/codeql.yml/badge.svg)](https://github.com/ulpati/atp-tennis-predictor/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

Predicts the winner of ATP matches with XGBoost on leakage-free features:
overall and surface Elo ratings, ranking, recent form, serve and return
statistics, workload and head-to-head. Every number below comes from a test
period that the model never saw during training or tuning; the
[methodology](docs/methodology.md) explains how leakage is ruled out.

## Contents

- [Results](#results)
- [Quick start](#quick-start)
- [Commands](#commands)
- [How it works](#how-it-works)
- [Data](#data)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Citation and license](#citation-and-license)

## Results

Test period from 13 May 2019 to 25 May 2026: 19,086 tour-level matches.
The model was trained on 1970–2012 and tuned on 2012–2019; the test period
was used only to write this table.

| Predictor | Accuracy | Log loss | Brier | AUC |
|---|---|---|---|---|
| **XGBoost model** | **66.3%** | **0.603** | **0.209** | **0.731** |
| Elo only | 64.3% | 0.640 | 0.223 | 0.702 |
| Surface Elo only | 63.7% | 0.648 | 0.226 | 0.694 |
| Better ranking wins | 63.9% | | | |

| Surface | Matches | Accuracy | Log loss |
|---|---|---|---|
| Hard | 11,328 | 66.7% | 0.599 |
| Clay | 5,806 | 64.9% | 0.615 |
| Grass | 1,900 | 67.7% | 0.596 |

Five-fold expanding-window cross-validation on 1970–2019 gives 69.5% ± 1.9%.
The model's main gain over Elo is better-calibrated probabilities, shown by
the lower log loss. Tennis is noisy: published models sit in the mid to high
60s, and betting markets are usually a few points better.

After this evaluation, `atp train` retrains the model with the same settings
on all 190,620 matches up to 25 May 2026, test period included, so
predictions also learn from the latest seasons.

## Quick start

```bash
git clone https://github.com/ulpati/atp-tennis-predictor.git
cd atp-tennis-predictor
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Reproducible install (Python 3.12+): every package, including the build backend,
# is checked against its SHA-256 and nothing else is downloaded
pip install --require-hashes -r requirements.txt
pip install --no-deps --no-build-isolation -e .
# Or, on Python 3.10 and 3.11: pip install -e .

atp download    # all 59 yearly files, 1968-2026, each checked against a known SHA-256
atp train       # evaluates on 2019-2026, then retrains on every match
atp predict "Jannik Sinner" "Carlos Alcaraz" Hard
```

```text
Jannik Sinner vs Carlos Alcaraz  |  Hard, best of 3  |  as of 2026-05-25
Player          Model  Elo only
--------------  -----  --------
Jannik Sinner   71.9%     60.2%
Carlos Alcaraz  28.1%     39.8%
Predicted winner: Jannik Sinner  (confidence high, model 'all')
```

The match files go to `data/` and the trained run to `models/`; Git ignores
both. On an 8-core desktop, training takes about a minute and 1.2 GB of
memory. On Linux the packages are a 420 MB download, mostly the NVIDIA NCCL
library that XGBoost's Linux wheels require.

## Commands

| Command | What it does |
|---|---|
| `atp download` | Downloads every `atp_matches_YYYY.csv` since 1968 over HTTPS from a pinned commit and verifies each file. |
| `atp train` | Builds features, trains, evaluates on the test period, retrains on every match and saves a run. |
| `atp evaluate` | Prints the metrics, baselines and splits of a run. |
| `atp predict A B SURFACE` | Predicts one match. `--best-of 5`, `--explain`, `--json`. |
| `atp batch FILE` | Predicts every `player_a,player_b,surface[,best_of]` row; CSV files saved by Excel, with `;`, work too. `--output results.csv`. |
| `atp list` | Strongest active players by Elo. `--surface Clay`, `--sort wins`. |
| `atp player NAME` | Record, surface breakdown, Elo and recent results. |
| `atp h2h A B` | Head-to-head by surface. |
| `atp importance` | Share of total gain per feature. |
| `atp shap` | Mean absolute SHAP value per feature on test rows. |
| `atp runs` | Lists saved runs with their test metrics. |

Every command accepts `--log-level`, `--log-file` and `--debug`. The
environment variables `ATP_DATA_DIR`, `ATP_MODELS_DIR` and `ATP_LOG_LEVEL`
change the defaults of `--data-dir`, `--models-dir` and `--log-level`.

Useful `train` options:

| Option | Default | Meaning |
|---|---|---|
| `--train-start DATE` | 3 years after the first match (1970) | Earlier matches only warm up ratings and statistics. |
| `--test-start DATE` | last 10% of matches | First date of the test period. |
| `--search N` | 0 | Random-search iterations, judged on the validation period. |
| `--cv-folds N` | 0 | Expanding-window cross-validation before the test period. |
| `--per-surface` | off | Hard, Clay and Grass models, each used only if it beats the global model on validation. |
| `--calibration isotonic` | `none` | Calibration fitted on the validation period; needs `--no-refit`. |
| `--no-refit` | off | Saves the evaluated model instead of retraining it on every period. |
| `--elo-k K` | dynamic | Fixed Elo K-factor instead of the experience-based one. |
| `--time-decay D` | 0 | Per-match decay for recent form, for example 0.9. |
| `--use-gpu` | off | Uses CUDA when XGBoost and a GPU support it, otherwise falls back to CPU. |

Names are matched without case, accents or hyphens. A surname works when it points to
one player (`Alcaraz`), or to only one player active in the last two years
(`Sinner` means Jannik, not Martin Sinner from the 1990s); the choice is
printed. When several active players match, the candidates are listed instead
of guessing. Unknown names are an error, never a 50/50.

## How it works

1. **Chronological state.** Matches are processed in date and round order.
   For each match the tool reads both players' pre-match state, then applies
   the result. Prediction reads the same state, so training and prediction
   features are identical.
2. **Mirrored rows.** Each match yields one row per player from the same
   function, so no feature can reveal the winner.
3. **Date-based evaluation.** Train, validation and test periods never
   overlap; all choices are made on validation.
4. **Symmetric prediction.** The probability is the average of
   `p(A beats B)` and `1 - p(B beats A)`.
5. **Final model.** After the evaluation, the chosen model is retrained with
   the same settings on every match, test period included. The reported
   metrics always come from the evaluated model.

Details and the full feature list are in
[docs/methodology.md](docs/methodology.md).

## Data

Match data: **Jeff Sackmann / Tennis Abstract**, licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):
attribution required, non-commercial use only.

The original `JeffSackmann/tennis_atp` repository is no longer online. Its
history survives in public forks. `atp download` uses commit
`712be0c5ade693cdab9e69c23a71a0edf5a23c44` ("thru 8 jun 2026", authored by
Jeff Sackmann), the latest one found in the public forks in October 2026.
It holds 59 yearly files, from 1968 to Roland Garros 2026, so predictions
describe the players as of 25 May 2026. A commit SHA identifies the same
content in every fork, and each file is also checked against the SHA-256
values in [atp_core/data_checksums.json](atp_core/data_checksums.json). Use
`--repo` and `--commit` to choose another fork or a newer commit.

## Security

The project follows a lightweight secure development life cycle. The
[threat model](docs/threat-model.md) lists every entry point, and
[SECURITY.md](SECURITY.md) explains how to report a vulnerability.

- **No pickle.** Models are XGBoost UBJSON files, state is JSON and test rows
  are `.npz` loaded with `allow_pickle=False`, so loading a run cannot execute code.
- **Integrity.** Every file of a run is checked against the SHA-256 in its
  manifest before parsing. Run ids and file names are validated and symbolic
  links are refused.
- **Safe download.** HTTPS only, same-host redirects, pinned commit, size
  limits, header validation, known checksums and atomic writes.
- **Input validation.** Typed, range-checked arguments and environment
  variables; size limits on every file; CSV formula injection neutralised in
  batch output; control characters in names are never printed to the
  terminal.
- **Supply chain.** Hash-pinned lock files, `pip-audit`, Dependabot, CodeQL,
  Bandit and GitHub Actions pinned by commit SHA with read-only permissions.

## Troubleshooting

| Problem | Solution |
|---|---|
| `python3 -m venv` says that `ensurepip` is not available | On Debian and Ubuntu install the venv module: `sudo apt install python3-venv` |
| `pip install --require-hashes -r requirements.txt` fails on Python 3.10 or 3.11 | The lock files target Python 3.12 and newer; use `pip install -e .` instead |
| `pip install` stops with `Read timed out` | Run the same command again: finished downloads are cached. `--timeout 120` gives a slow connection more time |
| `atp download` reports HTTP 429 or a timeout | GitHub limits anonymous downloads. Wait a few minutes and run it again; files already downloaded are skipped |
| `atp train` runs out of memory | Training needs about 1.2 GB. Use fewer years, for example `atp train --from-year 1990` |
| `No trained model in models` | Run `atp train` first, or point `--models-dir` or `ATP_MODELS_DIR` to your runs |
| `Player '...' is ambiguous` | Use the full name shown in the suggestions |
| `--use-gpu` prints that CUDA is not usable | XGBoost or the GPU driver has no CUDA support; training continues on the CPU. When the GPU is used, the log says "Training on the CUDA device" |

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
bandit -c pyproject.toml -r atp_core
pytest --cov
```

The suite has 170 tests and covers 97% of the code. It includes tests
that prove the absence of leakage, the parity between training and
prediction features, and every security control. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Citation and license

Code: [CC BY-NC-SA 4.0](LICENSE). GitHub's "Cite this repository" button
uses [CITATION.cff](CITATION.cff). When you use the data, also cite
Jeff Sackmann / Tennis Abstract.
