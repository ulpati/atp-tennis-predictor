# Threat model

This document records the threat model used for version 1.0, following the
design phase of a secure software development life cycle (SSDLC). It is
reviewed whenever a new input, output or dependency is added.

## System overview

`atp` is a local command-line tool. It downloads public CSV files, builds
features, trains XGBoost models, stores them on disk and answers prediction
queries. It runs no server, stores no personal data and needs no
credentials.

```text
 GitHub raw (HTTPS) ──download──▶ data/*.csv ──train──▶ models/<run>/ ──load──▶ predict, batch, evaluate
                                                          ▲                         ▲
                                       user: CLI args, environment variables, batch CSV files
```

## Assets

| Asset | Why it matters |
|---|---|
| The user's machine | Loading a model or data file must never execute code. |
| Model runs in `models/` | Wrong or swapped models silently produce wrong predictions. |
| Match data in `data/` | Poisoned data biases every model trained on it. |
| Reported metrics | Users rely on them to judge the model; they must be honest and reproducible. |

## Trust boundaries and entry points

1. **Network to disk**: the `download` command fetches CSV files.
2. **Disk to process**: CSV files and model runs are parsed on every command.
3. **User to process**: command-line arguments, environment variables and batch files.
4. **Supply chain**: Python packages from PyPI and GitHub Actions used by CI.

## Threats and mitigations (STRIDE)

| # | Category | Threat | Mitigation | Where |
|---|---|---|---|---|
| T1 | Elevation of privilege | A crafted model file executes code when loaded (insecure deserialisation, CWE-502). | No pickle anywhere. Boosters use XGBoost's UBJSON format, state is JSON, test rows use `np.load(allow_pickle=False)`. | `artifacts.py` |
| T2 | Tampering | A model file is corrupted or partly replaced. | Every file is checked against the SHA-256 in the manifest before it is parsed, with a constant-time comparison. | `artifacts.py` |
| T3 | Tampering | Path traversal through a run id or a file name in the manifest. | Run ids and file names must match strict patterns. Symbolic links are refused. | `artifacts.py` |
| T4 | Denial of service | Huge or highly compressed files exhaust memory. | Size limits on CSV files, manifests, model files and decompressed state. Batch files are limited to 16 MB and, by default, 10,000 rows. | `data.py`, `artifacts.py`, `cli.py` |
| T5 | Tampering | Downloaded data is altered in transit or by the hosting fork. | HTTPS only, redirects must stay on the same host, the commit is pinned to an immutable SHA, and every file of the pinned commit is verified against SHA-256 values shipped with the package, including files already on disk, which are downloaded again when they differ. Writes are atomic. | `download.py`, `data_checksums.json` |
| T6 | Spoofing | A look-alike URL or scheme (`file://`, `http://`) is used to read local files or downgrade transport. | The URL is built from a validated `owner/name` and a 40-character commit; only `https` is accepted. | `download.py` |
| T7 | Tampering | A batch output opened in a spreadsheet executes a formula (CSV injection, CWE-1236). | Cells starting with `=`, `+`, `-`, `@`, tab or carriage return are prefixed with a quote. | `utils.py`, `cli.py` |
| T8 | Information disclosure | Stack traces leak paths or data. | Expected errors print one line. Unexpected errors print only the exception type unless `--debug` is given. | `cli.py` |
| T9 | Repudiation and integrity of results | Metrics are inflated by data leakage. | Mirrored rows from one feature function, pre-match information only, date-based splits, decisions on validation only. Tests prove that a match's features do not depend on its result and that prediction features equal training features. | `features.py`, `model.py`, `tests/` |
| T10 | Tampering (supply chain) | A compromised dependency or CI action. | Hash-pinned lock files, `pip-audit` and Dependabot, GitHub Actions pinned by commit SHA, least-privilege workflow permissions, CodeQL and Bandit scans. | `requirements*.txt`, `.github/` |
| T11 | Input validation | Malformed CLI values or environment variables cause undefined behaviour. | Every argument is parsed by a typed, range-checked parser, and `download` validates the repository and commit before touching the disk or the network. An empty environment variable means the default; an invalid `ATP_LOG_LEVEL` is a usage error. | `cli.py`, `config.py`, `download.py` |
| T12 | Tampering | Player names in a batch file or in data from another fork contain escape sequences that change the terminal. | Control characters are replaced before anything is printed or logged; JSON output escapes them. | `cli.py`, `utils.py` |

## Residual risks

- The SHA-256 values in the manifest protect against corruption and partial
  tampering. An attacker who can rewrite a whole run directory can also
  rewrite its manifest. Because nothing is unpickled, the worst outcome is a
  wrong prediction, not code execution. Keep `models/` writable only by you.
- The match data comes from a fork of a repository that is no longer online.
  The pinned commit is authored by Jeff Sackmann and its files are checked by
  hash, but newer commits on any fork are not verified. Passing `--commit`
  with another SHA prints a warning.
- Predictions are statistical estimates. They must not be used for betting or
  any commercial purpose; the data license also forbids commercial use.
