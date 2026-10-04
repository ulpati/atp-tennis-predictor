# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.x | Yes |

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/ulpati/atp-tennis-predictor/security/advisories/new).
Do not open a public issue.

Include the affected version, the steps to reproduce and the impact you
expect. You will receive an acknowledgement within 7 days. Confirmed issues
are fixed in a new release and credited in the changelog unless you prefer
otherwise.

## How security is built in

The project follows a lightweight secure software development life cycle:

1. **Requirements**: the tool must never execute code from data or model
   files, must verify what it downloads and must report honest metrics.
2. **Design**: a STRIDE threat model lists every entry point, threat and
   mitigation. See [docs/threat-model.md](docs/threat-model.md).
3. **Implementation**: no pickle, `eval`, `exec` or shell calls; typed and
   range-checked inputs; size limits; atomic writes; checksum verification.
4. **Verification**: unit and end-to-end tests for every mitigation
   (tampered files, path traversal, oversized input, CSV injection, leakage),
   with at least 85% coverage enforced in CI. Ruff, Bandit and CodeQL run
   on every change.
5. **Release and maintenance**: hash-pinned lock files, `pip-audit`,
   Dependabot for Python packages and GitHub Actions, and actions pinned by
   commit SHA.
