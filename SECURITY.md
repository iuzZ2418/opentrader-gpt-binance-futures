# Security policy

## Reporting a vulnerability

Do not open a public issue for vulnerabilities that could expose credentials, bypass live gates, corrupt audit lineage, duplicate orders, or increase risk unexpectedly.

Use GitHub's private vulnerability reporting feature for this repository. Include affected version, impact, reproduction steps, and a minimal test when possible. Do not include real API keys or account data.

## Supported scope

The latest default branch is supported. Live trading remains disabled by default and is not considered production-ready without the external evidence and deployment controls listed in [docs/roadmap.md](docs/roadmap.md).

## Secrets

If a secret is committed, revoke and rotate it immediately. Removing it from the latest commit is not sufficient because Git history and forks may retain it.
