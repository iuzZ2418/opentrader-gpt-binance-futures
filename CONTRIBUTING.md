# Contributing to OpenTrader

Thanks for helping build safer AI-assisted trading infrastructure.

## Good contributions

- deterministic risk controls and failure-mode tests;
- Binance Futures protocol replay fixtures;
- point-in-time data lineage and backtest correctness;
- paper/Demo observability and reconciliation;
- documentation, translations, and reproducible research.

Strategy claims must include fees, slippage, funding, sample dates, out-of-sample methodology, and drawdown. Screenshots or isolated return percentages are not evidence.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,trader]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

Keep changes focused. Add tests for behavioral changes. Never weaken a hard risk limit, live gate, audit requirement, or fail-closed path without an explicit security rationale and adversarial tests.

## Pull requests

Describe the problem, safety impact, implementation, verification, and rollback behavior. Never include real credentials, private account data, copyrighted datasets, or code copied from a repository with an incompatible license.
