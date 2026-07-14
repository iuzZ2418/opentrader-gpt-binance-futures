# Company Event Monitor

An isolated Windows desktop research workspace for tracking public A-share company disclosures, management commitments, thesis evidence, document changes, and market reactions.

## Workflow

```text
Company code or name
→ exchange/CNINFO disclosure archive
→ document extraction and normalized events
→ management commitments and thesis evidence
→ price and benchmark updates
→ strengthen / weaken / conflict review
→ local company report and comparison
```

The application does not produce buy/sell instructions or target prices. API keys are stored through Windows Credential Manager rather than in the database. It remains isolated from the Futures trading services, credentials, databases, containers, and capital controls.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\company-events-desktop.exe
```

Build the Windows package with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```
