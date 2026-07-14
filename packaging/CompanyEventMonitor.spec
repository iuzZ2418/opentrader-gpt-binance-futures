# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPEC).resolve().parents[1]
httpx_datas, httpx_binaries, httpx_hidden = collect_all("httpx")
keyring_datas, keyring_binaries, keyring_hidden = collect_all("keyring")
docx_datas, docx_binaries, docx_hidden = collect_all("docx")

a = Analysis(
    [str(root / "packaging" / "windows_entry.py")],
    pathex=[str(root / "src")],
    binaries=httpx_binaries + keyring_binaries + docx_binaries,
    datas=httpx_datas
    + keyring_datas
    + docx_datas
    + [(str(root / "data" / "company_event_sample.json"), "data")],
    hiddenimports=httpx_hidden
    + keyring_hidden
    + docx_hidden
    + collect_submodules("pdfplumber")
    + [
        "company_event_monitor.native_app",
        "company_event_monitor.market",
        "tkinter",
        "tkinter.ttk",
        "tkinter.filedialog",
        "tkinter.messagebox",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "crypto_event_trader",
        "streamlit",
        "plotly",
        "pandas",
        "fastapi",
        "uvicorn",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CompanyEventMonitor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CompanyEventMonitor",
)
