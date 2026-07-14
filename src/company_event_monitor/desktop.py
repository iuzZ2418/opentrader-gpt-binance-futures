from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

from .bootstrap import bootstrap
from .native_app import NativeDesktopApp
from .querying import QueryCoordinator
from .storage import EventRepository

APP_NAME = "CompanyEventMonitor"


def user_data_dir() -> Path:
    if os.name == "nt":
        root = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path.home() / ".local" / "share"
    target = root / APP_NAME
    target.mkdir(parents=True, exist_ok=True)
    return target


def _enable_windows_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass


def create_native_app(*, database: Path | None = None, withdrawn: bool = False) -> NativeDesktopApp:
    import tkinter as tk

    _enable_windows_dpi_awareness()
    database = database or (user_data_dir() / "company_events.db")
    bootstrap(database, None)
    repository = EventRepository(database)
    repository.initialize()
    coordinator = QueryCoordinator(repository)
    root = tk.Tk()
    if withdrawn:
        root.withdraw()
    return NativeDesktopApp(root, repository, coordinator, database.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=not getattr(__import__("sys"), "frozen", False))
    parser.add_argument("--verify-imports", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--native-smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--database", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_imports:
        import tkinter

        import docx
        import httpx
        import keyring
        import pdfplumber

        from . import (
            market,  # noqa: F401
            native_app,  # noqa: F401
        )

        print(
            "Packaged native imports verified: "
            f"tk={tkinter.TkVersion}, httpx={httpx.__version__}, "
            f"keyring={keyring.__name__}, docx={docx.__version__}, "
            f"pdfplumber={pdfplumber.__version__}",
            flush=True,
        )
        return
    database = args.database or (user_data_dir() / "company_events.db")
    if args.bootstrap_only:
        bootstrap(database, None)
        return
    app = create_native_app(database=database, withdrawn=args.native_smoke_test)
    if args.native_smoke_test:
        app.root.update_idletasks()
        app.root.update()
        assert app.root.title() == "A股研究证据与观点跟踪工作台"
        assert set(app.nav_buttons) == {
            "query",
            "library",
            "theses",
            "batch",
            "compare",
            "settings",
        }
        print("Native desktop smoke test passed: no local web service required", flush=True)
        app.close()
        return
    app.root.mainloop()


if __name__ == "__main__":
    main()
