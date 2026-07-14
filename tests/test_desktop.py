import os
from pathlib import Path

import company_event_monitor.desktop as desktop


def test_native_desktop_uses_local_data_directory(monkeypatch, tmp_path: Path) -> None:
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        expected = tmp_path / "CompanyEventMonitor"
    else:
        monkeypatch.setenv("HOME", str(tmp_path))
        expected = tmp_path / ".local" / "share" / "CompanyEventMonitor"

    target = desktop.user_data_dir()

    assert target == expected
    assert target.is_dir()


def test_desktop_has_no_web_service_launcher() -> None:
    source = Path(desktop.__file__).read_text(encoding="utf-8")

    assert "internal_api" not in source
    assert "internal_ui" not in source
    assert "streamlit" not in source.lower()
    assert "NativeDesktopApp" in source
