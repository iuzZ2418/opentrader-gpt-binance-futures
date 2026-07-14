import sys
import traceback
from datetime import datetime

from company_event_monitor.desktop import main, user_data_dir


def safe_main() -> None:
    log = user_data_dir() / "runtime.log"
    if sys.stdout is None or sys.stderr is None:
        runtime_stream = log.open("a", encoding="utf-8", buffering=1)
        sys.stdout = runtime_stream
        sys.stderr = runtime_stream
    try:
        main()
    except BaseException:  # keep frozen GUI failures diagnosable
        crash_log = user_data_dir() / "crash.log"
        with crash_log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[{datetime.now().isoformat()}] {' '.join(sys.argv)}\n")
            traceback.print_exc(file=stream)
        try:
            from tkinter import messagebox

            messagebox.showerror("启动失败", f"软件启动失败。诊断日志：\n{crash_log}")
        except Exception:
            pass
        raise SystemExit(1) from None


if __name__ == "__main__":
    safe_main()
