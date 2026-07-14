from __future__ import annotations

import os
import threading
import tkinter as tk
import webbrowser
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .deepseek import DeepSeekClient, get_api_key, save_api_key
from .docx_reports import company_report, comparison_report
from .querying import AmbiguousCompanyError, QueryCoordinator
from .research import IMPACT_LABELS, STANCE_LABELS, THESIS_STATES
from .storage import EventRepository

COLORS = {
    "nav": "#10233F",
    "nav_hover": "#1B365D",
    "nav_active": "#2357D9",
    "primary": "#2563EB",
    "primary_dark": "#1D4ED8",
    "accent": "#0EA5A4",
    "bg": "#F3F6FB",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",
    "text": "#172033",
    "muted": "#64748B",
    "line": "#DCE3ED",
    "positive": "#087F5B",
    "negative": "#C92A2A",
    "warning": "#C2410C",
    "purple": "#6D28D9",
}

STAGE_LABELS = {
    "resolving": "识别公司",
    "fetching": "检索公告",
    "downloading": "筛选材料",
    "processing": "AI标准化",
    "market": "更新行情",
    "summarizing": "生成总结",
    "completed": "完成",
    "failed": "失败",
    "updating": "更新公司",
    "comparing": "生成比较",
    "preparing": "准备批量任务",
}


class ScrollableFrame(tk.Frame):
    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, bg=COLORS["bg"], **kwargs)
        self.canvas = tk.Canvas(self, bg=COLORS["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = tk.Frame(self.canvas, bg=COLORS["bg"])
        self.window_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.body.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_body)
        self.canvas.bind_all("<MouseWheel>", self._mousewheel)

    def _update_scroll_region(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_body(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _mousewheel(self, event: tk.Event) -> None:
        if self.winfo_exists() and self.winfo_ismapped():
            self.canvas.yview_scroll(int(-event.delta / 120), "units")


class PriceChart(tk.Canvas):
    def __init__(self, master: tk.Misc, prices: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(
            master,
            height=220,
            bg=COLORS["surface"],
            highlightthickness=0,
            **kwargs,
        )
        self.prices = prices[-90:]
        self.bind("<Configure>", lambda _event: self.draw())

    def draw(self) -> None:
        self.delete("all")
        if len(self.prices) < 2:
            self.create_text(20, 20, anchor="nw", text="暂无价格序列", fill=COLORS["muted"])
            return
        width = max(200, self.winfo_width())
        height = max(160, self.winfo_height())
        left, right, top, bottom = 54, 18, 18, 34
        values = [float(item["close"]) for item in self.prices]
        low, high = min(values), max(values)
        padding = max((high - low) * 0.12, high * 0.01)
        low -= padding
        high += padding
        plot_w = width - left - right
        plot_h = height - top - bottom
        for step in range(5):
            y = top + plot_h * step / 4
            value = high - (high - low) * step / 4
            self.create_line(left, y, width - right, y, fill="#E8EDF5")
            self.create_text(
                left - 8,
                y,
                anchor="e",
                text=f"{value:.2f}",
                fill=COLORS["muted"],
                font=("Segoe UI", 8),
            )
        points: list[float] = []
        for index, value in enumerate(values):
            x = left + plot_w * index / (len(values) - 1)
            y = top + (high - value) / (high - low) * plot_h
            points.extend((x, y))
        fill_points = [left, top + plot_h, *points, width - right, top + plot_h]
        self.create_polygon(fill_points, fill="#E7F0FF", outline="")
        self.create_line(points, fill=COLORS["primary"], width=2.2, smooth=True)
        first = self.prices[0]["trade_date"]
        last = self.prices[-1]["trade_date"]
        self.create_text(
            left, height - 10, anchor="sw", text=first, fill=COLORS["muted"], font=("Segoe UI", 8)
        )
        self.create_text(
            width - right,
            height - 10,
            anchor="se",
            text=last,
            fill=COLORS["muted"],
            font=("Segoe UI", 8),
        )


class NativeDesktopApp:
    def __init__(
        self,
        root: tk.Tk,
        repository: EventRepository,
        coordinator: QueryCoordinator,
        data_dir: Path,
    ) -> None:
        self.root = root
        self.repository = repository
        self.coordinator = coordinator
        self.data_dir = data_dir
        self.current_page = "query"
        self.nav_buttons: dict[str, tk.Button] = {}
        self.current_job_id = ""
        self.current_comparison_id = ""
        self.current_batch_id = ""
        self.query_view = "home"
        self.library_view_company_id = ""
        self.compare_view = "selection"
        self.batch_view = "new"
        self.last_comparison_id = ""
        self.last_batch_id = ""
        self.compare_selected_ids: set[str] = set()
        self.library_selected_ids: set[str] = set()
        self.batch_candidates: dict[str, Any] = {}
        self.batch_selected_ids: set[str] = set()
        self.thesis_rows: dict[str, dict[str, Any]] = {}
        self.thesis_company_map: dict[str, str] = {}
        self.report_event_rows: dict[str, dict[str, Any]] = {}
        self.library_rows: dict[str, dict[str, Any]] = {}
        self.compare_rows: dict[str, dict[str, Any]] = {}
        self._configure_root()
        self._build_shell()
        self.show_query()
        self.root.after(700, self._poll_background_tasks)

    def _configure_root(self) -> None:
        self.root.title("A股研究证据与观点跟踪工作台")
        self.root.geometry("1380x860")
        self.root.minsize(1120, 700)
        self.root.configure(bg=COLORS["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "App.Treeview",
            background=COLORS["surface"],
            fieldbackground=COLORS["surface"],
            foreground=COLORS["text"],
            rowheight=34,
            borderwidth=0,
        )
        style.configure(
            "App.Treeview.Heading",
            background="#EAF0F8",
            foreground=COLORS["text"],
            relief="flat",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map(
            "App.Treeview",
            background=[("selected", "#D9E7FF")],
            foreground=[("selected", COLORS["text"])],
        )
        style.configure(
            "Blue.Horizontal.TProgressbar", troughcolor="#DCE8FA", background=COLORS["primary"]
        )
        style.configure(
            "Compact.Treeview",
            background=COLORS["surface"],
            fieldbackground=COLORS["surface"],
            foreground=COLORS["text"],
            rowheight=30,
            borderwidth=0,
        )
        style.configure(
            "Compact.Treeview.Heading",
            background="#EAF0F8",
            foreground=COLORS["text"],
            relief="flat",
            font=("Microsoft YaHei UI", 9, "bold"),
        )

    def _build_shell(self) -> None:
        self.nav = tk.Frame(self.root, bg=COLORS["nav"], width=228)
        self.nav.pack(side="left", fill="y")
        self.nav.pack_propagate(False)
        brand = tk.Frame(self.nav, bg=COLORS["nav"], padx=22, pady=24)
        brand.pack(fill="x")
        tk.Label(
            brand,
            text="研报雷达",
            bg=COLORS["nav"],
            fg="white",
            font=("Microsoft YaHei UI", 19, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="公开信息 · 证据 · 观点",
            bg=COLORS["nav"],
            fg="#9FB3CE",
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(4, 0))
        nav_items = [
            ("query", "⌕  查询公司", self.show_query),
            ("library", "▣  已查公司", self.show_library),
            ("theses", "◎  研究观点", self.show_theses),
            ("batch", "▦  批量研究", self.show_batch),
            ("compare", "⇄  公司比较", self.show_compare),
            ("settings", "⚙  设置", self.show_settings),
        ]
        for key, label, command in nav_items:
            button = tk.Button(
                self.nav,
                text=label,
                command=command,
                anchor="w",
                padx=24,
                pady=13,
                bg=COLORS["nav"],
                fg="#D7E1EF",
                activebackground=COLORS["nav_hover"],
                activeforeground="white",
                relief="flat",
                bd=0,
                cursor="hand2",
                font=("Microsoft YaHei UI", 11),
            )
            button.pack(fill="x", padx=10, pady=3)
            self.nav_buttons[key] = button
        tk.Label(
            self.nav,
            text="数据仅保存在本机\n不构成投资建议",
            justify="left",
            bg=COLORS["nav"],
            fg="#7F94B0",
            font=("Microsoft YaHei UI", 8),
        ).pack(side="bottom", anchor="w", padx=24, pady=22)
        self.main = tk.Frame(self.root, bg=COLORS["bg"])
        self.main.pack(side="left", fill="both", expand=True)
        self.top = tk.Frame(self.main, bg=COLORS["surface"], height=64, padx=28)
        self.top.pack(fill="x")
        self.top.pack_propagate(False)
        self.page_title = tk.Label(
            self.top,
            text="",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 16, "bold"),
        )
        self.page_title.pack(side="left", pady=17)
        self.connection_label = tk.Label(
            self.top,
            text="● 本地数据库已连接",
            bg=COLORS["surface"],
            fg=COLORS["positive"],
            font=("Microsoft YaHei UI", 9),
        )
        self.connection_label.pack(side="right")
        self.task_strip = tk.Frame(
            self.main,
            bg="#EAF2FF",
            highlightbackground="#C9DBF6",
            highlightthickness=1,
            padx=28,
            pady=8,
        )
        self.task_strip_label = tk.Label(
            self.task_strip,
            text="",
            bg="#EAF2FF",
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self.task_strip_label.pack(side="left")
        self.task_strip_progress = ttk.Progressbar(
            self.task_strip,
            maximum=100,
            length=220,
            style="Blue.Horizontal.TProgressbar",
        )
        self.task_strip_progress.pack(side="left", padx=16)
        self.task_strip_open = self._button(
            self.task_strip, "查看进度", self._open_background_task, primary=False
        )
        self.task_strip_open.pack(side="right")
        self.content = tk.Frame(self.main, bg=COLORS["bg"])
        self.content.pack(fill="both", expand=True)

    def _set_page(self, key: str, title: str) -> None:
        self.current_page = key
        self.page_title.configure(text=title)
        for name, button in self.nav_buttons.items():
            button.configure(
                bg=COLORS["nav_active"] if name == key else COLORS["nav"],
                fg="white" if name == key else "#D7E1EF",
            )
        for child in self.content.winfo_children():
            child.destroy()

    def _poll_background_tasks(self) -> None:
        """Keep task progress visible even when the user navigates to another page."""
        try:
            tasks: list[tuple[str, str, float, str, str]] = []
            for job in self.repository.active_query_jobs():
                name = job.get("company_name") or job.get("query_text") or "公司查询"
                tasks.append(
                    (
                        "query",
                        str(job["id"]),
                        float(job.get("progress") or 0),
                        f"{name} · {STAGE_LABELS.get(job.get('stage', ''), job.get('stage', ''))}",
                        str(job.get("message") or ""),
                    )
                )
            for batch in self.repository.active_batch_queries():
                tasks.append(
                    (
                        "batch",
                        str(batch["id"]),
                        float(batch.get("progress") or 0),
                        f"批量研究 · {batch.get('name', '')}",
                        (
                            f"已完成 {batch.get('completed_companies', 0)}/"
                            f"{batch.get('total_companies', 0)} 家"
                        ),
                    )
                )
            for comparison in self.repository.active_comparisons():
                # Batch-owned comparisons are covered by their parent progress strip.
                if any(
                    batch.get("comparison_id") == comparison.get("id")
                    for batch in self.repository.active_batch_queries()
                ):
                    continue
                comparison_stage = comparison.get("stage", "")
                tasks.append(
                    (
                        "compare",
                        str(comparison["id"]),
                        float(comparison.get("progress") or 0),
                        (
                            "公司比较 · "
                            f"{STAGE_LABELS.get(comparison_stage, comparison_stage)}"
                        ),
                        f"{len(comparison.get('members') or [])} 家公司",
                    )
                )
            if tasks:
                task_type, task_id, progress, label, detail = tasks[0]
                self._background_focus = (task_type, task_id)
                suffix = f"（另有 {len(tasks) - 1} 个任务）" if len(tasks) > 1 else ""
                self.task_strip_label.configure(text=f"{label}  {detail} {suffix}".strip())
                self.task_strip_progress.configure(value=progress * 100)
                if not self.task_strip.winfo_ismapped():
                    self.task_strip.pack(fill="x", after=self.top)
            elif self.task_strip.winfo_ismapped():
                self.task_strip.pack_forget()
        except Exception:
            pass
        finally:
            self.root.after(800, self._poll_background_tasks)

    def _open_background_task(self) -> None:
        task_type, task_id = getattr(self, "_background_focus", ("", ""))
        if task_type == "query":
            self.current_job_id = task_id
            self.query_view = "progress"
            self._render_query_progress(self.repository.query_job(task_id))
        elif task_type == "compare":
            self.current_comparison_id = task_id
            self.compare_view = "progress"
            members = self.repository.comparison(task_id).get("members") or []
            self._render_comparison_starting(members)
        elif task_type == "batch":
            self.current_batch_id = task_id
            self.batch_view = "progress"
            self._render_batch_progress(self.repository.batch_query(task_id))

    def show_query(self) -> None:
        if self.current_job_id:
            try:
                job = self.repository.query_job(self.current_job_id)
                if job.get("status") in {"pending", "running"}:
                    self.query_view = "progress"
                    self._render_query_progress(job)
                    return
            except KeyError:
                self.current_job_id = ""
        self.query_view = "home"
        self._set_page("query", "查询公司")
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        body = scroll.body
        body.grid_columnconfigure(0, weight=1)
        hero = tk.Frame(body, bg="#174EA6", padx=38, pady=34)
        hero.grid(row=0, column=0, sticky="ew", padx=28, pady=(28, 18))
        tk.Label(
            hero,
            text="从一家公司开始研究",
            bg="#174EA6",
            fg="white",
            font=("Microsoft YaHei UI", 24, "bold"),
        ).pack(anchor="w")
        tk.Label(
            hero,
            text="输入股票代码或简称，软件将更新公告、标准化事件、行情与研究情景。",
            bg="#174EA6",
            fg="#D9E8FF",
            font=("Microsoft YaHei UI", 11),
        ).pack(anchor="w", pady=(8, 22))
        search_row = tk.Frame(hero, bg="#174EA6")
        search_row.pack(fill="x")
        self.search_entry = tk.Entry(
            search_row,
            font=("Microsoft YaHei UI", 13),
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=12, padx=(0, 10))
        self.search_entry.insert(0, "")
        self.search_entry.bind("<Return>", lambda _event: self._query_from_entry())
        self._button(
            search_row, "查询最新信息", self._query_from_entry, primary=False, bg="#0EA5A4"
        ).pack(side="right", ipady=4)
        self.search_entry.focus_set()
        info = self._surface(body, padx=22, pady=18)
        info.grid(row=1, column=0, sticky="ew", padx=28, pady=10)
        tk.Label(
            info,
            text="一次查询会完成什么",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 13, "bold"),
        ).pack(anchor="w")
        steps = (
            "官方公告检索  →  高价值材料筛选  →  DeepSeek标准化  →  行情更新  →  "
            "事件—价格联动  →  研究总结"
        )
        tk.Label(
            info,
            text=steps,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            wraplength=960,
            justify="left",
        ).pack(anchor="w", pady=(9, 0))
        recent = self.repository.library_companies()[:5]
        tk.Label(
            body,
            text="最近查询",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 14, "bold"),
        ).grid(row=2, column=0, sticky="w", padx=30, pady=(18, 5))
        recent_frame = tk.Frame(body, bg=COLORS["bg"])
        recent_frame.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 28))
        recent_frame.grid_columnconfigure(
            tuple(range(max(1, len(recent)))), weight=1, uniform="recent"
        )
        if not recent:
            empty = self._surface(recent_frame, padx=24, pady=24)
            empty.grid(row=0, column=0, sticky="ew")
            tk.Label(
                empty,
                text="尚无已查公司。完成首次查询后，结果将自动保存在本地。",
                bg=COLORS["surface"],
                fg=COLORS["muted"],
            ).pack(anchor="w")
        for index, item in enumerate(recent):
            card = self._surface(recent_frame, padx=18, pady=16)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 6, 6))
            tk.Label(
                card,
                text=item["name"],
                bg=COLORS["surface"],
                fg=COLORS["text"],
                font=("Microsoft YaHei UI", 11, "bold"),
            ).pack(anchor="w")
            tk.Label(
                card,
                text=f"{item['ticker']} · {item.get('market') or 'A股'}",
                bg=COLORS["surface"],
                fg=COLORS["muted"],
                font=("Microsoft YaHei UI", 8),
            ).pack(anchor="w", pady=(2, 8))
            tk.Label(
                card,
                text=f"{item.get('event_count', 0)} 项事件",
                bg=COLORS["surface"],
                fg=COLORS["primary"],
            ).pack(anchor="w")
            self._link_button(
                card, "打开报告", lambda company_id=item["company_id"]: self.show_report(company_id)
            ).pack(anchor="w", pady=(10, 0))

    def _query_from_entry(self) -> None:
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showinfo(
                "请输入公司",
                "请输入股票代码或公司简称，例如 002050 或 三花智控。",
                parent=self.root,
            )
            return
        self._launch_query(query)

    def _launch_query(self, query: str, company_id: str = "") -> None:
        self._show_query_starting(query)

        def worker() -> None:
            try:
                job = self.coordinator.start_query(query, company_id)
            except AmbiguousCompanyError as error:
                candidates = error.candidates
                self.root.after(
                    0,
                    lambda candidates=candidates: self._show_candidates(query, candidates),
                )
                return
            except Exception as error:
                self.root.after(0, lambda error=error: self._query_start_failed(str(error)))
                return
            self.root.after(0, lambda: self._begin_job_poll(str(job["id"])))

        threading.Thread(target=worker, daemon=True, name="desktop-query-start").start()

    def _show_query_starting(self, query: str) -> None:
        self.query_view = "progress"
        self._set_page("query", "查询公司")
        panel = self._surface(self.content, padx=34, pady=34)
        panel.pack(fill="x", padx=36, pady=36)
        tk.Label(
            panel,
            text=f"正在识别：{query}",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            panel, text="正在访问官方公司目录，请稍候……", bg=COLORS["surface"], fg=COLORS["muted"]
        ).pack(anchor="w", pady=(10, 20))
        bar = ttk.Progressbar(panel, mode="indeterminate", style="Blue.Horizontal.TProgressbar")
        bar.pack(fill="x")
        bar.start(12)

    def _show_candidates(self, query: str, candidates: list[Any]) -> None:
        self.show_query()
        dialog = tk.Toplevel(self.root)
        dialog.title("请选择公司")
        dialog.geometry("620x380")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = tk.Frame(dialog, bg=COLORS["surface"], padx=24, pady=22)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text="名称存在多个匹配项",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 15, "bold"),
        ).pack(anchor="w")
        tk.Label(
            frame, text="请选择正确的上市公司后继续查询。", bg=COLORS["surface"], fg=COLORS["muted"]
        ).pack(anchor="w", pady=(4, 14))
        tree = self._tree(frame, ("ticker", "name", "market"), ("代码", "公司", "市场"), height=8)
        tree.pack(fill="both", expand=True)
        mapping: dict[str, Any] = {}
        for company in candidates:
            iid = tree.insert("", "end", values=(company.ticker, company.name, company.market))
            mapping[iid] = company

        def confirm() -> None:
            selected = tree.selection()
            if not selected:
                return
            company = mapping[selected[0]]
            dialog.destroy()
            self._launch_query(query, company.company_id)

        buttons = tk.Frame(frame, bg=COLORS["surface"])
        buttons.pack(fill="x", pady=(14, 0))
        self._button(buttons, "取消", dialog.destroy, primary=False).pack(side="right")
        self._button(buttons, "继续查询", confirm).pack(side="right", padx=(0, 10))
        tree.bind("<Double-1>", lambda _event: confirm())

    def _query_start_failed(self, error: str) -> None:
        self.show_query()
        messagebox.showerror("无法开始查询", error, parent=self.root)

    def _begin_job_poll(self, job_id: str) -> None:
        self.current_job_id = job_id
        self.query_view = "progress"
        if self.current_page == "query":
            self._render_query_progress(self.repository.query_job(job_id))
        self.root.after(500, self._poll_query_job)

    def _render_query_progress(self, job: dict[str, Any]) -> None:
        self._set_page("query", "正在更新公司")
        panel = self._surface(self.content, padx=34, pady=30)
        panel.pack(fill="x", padx=36, pady=(36, 18))
        self.job_stage_label = tk.Label(
            panel,
            text=STAGE_LABELS.get(job["stage"], job["stage"]),
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 19, "bold"),
        )
        self.job_stage_label.pack(anchor="w")
        self.job_message_label = tk.Label(
            panel,
            text=job.get("message") or "正在处理",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            justify="left",
            wraplength=900,
        )
        self.job_message_label.pack(anchor="w", pady=(8, 18))
        self.job_progress = ttk.Progressbar(
            panel,
            maximum=100,
            value=float(job.get("progress") or 0) * 100,
            style="Blue.Horizontal.TProgressbar",
        )
        self.job_progress.pack(fill="x")
        stats = tk.Frame(panel, bg=COLORS["surface"])
        stats.pack(fill="x", pady=(18, 0))
        self.job_stats_label = tk.Label(stats, text="", bg=COLORS["surface"], fg=COLORS["text"])
        self.job_stats_label.pack(anchor="w")
        tk.Label(
            self.content,
            text="查询任务在本机后台运行。已完成的文档会即时保存，意外退出后不会丢失。",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
        ).pack(anchor="w", padx=40)
        self._update_job_widgets(job)

    def _update_job_widgets(self, job: dict[str, Any]) -> None:
        if not hasattr(self, "job_progress") or not self.job_progress.winfo_exists():
            return
        self.job_progress.configure(value=float(job.get("progress") or 0) * 100)
        self.job_stage_label.configure(
            text=STAGE_LABELS.get(job.get("stage", ""), job.get("stage", ""))
        )
        self.job_message_label.configure(text=job.get("message") or "正在处理")
        self.job_stats_label.configure(
            text=(
                f"发现材料 {job.get('discovered_documents', 0)}  ·  "
                f"历史跳过 {job.get('skipped_documents', 0)}  ·  "
                f"已处理 {job.get('processed_documents', 0)}  ·  "
                f"新增文档 {job.get('inserted_documents', 0)}  ·  "
                f"新增事件 {job.get('inserted_events', 0)}"
            )
        )

    def _poll_query_job(self) -> None:
        if not self.current_job_id:
            return
        try:
            job = self.repository.query_job(self.current_job_id)
        except KeyError:
            return
        self._update_job_widgets(job)
        if job["status"] == "completed":
            company_id = str(job.get("company_id") or "")
            self.current_job_id = ""
            self.query_view = "home"
            self.library_view_company_id = company_id
            if self.current_page == "query":
                self.root.after(250, lambda: self.show_report(company_id))
        elif job["status"] == "failed":
            self.current_job_id = ""
            self.query_view = "home"
            if self.current_page == "query":
                messagebox.showerror(
                    "查询失败",
                    f"{job.get('message', '')}\n\n{job.get('error', '')}",
                    parent=self.root,
                )
                self.show_library(reset=True)
        else:
            self.root.after(600, self._poll_query_job)

    def show_report(self, company_id: str) -> None:
        snapshot = self.repository.latest_snapshot(company_id)
        if snapshot is None:
            messagebox.showinfo("暂无报告", "该公司尚未完成查询。", parent=self.root)
            return
        summary = snapshot["summary"]
        company = summary.get("company") or {}
        self.library_view_company_id = company_id
        self._set_page("library", f"{company.get('name', '')} · 公司报告")
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        body = scroll.body
        body.grid_columnconfigure(0, weight=1)
        header = self._surface(body, padx=26, pady=22)
        header.grid(row=0, column=0, sticky="ew", padx=26, pady=(24, 12))
        title_row = tk.Frame(header, bg=COLORS["surface"])
        title_row.pack(fill="x")
        self._button(
            title_row, "返回公司库", lambda: self.show_library(reset=True), primary=False
        ).pack(side="right", padx=(8, 0))
        title_block = tk.Frame(title_row, bg=COLORS["surface"])
        title_block.pack(side="left", fill="x", expand=True)
        tk.Label(
            title_block,
            text=company.get("name", ""),
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 21, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text=(
                f"{company.get('ticker', '')} · {company.get('market', '')}　"
                f"数据截止 {self._date(summary.get('data_as_of'))}"
            ),
            bg=COLORS["surface"],
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(4, 0))
        buttons = tk.Frame(title_row, bg=COLORS["surface"])
        buttons.pack(side="right")
        self._button(
            buttons, "重新查询", lambda: self._launch_query(company.get("ticker", ""), company_id)
        ).pack(side="left", padx=4)
        self._button(
            buttons, "导出Word", lambda: self._export_company(snapshot), primary=False
        ).pack(side="left", padx=4)
        self._button(
            buttons,
            "研究观点",
            lambda: self.show_theses(company_id),
            primary=False,
        ).pack(side="left", padx=4)
        counts = summary.get("counts") or {}
        metrics = [
            ("发现材料", summary.get("discovered_document_count", 0), COLORS["primary"]),
            ("已分析", summary.get("processed_document_count", 0), COLORS["purple"]),
            ("正向事件", counts.get("positive", 0), COLORS["positive"]),
            ("负向风险", counts.get("negative", 0), COLORS["negative"]),
            ("风险升级", counts.get("escalation", 0), COLORS["warning"]),
            ("表述冲突", counts.get("conflict", 0), COLORS["purple"]),
        ]
        metric_row = tk.Frame(body, bg=COLORS["bg"])
        metric_row.grid(row=1, column=0, sticky="ew", padx=26, pady=4)
        for index in range(len(metrics)):
            metric_row.grid_columnconfigure(index, weight=1, uniform="metric")
        for index, (label, value, color) in enumerate(metrics):
            self._metric(metric_row, label, str(value), color).grid(
                row=0, column=index, sticky="nsew", padx=4
            )
        research = summary.get("research_summary") or {}
        executive = self._surface(body, padx=24, pady=20)
        executive.grid(row=2, column=0, sticky="ew", padx=26, pady=12)
        self._section_title(executive, "AI研究摘要", "基于公开事件与行情统计，不包含投资评级").pack(
            fill="x"
        )
        tk.Label(
            executive,
            text=research.get("summary") or "暂无总结",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            wraplength=1000,
            justify="left",
            font=("Microsoft YaHei UI", 11),
        ).pack(anchor="w", pady=(14, 8))
        for point in research.get("key_points") or []:
            tk.Label(
                executive,
                text=f"• {point}",
                bg=COLORS["surface"],
                fg=COLORS["text"],
                wraplength=1000,
                justify="left",
            ).pack(anchor="w", pady=2)
        workspace = summary.get("research_workspace") or {}
        market = summary.get("market_analysis") or {}
        next_row = 3
        if workspace:
            self._render_research_workspace(body, next_row, workspace)
            next_row += 1
        if market:
            self._render_market(body, next_row, market)
            next_row += 1
        events_panel = self._surface(body, padx=20, pady=18)
        events_panel.grid(row=next_row, column=0, sticky="ew", padx=26, pady=12)
        self._section_title(
            events_panel, "重要事件与原文证据", "双击事件查看标准化事实、原文、页码和官方链接"
        ).pack(fill="x", pady=(0, 12))
        event_columns = ("date", "direction", "change", "score", "event")
        tree = self._tree(
            events_panel, event_columns, ("日期", "方向", "变化", "关注度", "标准化事件"), height=12
        )
        tree.column("date", width=95, stretch=False)
        tree.column("direction", width=70, stretch=False)
        tree.column("change", width=80, stretch=False)
        tree.column("score", width=80, stretch=False)
        tree.column("event", width=650)
        tree.pack(fill="both", expand=True)
        self.report_event_rows = {}
        for event in summary.get("important_events") or []:
            direction = (
                "正向"
                if event.get("direction", 0) > 0
                else "负向"
                if event.get("direction", 0) < 0
                else "中性"
            )
            iid = tree.insert(
                "",
                "end",
                values=(
                    self._date(event.get("published_at")),
                    direction,
                    event.get("change_type", ""),
                    f"{float(event.get('value_score') or 0):.3f}",
                    event.get("standardized_text", ""),
                ),
            )
            self.report_event_rows[iid] = event
        tree.bind("<Double-1>", lambda _event: self._open_selected_event(tree))
        questions = research.get("research_questions") or []
        if questions:
            question_panel = self._surface(body, padx=22, pady=18)
            question_panel.grid(row=next_row + 1, column=0, sticky="ew", padx=26, pady=12)
            self._section_title(question_panel, "后续核验问题", "用于形成下一步研究清单").pack(
                fill="x", pady=(0, 10)
            )
            for index, question in enumerate(questions, start=1):
                tk.Label(
                    question_panel,
                    text=f"{index}. {question}",
                    bg=COLORS["surface"],
                    fg=COLORS["text"],
                    wraplength=1000,
                    justify="left",
                ).pack(anchor="w", pady=3)
            next_row += 1
        documents = self.repository.company_disclosures(company_id, 200)
        docs_panel = self._surface(body, padx=20, pady=18)
        docs_panel.grid(row=next_row + 1, column=0, sticky="ew", padx=26, pady=(12, 30))
        self._section_title(
            docs_panel, "公开材料覆盖", "记录发现、筛选和解析状态；双击打开官方原文"
        ).pack(fill="x", pady=(0, 12))
        doc_tree = self._tree(
            docs_panel,
            ("date", "source", "status", "title"),
            ("日期", "来源", "状态", "标题"),
            height=8,
        )
        doc_tree.column("date", width=95, stretch=False)
        doc_tree.column("source", width=135, stretch=False)
        doc_tree.column("status", width=90, stretch=False)
        doc_tree.column("title", width=700)
        doc_tree.pack(fill="both", expand=True)
        doc_map: dict[str, str] = {}
        for document in documents:
            iid = doc_tree.insert(
                "",
                "end",
                values=(
                    self._date(document.get("published_at")),
                    document.get("source_name"),
                    document.get("processing_status"),
                    document.get("title"),
                ),
            )
            doc_map[iid] = str(document.get("url") or "")
        doc_tree.bind("<Double-1>", lambda _event: self._open_tree_url(doc_tree, doc_map))

    def _render_research_workspace(
        self,
        parent: tk.Misc,
        row: int,
        workspace: dict[str, Any],
    ) -> None:
        panel = self._surface(parent, padx=22, pady=18)
        panel.grid(row=row, column=0, sticky="ew", padx=26, pady=12)
        self._section_title(
            panel,
            "观点—证据工作台",
            "新事件会自动判断对研究观点的支持、反对或分歧，并跟踪管理层承诺",
        ).pack(fill="x", pady=(0, 12))
        theses = workspace.get("theses") or []
        commitments = workspace.get("commitments") or []
        changes = workspace.get("document_changes") or []
        coverage = workspace.get("evidence_coverage") or {}
        metrics = tk.Frame(panel, bg=COLORS["surface"])
        metrics.pack(fill="x", pady=(0, 12))
        values = (
            ("研究观点", len(theses), COLORS["primary"]),
            (
                "证据加强",
                sum(item.get("state") == "strengthened" for item in theses),
                COLORS["positive"],
            ),
            (
                "证据削弱",
                sum(item.get("state") == "weakened" for item in theses),
                COLORS["negative"],
            ),
            (
                "待验证承诺",
                sum(item.get("status") == "open" for item in commitments),
                COLORS["warning"],
            ),
            ("文档变化", len(changes), COLORS["purple"]),
            (
                "证据覆盖",
                f"{float(coverage.get('grounded_ratio') or 0):.0%}",
                COLORS["accent"],
            ),
        )
        for index, (label, value, color) in enumerate(values):
            metrics.grid_columnconfigure(index, weight=1, uniform="researchmetric")
            self._mini_metric(metrics, label, str(value), color).grid(
                row=0, column=index, sticky="nsew", padx=3
            )
        for thesis in theses[:5]:
            row_frame = tk.Frame(panel, bg=COLORS["surface"])
            row_frame.pack(fill="x", pady=3)
            state = THESIS_STATES.get(str(thesis.get("state")), str(thesis.get("state") or ""))
            color = (
                COLORS["positive"]
                if thesis.get("state") == "strengthened"
                else COLORS["negative"]
                if thesis.get("state") == "weakened"
                else COLORS["warning"]
                if thesis.get("state") == "contested"
                else COLORS["muted"]
            )
            tk.Label(
                row_frame,
                text=f"{state}  {float(thesis.get('evidence_score') or 0):+.2f}",
                bg=COLORS["surface"],
                fg=color,
                width=17,
                anchor="w",
                font=("Microsoft YaHei UI", 9, "bold"),
            ).pack(side="left")
            tk.Label(
                row_frame,
                text=str(thesis.get("title") or ""),
                bg=COLORS["surface"],
                fg=COLORS["text"],
                wraplength=760,
                justify="left",
            ).pack(side="left", fill="x", expand=True)
            thesis_id = str(thesis.get("id") or "")
            self._link_button(
                row_frame,
                "证据",
                lambda value=thesis_id: self._thesis_dialog(value),
            ).pack(side="right")

    def _render_market(self, parent: tk.Misc, row: int, market: dict[str, Any]) -> None:
        panel = self._surface(parent, padx=22, pady=18)
        panel.grid(row=row, column=0, sticky="ew", padx=26, pady=12)
        self._section_title(
            panel, "价格与事件联动", f"{market.get('source', '')} · 截止 {market.get('as_of', '')}"
        ).pack(fill="x", pady=(0, 12))
        returns = market.get("returns") or {}
        benchmark = market.get("benchmark") or {}
        risk = market.get("risk") or {}
        market_metrics = [
            ("最新价格", f"{float(market.get('latest_price') or 0):.2f}", COLORS["text"]),
            ("近20日", self._pct(returns.get("20d")), self._return_color(returns.get("20d"))),
            ("近60日", self._pct(returns.get("60d")), self._return_color(returns.get("60d"))),
            (
                "相对基准20日",
                self._pct(benchmark.get("excess_20d")),
                self._return_color(benchmark.get("excess_20d")),
            ),
            (
                "20日年化波动",
                self._prob(risk.get("annualized_volatility_20d")),
                COLORS["warning"],
            ),
            ("60日最大回撤", self._pct(risk.get("max_drawdown_60d")), COLORS["negative"]),
        ]
        metric_row = tk.Frame(panel, bg=COLORS["surface"])
        metric_row.pack(fill="x")
        for index in range(len(market_metrics)):
            metric_row.grid_columnconfigure(index, weight=1, uniform="marketmetric")
        for index, (label, value, color) in enumerate(market_metrics):
            self._mini_metric(metric_row, label, value, color).grid(
                row=0, column=index, sticky="nsew", padx=3
            )
        valuation = market.get("valuation") or {}
        if valuation:
            valuation_text = (
                f"估值快照：动态PE {self._multiple(valuation.get('pe_dynamic'))}　"
                f"PB {self._multiple(valuation.get('pb'))}　"
                f"总市值 {self._number(valuation.get('total_market_cap_yi'))} 亿元　"
                f"换手率 {self._prob_from_percent(valuation.get('turnover_rate'))}　"
                f"量比 {self._number(valuation.get('volume_ratio'))}"
            )
            tk.Label(
                panel,
                text=valuation_text,
                bg=COLORS["surface"],
                fg=COLORS["muted"],
                justify="left",
            ).pack(anchor="w", pady=(12, 0))
        chart = PriceChart(panel, market.get("recent_prices") or [])
        chart.pack(fill="x", pady=(16, 10))
        forecast = market.get("forecast_20d") or {}
        forecast_box = tk.Frame(panel, bg="#F1F6FF", padx=18, pady=15)
        forecast_box.pack(fill="x", pady=(8, 4))
        tk.Label(
            forecast_box,
            text=f"20交易日研究情景：{forecast.get('regime', '暂无')}",
            bg="#F1F6FF",
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 12, "bold"),
        ).pack(anchor="w")
        probabilities = forecast.get("probabilities") or {}
        price_range = forecast.get("price_range") or {}
        range_text = (
            f"下行情景 {float(price_range.get('downside_p10') or 0):.2f}　"
            f"中位情景 {float(price_range.get('median_p50') or 0):.2f}　"
            f"上行情景 {float(price_range.get('upside_p90') or 0):.2f}　 |　"
            f"概率：上行 {self._prob(probabilities.get('up'))} / "
            f"震荡 {self._prob(probabilities.get('neutral'))} / "
            f"下行 {self._prob(probabilities.get('down'))}"
        )
        tk.Label(
            forecast_box,
            text=range_text,
            bg="#F1F6FF",
            fg=COLORS["text"],
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", pady=(7, 3))
        backtest = forecast.get("backtest") or {}
        hit_rate = backtest.get("direction_hit_rate")
        validation = (
            "暂无足够样本"
            if hit_rate is None
            else (
                f"历史方向命中率 {self._prob(hit_rate)}"
                f"（{backtest.get('sample_count', 0)}个滚动样本）"
            )
        )
        tk.Label(
            forecast_box,
            text=(
                f"可信度 {self._prob(forecast.get('confidence'))} · {validation} · "
                f"{forecast.get('calibration_note', '')}"
            ),
            bg="#F1F6FF",
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(3, 0))
        reasons = (market.get("signals") or {}).get("reasons") or []
        tk.Label(
            panel,
            text="分析依据：" + "；".join(reasons[:6]),
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", pady=(10, 2))
        tk.Label(
            panel,
            text=market.get("disclaimer", ""),
            bg=COLORS["surface"],
            fg=COLORS["warning"],
            wraplength=1000,
            justify="left",
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w", pady=(3, 0))
        links = market.get("event_price_links") or []
        if links:
            tk.Label(
                panel,
                text="公告后价格反应",
                bg=COLORS["surface"],
                fg=COLORS["text"],
                font=("Microsoft YaHei UI", 11, "bold"),
            ).pack(anchor="w", pady=(18, 8))
            reaction_tree = self._tree(
                panel,
                ("date", "event", "r5", "r20", "excess"),
                ("公告日", "标准化事件", "后5日", "后20日", "20日超额"),
                height=min(6, len(links)),
            )
            reaction_tree.column("date", width=95, stretch=False)
            reaction_tree.column("event", width=590)
            reaction_tree.column("r5", width=80, stretch=False)
            reaction_tree.column("r20", width=80, stretch=False)
            reaction_tree.column("excess", width=90, stretch=False)
            reaction_tree.pack(fill="x")
            for link in links:
                reaction_tree.insert(
                    "",
                    "end",
                    values=(
                        link.get("date", ""),
                        link.get("event", ""),
                        self._pct(link.get("forward_5d")),
                        self._pct(link.get("forward_20d")),
                        self._pct(link.get("excess_20d")),
                    ),
                )

    def show_library(self, reset: bool = False) -> None:
        if reset:
            self.library_view_company_id = ""
        elif self.library_view_company_id:
            snapshot = self.repository.latest_snapshot(self.library_view_company_id)
            if snapshot is not None:
                self.show_report(self.library_view_company_id)
                return
            self.library_view_company_id = ""
        self._set_page("library", "已查公司")
        frame = tk.Frame(self.content, bg=COLORS["bg"], padx=28, pady=26)
        frame.pack(fill="both", expand=True)
        toolbar = tk.Frame(frame, bg=COLORS["bg"])
        toolbar.pack(fill="x", pady=(0, 14))
        tk.Label(
            toolbar, text="所有查询记录均保存在本机", bg=COLORS["bg"], fg=COLORS["muted"]
        ).pack(side="left")
        self._button(
            toolbar, "刷新", lambda: self.show_library(reset=True), primary=False
        ).pack(side="right", padx=4)
        self._button(toolbar, "比较所选", self._compare_library_selected).pack(side="right", padx=4)
        self._button(toolbar, "更新所选", self._update_library_selected, primary=False).pack(
            side="right", padx=4
        )
        panel = self._surface(frame, padx=14, pady=14)
        panel.pack(fill="both", expand=True)
        columns = ("checked", "company", "last", "counts", "latest")
        tree = self._tree(
            panel,
            columns,
            ("选择", "公司", "最近查询", "材料 / 事件", "最近重要事件"),
            height=20,
            selectmode="none",
            style="Compact.Treeview",
        )
        tree.column("checked", width=54, stretch=False, anchor="center")
        tree.column("company", width=210)
        tree.column("last", width=120, stretch=False)
        tree.column("counts", width=125, stretch=False)
        tree.column("latest", width=620)
        tree.pack(fill="both", expand=True)
        self.library_tree = tree
        self.library_rows = {}
        for item in self.repository.library_companies():
            latest = item.get("latest_event") or {}
            iid = tree.insert(
                "",
                "end",
                values=(
                    "☑" if item["company_id"] in self.library_selected_ids else "☐",
                    f"{item['name']}  {item['ticker']}  {item.get('market', '')}",
                    self._date(item.get("last_queried_at")),
                    f"{item.get('document_count', 0)} / {item.get('event_count', 0)}",
                    latest.get("standardized_text", ""),
                ),
            )
            self.library_rows[iid] = item
        tree.bind("<Button-1>", lambda event: self._toggle_library_row(event))
        tree.bind("<Double-1>", lambda event: self._open_library_row(event))
        actions = tk.Frame(frame, bg=COLORS["bg"])
        actions.pack(fill="x", pady=(12, 0))
        self._button(actions, "查看报告", self._view_library_selected).pack(side="left")
        self._button(actions, "移除本地记录", self._remove_library_selected, primary=False).pack(
            side="right"
        )

    def _selected_library_items(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.library_rows.values()
            if str(item["company_id"]) in self.library_selected_ids
        ]

    def _toggle_library_row(self, event: tk.Event) -> str:
        iid = self.library_tree.identify_row(event.y)
        if not iid or iid not in self.library_rows:
            return "break"
        company_id = str(self.library_rows[iid]["company_id"])
        if company_id in self.library_selected_ids:
            self.library_selected_ids.remove(company_id)
        else:
            self.library_selected_ids.add(company_id)
        values = list(self.library_tree.item(iid, "values"))
        values[0] = "☑" if company_id in self.library_selected_ids else "☐"
        self.library_tree.item(iid, values=values)
        return "break"

    def _open_library_row(self, event: tk.Event) -> str:
        iid = self.library_tree.identify_row(event.y)
        if iid and iid in self.library_rows:
            self.show_report(str(self.library_rows[iid]["company_id"]))
        return "break"

    def _view_library_selected(self) -> None:
        items = self._selected_library_items()
        if items:
            self.show_report(str(items[0]["company_id"]))

    def _update_library_selected(self) -> None:
        items = self._selected_library_items()
        if len(items) != 1:
            messagebox.showinfo(
                "请选择一家公司",
                "更新时请选择一家公司；多家公司请使用“更新并比较”。",
                parent=self.root,
            )
            return
        item = items[0]
        self._launch_query(str(item["ticker"]), str(item["company_id"]))

    def _compare_library_selected(self) -> None:
        items = self._selected_library_items()
        if not 2 <= len(items) <= 6:
            messagebox.showinfo("请选择公司", "请选择2—6家公司进行比较。", parent=self.root)
            return
        self.show_compare([str(item["company_id"]) for item in items])

    def _remove_library_selected(self) -> None:
        items = self._selected_library_items()
        if not items:
            return
        names = "、".join(item["name"] for item in items)
        if not messagebox.askyesno(
            "确认移除", f"确认移除 {names} 的本地查询记录？", parent=self.root
        ):
            return
        for item in items:
            self.repository.delete_company(str(item["company_id"]))
        self.library_selected_ids.clear()
        self.show_library(reset=True)

    def show_batch(self, reset: bool = False) -> None:
        if reset:
            self.batch_view = "new"
            self.current_batch_id = ""
        elif self.current_batch_id:
            batch = self.repository.batch_query(self.current_batch_id)
            if batch.get("status") in {"pending", "running"}:
                self._render_batch_progress(batch)
                return
        elif self.batch_view == "detail" and self.last_batch_id:
            self._render_batch_detail(self.repository.batch_query(self.last_batch_id))
            return
        elif self.batch_view == "history":
            self._show_batch_history()
            return
        self.batch_view = "new"
        self._set_page("batch", "批量研究")
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        body = scroll.body
        body.grid_columnconfigure(0, weight=1)
        toolbar = tk.Frame(body, bg=COLORS["bg"])
        toolbar.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 10))
        tk.Label(
            toolbar,
            text="先预览并勾选公司，再逐家更新完整文本并自动比较",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
        ).pack(side="left")
        self._button(toolbar, "历史批次", self._show_batch_history, primary=False).pack(
            side="right"
        )

        form = self._surface(body, padx=22, pady=20)
        form.grid(row=1, column=0, sticky="ew", padx=28, pady=10)
        form.grid_columnconfigure(1, weight=1)
        tk.Label(form, text="批次名称", bg=COLORS["surface"], fg=COLORS["text"]).grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=6
        )
        self.batch_name_entry = tk.Entry(form, relief="solid", bd=1)
        self.batch_name_entry.grid(row=0, column=1, sticky="ew", pady=6)
        tk.Label(
            form, text="公司代码、简称或领域关键词", bg=COLORS["surface"], fg=COLORS["text"]
        ).grid(row=1, column=0, sticky="nw", padx=(0, 12), pady=6)
        self.batch_input = tk.Text(form, height=4, relief="solid", bd=1, wrap="word")
        self.batch_input.grid(row=1, column=1, sticky="ew", pady=6)
        self.batch_input.insert("1.0", "")
        tk.Label(
            form,
            text="可粘贴多行代码/简称；也可输入半导体、医药、光伏等关键词。系统会从本地官方目录缓存和官方公司搜索结果中生成候选。",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            wraplength=830,
            justify="left",
        ).grid(row=2, column=1, sticky="w", pady=(0, 8))
        filters = tk.Frame(form, bg=COLORS["surface"])
        filters.grid(row=3, column=1, sticky="w", pady=6)
        self.batch_market_vars = {
            "sse": tk.BooleanVar(value=True),
            "szse": tk.BooleanVar(value=True),
            "bse": tk.BooleanVar(value=True),
        }
        for market, label in (("sse", "沪市"), ("szse", "深市"), ("bse", "北交所")):
            tk.Checkbutton(
                filters,
                text=label,
                variable=self.batch_market_vars[market],
                bg=COLORS["surface"],
                activebackground=COLORS["surface"],
            ).pack(side="left", padx=(0, 12))
        self.batch_local_only_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            filters,
            text="仅从已查公司筛选",
            variable=self.batch_local_only_var,
            bg=COLORS["surface"],
            activebackground=COLORS["surface"],
        ).pack(side="left", padx=(12, 0))
        tk.Label(filters, text="候选上限", bg=COLORS["surface"], fg=COLORS["muted"]).pack(
            side="left", padx=(22, 6)
        )
        self.batch_limit_var = tk.IntVar(value=12)
        tk.Spinbox(filters, from_=2, to=12, width=4, textvariable=self.batch_limit_var).pack(
            side="left"
        )
        self._button(form, "生成候选公司", self._preview_batch_candidates).grid(
            row=4, column=1, sticky="e", pady=(10, 0)
        )

        panel = self._surface(body, padx=14, pady=14)
        panel.grid(row=2, column=0, sticky="ew", padx=28, pady=(10, 30))
        self.batch_preview_status = tk.Label(
            panel,
            text="输入名单或筛选条件后生成候选，候选不会自动开始查询。",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
        )
        self.batch_preview_status.pack(anchor="w", pady=(0, 10))
        self.batch_tree = self._tree(
            panel,
            ("checked", "company", "market", "industry", "local"),
            ("选择", "公司", "市场", "领域/行业", "本地状态"),
            height=10,
            selectmode="none",
        )
        self.batch_tree.column("checked", width=58, stretch=False, anchor="center")
        self.batch_tree.column("company", width=260)
        self.batch_tree.column("market", width=90, stretch=False)
        self.batch_tree.column("industry", width=280)
        self.batch_tree.column("local", width=120, stretch=False)
        self.batch_tree.pack(fill="x")
        self.batch_tree.bind("<Button-1>", self._toggle_batch_candidate)
        action = tk.Frame(panel, bg=COLORS["surface"])
        action.pack(fill="x", pady=(12, 0))
        self.batch_selected_label = tk.Label(
            action, text="已选 0 家", bg=COLORS["surface"], fg=COLORS["muted"]
        )
        self.batch_selected_label.pack(side="left")
        self._button(action, "开始批量更新并比较", self._start_batch_query).pack(side="right")

    @staticmethod
    def _split_batch_terms(text: str) -> list[str]:
        import re

        return [item.strip() for item in re.split(r"[\s,，;；]+", text) if item.strip()]

    def _preview_batch_candidates(self) -> None:
        terms = self._split_batch_terms(self.batch_input.get("1.0", "end"))
        markets = [key for key, value in self.batch_market_vars.items() if value.get()]
        if not terms and not self.batch_local_only_var.get():
            messagebox.showinfo(
                "请输入条件", "请输入公司代码、简称或领域关键词。", parent=self.root
            )
            return
        self.batch_preview_status.configure(text="正在识别候选公司，请稍候……")
        limit = max(2, min(int(self.batch_limit_var.get()), 12))
        local_only = bool(self.batch_local_only_var.get())

        def worker() -> None:
            try:
                companies = self.coordinator.resolve_batch_candidates(
                    terms,
                    markets=markets,
                    local_only=local_only,
                    limit=limit,
                )
            except Exception as error:
                message = str(error)
                self.root.after(
                    0,
                    lambda message=message: self.batch_preview_status.configure(
                        text=f"候选识别失败：{message}"
                    ),
                )
                return
            self.root.after(0, lambda: self._render_batch_candidates(companies))

        threading.Thread(target=worker, daemon=True, name="batch-candidate-preview").start()

    def _render_batch_candidates(self, companies: list[Any]) -> None:
        for iid in self.batch_tree.get_children():
            self.batch_tree.delete(iid)
        local_ids = {item["company_id"] for item in self.repository.library_companies()}
        self.batch_candidates = {}
        self.batch_selected_ids = {company.company_id for company in companies}
        for company in companies:
            iid = self.batch_tree.insert(
                "",
                "end",
                values=(
                    "☑",
                    f"{company.name}（{company.ticker}）",
                    company.market,
                    company.industry or "待查询后补充",
                    "已查询" if company.company_id in local_ids else "未查询",
                ),
            )
            self.batch_candidates[iid] = company
        self.batch_preview_status.configure(
            text=f"找到 {len(companies)} 家候选；请取消不需要的公司，至少保留 2 家。"
            if companies
            else "未找到候选，请改用股票代码或更具体的公司简称。"
        )
        self.batch_selected_label.configure(text=f"已选 {len(self.batch_selected_ids)} 家")

    def _toggle_batch_candidate(self, event: tk.Event) -> str:
        iid = self.batch_tree.identify_row(event.y)
        if not iid or iid not in self.batch_candidates:
            return "break"
        company_id = str(self.batch_candidates[iid].company_id)
        if company_id in self.batch_selected_ids:
            self.batch_selected_ids.remove(company_id)
        else:
            self.batch_selected_ids.add(company_id)
        values = list(self.batch_tree.item(iid, "values"))
        values[0] = "☑" if company_id in self.batch_selected_ids else "☐"
        self.batch_tree.item(iid, values=values)
        self.batch_selected_label.configure(text=f"已选 {len(self.batch_selected_ids)} 家")
        return "break"

    def _start_batch_query(self) -> None:
        companies = [
            company
            for company in self.batch_candidates.values()
            if company.company_id in self.batch_selected_ids
        ]
        if not 2 <= len(companies) <= 12:
            messagebox.showinfo("请选择公司", "批量研究请勾选 2—12 家公司。", parent=self.root)
            return
        terms = self._split_batch_terms(self.batch_input.get("1.0", "end"))
        criteria = {
            "terms": terms,
            "markets": [key for key, value in self.batch_market_vars.items() if value.get()],
            "local_only": bool(self.batch_local_only_var.get()),
        }
        name = self.batch_name_entry.get().strip()
        self._render_batch_progress(
            {"name": name or "批量研究", "stage": "preparing", "progress": 0, "members": []}
        )

        def worker() -> None:
            try:
                batch = self.coordinator.start_batch_query(
                    companies,
                    name=name,
                    query_mode="criteria" if len(terms) <= 3 else "list",
                    criteria=criteria,
                    window_days=180,
                )
            except Exception as error:
                message = str(error)
                self.root.after(0, lambda message=message: self._batch_failed(message))
                return
            self.root.after(0, lambda: self._begin_batch_poll(str(batch["id"])))

        threading.Thread(target=worker, daemon=True, name="desktop-batch-start").start()

    def _render_batch_progress(self, batch: dict[str, Any]) -> None:
        self.batch_view = "progress"
        self._set_page("batch", "批量研究进行中")
        panel = self._surface(self.content, padx=34, pady=30)
        panel.pack(fill="x", padx=36, pady=36)
        tk.Label(
            panel,
            text=batch.get("name") or "批量研究",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 19, "bold"),
        ).pack(anchor="w")
        self.batch_status_label = tk.Label(
            panel, text="正在准备任务", bg=COLORS["surface"], fg=COLORS["muted"]
        )
        self.batch_status_label.pack(anchor="w", pady=(8, 18))
        self.batch_progress = ttk.Progressbar(
            panel,
            maximum=100,
            value=float(batch.get("progress") or 0) * 100,
            style="Blue.Horizontal.TProgressbar",
        )
        self.batch_progress.pack(fill="x")
        tk.Label(
            self.content,
            text="可以切换到其他栏目；批量查询会继续在后台运行，顶部任务条会持续显示进度。",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
        ).pack(anchor="w", padx=40)
        self._update_batch_widgets(batch)

    def _update_batch_widgets(self, batch: dict[str, Any]) -> None:
        if not hasattr(self, "batch_progress") or not self.batch_progress.winfo_exists():
            return
        self.batch_progress.configure(value=float(batch.get("progress") or 0) * 100)
        self.batch_status_label.configure(
            text=(
                f"{STAGE_LABELS.get(batch.get('stage', ''), batch.get('stage', ''))} · "
                f"完成 {batch.get('completed_companies', 0)}/"
                f"{batch.get('total_companies', 0)} 家 · "
                f"失败 {batch.get('failed_companies', 0)} 家"
            )
        )

    def _begin_batch_poll(self, batch_id: str) -> None:
        self.current_batch_id = batch_id
        self.last_batch_id = batch_id
        self.batch_view = "progress"
        self.root.after(500, self._poll_batch)

    def _poll_batch(self) -> None:
        if not self.current_batch_id:
            return
        batch = self.repository.batch_query(self.current_batch_id)
        self._update_batch_widgets(batch)
        if batch["status"] == "completed":
            self.current_batch_id = ""
            self.last_batch_id = str(batch["id"])
            self.batch_view = "detail"
            if self.current_page == "batch":
                self._render_batch_detail(batch)
        elif batch["status"] == "failed":
            self.current_batch_id = ""
            self.batch_view = "detail"
            if self.current_page == "batch":
                self._batch_failed(str(batch.get("error") or "批量研究失败"))
        else:
            self.root.after(800, self._poll_batch)

    def _batch_failed(self, error: str) -> None:
        messagebox.showerror("批量研究失败", error, parent=self.root)
        self._show_batch_history()

    def _show_batch_history(self) -> None:
        self.batch_view = "history"
        self._set_page("batch", "历史批量研究")
        frame = tk.Frame(self.content, bg=COLORS["bg"], padx=28, pady=26)
        frame.pack(fill="both", expand=True)
        toolbar = tk.Frame(frame, bg=COLORS["bg"])
        toolbar.pack(fill="x", pady=(0, 12))
        tk.Label(
            toolbar,
            text="每次批量成员、失败信息和比较结果均保存在本地",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
        ).pack(side="left")
        self._button(toolbar, "新建批量研究", lambda: self.show_batch(reset=True)).pack(
            side="right"
        )
        panel = self._surface(frame, padx=14, pady=14)
        panel.pack(fill="both", expand=True)
        tree = self._tree(
            panel,
            ("name", "created", "status", "members", "failed"),
            ("批次", "创建时间", "状态", "公司数", "失败"),
            height=18,
        )
        tree.pack(fill="both", expand=True)
        self.batch_history_rows: dict[str, dict[str, Any]] = {}
        for batch in self.repository.recent_batch_queries(50):
            iid = tree.insert(
                "",
                "end",
                values=(
                    batch.get("name", ""),
                    self._date(batch.get("created_at")),
                    batch.get("status", ""),
                    batch.get("total_companies", 0),
                    batch.get("failed_companies", 0),
                ),
            )
            self.batch_history_rows[iid] = batch
        tree.bind("<Double-1>", lambda _event: self._open_batch_history(tree))
        self._button(panel, "查看批次", lambda: self._open_batch_history(tree)).pack(
            anchor="e", pady=(12, 0)
        )

    def _open_batch_history(self, tree: ttk.Treeview) -> None:
        selected = tree.selection()
        if not selected:
            return
        batch = self.batch_history_rows[selected[0]]
        self.last_batch_id = str(batch["id"])
        self._render_batch_detail(self.repository.batch_query(self.last_batch_id))

    def _render_batch_detail(self, batch: dict[str, Any]) -> None:
        self.batch_view = "detail"
        self.last_batch_id = str(batch.get("id") or self.last_batch_id)
        self._set_page("batch", "批量研究回看")
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        body = scroll.body
        body.grid_columnconfigure(0, weight=1)
        header = self._surface(body, padx=24, pady=20)
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 12))
        row = tk.Frame(header, bg=COLORS["surface"])
        row.pack(fill="x")
        tk.Label(
            row,
            text=batch.get("name") or "批量研究",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(side="left")
        self._button(row, "返回历史", self._show_batch_history, primary=False).pack(side="right")
        comparison_id = str(batch.get("comparison_id") or "")
        if comparison_id:
            self._button(
                row,
                "查看完整比较",
                lambda cid=comparison_id: self._open_saved_comparison(cid),
            ).pack(side="right", padx=8)
        tk.Label(
            header,
            text=(
                f"状态：{batch.get('status', '')} · 完成 {batch.get('completed_companies', 0)}/"
                f"{batch.get('total_companies', 0)} 家 · 失败 {batch.get('failed_companies', 0)} 家"
            ),
            bg=COLORS["surface"],
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(8, 0))
        result = batch.get("result") or {}
        if result.get("deepseek_summary"):
            tk.Label(
                header,
                text=result["deepseek_summary"],
                bg=COLORS["surface"],
                fg=COLORS["text"],
                wraplength=1000,
                justify="left",
            ).pack(anchor="w", pady=(14, 0))
        members_panel = self._surface(body, padx=16, pady=16)
        members_panel.grid(row=1, column=0, sticky="ew", padx=28, pady=(0, 28))
        tree = self._tree(
            members_panel,
            ("company", "status", "asof", "error"),
            ("公司", "更新状态", "实际数据截止", "错误"),
            height=max(5, len(batch.get("members") or [])),
        )
        tree.column("company", width=260)
        tree.column("status", width=110, stretch=False)
        tree.column("asof", width=150, stretch=False)
        tree.column("error", width=560)
        tree.pack(fill="x")
        for member in batch.get("members") or []:
            tree.insert(
                "",
                "end",
                values=(
                    f"{member.get('name')}（{member.get('ticker')}）",
                    member.get("status", ""),
                    self._date(member.get("data_as_of")),
                    member.get("error", ""),
                ),
            )

    def _open_saved_comparison(self, comparison_id: str) -> None:
        self.last_comparison_id = comparison_id
        self.compare_view = "result"
        self._render_comparison_result(self.repository.comparison(comparison_id))

    def show_compare(self, selected_ids: list[str] | None = None, reset: bool = False) -> None:
        if selected_ids is not None:
            self.compare_selected_ids = set(selected_ids)
            self.compare_view = "selection"
        elif reset:
            self.compare_view = "selection"
            self.current_comparison_id = ""
        elif self.current_comparison_id:
            comparison = self.repository.comparison(self.current_comparison_id)
            if comparison.get("status") in {"pending", "running"}:
                self._render_comparison_starting(comparison.get("members") or [])
                return
        elif self.compare_view == "result" and self.last_comparison_id:
            self._render_comparison_result(self.repository.comparison(self.last_comparison_id))
            return
        self._set_page("compare", "公司比较")
        frame = tk.Frame(self.content, bg=COLORS["bg"], padx=28, pady=26)
        frame.pack(fill="both", expand=True)
        intro = self._surface(frame, padx=22, pady=18)
        intro.pack(fill="x", pady=(0, 14))
        tk.Label(
            intro,
            text="勾选 2—12 家已查公司",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(anchor="w")
        tk.Label(
            intro,
            text="比较前会逐家公司更新公告与行情，再统一事件窗口并生成横向摘要。",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(5, 0))
        panel = self._surface(frame, padx=14, pady=14)
        panel.pack(fill="both", expand=True)
        tree = self._tree(
            panel,
            ("checked", "company", "market", "documents", "events", "asof"),
            ("选择", "公司", "市场", "已分析文档", "事件", "数据截止"),
            height=14,
            selectmode="none",
        )
        tree.pack(fill="both", expand=True)
        tree.column("checked", width=58, stretch=False, anchor="center")
        tree.column("company", width=260)
        tree.column("market", width=100)
        tree.column("documents", width=110)
        tree.column("events", width=90)
        tree.column("asof", width=180)
        self.compare_tree = tree
        self.compare_rows = {}
        for item in self.repository.library_companies():
            iid = tree.insert(
                "",
                "end",
                values=(
                    "☑" if item["company_id"] in self.compare_selected_ids else "☐",
                    f"{item['name']}（{item['ticker']}）",
                    item.get("market", ""),
                    item.get("document_count", 0),
                    item.get("event_count", 0),
                    self._date(item.get("data_as_of")),
                ),
            )
            self.compare_rows[iid] = item
        tree.bind("<Button-1>", self._toggle_compare_row)
        actions = tk.Frame(frame, bg=COLORS["bg"])
        actions.pack(fill="x", pady=(12, 0))
        self.compare_count_label = tk.Label(
            actions,
            text=f"已选 {len(self.compare_selected_ids)} 家",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
        )
        self.compare_count_label.pack(side="left")
        self._button(actions, "清空", self._clear_compare_selection, primary=False).pack(
            side="left", padx=8
        )
        self._button(actions, "更新并比较", self._start_comparison).pack(side="right")

    def _toggle_compare_row(self, event: tk.Event) -> str:
        iid = self.compare_tree.identify_row(event.y)
        if not iid or iid not in self.compare_rows:
            return "break"
        company_id = str(self.compare_rows[iid]["company_id"])
        if company_id in self.compare_selected_ids:
            self.compare_selected_ids.remove(company_id)
        elif len(self.compare_selected_ids) < 12:
            self.compare_selected_ids.add(company_id)
        else:
            messagebox.showinfo("选择已满", "一次最多比较 12 家公司。", parent=self.root)
        values = list(self.compare_tree.item(iid, "values"))
        values[0] = "☑" if company_id in self.compare_selected_ids else "☐"
        self.compare_tree.item(iid, values=values)
        self.compare_count_label.configure(text=f"已选 {len(self.compare_selected_ids)} 家")
        return "break"

    def _clear_compare_selection(self) -> None:
        self.compare_selected_ids.clear()
        self.show_compare(reset=True)

    def _start_comparison(self) -> None:
        selected = [
            item
            for item in self.compare_rows.values()
            if str(item["company_id"]) in self.compare_selected_ids
        ]
        if not 2 <= len(selected) <= 12:
            messagebox.showinfo("请选择公司", "请勾选 2—12 家公司。", parent=self.root)
            return
        company_ids = [str(item["company_id"]) for item in selected]
        self._render_comparison_starting(selected)

        def worker() -> None:
            try:
                comparison = self.coordinator.start_comparison(company_ids, 180)
            except Exception as error:
                self.root.after(0, lambda error=error: self._comparison_failed(str(error)))
                return
            self.root.after(0, lambda: self._begin_comparison_poll(str(comparison["id"])))

        threading.Thread(target=worker, daemon=True, name="desktop-comparison-start").start()

    def _render_comparison_starting(self, selected: list[dict[str, Any]]) -> None:
        self.compare_view = "progress"
        self._set_page("compare", "正在更新并比较")
        panel = self._surface(self.content, padx=34, pady=30)
        panel.pack(fill="x", padx=36, pady=36)
        tk.Label(
            panel,
            text="正在准备比较",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 19, "bold"),
        ).pack(anchor="w")
        tk.Label(
            panel,
            text="、".join(item["name"] for item in selected),
            bg=COLORS["surface"],
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(6, 18))
        self.comparison_progress = ttk.Progressbar(
            panel, maximum=100, value=1, style="Blue.Horizontal.TProgressbar"
        )
        self.comparison_progress.pack(fill="x")
        self.comparison_status = tk.Label(
            panel, text="正在创建比较任务", bg=COLORS["surface"], fg=COLORS["muted"]
        )
        self.comparison_status.pack(anchor="w", pady=(12, 0))

    def _begin_comparison_poll(self, comparison_id: str) -> None:
        self.current_comparison_id = comparison_id
        self.last_comparison_id = comparison_id
        self.compare_view = "progress"
        self.root.after(500, self._poll_comparison)

    def _poll_comparison(self) -> None:
        if not self.current_comparison_id:
            return
        comparison = self.repository.comparison(self.current_comparison_id)
        if hasattr(self, "comparison_progress") and self.comparison_progress.winfo_exists():
            self.comparison_progress.configure(value=float(comparison.get("progress") or 0) * 100)
            self.comparison_status.configure(
                text=STAGE_LABELS.get(comparison.get("stage", ""), comparison.get("stage", ""))
            )
        if comparison["status"] == "completed":
            self.current_comparison_id = ""
            self.last_comparison_id = str(comparison["id"])
            self.compare_view = "result"
            if self.current_page == "compare":
                self._render_comparison_result(comparison)
        elif comparison["status"] == "failed":
            self.current_comparison_id = ""
            self.compare_view = "selection"
            if self.current_page == "compare":
                self._comparison_failed(str(comparison.get("error") or "比较失败"))
        else:
            self.root.after(700, self._poll_comparison)

    def _comparison_failed(self, error: str) -> None:
        messagebox.showerror("比较失败", error, parent=self.root)
        self.show_compare(reset=True)

    def _render_comparison_result(self, comparison: dict[str, Any]) -> None:
        self.compare_view = "result"
        self.last_comparison_id = str(comparison.get("id") or self.last_comparison_id)
        self._set_page("compare", "公司比较结果")
        result = comparison.get("result") or {}
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        body = scroll.body
        body.grid_columnconfigure(0, weight=1)
        summary_panel = self._surface(body, padx=24, pady=20)
        summary_panel.grid(row=0, column=0, sticky="ew", padx=26, pady=(24, 12))
        title_row = tk.Frame(summary_panel, bg=COLORS["surface"])
        title_row.pack(fill="x")
        tk.Label(
            title_row,
            text="横向研究摘要",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 17, "bold"),
        ).pack(side="left")
        self._button(
            title_row, "导出Word", lambda: self._export_comparison(comparison), primary=False
        ).pack(side="right")
        self._button(
            title_row, "新建比较", lambda: self.show_compare(reset=True), primary=False
        ).pack(side="right", padx=8)
        if result.get("partial_update_failure"):
            tk.Label(
                summary_panel,
                text="部分公司更新失败，表格已保留各公司的实际数据截止日期。",
                bg="#FFF4E6",
                fg=COLORS["warning"],
                padx=10,
                pady=8,
            ).pack(fill="x", pady=(12, 4))
        tk.Label(
            summary_panel,
            text=result.get("deepseek_summary") or "暂无摘要",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            wraplength=1020,
            justify="left",
            font=("Microsoft YaHei UI", 11),
        ).pack(anchor="w", pady=(14, 0))
        table_panel = self._surface(body, padx=18, pady=18)
        table_panel.grid(row=1, column=0, sticky="ew", padx=26, pady=12)
        self._section_title(
            table_panel, "公司横向总表", "文本事件、相对市场表现与情景预测放在同一视图"
        ).pack(fill="x", pady=(0, 10))
        columns = (
            "company",
            "asof",
            "events",
            "negative",
            "pe",
            "r20",
            "excess",
            "regime",
            "up",
        )
        tree = self._tree(
            table_panel,
            columns,
            (
                "公司",
                "数据截止",
                "事件",
                "负向",
                "动态PE",
                "20日收益",
                "相对基准",
                "20日情景",
                "上行概率",
            ),
            height=max(4, len(result.get("companies") or [])),
        )
        tree.pack(fill="x")
        tree.column("company", width=190)
        tree.column("asof", width=100)
        for company in result.get("companies") or []:
            market = company.get("price_analysis") or {}
            forecast = market.get("forecast_20d") or {}
            tree.insert(
                "",
                "end",
                values=(
                    f"{company.get('name')} {company.get('ticker')}",
                    self._date(company.get("data_as_of")),
                    company.get("event_count", 0),
                    company.get("negative", 0),
                    self._multiple((market.get("valuation") or {}).get("pe_dynamic")),
                    self._pct((market.get("returns") or {}).get("20d")),
                    self._pct((market.get("benchmark") or {}).get("excess_20d")),
                    forecast.get("regime", "暂无"),
                    self._prob((forecast.get("probabilities") or {}).get("up")),
                ),
            )
        heat_panel = self._surface(body, padx=18, pady=18)
        heat_panel.grid(row=2, column=0, sticky="ew", padx=26, pady=12)
        self._section_title(
            heat_panel, "事件热力图", "单元格格式：事件数 / 负向数 / 累计关注度"
        ).pack(fill="x", pady=(0, 10))
        companies = result.get("companies") or []
        heat_columns = ("dimension", *[item.get("ticker", "") for item in companies])
        headings = ("比较维度", *[item.get("name", "") for item in companies])
        heat_tree = self._tree(heat_panel, heat_columns, headings, height=10)
        heat_tree.column("dimension", width=220)
        heat_tree.pack(fill="x")
        for row in result.get("heatmap") or []:
            values: list[str] = [row.get("dimension", "")]
            for company in companies:
                cell = row.get(company.get("ticker"), {})
                count = cell.get("count", 0)
                negative = cell.get("negative", 0)
                attention = float(cell.get("attention") or 0)
                values.append(f"{count} / {negative} / {attention:.2f}")
            heat_tree.insert("", "end", values=values)
        changes_panel = self._surface(body, padx=20, pady=18)
        changes_panel.grid(row=3, column=0, sticky="ew", padx=26, pady=(12, 30))
        self._section_title(changes_panel, "各公司最重要变化", "所有结论仍可回到原文证据").pack(
            fill="x", pady=(0, 8)
        )
        for company in companies:
            tk.Label(
                changes_panel,
                text=f"{company.get('name')}（{company.get('ticker')}）",
                bg=COLORS["surface"],
                fg=COLORS["text"],
                font=("Microsoft YaHei UI", 11, "bold"),
            ).pack(anchor="w", pady=(10, 4))
            for event in company.get("top_events") or []:
                row = tk.Frame(changes_panel, bg=COLORS["surface"])
                row.pack(fill="x", pady=2)
                tk.Label(
                    row,
                    text=f"• {event.get('standardized_text', '')}",
                    bg=COLORS["surface"],
                    fg=COLORS["text"],
                    wraplength=900,
                    justify="left",
                ).pack(side="left", anchor="w")
                if event.get("url"):
                    event_url = str(event["url"])
                    self._link_button(row, "原文", lambda url=event_url: webbrowser.open(url)).pack(
                        side="right"
                    )

    def show_theses(self, selected_company_id: str | None = None) -> None:
        self._set_page("theses", "研究观点")
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        body = scroll.body
        body.grid_columnconfigure(0, weight=1)
        companies = self.repository.library_companies()
        if not companies:
            panel = self._surface(body, padx=28, pady=28)
            panel.grid(row=0, column=0, sticky="ew", padx=28, pady=28)
            self._section_title(panel, "尚无本地公司", "先查询公司，再建立需要持续验证的观点").pack(
                fill="x"
            )
            self._button(panel, "前往查询", self.show_query).pack(anchor="w", pady=(16, 0))
            return

        form = self._surface(body, padx=24, pady=20)
        form.grid(row=0, column=0, sticky="ew", padx=28, pady=(28, 12))
        self._section_title(
            form,
            "建立可验证的研究观点",
            "选择影响维度和失效条件；历史与新增事件会自动成为支持或反对证据",
        ).pack(fill="x", pady=(0, 14))
        grid = tk.Frame(form, bg=COLORS["surface"])
        grid.pack(fill="x")
        grid.grid_columnconfigure(1, weight=1)
        grid.grid_columnconfigure(3, weight=1)

        company_values = [f"{item['name']}（{item['ticker']}）" for item in companies]
        self.thesis_company_map = {
            value: str(item["company_id"])
            for value, item in zip(company_values, companies, strict=True)
        }
        default_company = next(
            (
                value
                for value, item in zip(company_values, companies, strict=True)
                if str(item["company_id"]) == selected_company_id
            ),
            company_values[0],
        )
        self.thesis_company_var = tk.StringVar(value=default_company)
        self.thesis_direction_var = tk.StringVar(value="正向观点")
        impact_values = [f"{label}（{key}）" for key, label in IMPACT_LABELS.items()]
        self.thesis_impact_map = {
            value: key for value, key in zip(impact_values, IMPACT_LABELS, strict=True)
        }
        self.thesis_impact_var = tk.StringVar(value=impact_values[0])

        tk.Label(grid, text="公司", bg=COLORS["surface"], fg=COLORS["muted"]).grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=5
        )
        ttk.Combobox(
            grid,
            values=company_values,
            textvariable=self.thesis_company_var,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=(0, 18), pady=5)
        tk.Label(grid, text="观点方向", bg=COLORS["surface"], fg=COLORS["muted"]).grid(
            row=0, column=2, sticky="w", padx=(0, 8), pady=5
        )
        ttk.Combobox(
            grid,
            values=("正向观点", "谨慎观点"),
            textvariable=self.thesis_direction_var,
            state="readonly",
        ).grid(row=0, column=3, sticky="ew", pady=5)

        tk.Label(grid, text="观点标题", bg=COLORS["surface"], fg=COLORS["muted"]).grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=5
        )
        self.thesis_title_entry = tk.Entry(grid, relief="solid", bd=1)
        self.thesis_title_entry.grid(row=1, column=1, columnspan=3, sticky="ew", pady=5, ipady=6)
        tk.Label(grid, text="验证维度", bg=COLORS["surface"], fg=COLORS["muted"]).grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=5
        )
        ttk.Combobox(
            grid,
            values=impact_values,
            textvariable=self.thesis_impact_var,
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=(0, 18), pady=5)
        tk.Label(grid, text="失效条件", bg=COLORS["surface"], fg=COLORS["muted"]).grid(
            row=2, column=2, sticky="w", padx=(0, 8), pady=5
        )
        self.thesis_invalidation_entry = tk.Entry(grid, relief="solid", bd=1)
        self.thesis_invalidation_entry.grid(row=2, column=3, sticky="ew", pady=5, ipady=6)
        tk.Label(grid, text="观点说明", bg=COLORS["surface"], fg=COLORS["muted"]).grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=5
        )
        self.thesis_description_entry = tk.Entry(grid, relief="solid", bd=1)
        self.thesis_description_entry.grid(
            row=3, column=1, columnspan=3, sticky="ew", pady=5, ipady=6
        )
        action = tk.Frame(form, bg=COLORS["surface"])
        action.pack(fill="x", pady=(14, 0))
        self._button(action, "创建观点并匹配证据", self._create_thesis).pack(side="right")
        self._button(
            action,
            "重新计算全部证据",
            self._refresh_all_theses,
            primary=False,
        ).pack(side="right", padx=(0, 8))

        panel = self._surface(body, padx=20, pady=18)
        panel.grid(row=1, column=0, sticky="ew", padx=28, pady=(12, 28))
        self._section_title(
            panel,
            "观点状态",
            "双击查看每项支持与反对证据，并可人工确认、失效或恢复自动判断",
        ).pack(fill="x", pady=(0, 12))
        columns = ("company", "state", "score", "support", "contradict", "title")
        tree = self._tree(
            panel,
            columns,
            ("公司", "状态", "证据分", "支持", "反对", "观点"),
            height=14,
        )
        tree.column("company", width=140, stretch=False)
        tree.column("state", width=100, stretch=False)
        tree.column("score", width=80, stretch=False)
        tree.column("support", width=65, stretch=False)
        tree.column("contradict", width=65, stretch=False)
        tree.column("title", width=620)
        tree.pack(fill="both", expand=True)
        self.thesis_rows = {}
        for thesis in self.repository.list_theses():
            iid = tree.insert(
                "",
                "end",
                values=(
                    thesis.get("company_name", ""),
                    THESIS_STATES.get(str(thesis.get("state")), thesis.get("state", "")),
                    f"{float(thesis.get('evidence_score') or 0):+.2f}",
                    thesis.get("support_count", 0),
                    thesis.get("contradict_count", 0),
                    thesis.get("title", ""),
                ),
            )
            self.thesis_rows[iid] = thesis
        tree.bind("<Double-1>", lambda _event: self._open_selected_thesis(tree))

    def _create_thesis(self) -> None:
        company_id = self.thesis_company_map.get(self.thesis_company_var.get(), "")
        title = self.thesis_title_entry.get().strip()
        impact = self.thesis_impact_map.get(self.thesis_impact_var.get(), "")
        try:
            thesis = self.repository.create_thesis(
                company_id,
                title,
                description=self.thesis_description_entry.get().strip(),
                thesis_direction=1 if self.thesis_direction_var.get() == "正向观点" else -1,
                impact_dimensions=[impact],
                invalidation_criteria=self.thesis_invalidation_entry.get().strip(),
            )
        except (KeyError, ValueError) as error:
            messagebox.showerror("无法创建观点", str(error), parent=self.root)
            return
        self.show_theses(company_id)
        self.root.after(100, lambda: self._thesis_dialog(str(thesis["id"])))

    def _refresh_all_theses(self) -> None:
        for company in self.repository.library_companies():
            self.repository.refresh_research_workspace(str(company["company_id"]))
        selected = self.thesis_company_map.get(self.thesis_company_var.get(), "")
        self.show_theses(selected)

    def _open_selected_thesis(self, tree: ttk.Treeview) -> None:
        selected = tree.selection()
        if selected and selected[0] in self.thesis_rows:
            self._thesis_dialog(str(self.thesis_rows[selected[0]]["id"]))

    def _thesis_dialog(self, thesis_id: str) -> None:
        try:
            thesis = self.repository.thesis(thesis_id)
        except KeyError:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("研究观点证据")
        dialog.geometry("980x690")
        dialog.transient(self.root)
        frame = tk.Frame(dialog, bg=COLORS["surface"], padx=24, pady=22)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text=thesis.get("title", ""),
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 15, "bold"),
            wraplength=900,
            justify="left",
        ).pack(anchor="w")
        state = THESIS_STATES.get(str(thesis.get("state")), thesis.get("state", ""))
        tk.Label(
            frame,
            text=(
                f"{thesis.get('company_name', '')} · {state} · "
                f"证据分 {float(thesis.get('evidence_score') or 0):+.2f} · "
                f"支持 {thesis.get('support_count', 0)} / 反对 "
                f"{thesis.get('contradict_count', 0)}"
            ),
            bg=COLORS["surface"],
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(6, 10))
        if thesis.get("description"):
            tk.Label(
                frame,
                text=f"观点说明：{thesis['description']}",
                bg=COLORS["surface"],
                fg=COLORS["text"],
                wraplength=900,
                justify="left",
            ).pack(anchor="w", pady=3)
        if thesis.get("invalidation_criteria"):
            tk.Label(
                frame,
                text=f"失效条件：{thesis['invalidation_criteria']}",
                bg=COLORS["surface"],
                fg=COLORS["negative"],
                wraplength=900,
                justify="left",
            ).pack(anchor="w", pady=3)
        tree = self._tree(
            frame,
            ("date", "stance", "relevance", "source", "evidence"),
            ("日期", "作用", "相关度", "来源", "标准化证据"),
            height=14,
        )
        tree.column("date", width=90, stretch=False)
        tree.column("stance", width=70, stretch=False)
        tree.column("relevance", width=70, stretch=False)
        tree.column("source", width=120, stretch=False)
        tree.column("evidence", width=570)
        tree.pack(fill="both", expand=True, pady=(16, 10))
        url_map: dict[str, str] = {}
        for evidence in thesis.get("evidence") or []:
            iid = tree.insert(
                "",
                "end",
                values=(
                    self._date(evidence.get("published_at")),
                    STANCE_LABELS.get(str(evidence.get("stance")), evidence.get("stance", "")),
                    f"{float(evidence.get('relevance') or 0):.0%}",
                    evidence.get("source_name", ""),
                    evidence.get("standardized_text", ""),
                ),
            )
            url_map[iid] = str(evidence.get("url") or "")
        tree.bind("<Double-1>", lambda _event: self._open_tree_url(tree, url_map))
        actions = tk.Frame(frame, bg=COLORS["surface"])
        actions.pack(fill="x")

        def update_state(value: str, archived: bool = False) -> None:
            self.repository.update_thesis(
                thesis_id,
                manual_state=value,
                archived=archived,
            )
            dialog.destroy()
            self.show_theses(str(thesis["company_id"]))

        self._button(
            actions, "人工确认", lambda: update_state("confirmed"), primary=False
        ).pack(side="left", padx=(0, 6))
        self._button(
            actions, "标记失效", lambda: update_state("invalidated"), primary=False
        ).pack(side="left", padx=6)
        self._button(actions, "恢复自动", lambda: update_state(""), primary=False).pack(
            side="left", padx=6
        )
        self._button(actions, "归档观点", lambda: update_state("", True), primary=False).pack(
            side="right"
        )

    def show_settings(self) -> None:
        self._set_page("settings", "设置")
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        body = scroll.body
        body.grid_columnconfigure(0, weight=1)
        api = self._surface(body, padx=24, pady=22)
        api.grid(row=0, column=0, sticky="ew", padx=28, pady=(28, 12))
        self._section_title(
            api, "DeepSeek V4", "密钥保存在Windows凭据管理器，不写入数据库或安装包"
        ).pack(fill="x", pady=(0, 14))
        self.api_status = tk.Label(
            api,
            text="已配置" if get_api_key() else "未配置",
            bg=COLORS["surface"],
            fg=COLORS["positive"] if get_api_key() else COLORS["warning"],
        )
        self.api_status.pack(anchor="w", pady=(0, 8))
        key_row = tk.Frame(api, bg=COLORS["surface"])
        key_row.pack(fill="x")
        self.api_key_entry = tk.Entry(
            key_row, show="●", font=("Segoe UI", 11), relief="solid", bd=1
        )
        self.api_key_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
        self._button(key_row, "保存并测试", self._save_and_test_key).pack(side="right")
        data = self._surface(body, padx=24, pady=22)
        data.grid(row=1, column=0, sticky="ew", padx=28, pady=12)
        self._section_title(
            data, "本地数据", "所有已查公司、文档、事件、价格和比较快照均保存在本机"
        ).pack(fill="x", pady=(0, 12))
        tk.Label(
            data,
            text=str(self.data_dir),
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            wraplength=900,
            justify="left",
        ).pack(anchor="w")
        self._button(data, "打开数据目录", self._open_data_dir, primary=False).pack(
            anchor="w", pady=(12, 0)
        )
        source = self._surface(body, padx=24, pady=22)
        source.grid(row=2, column=0, sticky="ew", padx=28, pady=(12, 28))
        self._section_title(
            source, "数据源与模型边界", "供应商可替换，输出保留来源和截止日期"
        ).pack(fill="x", pady=(0, 12))
        text = (
            "公告：上海证券交易所、深圳证券交易所、巨潮资讯（北交所当前主要使用巨潮）。\n"
            "行情：腾讯证券公开A股复权日线；用于收益、波动、相对指数和情景预测。\n"
            "预测：不输出目标价或买卖评级；展示历史分布、场景概率、回测命中率和方法限制。"
        )
        tk.Label(
            source,
            text=text,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            justify="left",
            wraplength=980,
        ).pack(anchor="w")

    def _save_and_test_key(self) -> None:
        key = self.api_key_entry.get().strip()
        if not key:
            messagebox.showinfo("请输入密钥", "请输入DeepSeek API密钥。", parent=self.root)
            return
        self.api_status.configure(text="正在测试连接……", fg=COLORS["primary"])

        def worker() -> None:
            try:
                save_api_key(key)
                result = DeepSeekClient(self.repository, api_key=key).test_connection()
                message = f"连接成功：{result.get('model', '')}"
                self.root.after(
                    0, lambda: self.api_status.configure(text=message, fg=COLORS["positive"])
                )
            except Exception as error:
                self.root.after(
                    0,
                    lambda error=error: self.api_status.configure(
                        text=f"连接失败：{error}", fg=COLORS["negative"]
                    ),
                )

        threading.Thread(target=worker, daemon=True, name="deepseek-test").start()

    def _open_selected_event(self, tree: ttk.Treeview) -> None:
        selected = tree.selection()
        if not selected:
            return
        event = self.report_event_rows.get(selected[0])
        if event:
            self._event_dialog(event)

    def _event_dialog(self, event: dict[str, Any]) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("事件证据")
        dialog.geometry("820x620")
        dialog.transient(self.root)
        frame = tk.Frame(dialog, bg=COLORS["surface"], padx=24, pady=22)
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text=event.get("standardized_text", ""),
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 14, "bold"),
            wraplength=750,
            justify="left",
        ).pack(anchor="w")
        meta = (
            f"研究关注度 {float(event.get('value_score') or 0):.3f}　·　"
            f"{event.get('source_name', '')}　·　"
            f"{self._date(event.get('published_at'))}"
        )
        tk.Label(frame, text=meta, bg=COLORS["surface"], fg=COLORS["muted"]).pack(
            anchor="w", pady=(7, 18)
        )
        tk.Label(
            frame,
            text="原文证据",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w")
        evidence = tk.Text(
            frame,
            height=13,
            wrap="word",
            bg=COLORS["surface_alt"],
            fg=COLORS["text"],
            relief="flat",
            padx=12,
            pady=10,
        )
        evidence.pack(fill="both", expand=True, pady=(6, 12))
        evidence.insert("1.0", event.get("evidence_text", ""))
        evidence.configure(state="disabled")
        location = (
            f"页码：{event.get('evidence_page') or '正文'}　"
            f"章节：{event.get('evidence_section') or '未标注'}"
        )
        tk.Label(frame, text=location, bg=COLORS["surface"], fg=COLORS["muted"]).pack(anchor="w")
        if event.get("importance_reason"):
            tk.Label(
                frame,
                text=f"重要性：{event['importance_reason']}",
                bg=COLORS["surface"],
                fg=COLORS["text"],
                wraplength=750,
                justify="left",
            ).pack(anchor="w", pady=(8, 0))
        buttons = tk.Frame(frame, bg=COLORS["surface"])
        buttons.pack(fill="x", pady=(16, 0))
        if event.get("url"):
            self._button(buttons, "打开官方原文", lambda: webbrowser.open(str(event["url"]))).pack(
                side="left"
            )
        self._button(buttons, "关闭", dialog.destroy, primary=False).pack(side="right")

    def _export_company(self, snapshot: dict[str, Any]) -> None:
        company = (snapshot.get("summary") or {}).get("company") or {}
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出公司报告",
            defaultextension=".docx",
            initialfile=f"{company.get('ticker', 'company')}-研究报告.docx",
            filetypes=[("Word文档", "*.docx")],
        )
        if filename:
            Path(filename).write_bytes(company_report(snapshot))
            messagebox.showinfo("导出完成", f"报告已保存至：\n{filename}", parent=self.root)

    def _export_comparison(self, comparison: dict[str, Any]) -> None:
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出比较报告",
            defaultextension=".docx",
            initialfile="上市公司比较报告.docx",
            filetypes=[("Word文档", "*.docx")],
        )
        if filename:
            Path(filename).write_bytes(comparison_report(comparison))
            messagebox.showinfo("导出完成", f"报告已保存至：\n{filename}", parent=self.root)

    def _open_tree_url(self, tree: ttk.Treeview, mapping: dict[str, str]) -> None:
        selected = tree.selection()
        if selected and mapping.get(selected[0]):
            webbrowser.open(mapping[selected[0]])

    def _open_data_dir(self) -> None:
        if os.name == "nt":
            os.startfile(self.data_dir)  # type: ignore[attr-defined]  # noqa: S606
        else:
            webbrowser.open(self.data_dir.as_uri())

    def close(self) -> None:
        try:
            self.coordinator.executor.shutdown(wait=False, cancel_futures=True)
        finally:
            self.root.destroy()

    @staticmethod
    def _surface(master: tk.Misc, **kwargs: Any) -> tk.Frame:
        return tk.Frame(
            master,
            bg=COLORS["surface"],
            highlightbackground=COLORS["line"],
            highlightthickness=1,
            **kwargs,
        )

    @staticmethod
    def _button(
        master: tk.Misc,
        text: str,
        command: Callable[[], None],
        *,
        primary: bool = True,
        bg: str | None = None,
    ) -> tk.Button:
        background = bg or (COLORS["primary"] if primary else COLORS["surface_alt"])
        foreground = "white" if primary or bg else COLORS["text"]
        return tk.Button(
            master,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground=COLORS["primary_dark"] if primary else "#E5EAF2",
            activeforeground="white" if primary else COLORS["text"],
            relief="flat",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            font=("Microsoft YaHei UI", 9, "bold"),
        )

    @staticmethod
    def _link_button(master: tk.Misc, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            master,
            text=text,
            command=command,
            bg=COLORS["surface"],
            fg=COLORS["primary"],
            activebackground=COLORS["surface"],
            activeforeground=COLORS["primary_dark"],
            relief="flat",
            bd=0,
            padx=0,
            pady=0,
            cursor="hand2",
            font=("Microsoft YaHei UI", 9, "underline"),
        )

    @staticmethod
    def _tree(
        master: tk.Misc,
        columns: tuple[str, ...],
        headings: tuple[str, ...],
        *,
        height: int = 10,
        selectmode: str = "browse",
        style: str = "App.Treeview",
    ) -> ttk.Treeview:
        tree = ttk.Treeview(
            master,
            columns=columns,
            show="headings",
            height=height,
            style=style,
            selectmode=selectmode,
        )
        for column, heading in zip(columns, headings, strict=True):
            tree.heading(column, text=heading)
            tree.column(column, width=120, anchor="w")
        return tree

    @staticmethod
    def _metric(master: tk.Misc, label: str, value: str, color: str) -> tk.Frame:
        card = tk.Frame(
            master,
            bg=COLORS["surface"],
            highlightbackground=COLORS["line"],
            highlightthickness=1,
            padx=14,
            pady=13,
        )
        tk.Label(
            card,
            text=label,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w")
        tk.Label(
            card, text=value, bg=COLORS["surface"], fg=color, font=("Segoe UI", 18, "bold")
        ).pack(anchor="w", pady=(3, 0))
        return card

    @staticmethod
    def _mini_metric(master: tk.Misc, label: str, value: str, color: str) -> tk.Frame:
        card = tk.Frame(master, bg=COLORS["surface_alt"], padx=10, pady=9)
        tk.Label(
            card,
            text=label,
            bg=COLORS["surface_alt"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w")
        tk.Label(
            card, text=value, bg=COLORS["surface_alt"], fg=color, font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(2, 0))
        return card

    @staticmethod
    def _section_title(master: tk.Misc, title: str, subtitle: str = "") -> tk.Frame:
        frame = tk.Frame(master, bg=COLORS["surface"])
        tk.Label(
            frame,
            text=title,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 13, "bold"),
        ).pack(side="left")
        if subtitle:
            tk.Label(
                frame,
                text=subtitle,
                bg=COLORS["surface"],
                fg=COLORS["muted"],
                font=("Microsoft YaHei UI", 8),
            ).pack(side="right")
        return frame

    @staticmethod
    def _date(value: Any) -> str:
        if not value:
            return "暂无"
        return str(value)[:10]

    @staticmethod
    def _pct(value: Any) -> str:
        try:
            return f"{float(value):+.1%}"
        except (TypeError, ValueError):
            return "暂无"

    @staticmethod
    def _prob(value: Any) -> str:
        try:
            return f"{float(value):.1%}"
        except (TypeError, ValueError):
            return "暂无"

    @staticmethod
    def _prob_from_percent(value: Any) -> str:
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return "暂无"

    @staticmethod
    def _multiple(value: Any) -> str:
        try:
            return f"{float(value):.2f}x"
        except (TypeError, ValueError):
            return "暂无"

    @staticmethod
    def _number(value: Any) -> str:
        try:
            return f"{float(value):,.2f}"
        except (TypeError, ValueError):
            return "暂无"

    @staticmethod
    def _return_color(value: Any) -> str:
        try:
            return COLORS["positive"] if float(value) >= 0 else COLORS["negative"]
        except (TypeError, ValueError):
            return COLORS["muted"]
