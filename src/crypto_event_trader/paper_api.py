from __future__ import annotations

from fastapi import FastAPI

from .api import create_app
from .audit import AuditRepository
from .config import Settings
from .distributed_control import RedisTradingControl
from .paper_runtime import validate_paper_runtime_settings


def create_paper_app() -> FastAPI:
    """Control/audit API factory for the worker-owned internal paper account."""

    settings = Settings.from_env()
    # Reject a wrong stage, external venue, live switch, or account key before any service opens.
    validate_paper_runtime_settings(settings)
    control = RedisTradingControl(settings, lock_on_start=False)
    audit = AuditRepository(settings.audit_database_url)
    audit.initialize()
    app = create_app(
        settings=settings,
        audit=audit,
        control=control,
        allow_http_paper_execution=False,
        enable_legacy_pipeline=False,
        worker_managed_execution=True,
    )
    app.state.worker_managed_paper = True
    app.router.add_event_handler("shutdown", audit.close)
    app.router.add_event_handler("shutdown", control.close)
    return app


__all__ = ["create_paper_app"]
