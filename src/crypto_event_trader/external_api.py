from __future__ import annotations

from fastapi import FastAPI

from .api import create_app
from .binance_runtime import BinanceApprovalRuntime, build_binance_approval_runtime
from .config import Settings
from .distributed_control import RedisTradingControl


def create_external_app() -> FastAPI:
    """Uvicorn factory for explicit Demo/Live deployments.

    It is intentionally separate from ``crypto_event_trader.api:app`` so a paper API import can
    never initialize an exchange account by accident.
    """

    settings = Settings.from_env()
    control = RedisTradingControl(settings, lock_on_start=True)
    external: BinanceApprovalRuntime = build_binance_approval_runtime(
        settings, control=control
    )
    app = create_app(
        settings=settings,
        approval_service=external.approvals,
        audit=external.audit,
        control=control,
    )
    app.state.binance_runtime = external
    app.router.add_event_handler("shutdown", external.close)
    app.router.add_event_handler("shutdown", control.close)
    return app
