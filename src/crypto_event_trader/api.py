from __future__ import annotations

import hmac
import inspect
import math
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .approval import ApprovalTradingService
from .audit import AuditRepository
from .binance import BinanceApiError
from .config import Settings
from .contracts import PositionThesis, TradeCandidate
from .control import TradingControl
from .domain import MarketQuote
from .research import performance_summary
from .runtime import build_api_runtime
from .service import TradingService
from .strategy_registry import StrategyRegistry

ControlToken = Annotated[str | None, Header(alias="X-Control-Token")]


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KillSwitchRequest(StrictRequest):
    reason: str = Field(default="manual_kill_switch", min_length=1, max_length=200)


class EvidencePacket(StrictRequest):
    evidence_id: str = Field(min_length=1, max_length=160)
    source_type: str = Field(min_length=1, max_length=80)
    observed_at: datetime
    summary: str = Field(min_length=1, max_length=4_000)
    confidence: float = Field(ge=0, le=1)
    source_url: str | None = Field(default=None, max_length=2_000)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_aware_timestamp(self) -> EvidencePacket:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        return self


class QuoteRequest(StrictRequest):
    symbol: str = Field(min_length=1, max_length=30)
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    last: float = Field(gt=0)
    volume_24h: float = Field(default=0, ge=0)
    timestamp: datetime

    @model_validator(mode="after")
    def validate_quote(self) -> QuoteRequest:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.ask <= self.bid:
            raise ValueError("ask must be greater than bid")
        if not all(
            math.isfinite(value)
            for value in (self.bid, self.ask, self.last, self.volume_24h)
        ):
            raise ValueError("quote values must be finite")
        self.symbol = self.symbol.upper()
        self.timestamp = self.timestamp.astimezone(UTC)
        return self

    def to_domain(self) -> MarketQuote:
        return MarketQuote(
            symbol=self.symbol,
            bid=self.bid,
            ask=self.ask,
            last=self.last,
            volume_24h=self.volume_24h,
            timestamp=self.timestamp,
        )


class ApprovalEvaluationRequest(StrictRequest):
    candidate: TradeCandidate
    quote: QuoteRequest
    evidence: list[EvidencePacket] = Field(min_length=1, max_length=100)
    position: PositionThesis | None = None
    signal_strengthening: bool = False

    @model_validator(mode="after")
    def validate_lineage(self) -> ApprovalEvaluationRequest:
        if not self.candidate.symbol.endswith("USDT"):
            raise ValueError("only USDT-margined perpetual candidates are accepted")
        if self.candidate.symbol != self.quote.symbol:
            raise ValueError("candidate and quote symbols must match")
        if not math.isfinite(self.candidate.max_quantity):
            raise ValueError("candidate max_quantity must be finite")
        atr = self.candidate.feature_snapshot.get("atr_1h")
        if (
            isinstance(atr, bool)
            or not isinstance(atr, (int, float))
            or not math.isfinite(atr)
            or atr <= 0
        ):
            raise ValueError("candidate requires a positive finite atr_1h")
        identifiers = [item.evidence_id for item in self.evidence]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evidence_id values must be unique")
        latest_permitted = self.candidate.created_at.timestamp() + 2
        if any(item.observed_at.timestamp() > latest_permitted for item in self.evidence):
            raise ValueError("evidence observed after candidate creation is not point-in-time safe")
        if self.position is not None:
            if self.position.symbol != self.candidate.symbol:
                raise ValueError("position and candidate symbols must match")
            if self.position.direction is not self.candidate.direction:
                raise ValueError("position and candidate directions must match")
            if not math.isfinite(self.position.pnl_r):
                raise ValueError("position pnl_r must be finite")
        return self


class RiskBaselineConfirmationRequest(StrictRequest):
    confirmation: Literal["CONFIRM_RISK_BASELINE"]
    reason: str = Field(min_length=4, max_length=500)


class ReconciliationRequest(StrictRequest):
    expected_open_client_ids: tuple[str, ...] = Field(default=(), max_length=1_000)
    expected_positions: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_expected_positions(self) -> ReconciliationRequest:
        if any(
            not symbol.upper().endswith("USDT") or not math.isfinite(quantity)
            for symbol, quantity in self.expected_positions.items()
        ):
            raise ValueError("expected positions require finite USDT contract quantities")
        self.expected_positions = {
            symbol.upper(): quantity for symbol, quantity in self.expected_positions.items()
        }
        return self


def create_app(
    settings: Settings | None = None,
    approval_service: ApprovalTradingService | None = None,
    audit: AuditRepository | None = None,
    control: TradingControl | None = None,
    *,
    allow_http_paper_execution: bool = True,
    enable_legacy_pipeline: bool = True,
    worker_managed_execution: bool = False,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    runtime = build_api_runtime(
        runtime_settings,
        approval_service=approval_service,
        audit=audit,
        control=control,
        worker_managed_execution=worker_managed_execution,
    )
    legacy_service, legacy_error = (
        _build_legacy_service(runtime_settings)
        if enable_legacy_pipeline
        else (None, "legacy pipeline disabled by API factory")
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if legacy_service is not None:
            legacy_service.binance_client.close()
        runtime.close()

    app = FastAPI(
        title="Crypto Event Trader",
        description="Auditable GPT-approved USD-M futures control API",
        version="0.4.0",
        lifespan=lifespan,
    )
    app.state.settings = runtime_settings
    app.state.runtime = runtime
    app.state.approval_service = runtime.approval_service
    app.state.audit = runtime.audit
    app.state.control = runtime.control
    app.state.legacy_service = legacy_service

    def require_control_token(token: str | None) -> str:
        expected = runtime_settings.control_api_token
        if not expected or not token or not hmac.compare_digest(expected, token):
            raise HTTPException(status_code=401, detail="invalid control token")
        return token

    def require_legacy_service() -> TradingService:
        if legacy_service is None:
            raise HTTPException(
                status_code=503,
                detail=f"legacy pipeline unavailable: {legacy_error or 'not configured'}",
            )
        return legacy_service

    def require_audit() -> AuditRepository:
        if any(reason.startswith("audit_unavailable:") for reason in runtime.errors):
            raise HTTPException(status_code=503, detail="audit repository unavailable")
        return runtime.audit

    def authorize_external_mutation(token: str | None) -> None:
        require_control_token(token)
        if runtime_settings.execution_venue != "internal":
            raise HTTPException(
                status_code=423,
                detail=(
                    "legacy external execution is disabled; Binance orders must use the "
                    "audited approval worker"
                ),
            )

    @app.get("/health")
    def health() -> dict[str, Any]:
        readiness = runtime.readiness()
        control_snapshot = runtime.control.snapshot()
        return {
            "status": "ok",
            "environment": runtime_settings.app_env,
            "mode": "gpt_approval_gated",
            "trading_stage": runtime_settings.trading_stage,
            "execution_venue": runtime_settings.execution_venue,
            "binance_credentials_configured": runtime_settings.binance_credentials_ready,
            "openai_credentials_configured": runtime_settings.openai_credentials_ready,
            "readiness": readiness,
            "control": {
                "new_positions_enabled": control_snapshot.new_positions_enabled,
                "kill_switch_active": control_snapshot.kill_switch_active,
                "permanent_risk_lock": control_snapshot.permanent_risk_lock,
            },
            "legacy_pipeline_available": legacy_service is not None,
            "worker_managed_execution": worker_managed_execution,
        }

    @app.get("/control/status")
    def control_status(x_control_token: ControlToken = None) -> dict[str, Any]:
        require_control_token(x_control_token)
        return runtime.control.as_dict()

    @app.post("/control/kill")
    def engage_kill_switch(
        request: KillSwitchRequest,
        x_control_token: ControlToken = None,
    ) -> dict[str, Any]:
        require_control_token(x_control_token)
        return asdict(runtime.control.engage_kill_switch(request.reason))

    @app.post("/control/reset")
    def reset_kill_switch(x_control_token: ControlToken = None) -> dict[str, Any]:
        token = require_control_token(x_control_token)
        try:
            return asdict(runtime.control.reset_kill_switch(token))
        except (PermissionError, RuntimeError) as error:
            raise HTTPException(status_code=423, detail=str(error)) from error

    @app.post("/control/unlock-live")
    def unlock_live(x_control_token: ControlToken = None) -> dict[str, Any]:
        token = require_control_token(x_control_token)
        try:
            return asdict(runtime.control.unlock_live(token))
        except (PermissionError, RuntimeError) as error:
            raise HTTPException(status_code=423, detail=str(error)) from error

    @app.post("/runtime/risk-baseline/confirm")
    def confirm_risk_baseline(
        _: RiskBaselineConfirmationRequest,
        x_control_token: ControlToken = None,
    ) -> Any:
        require_control_token(x_control_token)
        service = runtime.approval_service
        source = getattr(service, "account_source", None) if service else None
        handler = getattr(source, "confirm_risk_baseline", None)
        if not callable(handler):
            raise HTTPException(
                status_code=501,
                detail="the configured account source has no risk-baseline hook",
            )
        try:
            return _jsonable_result(handler())
        except (BinanceApiError, OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/runtime/reconcile")
    def reconcile(
        request: ReconciliationRequest,
        x_control_token: ControlToken = None,
    ) -> Any:
        require_control_token(x_control_token)
        # In an external deployment the full runtime reconciliation also validates account
        # mode/leverage, reconstructs child and protective orders, accounts exact fills and
        # funding, and only then refreshes the expected-position baseline.  Falling back to the
        # bare gateway hook is reserved for injected paper/test services.
        external_runtime = getattr(app.state, "binance_runtime", None)
        handler = getattr(external_runtime, "reconcile", None)
        if not callable(handler):
            service = runtime.approval_service
            gateway = getattr(service, "gateway", None) if service else None
            handler = getattr(gateway, "reconcile", None)
        if not callable(handler):
            raise HTTPException(
                status_code=501,
                detail="the configured gateway has no reconciliation hook",
            )
        arguments = {
            "expected_open_client_ids": request.expected_open_client_ids or None,
            "expected_positions": request.expected_positions or None,
        }
        try:
            result = _call_supported_keywords(handler, arguments)
            return _jsonable_result(result)
        except (BinanceApiError, OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/approval/evaluate")
    def evaluate_paper_candidate(
        request: ApprovalEvaluationRequest,
        x_control_token: ControlToken = None,
    ) -> Any:
        require_control_token(x_control_token)
        service = runtime.approval_service
        gateway = getattr(service, "gateway", None) if service else None
        gateway_venue = getattr(gateway, "venue", None)
        if (
            not allow_http_paper_execution
            or
            service is None
            or runtime_settings.execution_venue != "internal"
            or runtime_settings.trading_stage not in {"backtest", "paper"}
            or gateway_venue != "internal-paper"
        ):
            raise HTTPException(
                status_code=423,
                detail=(
                    "HTTP approval evaluation is disabled for worker-managed paper trading "
                    "or the runtime is not explicit internal-paper"
                ),
            )
        readiness = runtime.readiness()
        if not readiness["ready_for_new_positions"]:
            raise HTTPException(
                status_code=423,
                detail={
                    "message": "approval runtime is fail-closed",
                    "reasons": readiness["reasons"],
                },
            )
        evidence = [
            _evidence_for_model(item, allow_x_content=runtime_settings.x_content_to_openai_allowed)
            for item in request.evidence
        ]
        result = service.review_candidate(
            request.candidate,
            quote=request.quote.to_domain(),
            evidence=evidence,
            position=request.position,
            signal_strengthening=request.signal_strengthening,
        )
        return jsonable_encoder(asdict(result))

    @app.get("/audit/traces")
    def list_audit_traces(
        limit: int = Query(100, ge=1, le=1_000),
        x_control_token: ControlToken = None,
    ) -> list[dict[str, Any]]:
        require_control_token(x_control_token)
        try:
            return require_audit().list_traces(limit=limit)
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/audit/traces/{trace_id}")
    def get_audit_trace(
        trace_id: str,
        x_control_token: ControlToken = None,
    ) -> dict[str, Any]:
        require_control_token(x_control_token)
        if len(trace_id) > 128:
            raise HTTPException(status_code=422, detail="trace_id is too long")
        try:
            result = require_audit().get_trace(trace_id)
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        if not any(result.get(table) for table in result if table != "trace_id"):
            raise HTTPException(status_code=404, detail="trace not found")
        return result

    @app.get("/audit/orders/{venue_order_id}")
    def get_order_trace(
        venue_order_id: str,
        x_control_token: ControlToken = None,
    ) -> dict[str, Any]:
        require_control_token(x_control_token)
        try:
            return require_audit().trace_for_order(venue_order_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/strategies/champion")
    def champion_strategy(x_control_token: ControlToken = None) -> dict[str, Any]:
        require_control_token(x_control_token)
        try:
            # The trading worker and control API are separate processes.  Reload the atomic
            # registry file so an authenticated operator sees the durable post-promotion state,
            # not the API process's startup snapshot.
            registry = StrategyRegistry(runtime_settings.strategy_registry_file())
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=503, detail="strategy registry unavailable") from error
        return {
            "champion": registry.champion.model_dump(mode="json"),
            "challengers": [item.model_dump(mode="json") for item in registry.challengers],
            "promotion_records": [
                item.model_dump(mode="json") for item in registry.promotion_records[-20:]
            ],
        }

    @app.get("/assets")
    def assets(x_control_token: ControlToken = None) -> list[dict[str, Any]]:
        require_control_token(x_control_token)
        return require_legacy_service().repository.list_assets()

    @app.get("/events")
    def events(
        limit: int = Query(100, ge=1, le=1_000),
        x_control_token: ControlToken = None,
    ) -> list[dict[str, Any]]:
        require_control_token(x_control_token)
        return require_legacy_service().repository.list_events(limit)

    @app.get("/signals")
    def signals(
        limit: int = Query(100, ge=1, le=1_000),
        x_control_token: ControlToken = None,
    ) -> list[dict[str, Any]]:
        require_control_token(x_control_token)
        return require_legacy_service().repository.list_signals(limit)

    @app.get("/portfolio")
    def portfolio(x_control_token: ControlToken = None) -> dict[str, Any]:
        require_control_token(x_control_token)
        return require_legacy_service().repository.portfolio()

    @app.get("/orders")
    def orders(
        limit: int = Query(100, ge=1, le=1_000),
        x_control_token: ControlToken = None,
    ) -> list[dict[str, Any]]:
        require_control_token(x_control_token)
        return require_legacy_service().repository.list_orders(limit)

    @app.get("/research/summary")
    def research_summary(x_control_token: ControlToken = None) -> dict[str, Any]:
        require_control_token(x_control_token)
        repository = require_legacy_service().repository
        account = repository.account()
        return performance_summary(
            repository.list_signals(10_000),
            repository.list_orders(10_000),
            repository.equity_curve(100_000),
            account["initial_cash"],
        )

    @app.get("/binance/status")
    def binance_status(x_control_token: ControlToken = None) -> dict[str, Any]:
        require_control_token(x_control_token)
        service = require_legacy_service()
        try:
            offset = service.binance_client.sync_time()
            result: dict[str, Any] = {
                "status": "ok",
                "environment": "USD-M Futures Demo",
                "base_url": runtime_settings.binance_futures_demo_url,
                "server_time_offset_ms": offset,
                "credentials_configured": runtime_settings.binance_credentials_ready,
            }
            if runtime_settings.binance_credentials_ready:
                balances = service.binance_client.account_balance()
                result["usdt_balance"] = next(
                    (item for item in balances if item.get("asset") == "USDT"), None
                )
            return result
        except BinanceApiError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.get("/market/binance-quotes")
    def binance_quotes(x_control_token: ControlToken = None) -> dict[str, Any]:
        require_control_token(x_control_token)
        service = require_legacy_service()
        try:
            return {symbol: asdict(quote) for symbol, quote in service.fetch_live_quotes().items()}
        except BinanceApiError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/pipeline/sample")
    def run_sample(x_control_token: ControlToken = None) -> dict[str, Any]:
        authorize_external_mutation(x_control_token)
        service = require_legacy_service()
        try:
            return asdict(service.run_sample_cycle())
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/pipeline/process-live-prices")
    @app.post("/pipeline/binance")
    def process_live_prices(x_control_token: ControlToken = None) -> dict[str, Any]:
        authorize_external_mutation(x_control_token)
        if not runtime.control.snapshot().new_positions_enabled:
            raise HTTPException(status_code=423, detail="new positions are disabled")
        service = require_legacy_service()
        try:
            quotes = service.fetch_live_quotes()
            return asdict(service.process(quotes))
        except (BinanceApiError, OSError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    return app


def _build_legacy_service(settings: Settings) -> tuple[TradingService | None, str | None]:
    exact_internal = settings.execution_venue == "internal" and settings.trading_stage in {
        "backtest",
        "paper",
    }
    exact_demo = (
        settings.execution_venue == "binance_futures_demo"
        and settings.trading_stage == "demo"
    )
    if not (exact_internal or exact_demo):
        return None, "stage and venue do not identify a supported legacy runtime"
    try:
        return TradingService(settings), None
    except Exception as error:
        return None, f"{type(error).__name__}:{error}"


def _call_supported_keywords(handler: Any, arguments: Mapping[str, Any]) -> Any:
    signature = inspect.signature(handler)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return handler(**arguments)
    supported = {key: value for key, value in arguments.items() if key in signature.parameters}
    return handler(**supported)


def _jsonable_result(result: Any) -> Any:
    payload = asdict(result) if is_dataclass(result) else result
    if hasattr(result, "consistent"):
        payload = dict(payload)
        payload["consistent"] = bool(result.consistent)
    return jsonable_encoder(payload)


def _evidence_for_model(
    packet: EvidencePacket, *, allow_x_content: bool
) -> dict[str, JsonValue]:
    payload = packet.model_dump(mode="json")
    if packet.source_type.strip().lower() not in {"x", "twitter", "x_post"}:
        return payload
    if allow_x_content:
        return payload
    allowed_attribute_keys = {
        "account_id",
        "aggregate_count",
        "credibility",
        "event_type",
        "is_verified_source",
        "sentiment",
        "source_id",
        "topic",
    }
    payload["summary"] = "X raw content withheld by local policy."
    payload["source_url"] = None
    payload["attributes"] = {
        key: value
        for key, value in packet.attributes.items()
        if key in allowed_attribute_keys
    }
    return payload


app = create_app()
