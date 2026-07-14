from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from .approval import ApprovalResult, ApprovalTradingService, PaperFuturesGateway
from .audit import (
    PAPER_FUNDING_COVERAGE_SCHEMA,
    PAPER_FUNDING_COVERAGE_SOURCE,
    AuditRepository,
    FundingAttribution,
    paper_funding_coverage_evidence_id,
)
from .binance import BinanceFuturesClient
from .config import Settings
from .control import TradingControl
from .domain import MarketQuote
from .futures_portfolio import FuturesPortfolio
from .market_data import BinanceFuturesMarketDataProvider
from .openai_decision import OpenAIResponsesDecisionProvider


class PaperRuntimeBoundaryError(ValueError):
    """Raised before startup when paper/external authority boundaries are mixed."""


class PaperLedgerError(RuntimeError):
    """Raised when append-only paper facts cannot reconstruct one exact account."""


@dataclass(frozen=True, slots=True)
class PaperProtectiveStop:
    symbol: str
    side: str
    trigger_price: float
    client_order_id: str
    external_order_id: str
    order_event_id: str
    trace_id: str
    venue_order_id: str
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class PaperFundingEpisode:
    episode_id: str
    symbol: str
    opened_at: datetime
    closed_at: datetime | None


@dataclass(slots=True)
class PaperApprovalRuntime:
    settings: Settings
    client: Any
    market_data: BinanceFuturesMarketDataProvider
    audit: AuditRepository
    portfolio: FuturesPortfolio
    gateway: PaperFuturesGateway
    decision_provider: Any
    approvals: ApprovalTradingService
    control: Any
    owns_client: bool = True
    owns_audit: bool = True
    owns_decision_provider: bool = True
    protective_stops: dict[str, PaperProtectiveStop] = field(default_factory=dict)
    operation_lock: Any = field(default_factory=RLock, repr=False)

    @property
    def account_source(self) -> FuturesPortfolio:
        return self.portfolio

    def run_serialized(self, operation: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run one portfolio/order operation under the runtime's re-entrant mutex."""

        with self.operation_lock:
            return operation(*args, **kwargs)

    def startup_check(self) -> dict[str, Any]:
        """Verify exact model identity while leaving deterministic exits operational."""

        model_ready = self._decision_model_ready()
        if not model_ready:
            self.control.engage_kill_switch("paper_decision_model_unavailable")
        protection_ready = True
        try:
            self.refresh_protective_stops(require_all=True)
        except Exception:
            protection_ready = False
            self.control.engage_kill_switch("paper_protective_ledger_unresolved")
        return {
            "stage": self.settings.trading_stage,
            "venue": self.settings.execution_venue,
            "gateway_venue": self.gateway.venue,
            "public_market_data_only": not bool(
                getattr(self.client, "api_key", None) or getattr(self.client, "api_secret", None)
            ),
            "model_access_verified": model_ready,
            "protective_ledger_verified": protection_ready,
        }

    def persist_account_snapshot(self, *, observed_at: datetime | None = None) -> None:
        with self.operation_lock:
            snapshot = self.portfolio.snapshot(timestamp=observed_at)
            self.audit.append_account_snapshot(
                equity=snapshot.equity,
                cash=snapshot.wallet_balance,
                gross_exposure=snapshot.gross_notional,
                net_exposure=snapshot.net_notional,
                daily_pnl=snapshot.daily_pnl_fraction,
                drawdown=snapshot.drawdown,
                positions=snapshot.positions,
                source=self.gateway.venue,
                observed_at=snapshot.timestamp,
            )

    def mark_positions(self) -> dict[str, MarketQuote]:
        """Mark all positions from one credential-free Binance premium-index snapshot."""

        with self.operation_lock:
            positions = {
                str(raw["symbol"]).upper()
                for raw in self.portfolio.snapshot().positions
                if float(raw["quantity"])
            }
            if not positions:
                return {}
            premium = self.client.premium_index()
            rows = [premium] if isinstance(premium, dict) else premium
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise PaperLedgerError("public premium index must be an object array")
            received_at = datetime.now(UTC)
            by_symbol: dict[str, dict[str, Any]] = {}
            for row in rows:
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol in by_symbol:
                    raise PaperLedgerError(f"duplicate public mark price: {symbol}")
                if symbol:
                    by_symbol[symbol] = row
            quotes: dict[str, MarketQuote] = {}
            for symbol in sorted(positions):
                row = by_symbol.get(symbol)
                if row is None:
                    raise PaperLedgerError(f"public mark price is missing: {symbol}")
                mark_price = _finite_positive(
                    row.get("markPrice"), "public mark price"
                )
                raw_time = row.get("time")
                if raw_time is None:
                    raise PaperLedgerError(f"public mark time is missing: {symbol}")
                try:
                    observed_at = datetime.fromtimestamp(int(raw_time) / 1_000, UTC)
                except (TypeError, ValueError, OSError) as error:
                    raise PaperLedgerError(f"public mark time is invalid: {symbol}") from error
                age_seconds = (received_at - observed_at).total_seconds()
                if age_seconds < -1 or age_seconds > self.settings.market_data_max_age_seconds:
                    raise PaperLedgerError(f"public mark price is stale: {symbol}")
                quote = MarketQuote(
                    symbol=symbol,
                    # A trigger snapshot deliberately carries no synthetic book. If it
                    # crosses, `_enforce_protective_stops` fetches the executable bid/ask.
                    bid=mark_price,
                    ask=mark_price,
                    last=mark_price,
                    volume_24h=0.0,
                    timestamp=observed_at,
                )
                self.portfolio.mark(symbol, mark_price)
                quotes[symbol] = quote
            return quotes

    def refresh_protective_stops(
        self, *, require_all: bool = True
    ) -> dict[str, PaperProtectiveStop]:
        with self.operation_lock:
            return self._refresh_protective_stops(require_all=require_all)

    def _refresh_protective_stops(
        self, *, require_all: bool
    ) -> dict[str, PaperProtectiveStop]:
        active: dict[str, PaperProtectiveStop] = {}
        for row in self.audit.list_protective_order_events(venue=self.gateway.venue):
            event_type = str(row["event_type"]).upper()
            raw = row.get("raw_response")
            raw = raw if isinstance(raw, dict) else {}
            child = raw.get("gateway_order_event")
            child = child if isinstance(child, dict) else {}
            symbol = str(row["symbol"]).upper()
            if event_type == "PROTECTIVE_CREATED":
                if str(child.get("role", "")).upper() != "PROTECTIVE":
                    raise PaperLedgerError("protective event role is invalid")
                trigger = _finite_positive(child.get("trigger_price"), "protective trigger")
                side = str(child.get("side", "")).upper()
                client_order_id = str(child.get("child_client_order_id", ""))
                external_order_id = str(row.get("external_order_id") or "")
                if side not in {"BUY", "SELL"} or not client_order_id or not external_order_id:
                    raise PaperLedgerError("protective event identity is incomplete")
                active[symbol] = PaperProtectiveStop(
                    symbol=symbol,
                    side=side,
                    trigger_price=trigger,
                    client_order_id=client_order_id,
                    external_order_id=external_order_id,
                    order_event_id=str(row["order_event_id"]),
                    trace_id=str(row["trace_id"]),
                    venue_order_id=str(row["venue_order_id"]),
                    observed_at=_datetime(row["observed_at"]),
                )
            elif event_type in {"PROTECTIVE_CONSUMED", "PROTECTIVE_CANCELLED"}:
                target = str(raw.get("protective_client_order_id") or "")
                current = active.get(symbol)
                if current is not None and (not target or target == current.client_order_id):
                    active.pop(symbol, None)

        positions = {
            str(item["symbol"]).upper(): float(item["quantity"])
            for item in self.portfolio.snapshot().positions
            if float(item["quantity"])
        }
        active = {symbol: stop for symbol, stop in active.items() if symbol in positions}
        for symbol, quantity in positions.items():
            stop = active.get(symbol)
            if stop is None:
                if require_all:
                    raise PaperLedgerError(f"open paper position has no audited stop: {symbol}")
                continue
            expected_side = "SELL" if quantity > 0 else "BUY"
            if stop.side != expected_side:
                raise PaperLedgerError(f"paper stop side mismatch: {symbol}")
        self.protective_stops = active
        self.gateway.protective_stops = {
            symbol: stop.trigger_price for symbol, stop in active.items()
        }
        return dict(active)

    def enforce_protective_stops(
        self,
        quotes: dict[str, MarketQuote] | None = None,
        *,
        now: datetime | None = None,
    ) -> tuple[ApprovalResult, ...]:
        with self.operation_lock:
            return self._enforce_protective_stops(quotes, now=now)

    def _enforce_protective_stops(
        self,
        quotes: dict[str, MarketQuote] | None,
        *,
        now: datetime | None,
    ) -> tuple[ApprovalResult, ...]:
        reference = (now or datetime.now(UTC)).astimezone(UTC)
        available = quotes if quotes is not None else self.mark_positions()
        stops = self._refresh_protective_stops(require_all=True)
        results: list[ApprovalResult] = []
        for symbol, stop in stops.items():
            quote = available.get(symbol)
            if quote is None:
                quote = self.market_data.quote(symbol)
                self.portfolio.mark(symbol, quote.last)
            quantity = self.portfolio.position(symbol).quantity
            crossed = (
                quote.last <= stop.trigger_price
                if quantity > 0
                else quote.last >= stop.trigger_price
            )
            if not quantity or not crossed:
                continue
            if quote.bid == quote.ask == quote.last:
                execution_quote = self.market_data.quote(symbol)
                if execution_quote.symbol.upper() != symbol:
                    raise PaperLedgerError(f"paper execution quote symbol mismatch: {symbol}")
                quote = MarketQuote(
                    symbol=symbol,
                    bid=_finite_positive(execution_quote.bid, "execution bid"),
                    ask=_finite_positive(execution_quote.ask, "execution ask"),
                    last=quote.last,
                    volume_24h=_finite_number(
                        execution_quote.volume_24h, "24h quote volume"
                    ),
                    timestamp=quote.timestamp,
                )
                if quote.bid > quote.ask or quote.volume_24h < 0:
                    raise PaperLedgerError("public execution quote is invalid")
            trigger_source = f"paper-protective:{stop.external_order_id}:triggering"
            self.audit.append_venue_order_event(
                trace_id=stop.trace_id,
                venue_order_id=stop.venue_order_id,
                event_type="PROTECTIVE_TRIGGERING",
                status="TRIGGERED",
                source_event_id=trigger_source,
                external_order_id=stop.external_order_id,
                raw_response={
                    "protective_client_order_id": stop.client_order_id,
                    "stop_price": stop.trigger_price,
                    "mark_price": quote.last,
                    "reduce_only_close_required": True,
                },
                observed_at=reference,
            )
            result = self.approvals.protective_stop_close(
                symbol=symbol,
                quote=quote,
                stop_price=stop.trigger_price,
                source_order_event_id=stop.order_event_id,
                source_trace_id=stop.trace_id,
                source_venue_order_id=stop.venue_order_id,
                now=reference,
            )
            if result is None:
                continue
            results.append(result)
            remaining = self.portfolio.position(symbol).quantity
            if result.status != "FILLED" or abs(remaining) > 1e-12:
                self.control.engage_kill_switch(f"paper_protective_stop_unresolved:{symbol}")
                continue
            self.audit.append_venue_order_event(
                trace_id=stop.trace_id,
                venue_order_id=stop.venue_order_id,
                event_type="PROTECTIVE_CONSUMED",
                status="FILLED",
                source_event_id=f"paper-protective:{stop.external_order_id}:consumed",
                external_order_id=stop.external_order_id,
                executed_quantity=abs(quantity),
                average_price=(
                    result.submission.fills[0].price
                    if result.submission is not None and result.submission.fills
                    else quote.last
                ),
                raw_response={
                    "protective_client_order_id": stop.client_order_id,
                    "reduce_only_venue_order_id": result.venue_order_id,
                    "reduce_only_trace_id": result.trace_id,
                },
                observed_at=reference,
            )
        self._refresh_protective_stops(require_all=True)
        return tuple(results)

    def sync_funding(self, *, now: datetime | None = None) -> int:
        """Book and watermark public funding for every audited paper position episode."""

        with self.operation_lock:
            return self._sync_funding(now=now)

    def _sync_funding(self, *, now: datetime | None) -> int:
        reference = (now or datetime.now(UTC)).astimezone(UTC)
        records = self.audit.audited_performance_records(venue=self.gateway.venue)
        fills = records["fills"]
        episodes = _funding_episodes(fills)
        booked = 0
        reference_ms = int(reference.timestamp() * 1_000)
        for episode in episodes:
            opened_ms = int(episode.opened_at.timestamp() * 1_000)
            closed_ms = (
                int(episode.closed_at.timestamp() * 1_000)
                if episode.closed_at is not None
                else None
            )
            target = (
                min(episode.closed_at, reference)
                if episode.closed_at is not None
                else reference
            )
            target_ms = min(closed_ms, reference_ms) if closed_ms is not None else reference_ms
            if target_ms < opened_ms:
                continue
            evidence_id = paper_funding_coverage_evidence_id(
                self.gateway.venue, episode.episode_id
            )
            prior = self.audit.latest_external_evidence(evidence_id)
            prior_covered_ms = _validated_coverage_watermark(
                prior,
                venue=self.gateway.venue,
                episode=episode,
            )
            cursor_ms = max(opened_ms, (prior_covered_ms or opened_ms - 1) + 1)
            while cursor_ms <= target_ms:
                page = self.client.funding_rate_history(
                    episode.symbol,
                    start_time=cursor_ms,
                    end_time=target_ms,
                    limit=1_000,
                )
                ordered = _validated_funding_page(
                    page,
                    start_ms=cursor_ms,
                    end_ms=target_ms,
                )
                for funding_time_ms, item in ordered:
                    quantity = _position_quantity_at(
                        fills, episode.symbol, funding_time_ms
                    )
                    if abs(quantity) <= 1e-12:
                        continue
                    rate = _finite_number(item.get("fundingRate"), "funding rate")
                    mark_price = _finite_positive(
                        item.get("markPrice"), "funding mark price"
                    )
                    amount = -quantity * mark_price * rate
                    transaction_time = datetime.fromtimestamp(
                        funding_time_ms / 1_000, UTC
                    )
                    external_id = f"paper-funding:{episode.symbol}:{funding_time_ms}"
                    is_new = (
                        self.audit.venue_accounting_event_by_external_id(
                            venue=self.gateway.venue,
                            external_income_id=external_id,
                        )
                        is None
                    )
                    event_id = self.audit.append_venue_accounting_event(
                        venue=self.gateway.venue,
                        external_income_id=external_id,
                        income_type="FUNDING_FEE",
                        asset="USDT",
                        amount=amount,
                        transaction_time=transaction_time,
                        symbol=episode.symbol,
                        raw_response={
                            "source": PAPER_FUNDING_COVERAGE_SOURCE,
                            "funding_rate": rate,
                            "mark_price": mark_price,
                            "position_quantity": quantity,
                            "binance_payload": item,
                        },
                    )
                    attribution = self.audit.resolve_funding_attribution(
                        venue=self.gateway.venue,
                        symbol=episode.symbol,
                        transaction_time=transaction_time,
                    )
                    self._ensure_funding_attribution(
                        event_id, attribution, transaction_time
                    )
                    if attribution.status != "ATTRIBUTED":
                        raise PaperLedgerError(
                            "paper funding attribution failed: "
                            f"{episode.symbol}:{attribution.reason}"
                        )
                    if is_new:
                        self.portfolio.apply_funding(episode.symbol, amount)
                        booked += 1
                if len(ordered) < 1_000:
                    break
                advanced = ordered[-1][0] + 1
                if advanced <= cursor_ms:
                    raise PaperLedgerError("funding history pagination did not advance")
                cursor_ms = advanced
            self._persist_funding_coverage(
                episode,
                covered_through=target,
                observed_at=reference,
            )
        return booked

    def _persist_funding_coverage(
        self,
        episode: PaperFundingEpisode,
        *,
        covered_through: datetime,
        observed_at: datetime,
    ) -> None:
        evidence_id = paper_funding_coverage_evidence_id(
            self.gateway.venue, episode.episode_id
        )
        self.audit.ensure_external_evidence(
            source=PAPER_FUNDING_COVERAGE_SOURCE,
            source_id=f"{self.gateway.venue}:{episode.episode_id}",
            evidence_id=evidence_id,
            occurred_at=covered_through,
            first_observed_at=observed_at,
            created_at=observed_at,
            payload={
                "schema": PAPER_FUNDING_COVERAGE_SCHEMA,
                "venue": self.gateway.venue,
                "episode_id": episode.episode_id,
                "symbol": episode.symbol,
                "episode_opened_at": _iso(episode.opened_at),
                "episode_closed_at": (
                    _iso(episode.closed_at) if episode.closed_at is not None else None
                ),
                "covered_through": _iso(covered_through),
                "coverage_semantics": "inclusive_public_funding_history_query",
                "endpoint": "/fapi/v1/fundingRate",
            },
        )

    def cancel_all_entries(self) -> None:
        # Internal entry fills are synchronous; audited protective stops must remain active.
        with self.operation_lock:
            self.gateway.cancel_all()

    def close(self) -> None:
        if self.owns_decision_provider:
            close = getattr(self.decision_provider, "close", None)
            if callable(close):
                close()
        if self.owns_client:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()
        if self.owns_audit:
            self.audit.close()

    def _decision_model_ready(self) -> bool:
        configured = getattr(self.decision_provider, "model", None)
        if configured is not None and configured != self.settings.openai_decision_model:
            return False
        checker = getattr(self.decision_provider, "check_model_access", None)
        if not callable(checker):
            return True  # deterministic injected test provider
        try:
            return bool(checker())
        except Exception:
            return False

    def _append_funding_attribution(
        self,
        event_id: str,
        attribution: FundingAttribution,
        transaction_time: datetime,
    ) -> None:
        self.audit.append_venue_accounting_attribution(
            accounting_event_id=event_id,
            status=attribution.status,
            reason=attribution.reason,
            trace_id=attribution.trace_id,
            venue_order_id=attribution.venue_order_id,
            resolved_at=transaction_time,
        )

    def _ensure_funding_attribution(
        self,
        event_id: str,
        attribution: FundingAttribution,
        transaction_time: datetime,
    ) -> None:
        prior = self.audit.latest_venue_accounting_attribution(event_id)
        if prior is None:
            self._append_funding_attribution(event_id, attribution, transaction_time)
            return
        immutable = {
            "status": attribution.status,
            "reason": attribution.reason,
            "trace_id": attribution.trace_id,
            "venue_order_id": attribution.venue_order_id,
        }
        if any(prior.get(key) != value for key, value in immutable.items()):
            raise PaperLedgerError(f"funding attribution replay conflicts: {event_id}")


def validate_paper_runtime_settings(settings: Settings) -> None:
    if settings.trading_stage != "paper" or settings.execution_venue != "internal":
        raise PaperRuntimeBoundaryError(
            "paper worker requires TRADING_STAGE=paper and EXECUTION_VENUE=internal"
        )
    if settings.binance_api_key or settings.binance_api_secret:
        raise PaperRuntimeBoundaryError("paper worker refuses Binance account credentials")
    if settings.live_trading_enabled or settings.allow_binance_production:
        raise PaperRuntimeBoundaryError("paper worker refuses production trading switches")
    if settings.decision_cycle_seconds != 900:
        raise PaperRuntimeBoundaryError("paper worker decision cycle must be exactly 900 seconds")


def restore_paper_portfolio(
    settings: Settings,
    audit: AuditRepository,
    *,
    now: datetime | None = None,
) -> FuturesPortfolio:
    """Rebuild paper cash and positions only from authoritative fills/funding."""

    reference = (now or datetime.now(UTC)).astimezone(UTC)
    portfolio = FuturesPortfolio(
        settings.initial_cash,
        default_leverage=settings.max_leverage,
    )
    records = audit.audited_performance_records(venue=PaperFuturesGateway.venue)
    for fill in records["fills"]:
        if str(fill.get("fee_asset", "")).upper() != "USDT":
            raise PaperLedgerError("paper fill fee asset must be USDT")
        symbol = str(fill.get("symbol") or "").upper()
        side = str(fill.get("side") or "").upper()
        quantity = _finite_positive(fill.get("quantity"), "fill quantity")
        price = _finite_positive(fill.get("price"), "fill price")
        fee = _finite_number(fill.get("fee"), "fill fee")
        if fee < 0 or not symbol or side not in {"BUY", "SELL"}:
            raise PaperLedgerError("paper fill shape is invalid")
        before_quantity = portfolio.position(symbol).quantity
        signed = quantity if side == "BUY" else -quantity
        after_quantity = before_quantity + signed
        tolerance = max(1e-12, quantity * 1e-10)
        if bool(fill.get("reduce_only")):
            if (
                abs(before_quantity) <= tolerance
                or before_quantity * signed >= 0
                or abs(after_quantity) > abs(before_quantity) + tolerance
                or before_quantity * after_quantity < -tolerance
            ):
                raise PaperLedgerError("invalid audited paper reduce-only transition")
        elif abs(before_quantity) > tolerance and before_quantity * signed < 0:
            raise PaperLedgerError("non-reduce paper fill changed position direction")
        before_realized = portfolio.position(symbol).realized_pnl
        position = portfolio.apply_fill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            leverage=settings.max_leverage,
            correlation_cluster=(
                str(fill.get("raw_response", {}).get("correlation_cluster"))
                if isinstance(fill.get("raw_response"), dict)
                and fill.get("raw_response", {}).get("correlation_cluster")
                else None
            ),
        )
        observed_realized = fill.get("realized_pnl")
        if observed_realized is None:
            raise PaperLedgerError("paper fill is missing authoritative realized PnL")
        expected_realized = position.realized_pnl - before_realized
        if not math.isclose(
            expected_realized,
            _finite_number(observed_realized, "realized PnL"),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise PaperLedgerError("paper fill realized PnL conflicts with replay")

    for event in sorted(
        records["funding"],
        key=lambda item: (_datetime(item["transaction_time"]), str(item["accounting_event_id"])),
    ):
        if str(event.get("asset", "")).upper() != "USDT":
            raise PaperLedgerError("paper funding asset must be USDT")
        attribution = event.get("attribution")
        if not isinstance(attribution, dict):
            resolved = audit.resolve_funding_attribution(
                venue=PaperFuturesGateway.venue,
                symbol=str(event.get("symbol") or ""),
                transaction_time=event["transaction_time"],
            )
            audit.append_venue_accounting_attribution(
                accounting_event_id=str(event["accounting_event_id"]),
                status=resolved.status,
                reason=resolved.reason,
                trace_id=resolved.trace_id,
                venue_order_id=resolved.venue_order_id,
                resolved_at=event["transaction_time"],
            )
            attribution = {"status": resolved.status, "reason": resolved.reason}
        if attribution.get("status") != "ATTRIBUTED":
            raise PaperLedgerError(
                "paper funding is not attributed: " + str(attribution.get("reason") or "missing")
            )
        portfolio.apply_funding(
            str(event.get("symbol") or "").upper(),
            _finite_number(event.get("amount"), "funding amount"),
        )

    risk = audit.account_risk_state(PaperFuturesGateway.venue, now=reference)
    current_equity = portfolio.snapshot(timestamp=reference).equity
    high_water = risk.get("historical_high_water_equity")
    if high_water is not None:
        portfolio.high_water_equity = max(
            portfolio.high_water_equity,
            _finite_positive(high_water, "historical high-water equity"),
        )
    day_start = risk.get("utc_day_start_equity")
    latest = risk.get("latest") if isinstance(risk.get("latest"), dict) else None
    if day_start is None and latest is not None:
        day_start = latest.get("equity")
    portfolio.daily_start_equity = (
        _finite_positive(day_start, "daily start equity")
        if day_start is not None
        else current_equity
    )
    portfolio.current_day = (
        reference.date()
        if risk.get("utc_day_start_equity") is not None or latest is None
        else _datetime(latest["observed_at"]).date()
    )
    return portfolio


def build_paper_approval_runtime(
    settings: Settings,
    *,
    control: Any | None = None,
    audit: AuditRepository | None = None,
    client: Any | None = None,
    decision_provider: Any | None = None,
) -> PaperApprovalRuntime:
    validate_paper_runtime_settings(settings)
    runtime_audit = audit or AuditRepository(settings.audit_database_url)
    runtime_audit.initialize()
    portfolio = restore_paper_portfolio(settings, runtime_audit)
    runtime_client = client or BinanceFuturesClient(
        None,
        None,
        base_url=settings.binance_futures_live_url,
        environment="production",
        allow_production_trading=False,
        max_leverage=settings.max_leverage,
        recv_window_ms=settings.binance_recv_window_ms,
    )
    if getattr(runtime_client, "api_key", None) or getattr(runtime_client, "api_secret", None):
        raise PaperRuntimeBoundaryError("paper market-data client must be credential-free")
    provider = decision_provider or OpenAIResponsesDecisionProvider(
        api_key=settings.openai_api_key,
        project=settings.openai_project,
        model=settings.openai_decision_model,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.openai_request_timeout_seconds,
        allow_web_search=False,
        x_content_to_openai_allowed=settings.x_content_to_openai_allowed,
    )
    runtime_control = control or TradingControl(settings)
    gateway = PaperFuturesGateway(portfolio, settings)
    approvals = ApprovalTradingService(
        settings=settings,
        decision_provider=provider,
        account_source=portfolio,
        gateway=gateway,
        audit=runtime_audit,
        control=runtime_control,
    )
    runtime = PaperApprovalRuntime(
        settings=settings,
        client=runtime_client,
        market_data=BinanceFuturesMarketDataProvider(runtime_client),
        audit=runtime_audit,
        portfolio=portfolio,
        gateway=gateway,
        decision_provider=provider,
        approvals=approvals,
        control=runtime_control,
        owns_client=client is None,
        owns_audit=audit is None,
        owns_decision_provider=decision_provider is None,
    )
    runtime.refresh_protective_stops(require_all=True)
    return runtime


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _datetime(value).isoformat().replace("+00:00", "Z")


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PaperLedgerError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise PaperLedgerError(f"{name} must be finite")
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0:
        raise PaperLedgerError(f"{name} must be positive")
    return result


def _position_quantity_at(
    fills: list[dict[str, Any]], symbol: str, timestamp_ms: int
) -> float:
    quantity = 0.0
    transaction_time = datetime.fromtimestamp(timestamp_ms / 1_000, UTC)
    for fill in fills:
        if str(fill.get("symbol") or "").upper() != symbol:
            continue
        if _datetime(fill["filled_at"]) > transaction_time:
            break
        size = _finite_positive(fill.get("quantity"), "fill quantity")
        quantity += size if str(fill.get("side")).upper() == "BUY" else -size
        if abs(quantity) <= 1e-12:
            quantity = 0.0
    return quantity


def _funding_episodes(fills: list[dict[str, Any]]) -> tuple[PaperFundingEpisode, ...]:
    ordered = sorted(
        fills,
        key=lambda item: (
            _datetime(item["filled_at"]),
            str(item.get("venue_fill_id") or ""),
        ),
    )
    states: dict[str, dict[str, Any]] = {}
    episodes: list[PaperFundingEpisode] = []
    for fill in ordered:
        symbol = str(fill.get("symbol") or "").strip().upper()
        fill_id = str(fill.get("venue_fill_id") or "").strip()
        side = str(fill.get("side") or "").upper()
        if not symbol or not fill_id or side not in {"BUY", "SELL"}:
            raise PaperLedgerError("paper funding episode fill identity is incomplete")
        size = _finite_positive(fill.get("quantity"), "fill quantity")
        signed = size if side == "BUY" else -size
        filled_at = _datetime(fill["filled_at"])
        state = states.setdefault(
            symbol,
            {
                "quantity": 0.0,
                "total_quantity": 0.0,
                "first_fill_id": None,
                "opened_at": None,
            },
        )
        previous = float(state["quantity"])
        state["total_quantity"] = float(state["total_quantity"]) + size
        tolerance = max(1e-12, float(state["total_quantity"]) * 1e-10)
        reduce_only = bool(fill.get("reduce_only"))
        if abs(previous) <= tolerance:
            if reduce_only:
                raise PaperLedgerError("paper reduce-only fill cannot open a funding episode")
            state["first_fill_id"] = fill_id
            state["opened_at"] = filled_at
        next_quantity = previous + signed
        if reduce_only:
            if (
                abs(previous) <= tolerance
                or previous * signed >= 0
                or abs(next_quantity) > abs(previous) + tolerance
                or previous * next_quantity < -tolerance
            ):
                raise PaperLedgerError("invalid paper reduce-only funding episode transition")
        elif abs(previous) > tolerance and previous * signed < 0:
            raise PaperLedgerError("paper fill reverses a funding episode without reduce-only")
        state["quantity"] = 0.0 if abs(next_quantity) <= tolerance else next_quantity
        if state["quantity"] == 0.0:
            first_fill_id = str(state["first_fill_id"] or "")
            opened_at = state["opened_at"]
            if not first_fill_id or not isinstance(opened_at, datetime):
                raise PaperLedgerError("closed paper funding episode has no opening identity")
            episodes.append(
                PaperFundingEpisode(
                    episode_id=f"{symbol}:{first_fill_id}",
                    symbol=symbol,
                    opened_at=opened_at,
                    closed_at=filled_at,
                )
            )
            states[symbol] = {
                "quantity": 0.0,
                "total_quantity": 0.0,
                "first_fill_id": None,
                "opened_at": None,
            }
    for symbol, state in states.items():
        if abs(float(state["quantity"])) <= 1e-12:
            continue
        first_fill_id = str(state["first_fill_id"] or "")
        opened_at = state["opened_at"]
        if not first_fill_id or not isinstance(opened_at, datetime):
            raise PaperLedgerError("open paper funding episode has no opening identity")
        episodes.append(
            PaperFundingEpisode(
                episode_id=f"{symbol}:{first_fill_id}",
                symbol=symbol,
                opened_at=opened_at,
                closed_at=None,
            )
        )
    return tuple(sorted(episodes, key=lambda item: (item.opened_at, item.episode_id)))


def _validated_coverage_watermark(
    evidence: dict[str, Any] | None,
    *,
    venue: str,
    episode: PaperFundingEpisode,
) -> int | None:
    if evidence is None:
        return None
    if evidence.get("deleted_at") is not None:
        raise PaperLedgerError(f"paper funding coverage is deleted: {episode.episode_id}")
    if evidence.get("source") != PAPER_FUNDING_COVERAGE_SOURCE:
        raise PaperLedgerError(f"paper funding coverage source mismatch: {episode.episode_id}")
    payload = evidence.get("payload")
    if not isinstance(payload, dict):
        raise PaperLedgerError(f"paper funding coverage payload is invalid: {episode.episode_id}")
    expected = {
        "schema": PAPER_FUNDING_COVERAGE_SCHEMA,
        "venue": venue,
        "episode_id": episode.episode_id,
        "symbol": episode.symbol,
        "episode_opened_at": _iso(episode.opened_at),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise PaperLedgerError(f"paper funding coverage identity mismatch: {episode.episode_id}")
    recorded_close = payload.get("episode_closed_at")
    expected_close = _iso(episode.closed_at) if episode.closed_at is not None else None
    if (
        (expected_close is None and recorded_close is not None)
        or (expected_close is not None and recorded_close not in (None, expected_close))
    ):
        raise PaperLedgerError(f"paper funding coverage close mismatch: {episode.episode_id}")
    try:
        covered = _datetime(payload["covered_through"])
    except (KeyError, TypeError, ValueError) as error:
        raise PaperLedgerError(
            f"paper funding coverage watermark is invalid: {episode.episode_id}"
        ) from error
    if covered < episode.opened_at:
        raise PaperLedgerError(f"paper funding coverage predates episode: {episode.episode_id}")
    if episode.closed_at is not None and covered > episode.closed_at:
        raise PaperLedgerError(
            f"paper funding coverage exceeds episode close: {episode.episode_id}"
        )
    return int(covered.timestamp() * 1_000)


def _validated_funding_page(
    page: Any,
    *,
    start_ms: int,
    end_ms: int,
) -> list[tuple[int, dict[str, Any]]]:
    if not isinstance(page, list) or len(page) > 1_000:
        raise PaperLedgerError("funding history response must be a list of at most 1000 rows")
    ordered: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for raw in page:
        if not isinstance(raw, dict):
            raise PaperLedgerError("funding history row must be an object")
        try:
            funding_time_ms = int(raw["fundingTime"])
        except (KeyError, TypeError, ValueError) as error:
            raise PaperLedgerError("funding history row has no valid fundingTime") from error
        if funding_time_ms in seen:
            raise PaperLedgerError("funding history response contains duplicate timestamps")
        if not start_ms <= funding_time_ms <= end_ms:
            raise PaperLedgerError("funding history row is outside the requested coverage range")
        seen.add(funding_time_ms)
        ordered.append((funding_time_ms, raw))
    ordered.sort(key=lambda item: item[0])
    return ordered


__all__ = [
    "PaperApprovalRuntime",
    "PaperFundingEpisode",
    "PaperLedgerError",
    "PaperProtectiveStop",
    "PaperRuntimeBoundaryError",
    "build_paper_approval_runtime",
    "restore_paper_portfolio",
    "validate_paper_runtime_settings",
]
