from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

from pydantic import JsonValue

from .audit import AuditRepository
from .config import Settings
from .contracts import (
    DecisionProvider,
    PositionThesis,
    TradeAction,
    TradeCandidate,
    TradeDecision,
    TradeDirection,
)
from .control import TradingControl
from .domain import MarketQuote
from .futures_portfolio import FuturesAccountSnapshot, FuturesPortfolio
from .futures_risk import ExecutionIntent, FuturesHardRisk, emergency_exit_reason
from .openai_decision import safe_rejection


@dataclass(frozen=True, slots=True)
class GatewayFill:
    fill_id: str
    price: float
    quantity: float
    fee: float
    fee_asset: str = "USDT"
    realized_pnl: float | None = None
    filled_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    client_order_id: str | None = None
    side: str | None = None
    role: str = "PRIMARY"
    # Binance order responses expose only cumulative quantity/average price.  They are useful for
    # execution control but must not be booked alongside authoritative private-WS/userTrades
    # trade records.  Paper gateways and exact trade adapters remain authoritative by default.
    authoritative: bool = True
    raw_response: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class GatewayOrderEvent:
    """One venue-side child-order observation returned by an execution gateway.

    A strategy decision has one audited parent order, while a safe execution may create two
    entry attempts, a protective algo order, and an emergency reduce-only order.  Keeping those
    children as structured append-only events makes the complete mutation chain recoverable
    without pretending that every child is a new strategy decision.
    """

    role: str
    event_type: str
    status: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    reduce_only: bool
    external_order_id: str | None = None
    executed_quantity: float = 0
    average_price: float | None = None
    trigger_price: float | None = None
    source_event_id: str | None = None
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_response: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class GatewaySubmission:
    status: str
    client_order_id: str
    external_order_id: str | None = None
    fills: tuple[GatewayFill, ...] = ()
    protective_order_id: str | None = None
    order_events: tuple[GatewayOrderEvent, ...] = ()
    raw_response: Mapping[str, Any] = field(default_factory=dict, repr=False)


class GatewaySubmissionUnresolved(RuntimeError):
    """Carries the venue observations collected before execution became uncertain."""

    def __init__(self, message: str, submission: GatewaySubmission) -> None:
        super().__init__(message)
        self.submission = submission


class FuturesExecutionGateway(Protocol):
    venue: str

    def submit(
        self,
        *,
        intent: ExecutionIntent,
        quote: MarketQuote,
        client_order_id: str,
    ) -> GatewaySubmission: ...

    def cancel_all(self) -> Sequence[GatewayOrderEvent] | None: ...


class FuturesAccountSource(Protocol):
    def snapshot(self, *, timestamp: datetime | None = None) -> FuturesAccountSnapshot: ...


class PaperFuturesGateway:
    """Deterministic paper gateway using conservative taker costs.

    The production gateway has a two-attempt price-protected limit workflow. Paper fills are
    immediate so replay tests remain deterministic; protective stops are retained as venue-side
    state and can be triggered by the replay/worker layer.
    """

    venue = "internal-paper"

    def __init__(self, portfolio: FuturesPortfolio, settings: Settings) -> None:
        self.portfolio = portfolio
        self.settings = settings
        self.protective_stops: dict[str, float] = {}

    def submit(
        self,
        *,
        intent: ExecutionIntent,
        quote: MarketQuote,
        client_order_id: str,
    ) -> GatewaySubmission:
        if not intent.approved or not intent.side or intent.quantity <= 0:
            raise ValueError("paper gateway received an unapproved execution intent")
        slip = self.settings.base_slippage_bps / 10_000
        if intent.side == "BUY":
            price = quote.ask * (1 + slip)
        else:
            price = quote.bid * (1 - slip)
        notional = price * intent.quantity
        fee = notional * self.settings.taker_fee_bps / 10_000
        before = self.portfolio.position(intent.symbol).realized_pnl
        position = self.portfolio.apply_fill(
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            price=price,
            fee=fee,
            leverage=self.settings.max_leverage,
            correlation_cluster=intent.correlation_cluster,
        )
        realized = position.realized_pnl - before
        if intent.protective_stop_price is not None:
            self.protective_stops[intent.symbol] = intent.protective_stop_price
        if not position.quantity:
            self.protective_stops.pop(intent.symbol, None)
        now = quote.timestamp.astimezone(UTC)
        protective_order_id: str | None = None
        order_events: tuple[GatewayOrderEvent, ...] = ()
        if intent.protective_stop_price is not None:
            protective_order_id = f"paper-stop-{uuid4().hex}"
            protective_client_id = f"{client_order_id[:34]}-s"
            protective_side = "SELL" if position.quantity > 0 else "BUY"
            order_events = (
                GatewayOrderEvent(
                    role="PROTECTIVE",
                    event_type="CREATED",
                    status="ACTIVE",
                    client_order_id=protective_client_id,
                    symbol=intent.symbol,
                    side=protective_side,
                    order_type="STOP_MARKET",
                    quantity=abs(position.quantity),
                    reduce_only=True,
                    external_order_id=protective_order_id,
                    trigger_price=intent.protective_stop_price,
                    source_event_id=f"paper-protective:{protective_order_id}:created",
                    observed_at=now,
                    raw_response={
                        "simulated": True,
                        "working_type": "MARK_PRICE",
                        "price_protect": True,
                    },
                ),
            )
        fill = GatewayFill(
            fill_id=f"paper-fill-{uuid4().hex}",
            price=price,
            quantity=intent.quantity,
            fee=fee,
            realized_pnl=realized,
            filled_at=now,
            client_order_id=client_order_id,
            side=intent.side,
            raw_response={"simulated": True, "reduce_only": intent.reduce_only},
        )
        return GatewaySubmission(
            status="FILLED",
            client_order_id=client_order_id,
            external_order_id=f"paper-order-{uuid4().hex}",
            fills=(fill,),
            protective_order_id=protective_order_id,
            order_events=order_events,
            raw_response={"simulated": True, "filled_notional": notional},
        )

    def cancel_all(self) -> None:
        # Paper entry orders fill synchronously; protective exits are intentionally retained.
        return None

    def protective_stop_price(self, symbol: str) -> float | None:
        return self.protective_stops.get(symbol.strip().upper())


@dataclass(frozen=True, slots=True)
class ApprovalResult:
    trace_id: str
    candidate_id: str
    decision: TradeDecision
    intent: ExecutionIntent
    status: str
    venue_order_id: str | None = None
    client_order_id: str | None = None
    submission: GatewaySubmission | None = None
    error: str | None = None


class ApprovalTradingService:
    """Audited GPT approval coordinator; it is the only path to an order gateway.

    Every external mutation follows a prepare/audit/validate/submit sequence. Provider failures,
    stale data, missing evidence, malformed decisions and incomplete traces all fail closed.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        decision_provider: DecisionProvider,
        account_source: FuturesAccountSource,
        gateway: FuturesExecutionGateway,
        audit: AuditRepository,
        control: TradingControl,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.decision_provider = decision_provider
        self.account_source = account_source
        self.gateway = gateway
        self.audit = audit
        self.control = control
        self.clock = clock or (lambda: datetime.now(UTC))
        self.risk = FuturesHardRisk(settings)
        self.theses: dict[str, PositionThesis] = {}
        self._initial_risk_distance: dict[str, float] = {}
        self._results: dict[str, ApprovalResult] = {}
        self.reconciliation_hook: Any | None = None
        self._lock = threading.RLock()
        self.audit.initialize()

    @classmethod
    def paper(
        cls,
        *,
        settings: Settings,
        decision_provider: DecisionProvider,
        audit: AuditRepository,
        control: TradingControl | None = None,
        portfolio: FuturesPortfolio | None = None,
    ) -> ApprovalTradingService:
        paper_portfolio = portfolio or FuturesPortfolio(
            settings.initial_cash, default_leverage=settings.max_leverage
        )
        return cls(
            settings=settings,
            decision_provider=decision_provider,
            account_source=paper_portfolio,
            gateway=PaperFuturesGateway(paper_portfolio, settings),
            audit=audit,
            control=control or TradingControl(settings),
        )

    def review_candidate(
        self,
        candidate: TradeCandidate,
        *,
        quote: MarketQuote,
        evidence: Sequence[Mapping[str, JsonValue]],
        raw_evidence: Sequence[Mapping[str, JsonValue]] | None = None,
        position: PositionThesis | None = None,
        signal_strengthening: bool = False,
        now: datetime | None = None,
    ) -> ApprovalResult:
        reference = (now or self.clock()).astimezone(UTC)
        with self._lock:
            prior = self._results.get(candidate.candidate_id)
            if prior is not None:
                return prior
            trace_id = f"trace_{uuid4().hex}"
            evidence_ids = self._evidence_ids(evidence)
            persisted_evidence = raw_evidence if raw_evidence is not None else evidence
            if set(self._evidence_ids(persisted_evidence)) != set(evidence_ids):
                raise ValueError("raw/model evidence IDs must match exactly")
            self._audit_candidate(trace_id, candidate, persisted_evidence)

            if not evidence_ids:
                decision = safe_rejection(
                    "evidence_missing",
                    candidate=candidate,
                    position=position,
                    now=reference,
                )
            else:
                decision = self.decision_provider.decide(
                    candidate,
                    position=position,
                    evidence=evidence,
                    signal_strengthening=signal_strengthening,
                    now=reference,
                )
            execution_reference = reference if now is not None else self.clock().astimezone(UTC)
            if decision.action in {TradeAction.OPEN, TradeAction.ADD} and (
                not decision.position_thesis.strip() or not decision.invalidation_conditions
            ):
                decision = safe_rejection(
                    "approved_decision_missing_thesis_or_invalidation",
                    candidate=candidate,
                    position=position,
                    now=reference,
                    provider_model=decision.provider_model,
                    response_id=decision.response_id,
                    prompt_version=decision.prompt_version,
                    latency_ms=decision.latency_ms,
                    source_urls=decision.source_urls,
                )
            result = self._process_decision(
                trace_id=trace_id,
                candidate=candidate,
                decision=decision,
                quote=self._fresh_execution_quote(
                    candidate.symbol, quote, decision.action, deterministic_now=now
                ),
                position=position,
                signal_strengthening=signal_strengthening,
                now=execution_reference,
            )
            self._results[candidate.candidate_id] = result
            return result

    def emergency_close_all(
        self,
        quotes: Mapping[str, MarketQuote],
        *,
        now: datetime | None = None,
    ) -> tuple[ApprovalResult, ...]:
        """Cancel outstanding entries and close every position outside of the model path."""

        reference = (now or datetime.now(UTC)).astimezone(UTC)
        account = self.account_source.snapshot()
        reason = emergency_exit_reason(account, self.settings)
        if reason is None:
            return ()
        self.control.engage_risk_lock(reason, at=reference)
        self.audit.append_account_snapshot(
            equity=account.equity,
            cash=account.wallet_balance,
            gross_exposure=account.gross_notional,
            net_exposure=account.net_notional,
            daily_pnl=account.daily_pnl_fraction,
            drawdown=account.drawdown,
            positions=account.positions,
            source=f"deterministic_hard_risk:CANCEL_ALL_PREPARED:{reason}",
            observed_at=reference,
        )
        try:
            self.gateway.cancel_all()
        except Exception as error:
            self.control.engage_kill_switch("emergency_cancel_all_unresolved")
            observed = datetime.now(UTC)
            self.audit.append_account_snapshot(
                equity=account.equity,
                cash=account.wallet_balance,
                gross_exposure=account.gross_notional,
                net_exposure=account.net_notional,
                daily_pnl=account.daily_pnl_fraction,
                drawdown=account.drawdown,
                positions=account.positions,
                source=(f"deterministic_hard_risk:CANCEL_ALL_UNRESOLVED:{type(error).__name__}"),
                observed_at=observed,
            )
        results: list[ApprovalResult] = []
        for raw_position in account.positions:
            quantity = float(raw_position["quantity"])
            symbol = str(raw_position["symbol"]).upper()
            quote = quotes.get(symbol)
            if not quantity:
                continue
            if quote is None:
                mark = float(raw_position.get("mark_price") or raw_position.get("entry_price") or 0)
                if mark <= 0:
                    self.control.engage_kill_switch(
                        f"emergency_exit_missing_reference_price:{symbol}"
                    )
                    continue
                quote = MarketQuote(symbol, mark, mark, mark, 0, reference)
            direction = TradeDirection.LONG if quantity > 0 else TradeDirection.SHORT
            evidence_id = f"risk:{reason}:{int(reference.timestamp())}"
            candidate = TradeCandidate(
                strategy_version="deterministic-circuit-breaker-v1",
                symbol=symbol,
                direction=direction,
                max_quantity=abs(quantity),
                max_risk_fraction=self.settings.add_position_risk,
                feature_snapshot={"trigger": reason, "atr_1h": quote.last * 0.01},
                created_at=reference,
                expires_at=reference + timedelta(seconds=self.settings.candidate_ttl_seconds),
            )
            trace_id = f"trace_{uuid4().hex}"
            self._audit_candidate(
                trace_id,
                candidate,
                (
                    {
                        "evidence_id": evidence_id,
                        "source": "deterministic_hard_risk",
                        "source_id": evidence_id,
                        "occurred_at": reference.isoformat(),
                        "first_observed_at": reference.isoformat(),
                        "payload": {
                            "trigger": reason,
                            "equity": account.equity,
                            "daily_pnl_fraction": account.daily_pnl_fraction,
                            "drawdown": account.drawdown,
                        },
                    },
                ),
            )
            thesis = self.theses.get(symbol) or PositionThesis(
                symbol=symbol,
                direction=direction,
                entry_reason="Recovered position under deterministic risk management",
                expected_horizon_minutes=15,
                supporting_evidence_ids=(evidence_id,),
                pnl_r=0,
            )
            decision = TradeDecision(
                candidate_id=candidate.candidate_id,
                symbol=symbol,
                action=TradeAction.CLOSE,
                direction=direction,
                position_multiplier=0,
                confidence=1,
                evidence_ids=(evidence_id,),
                position_thesis=f"Emergency exit: {reason}",
                invalidation_conditions=(reason,),
                next_review_at=reference + timedelta(minutes=15),
                reason=reason,
                provider_model="deterministic-hard-risk",
                decided_at=reference,
            )
            result = self._process_decision(
                trace_id=trace_id,
                candidate=candidate,
                decision=decision,
                quote=quote,
                position=thesis,
                signal_strengthening=False,
                now=reference,
            )
            self._results[candidate.candidate_id] = result
            results.append(result)
        return tuple(results)

    def protective_stop_close(
        self,
        *,
        symbol: str,
        quote: MarketQuote,
        stop_price: float,
        source_order_event_id: str,
        source_trace_id: str,
        source_venue_order_id: str,
        now: datetime | None = None,
    ) -> ApprovalResult | None:
        """Close one paper position through the ordinary audited reduce-only path.

        This deterministic path deliberately bypasses the model only after an already-audited
        ATR stop has crossed.  It still creates candidate, evidence, decision, hard-risk, order,
        event, fill, thesis, and account-snapshot records before returning.
        """

        reference = (now or quote.timestamp).astimezone(UTC)
        normalized = symbol.strip().upper()
        if stop_price <= 0 or quote.symbol != normalized:
            raise ValueError("protective stop requires a positive matching quote")
        account = self.account_source.snapshot(timestamp=reference)
        raw_position = next(
            (
                item
                for item in account.positions
                if str(item["symbol"]).upper() == normalized and float(item["quantity"])
            ),
            None,
        )
        if raw_position is None:
            return None
        quantity = float(raw_position["quantity"])
        triggered = quote.last <= stop_price if quantity > 0 else quote.last >= stop_price
        if not triggered:
            raise ValueError("protective stop has not crossed the mark price")
        direction = TradeDirection.LONG if quantity > 0 else TradeDirection.SHORT
        evidence_id = f"paper-stop:{source_order_event_id}"
        candidate = TradeCandidate(
            strategy_version="deterministic-paper-atr-stop-v1",
            symbol=normalized,
            direction=direction,
            max_quantity=abs(quantity),
            max_risk_fraction=min(self.settings.add_position_risk, 0.01),
            feature_snapshot={
                "position_management_only": True,
                "protective_stop_triggered": True,
                "atr_1h": max(abs(float(raw_position["entry_price"]) - stop_price) / 2, 1e-12),
                "protective_stop_price": stop_price,
                "trigger_mark_price": quote.last,
                "source_order_event_id": source_order_event_id,
                "source_trace_id": source_trace_id,
                "source_venue_order_id": source_venue_order_id,
            },
            created_at=reference,
            expires_at=reference + timedelta(seconds=self.settings.candidate_ttl_seconds),
        )
        trace_id = f"trace_{uuid4().hex}"
        evidence = (
            {
                "evidence_id": evidence_id,
                "source": "deterministic_paper_atr_stop",
                "source_id": source_order_event_id,
                "occurred_at": reference.isoformat(),
                "first_observed_at": reference.isoformat(),
                "payload": {
                    "symbol": normalized,
                    "stop_price": stop_price,
                    "mark_price": quote.last,
                    "source_trace_id": source_trace_id,
                    "source_venue_order_id": source_venue_order_id,
                },
            },
        )
        self._audit_candidate(trace_id, candidate, evidence)
        thesis = self.current_thesis(normalized, mark_price=quote.last) or PositionThesis(
            symbol=normalized,
            direction=direction,
            entry_reason="Recovered paper position under audited ATR-stop protection",
            expected_horizon_minutes=15,
            supporting_evidence_ids=(evidence_id,),
            invalidation_conditions=("audited ATR protective stop crossed",),
            add_count=1,
        )
        decision = TradeDecision(
            candidate_id=candidate.candidate_id,
            symbol=normalized,
            action=TradeAction.CLOSE,
            direction=direction,
            position_multiplier=0,
            confidence=1,
            evidence_ids=(evidence_id,),
            position_thesis="Audited paper ATR protective stop crossed; close immediately.",
            invalidation_conditions=("protective stop crossed",),
            next_review_at=reference + timedelta(minutes=15),
            reason="deterministic_paper_atr_protective_stop",
            provider_model="deterministic-paper-risk",
            decided_at=reference,
        )
        result = self._process_decision(
            trace_id=trace_id,
            candidate=candidate,
            decision=decision,
            quote=quote,
            position=thesis,
            signal_strengthening=False,
            now=reference,
        )
        self._results[candidate.candidate_id] = result
        return result

    def confirm_risk_baseline(self) -> Mapping[str, Any]:
        confirm = getattr(self.account_source, "confirm_risk_baseline", None)
        if not callable(confirm):
            raise NotImplementedError("this account source does not require a risk baseline")
        snapshot = confirm()
        return asdict(snapshot)

    def current_thesis(
        self, symbol: str, *, mark_price: float | None = None
    ) -> PositionThesis | None:
        normalized = symbol.upper()
        thesis = self.theses.get(normalized)
        if thesis is None:
            thesis = self._restore_thesis(normalized)
        if thesis is None or mark_price is None:
            return thesis
        snapshot = self.account_source.snapshot()
        position = next(
            (item for item in snapshot.positions if str(item["symbol"]).upper() == symbol.upper()),
            None,
        )
        risk_distance = self._initial_risk_distance.get(symbol.upper())
        if not position or not risk_distance:
            return thesis
        direction = 1 if float(position["quantity"]) > 0 else -1
        pnl_r = direction * (mark_price - float(position["entry_price"])) / risk_distance
        return thesis.model_copy(update={"pnl_r": pnl_r})

    def reconcile(self) -> Mapping[str, Any]:
        reconcile = self.reconciliation_hook or getattr(self.gateway, "reconcile", None)
        if not callable(reconcile):
            raise NotImplementedError("this execution gateway has no remote reconciliation")
        result = reconcile()
        return {
            "consistent": bool(result.consistent),
            "issues": [asdict(issue) for issue in result.issues],
            "observed_at_ms": result.snapshot.observed_at_ms,
        }

    def _process_decision(
        self,
        *,
        trace_id: str,
        candidate: TradeCandidate,
        decision: TradeDecision,
        quote: MarketQuote,
        position: PositionThesis | None,
        signal_strengthening: bool,
        now: datetime,
    ) -> ApprovalResult:
        decision_id = self.audit.append_llm_decision(
            trace_id=trace_id,
            candidate_id=candidate.candidate_id,
            action=decision.action.value,
            direction=decision.direction.value if decision.direction else None,
            position_multiplier=decision.position_multiplier,
            confidence=decision.confidence,
            evidence_ids=decision.evidence_ids,
            thesis=decision.position_thesis,
            invalidation_conditions=decision.invalidation_conditions,
            model=decision.provider_model or "deterministic-fail-closed",
            prompt_version=decision.prompt_version,
            next_review_at=decision.next_review_at,
            response_id=decision.response_id,
            latency_ms=decision.latency_ms,
            raw_response=decision.model_dump(mode="json"),
            decision_id=decision.decision_id,
            created_at=decision.decided_at,
        )
        self._mark_account(quote)
        try:
            account = self.account_source.snapshot()
        except Exception as error:
            intent = ExecutionIntent(
                False,
                "account_snapshot_unavailable",
                decision.action,
                decision.symbol,
            )
            self.audit.append_risk_decision(
                trace_id=trace_id,
                candidate_id=candidate.candidate_id,
                decision_id=decision_id,
                outcome="REJECT",
                approved_quantity=0,
                reason_codes=("account_snapshot_unavailable",),
                limits_snapshot={"error": str(error), "fail_closed": True},
            )
            return ApprovalResult(
                trace_id=trace_id,
                candidate_id=candidate.candidate_id,
                decision=decision,
                intent=intent,
                status="REJECTED",
                error=str(error),
            )
        control_snapshot = self.control.snapshot()
        source_ready = getattr(self.account_source, "ready_for_new_orders", True)
        if callable(source_ready):
            source_ready = source_ready()
        if not source_ready:
            control_snapshot = replace(
                control_snapshot,
                new_positions_enabled=False,
                reason="account_or_private_stream_not_ready",
            )
        intent = self.risk.evaluate(
            decision=decision,
            candidate=candidate,
            quote=quote,
            account=account,
            control=control_snapshot,
            thesis=position,
            signal_strengthening=signal_strengthening,
            existing_protective_stop=self._existing_protective_stop(candidate.symbol),
            now=now,
        )
        requested = candidate.max_quantity * decision.position_multiplier
        if not intent.approved:
            outcome = "REJECT"
        elif intent.action in {TradeAction.REDUCE, TradeAction.CLOSE}:
            outcome = "EXIT"
        elif intent.quantity + 1e-12 < requested:
            outcome = "RESIZE"
        else:
            outcome = "ALLOW"
        risk_id = self.audit.append_risk_decision(
            trace_id=trace_id,
            candidate_id=candidate.candidate_id,
            decision_id=decision_id,
            outcome=outcome,
            approved_quantity=intent.quantity,
            reason_codes=(intent.reason,),
            limits_snapshot=self._limits_snapshot(account, quote=quote),
        )

        successor = self._successor_thesis(position, candidate, decision, intent, now)
        if successor is not None and (intent.approved or position is not None):
            self._audit_thesis(trace_id, decision_id, successor, decision)
            # OPEN and ADD theses are proposed before the mutation so the order trace is
            # complete, but they do not become active position state until an authoritative
            # fill exists.  A cancelled/unfilled entry must not create a phantom position or
            # consume the sole add allowance.
            if decision.action not in {TradeAction.OPEN, TradeAction.ADD}:
                self.theses[candidate.symbol] = successor
        if not intent.approved:
            return ApprovalResult(
                trace_id=trace_id,
                candidate_id=candidate.candidate_id,
                decision=decision,
                intent=intent,
                status="REJECTED",
            )

        if successor is None:
            return ApprovalResult(
                trace_id=trace_id,
                candidate_id=candidate.candidate_id,
                decision=decision,
                intent=ExecutionIntent(
                    False,
                    "position_thesis_missing",
                    intent.action,
                    intent.symbol,
                ),
                status="REJECTED",
                error="position_thesis_missing",
            )

        client_order_id = self._client_order_id(trace_id, decision.action)
        venue_order_id = self.audit.append_venue_order(
            trace_id=trace_id,
            candidate_id=candidate.candidate_id,
            decision_id=decision_id,
            risk_decision_id=risk_id,
            venue=self.gateway.venue,
            client_order_id=client_order_id,
            symbol=intent.symbol,
            side=intent.side or "BUY",
            order_type=("MARKET" if intent.reduce_only else "LIMIT"),
            quantity=intent.quantity,
            price=(None if intent.reduce_only else quote.last),
            reduce_only=intent.reduce_only,
            status="PREPARED",
            raw_response={
                "intent": asdict(intent),
                "execution_quote": self._quote_snapshot(quote),
            },
            observed_at=now,
        )
        complete, missing = self.audit.validate_order_trace(venue_order_id)
        if not complete:
            self._append_order_event(
                trace_id,
                venue_order_id,
                "TRACE_REJECTED",
                {"missing": list(missing)},
            )
            return ApprovalResult(
                trace_id=trace_id,
                candidate_id=candidate.candidate_id,
                decision=decision,
                intent=intent,
                status="TRACE_INCOMPLETE",
                venue_order_id=venue_order_id,
                client_order_id=client_order_id,
                error=",".join(missing),
            )

        planner = getattr(self.gateway, "planned_order_events", None)
        if callable(planner):
            planned_events = tuple(
                planner(
                    intent=intent,
                    quote=quote,
                    client_order_id=client_order_id,
                )
            )
            if planned_events:
                # Register every deterministic child ID before the first network mutation.  The
                # private stream can therefore resolve a very fast fill even if it arrives while
                # submit() is still waiting on the five-second entry barrier.
                self._audit_gateway_details(
                    trace_id,
                    venue_order_id,
                    GatewaySubmission(
                        status="PLANNED",
                        client_order_id=client_order_id,
                        order_events=planned_events,
                    ),
                )
        self._append_order_event(trace_id, venue_order_id, "SUBMITTING", {})
        try:
            submission = self.gateway.submit(
                intent=intent,
                quote=quote,
                client_order_id=client_order_id,
            )
        except Exception as error:  # the order may be unknown; never blind-retry here
            partial = getattr(error, "submission", None)
            if isinstance(partial, GatewaySubmission):
                self._audit_gateway_details(trace_id, venue_order_id, partial)
                self._audit_gateway_fills(trace_id, venue_order_id, partial.fills)
            self._append_order_event(
                trace_id,
                venue_order_id,
                "SUBMISSION_UNRESOLVED",
                {"error": str(error), "requires_reconciliation": True},
            )
            self.control.engage_kill_switch("venue_execution_uncertain")
            self._reconcile_after_uncertain_submission(trace_id, venue_order_id)
            self._audit_account_snapshot_best_effort(trace_id)
            return ApprovalResult(
                trace_id=trace_id,
                candidate_id=candidate.candidate_id,
                decision=decision,
                intent=intent,
                status="SUBMISSION_UNRESOLVED",
                venue_order_id=venue_order_id,
                client_order_id=client_order_id,
                error=str(error),
            )

        try:
            self._audit_gateway_details(trace_id, venue_order_id, submission)
            self._append_order_event(
                trace_id,
                venue_order_id,
                submission.status,
                dict(submission.raw_response),
                executed_quantity=sum(fill.quantity for fill in submission.fills),
                average_price=(
                    sum(fill.price * fill.quantity for fill in submission.fills)
                    / sum(fill.quantity for fill in submission.fills)
                    if submission.fills
                    else None
                ),
                external_order_id=submission.external_order_id,
            )
            self._audit_gateway_fills(trace_id, venue_order_id, submission.fills)
            after = self.account_source.snapshot()
            self.audit.append_account_snapshot(
                trace_id=trace_id,
                equity=after.equity,
                cash=after.wallet_balance,
                gross_exposure=after.gross_notional,
                net_exposure=after.net_notional,
                daily_pnl=after.daily_pnl_fraction,
                drawdown=after.drawdown,
                positions=after.positions,
                source=self.gateway.venue,
                observed_at=after.timestamp,
            )
        except Exception as error:
            # The venue may already have mutated.  An incomplete local lineage is therefore an
            # execution-uncertain incident, never an ordinary cycle failure that may continue.
            self.control.engage_kill_switch("post_execution_audit_failed")
            try:
                self._reconcile_after_uncertain_submission(trace_id, venue_order_id)
            except Exception:
                pass
            self._audit_account_snapshot_best_effort(trace_id)
            raise RuntimeError("post-execution audit failed; trading is locked") from error

        authoritative_fills = tuple(fill for fill in submission.fills if fill.authoritative)
        if authoritative_fills:
            remaining = next(
                (
                    abs(float(item["quantity"]))
                    for item in after.positions
                    if str(item["symbol"]).upper() == candidate.symbol
                ),
                0.0,
            )
            if decision.action is TradeAction.CLOSE and remaining <= 1e-12:
                self.theses.pop(candidate.symbol, None)
                self._initial_risk_distance.pop(candidate.symbol, None)
            else:
                self.theses[candidate.symbol] = successor
                if decision.action is TradeAction.OPEN:
                    distance = float(
                        candidate.feature_snapshot.get("suggested_stop_distance")
                        or candidate.feature_snapshot.get("atr_1h", 0) * 2
                        or 0
                    )
                    if distance > 0:
                        self._initial_risk_distance[candidate.symbol] = distance
        return ApprovalResult(
            trace_id=trace_id,
            candidate_id=candidate.candidate_id,
            decision=decision,
            intent=intent,
            status=submission.status,
            venue_order_id=venue_order_id,
            client_order_id=client_order_id,
            submission=submission,
        )

    def _audit_candidate(
        self,
        trace_id: str,
        candidate: TradeCandidate,
        evidence: Sequence[Mapping[str, JsonValue]],
    ) -> None:
        supplied_ids = self._evidence_ids(evidence)
        evidence_ids = (candidate.candidate_id, *supplied_ids) if supplied_ids else ()
        record_ids: list[str] = []
        if supplied_ids:
            record_ids.append(
                self.audit.append_external_evidence(
                    trace_id=trace_id,
                    source="strategy_feature_snapshot",
                    source_id=candidate.candidate_id,
                    evidence_id=candidate.candidate_id,
                    occurred_at=candidate.created_at,
                    first_observed_at=candidate.created_at,
                    payload=dict(candidate.feature_snapshot),
                )
            )
        for item in evidence:
            evidence_id = self._item_evidence_id(item)
            if evidence_id is None:
                continue
            existing_record_id = item.get("evidence_record_id")
            if isinstance(existing_record_id, str) and existing_record_id:
                latest = self.audit.latest_external_evidence(evidence_id)
                if latest is None or str(latest["evidence_record_id"]) != existing_record_id:
                    raise ValueError("external evidence is not the latest audited version")
                if latest.get("deleted_at") is not None:
                    raise ValueError("deleted external evidence cannot support a candidate")
                first_observed = datetime.fromisoformat(
                    str(latest["first_observed_at"]).replace("Z", "+00:00")
                )
                if first_observed.astimezone(UTC) > candidate.created_at:
                    raise ValueError("external evidence was observed after candidate creation")
                supplied_hash = item.get("content_hash")
                latest_payload = latest.get("payload")
                expected_hash = (
                    latest_payload.get("source_content_hash")
                    if isinstance(latest_payload, Mapping)
                    else None
                ) or latest.get("content_hash")
                if (
                    isinstance(supplied_hash, str)
                    and supplied_hash
                    and supplied_hash != expected_hash
                ):
                    raise ValueError("external evidence content hash does not match audit")
                record_ids.append(existing_record_id)
                continue
            source = str(item.get("source") or "local")
            source_id = str(item.get("source_id") or item.get("id") or evidence_id)
            occurred_at = item.get("occurred_at") or item.get("first_observed_at")
            first_observed_at = item.get("first_observed_at") or candidate.created_at
            raw_payload = item.get("payload")
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else dict(item)
            source_url = item.get("source_url") or item.get("url")
            deleted_at = item.get("deleted_at")
            record_ids.append(
                self.audit.append_external_evidence(
                    trace_id=trace_id,
                    source=source,
                    source_id=source_id,
                    evidence_id=evidence_id,
                    occurred_at=(str(occurred_at) if occurred_at else candidate.created_at),
                    first_observed_at=(
                        str(first_observed_at)
                        if not isinstance(first_observed_at, datetime)
                        else first_observed_at
                    ),
                    source_url=str(source_url) if source_url else None,
                    deleted_at=str(deleted_at) if deleted_at else None,
                    payload=payload,
                )
            )
        self.audit.append_trade_candidate(
            trace_id=trace_id,
            strategy_version=candidate.strategy_version,
            symbol=candidate.symbol,
            direction=candidate.direction.value,
            max_quantity=candidate.max_quantity,
            max_risk_fraction=candidate.max_risk_fraction,
            feature_snapshot=candidate.feature_snapshot,
            evidence_ids=evidence_ids,
            evidence_record_ids=record_ids,
            valid_until=candidate.expires_at,
            candidate_id=candidate.candidate_id,
            created_at=candidate.created_at,
        )

    def _audit_thesis(
        self,
        trace_id: str,
        decision_id: str,
        thesis: PositionThesis,
        decision: TradeDecision,
    ) -> None:
        self.audit.append_position_thesis(
            trace_id=trace_id,
            position_id=f"{thesis.symbol}:BOTH",
            decision_id=decision_id,
            entry_reason=thesis.entry_reason,
            expected_horizon=f"{thesis.expected_horizon_minutes} minutes",
            supporting_evidence=thesis.supporting_evidence_ids,
            opposing_evidence=thesis.contradicting_evidence_ids,
            add_count=thesis.add_count,
            pnl_r=thesis.pnl_r,
            invalidation_conditions=(
                thesis.invalidation_conditions
                or decision.invalidation_conditions
                or ("position_requires_continuing_deterministic_risk_protection",)
            ),
            thesis_id=thesis.thesis_id,
            created_at=thesis.created_at,
        )

    @staticmethod
    def _successor_thesis(
        current: PositionThesis | None,
        candidate: TradeCandidate,
        decision: TradeDecision,
        intent: ExecutionIntent,
        now: datetime,
    ) -> PositionThesis | None:
        if current is not None:
            return current.append_version(
                entry_reason=decision.position_thesis or current.entry_reason,
                supporting_evidence_ids=(decision.evidence_ids or current.supporting_evidence_ids),
                contradicting_evidence_ids=(),
                invalidation_conditions=(
                    decision.invalidation_conditions
                    or current.invalidation_conditions
                    or ("position_requires_continuing_deterministic_risk_protection",)
                ),
                pnl_r=current.pnl_r,
                decision_id=decision.decision_id,
                add_count=(
                    current.add_count + 1
                    if decision.action is TradeAction.ADD and intent.approved
                    else current.add_count
                ),
                created_at=now,
            )
        if decision.action is not TradeAction.OPEN or not intent.approved:
            return None
        return PositionThesis(
            symbol=candidate.symbol,
            direction=candidate.direction,
            entry_reason=decision.position_thesis,
            expected_horizon_minutes=240,
            supporting_evidence_ids=decision.evidence_ids,
            invalidation_conditions=decision.invalidation_conditions,
            add_count=0,
            pnl_r=0,
            decision_history=(decision.decision_id,),
            created_at=now,
        )

    def _restore_thesis(self, symbol: str) -> PositionThesis | None:
        row = self.audit.latest_position_thesis(f"{symbol}:BOTH")
        if row is None:
            return None
        try:
            snapshot = self.account_source.snapshot()
            raw_position = next(
                (
                    item
                    for item in snapshot.positions
                    if str(item["symbol"]).upper() == symbol and float(item["quantity"])
                ),
                None,
            )
        except Exception:
            return None
        if raw_position is None:
            return None
        horizon_text = str(row.get("expected_horizon") or "15 minutes")
        try:
            horizon = max(1, int(horizon_text.split()[0]))
        except (TypeError, ValueError):
            horizon = 15
        created_at = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
        thesis = PositionThesis(
            thesis_id=str(row["thesis_id"]),
            previous_version_id=(
                str(row["prior_thesis_id"]) if row.get("prior_thesis_id") else None
            ),
            version=int(row["version"]),
            symbol=symbol,
            direction=(
                TradeDirection.LONG if float(raw_position["quantity"]) > 0 else TradeDirection.SHORT
            ),
            entry_reason=str(row["entry_reason"]),
            expected_horizon_minutes=horizon,
            supporting_evidence_ids=tuple(row.get("supporting_evidence") or ()),
            contradicting_evidence_ids=tuple(row.get("opposing_evidence") or ()),
            invalidation_conditions=tuple(row.get("invalidation_conditions") or ()),
            # Initial R is not reconstructed from a mutable guess. A recovered process may
            # manage/exit the position, but may not consume its one add allowance.
            add_count=1,
            pnl_r=float(row.get("pnl_r") or 0),
            decision_history=(str(row["decision_id"]),),
            created_at=created_at,
        )
        self.theses[symbol] = thesis
        return thesis

    def _fresh_execution_quote(
        self,
        symbol: str,
        fallback: MarketQuote,
        action: TradeAction,
        *,
        deterministic_now: datetime | None,
    ) -> MarketQuote:
        if deterministic_now is not None or action not in {TradeAction.OPEN, TradeAction.ADD}:
            return fallback
        provider = getattr(self.gateway, "current_quote", None)
        if not callable(provider):
            return fallback
        return provider(symbol)

    def _mark_account(self, quote: MarketQuote) -> None:
        marker = getattr(self.account_source, "mark", None)
        if callable(marker):
            marker(quote.symbol, quote.last)

    def _existing_protective_stop(self, symbol: str) -> float | None:
        getter = getattr(self.gateway, "protective_stop_price", None)
        if not callable(getter):
            return None
        value = getter(symbol)
        if value is None:
            return None
        try:
            stop = float(value)
        except (TypeError, ValueError):
            return None
        return stop if stop > 0 else None

    def _append_order_event(
        self,
        trace_id: str,
        venue_order_id: str,
        status: str,
        payload: Mapping[str, Any],
        *,
        executed_quantity: float = 0,
        average_price: float | None = None,
        external_order_id: str | None = None,
        event_type: str | None = None,
        source_event_id: str | None = None,
        observed_at: datetime | None = None,
    ) -> None:
        append = getattr(self.audit, "append_venue_order_event", None)
        if callable(append):
            append(
                trace_id=trace_id,
                venue_order_id=venue_order_id,
                event_type=event_type or status,
                status=status,
                source_event_id=source_event_id,
                executed_quantity=executed_quantity,
                average_price=average_price,
                external_order_id=external_order_id,
                raw_response=payload,
                observed_at=observed_at,
            )

    def _audit_gateway_details(
        self,
        trace_id: str,
        venue_order_id: str,
        submission: GatewaySubmission,
    ) -> None:
        for event in submission.order_events:
            payload = {
                "gateway_order_event": {
                    "role": event.role,
                    "child_client_order_id": event.client_order_id,
                    "symbol": event.symbol,
                    "side": event.side,
                    "order_type": event.order_type,
                    "quantity": event.quantity,
                    "reduce_only": event.reduce_only,
                    "trigger_price": event.trigger_price,
                },
                "venue_response": dict(event.raw_response),
            }
            self._append_order_event(
                trace_id,
                venue_order_id,
                event.status,
                payload,
                event_type=f"{event.role}_{event.event_type}",
                source_event_id=event.source_event_id,
                executed_quantity=event.executed_quantity,
                average_price=event.average_price,
                external_order_id=event.external_order_id,
                observed_at=event.observed_at,
            )

    def _audit_gateway_fills(
        self,
        trace_id: str,
        venue_order_id: str,
        fills: Sequence[GatewayFill],
    ) -> None:
        for fill in fills:
            if not fill.authoritative:
                continue
            raw = dict(fill.raw_response)
            raw.setdefault("client_order_id", fill.client_order_id)
            raw.setdefault("side", fill.side)
            raw.setdefault("role", fill.role)
            self.audit.append_venue_fill(
                trace_id=trace_id,
                venue_order_id=venue_order_id,
                external_fill_id=fill.fill_id,
                price=fill.price,
                quantity=fill.quantity,
                fee=fill.fee,
                fee_asset=fill.fee_asset,
                realized_pnl=fill.realized_pnl,
                raw_response=raw,
                filled_at=fill.filled_at,
            )

    def _reconcile_after_uncertain_submission(self, trace_id: str, venue_order_id: str) -> None:
        reconcile = self.reconciliation_hook or getattr(self.gateway, "reconcile", None)
        if not callable(reconcile):
            self._append_order_event(
                trace_id,
                venue_order_id,
                "RECONCILIATION_REQUIRED",
                {"available": False},
            )
            return
        try:
            result = reconcile()
            consistent = bool(getattr(result, "consistent", False))
            self._append_order_event(
                trace_id,
                venue_order_id,
                "RECONCILED" if consistent else "RECONCILIATION_MISMATCH",
                {"consistent": consistent},
            )
        except Exception as error:
            self._append_order_event(
                trace_id,
                venue_order_id,
                "RECONCILIATION_FAILED",
                {"error": str(error)},
            )

    def _audit_account_snapshot_best_effort(self, trace_id: str) -> None:
        try:
            snapshot = self.account_source.snapshot()
            self.audit.append_account_snapshot(
                trace_id=trace_id,
                equity=snapshot.equity,
                cash=snapshot.wallet_balance,
                gross_exposure=snapshot.gross_notional,
                net_exposure=snapshot.net_notional,
                daily_pnl=snapshot.daily_pnl_fraction,
                drawdown=snapshot.drawdown,
                positions=snapshot.positions,
                # Never let an uncertain observation become the next authoritative expected
                # position baseline during startup reconciliation.
                source=f"{self.gateway.venue}:unresolved_observation",
                observed_at=snapshot.timestamp,
            )
        except Exception:
            return

    def _limits_snapshot(
        self,
        account: FuturesAccountSnapshot,
        *,
        quote: MarketQuote | None = None,
    ) -> dict[str, Any]:
        control = asdict(self.control.snapshot())
        for key, value in tuple(control.items()):
            if isinstance(value, datetime):
                control[key] = value.isoformat()
        snapshot = {
            "equity": account.equity,
            "daily_pnl_fraction": account.daily_pnl_fraction,
            "drawdown": account.drawdown,
            "gross_notional": account.gross_notional,
            "net_notional": account.net_notional,
            "max_gross_exposure": self.settings.max_gross_exposure,
            "max_net_exposure": self.settings.max_net_exposure,
            "max_asset_exposure": self.settings.max_asset_exposure,
            "control": control,
        }
        if quote is not None:
            snapshot["execution_quote"] = self._quote_snapshot(quote)
        return snapshot

    @staticmethod
    def _quote_snapshot(quote: MarketQuote) -> dict[str, Any]:
        timestamp = quote.timestamp
        return {
            "source": "fresh_execution_quote",
            "symbol": quote.symbol.upper(),
            "bid": quote.bid,
            "ask": quote.ask,
            "last": quote.last,
            "volume_24h": quote.volume_24h,
            "timestamp": (
                timestamp.astimezone(UTC).isoformat()
                if timestamp.tzinfo is not None
                else timestamp.isoformat()
            ),
        }

    @staticmethod
    def _evidence_ids(evidence: Sequence[Mapping[str, JsonValue]]) -> tuple[str, ...]:
        identifiers: list[str] = []
        for item in evidence:
            for key in ("evidence_id", "source_id", "id"):
                value = item.get(key)
                if isinstance(value, str) and value and value not in identifiers:
                    identifiers.append(value)
                    break
        return tuple(identifiers)

    @staticmethod
    def _item_evidence_id(item: Mapping[str, JsonValue]) -> str | None:
        for key in ("evidence_id", "source_id", "id"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _client_order_id(trace_id: str, action: TradeAction) -> str:
        # Binance permits 36 characters and a restricted ASCII alphabet.
        return f"gpt-{action.value.lower()}-{trace_id.removeprefix('trace_')[:24]}"[:36]
