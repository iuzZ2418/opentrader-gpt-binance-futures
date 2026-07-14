from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import JsonValue

from crypto_event_trader.approval import (
    ApprovalTradingService,
    GatewayFill,
    GatewayOrderEvent,
    GatewaySubmission,
)
from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.config import Settings
from crypto_event_trader.contracts import (
    PositionThesis,
    TradeAction,
    TradeCandidate,
    TradeDecision,
    TradeDirection,
)
from crypto_event_trader.control import TradingControl
from crypto_event_trader.domain import MarketQuote
from crypto_event_trader.futures_portfolio import FuturesPortfolio
from crypto_event_trader.futures_risk import ExecutionIntent


def _settings(tmp_path: Path) -> Settings:
    return replace(
        Settings.from_env(),
        audit_database_url=f"sqlite:///{tmp_path / 'approval-audit.db'}",
        execution_venue="internal",
    )


def _audit(settings: Settings) -> AuditRepository:
    return AuditRepository(settings.audit_database_url)


def _quote(now: datetime, price: float = 50_000) -> MarketQuote:
    return MarketQuote(
        symbol="BTCUSDT",
        bid=price - 1,
        ask=price + 1,
        last=price,
        volume_24h=2_000_000_000,
        timestamp=now,
    )


def _candidate(
    now: datetime,
    candidate_id: str,
    *,
    direction: TradeDirection = TradeDirection.LONG,
    max_quantity: float = 0.1,
    max_risk_fraction: float = 0.0075,
    action: TradeAction = TradeAction.OPEN,
) -> TradeCandidate:
    return TradeCandidate(
        candidate_id=candidate_id,
        strategy_version="champion-trend-v1",
        symbol="BTCUSDT",
        direction=direction,
        max_quantity=max_quantity,
        max_risk_fraction=max_risk_fraction,
        feature_snapshot={
            "atr_1h": 500,
            "closed_bar": now.isoformat(),
            "requested_action": action.value,
        },
        created_at=now,
        expires_at=now + timedelta(seconds=120),
    )


class ScriptedDecisionProvider:
    def __init__(self, *, reverse_direction: bool = False) -> None:
        self.reverse_direction = reverse_direction
        self.calls = 0

    def decide(
        self,
        candidate: TradeCandidate | None,
        *,
        position: PositionThesis | None = None,
        evidence: Sequence[Mapping[str, JsonValue]] = (),
        signal_strengthening: bool = False,
        now: datetime | None = None,
    ) -> TradeDecision:
        del position, signal_strengthening
        assert candidate is not None
        self.calls += 1
        reference = now or datetime.now(UTC)
        action = TradeAction(str(candidate.feature_snapshot["requested_action"]))
        direction = candidate.direction
        if self.reverse_direction:
            direction = (
                TradeDirection.SHORT
                if candidate.direction is TradeDirection.LONG
                else TradeDirection.LONG
            )
        evidence_ids = tuple(
            str(item["evidence_id"])
            for item in evidence
            if isinstance(item.get("evidence_id"), str)
        )
        return TradeDecision(
            candidate_id=candidate.candidate_id,
            symbol=candidate.symbol,
            action=action,
            direction=direction,
            position_multiplier=1,
            confidence=0.91,
            evidence_ids=evidence_ids,
            position_thesis="Closed-bar trend and breakout votes remain aligned.",
            invalidation_conditions=("directional vote count falls below three",),
            next_review_at=reference + timedelta(minutes=15),
            reason="quantitative candidate approved with bounded size",
            provider_model="fake-gpt-decision",
            response_id=f"resp-{self.calls}",
            decided_at=reference,
        )


def test_precollected_external_evidence_is_linked_without_creating_a_new_version(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    audit = _audit(settings)
    audit.initialize()
    record_id = audit.append_external_evidence(
        source="github:acme/protocol",
        source_id="release:1",
        evidence_id="github:acme/protocol:release:1",
        occurred_at=now - timedelta(minutes=2),
        first_observed_at=now - timedelta(minutes=1),
        payload={
            "source_content_hash": "a" * 64,
            "notification": {"event_type": "SECURITY"},
        },
    )
    latest = audit.latest_external_evidence("github:acme/protocol:release:1")
    assert latest is not None
    service = ApprovalTradingService.paper(
        settings=settings,
        decision_provider=ScriptedDecisionProvider(),
        audit=audit,
    )

    result = service.review_candidate(
        _candidate(now, "candidate-existing-evidence"),
        quote=_quote(now),
        evidence=(
            {
                "evidence_id": "github:acme/protocol:release:1",
                "evidence_record_id": record_id,
                "source": "github:acme/protocol",
                "source_id": "release:1",
                "occurred_at": (now - timedelta(minutes=2)).isoformat(),
                "first_observed_at": (now - timedelta(minutes=1)).isoformat(),
                "content_hash": "a" * 64,
                "summary": "Normalized security event; source content is untrusted.",
                "confidence": 0.9,
            },
        ),
        now=now,
    )

    assert result.status == "FILLED"
    latest_after = audit.latest_external_evidence("github:acme/protocol:release:1")
    assert latest_after is not None and latest_after["version"] == 1
    trace = audit.get_trace(result.trace_id)
    assert record_id in {
        item["evidence_record_id"] for item in trace["linked_external_evidence"]
    }


@dataclass
class AuditCheckingGateway:
    portfolio: FuturesPortfolio
    audit: AuditRepository
    venue: str = "audit-checking-paper"
    submit_calls: int = 0
    cancel_calls: int = 0
    trace_was_complete_before_submit: bool = False

    def submit(
        self,
        *,
        intent: ExecutionIntent,
        quote: MarketQuote,
        client_order_id: str,
    ) -> GatewaySubmission:
        self.submit_calls += 1
        summaries = self.audit.list_traces(limit=10)
        assert len(summaries) == 1
        trace = self.audit.get_trace(str(summaries[0]["trace_id"]))
        assert len(trace["trade_candidates"]) == 1
        assert len(trace["llm_decisions"]) == 1
        assert len(trace["position_theses"]) == 1
        assert len(trace["risk_decisions"]) == 1
        assert len(trace["venue_orders"]) == 1
        venue_order_id = str(trace["venue_orders"][0]["venue_order_id"])
        complete, missing = self.audit.validate_order_trace(venue_order_id)
        assert complete, missing
        self.trace_was_complete_before_submit = True

        position = self.portfolio.apply_fill(
            symbol=intent.symbol,
            side=intent.side or "BUY",
            quantity=intent.quantity,
            price=quote.last,
            leverage=3,
        )
        return GatewaySubmission(
            status="FILLED",
            client_order_id=client_order_id,
            external_order_id="fake-order-1",
            fills=(
                GatewayFill(
                    fill_id="fake-fill-1",
                    price=quote.last,
                    quantity=intent.quantity,
                    fee=0,
                    realized_pnl=position.realized_pnl,
                ),
            ),
            protective_order_id="fake-stop-1",
            order_events=(
                GatewayOrderEvent(
                    role="ENTRY_ATTEMPT",
                    event_type="OBSERVED",
                    status="FILLED",
                    client_order_id=client_order_id,
                    symbol=intent.symbol,
                    side=intent.side or "BUY",
                    order_type="LIMIT",
                    quantity=intent.quantity,
                    reduce_only=False,
                    external_order_id="fake-order-1",
                    executed_quantity=intent.quantity,
                    average_price=quote.last,
                    source_event_id=f"fake-child:{client_order_id}:filled",
                    raw_response={"status": "FILLED"},
                ),
            ),
            raw_response={"status": "FILLED"},
        )

    def cancel_all(self) -> None:
        self.cancel_calls += 1


def test_order_trace_is_complete_before_gateway_execution(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)
    gateway = AuditCheckingGateway(portfolio, repository)
    provider = ScriptedDecisionProvider()
    service = ApprovalTradingService(
        settings=settings,
        decision_provider=provider,
        account_source=portfolio,
        gateway=gateway,
        audit=repository,
        control=TradingControl(settings),
    )

    result = service.review_candidate(
        _candidate(now, "candidate-audit-chain"),
        quote=_quote(now),
        evidence=({"evidence_id": "binance:closed-kline:btc:1h"},),
        now=now,
    )

    assert result.status == "FILLED"
    assert gateway.trace_was_complete_before_submit is True
    assert gateway.submit_calls == provider.calls == 1
    trace = repository.get_trace(result.trace_id)
    assert trace["venue_fills"][0]["external_fill_id"] == "fake-fill-1"
    assert trace["account_snapshots"][0]["source"] == gateway.venue
    assert trace["risk_decisions"][0]["limits_snapshot"]["execution_quote"] == {
        "source": "fresh_execution_quote",
        "symbol": "BTCUSDT",
        "bid": 49_999,
        "ask": 50_001,
        "last": 50_000,
        "volume_24h": 2_000_000_000,
        "timestamp": now.isoformat(),
    }
    assert trace["venue_orders"][0]["raw_response"]["execution_quote"]["last"] == 50_000
    event_types = [event["event_type"] for event in trace["venue_order_events"]]
    assert event_types == [
        "ORDER_RECORDED",
        "SUBMITTING",
        "ENTRY_ATTEMPT_OBSERVED",
        "FILLED",
    ]
    child_event = trace["venue_order_events"][2]
    assert child_event["raw_response"]["gateway_order_event"] == {
        "role": "ENTRY_ATTEMPT",
        "child_client_order_id": result.client_order_id,
        "symbol": "BTCUSDT",
        "side": "BUY",
        "order_type": "LIMIT",
        "quantity": pytest.approx(result.intent.quantity),
        "reduce_only": False,
        "trigger_price": None,
    }


def test_provisional_gateway_aggregate_is_not_booked_as_authoritative_fill(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)

    class ProvisionalGateway(AuditCheckingGateway):
        def submit(
            self,
            *,
            intent: ExecutionIntent,
            quote: MarketQuote,
            client_order_id: str,
        ) -> GatewaySubmission:
            submission = super().submit(intent=intent, quote=quote, client_order_id=client_order_id)
            return replace(
                submission,
                fills=tuple(replace(fill, authoritative=False) for fill in submission.fills),
            )

    gateway = ProvisionalGateway(portfolio, repository)
    service = ApprovalTradingService(
        settings=settings,
        decision_provider=ScriptedDecisionProvider(),
        account_source=portfolio,
        gateway=gateway,
        audit=repository,
        control=TradingControl(settings),
    )

    result = service.review_candidate(
        _candidate(now, "candidate-provisional-fill"),
        quote=_quote(now),
        evidence=({"evidence_id": "binance:closed-kline:btc:1h"},),
        now=now,
    )

    trace = repository.get_trace(result.trace_id)
    assert result.submission is not None and result.submission.fills
    assert trace["venue_fills"] == []
    assert trace["venue_order_events"][-1]["executed_quantity"] == pytest.approx(
        result.intent.quantity
    )


def test_missing_evidence_fails_closed_without_calling_model_or_gateway(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)
    gateway = AuditCheckingGateway(portfolio, repository)
    provider = ScriptedDecisionProvider()
    service = ApprovalTradingService(
        settings=settings,
        decision_provider=provider,
        account_source=portfolio,
        gateway=gateway,
        audit=repository,
        control=TradingControl(settings),
    )

    result = service.review_candidate(
        _candidate(now, "candidate-no-evidence"),
        quote=_quote(now),
        evidence=(),
        now=now,
    )

    assert result.status == "REJECTED"
    assert result.decision.action is TradeAction.REJECT
    assert result.decision.reason == "evidence_missing"
    assert result.intent.reason == "no_execution_required"
    assert provider.calls == gateway.submit_calls == 0
    trace = repository.get_trace(result.trace_id)
    assert trace["trade_candidates"][0]["evidence_ids"] == []
    assert trace["risk_decisions"][0]["outcome"] == "REJECT"
    assert trace["venue_orders"] == []


def test_model_cannot_reverse_candidate_direction(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)
    gateway = AuditCheckingGateway(portfolio, repository)
    provider = ScriptedDecisionProvider(reverse_direction=True)
    service = ApprovalTradingService(
        settings=settings,
        decision_provider=provider,
        account_source=portfolio,
        gateway=gateway,
        audit=repository,
        control=TradingControl(settings),
    )

    result = service.review_candidate(
        _candidate(now, "candidate-reverse-attempt"),
        quote=_quote(now),
        evidence=({"evidence_id": "source:one"},),
        now=now,
    )

    assert result.status == "REJECTED"
    assert result.intent.reason == "candidate_direction_mismatch"
    assert gateway.submit_calls == 0
    trace = repository.get_trace(result.trace_id)
    assert trace["llm_decisions"][0]["direction"] == "SHORT"
    assert trace["risk_decisions"][0]["outcome"] == "REJECT"


def test_candidate_review_is_idempotent_across_model_audit_and_execution(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)
    gateway = AuditCheckingGateway(portfolio, repository)
    provider = ScriptedDecisionProvider()
    service = ApprovalTradingService(
        settings=settings,
        decision_provider=provider,
        account_source=portfolio,
        gateway=gateway,
        audit=repository,
        control=TradingControl(settings),
    )
    candidate = _candidate(now, "candidate-idempotent")
    evidence = ({"evidence_id": "source:idempotent"},)

    first = service.review_candidate(candidate, quote=_quote(now), evidence=evidence, now=now)
    second = service.review_candidate(candidate, quote=_quote(now), evidence=evidence, now=now)

    assert second is first
    assert provider.calls == gateway.submit_calls == 1
    assert len(repository.list_traces()) == 1
    trace = repository.get_trace(first.trace_id)
    assert len(trace["trade_candidates"]) == 1
    assert len(trace["venue_orders"]) == 1
    assert len(trace["venue_fills"]) == 1


def test_cancelled_open_does_not_activate_proposed_position_thesis(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)

    class CancelledGateway(AuditCheckingGateway):
        def submit(
            self,
            *,
            intent: ExecutionIntent,
            quote: MarketQuote,
            client_order_id: str,
        ) -> GatewaySubmission:
            del intent, quote
            self.submit_calls += 1
            return GatewaySubmission(
                status="CANCELED",
                client_order_id=client_order_id,
                external_order_id="cancelled-without-fill",
                raw_response={"status": "CANCELED", "executedQty": "0"},
            )

    service = ApprovalTradingService(
        settings=settings,
        decision_provider=ScriptedDecisionProvider(),
        account_source=portfolio,
        gateway=CancelledGateway(portfolio, repository),
        audit=repository,
        control=TradingControl(settings),
    )

    result = service.review_candidate(
        _candidate(now, "candidate-cancelled-open"),
        quote=_quote(now),
        evidence=({"evidence_id": "source:cancelled-open"},),
        now=now,
    )

    assert result.status == "CANCELED"
    assert service.current_thesis("BTCUSDT") is None
    assert portfolio.snapshot().positions == ()
    # The proposed thesis remains immutable evidence of what was approved, but is not active.
    assert len(repository.get_trace(result.trace_id)["position_theses"]) == 1


def test_post_execution_audit_failure_locks_trading_and_forces_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)
    control = TradingControl(settings)
    gateway = AuditCheckingGateway(portfolio, repository)
    service = ApprovalTradingService(
        settings=settings,
        decision_provider=ScriptedDecisionProvider(),
        account_source=portfolio,
        gateway=gateway,
        audit=repository,
        control=control,
    )

    def fail_snapshot(**_: object) -> str:
        raise OSError("audit storage unavailable after venue fill")

    monkeypatch.setattr(repository, "append_account_snapshot", fail_snapshot)
    with pytest.raises(RuntimeError, match="post-execution audit failed"):
        service.review_candidate(
            _candidate(now, "candidate-post-audit-failure"),
            quote=_quote(now),
            evidence=({"evidence_id": "source:post-audit-failure"},),
            now=now,
        )

    snapshot = control.snapshot()
    assert snapshot.kill_switch_active is True
    assert snapshot.reason == "post_execution_audit_failed"
    assert gateway.submit_calls == 1


def test_add_requires_plus_one_r_and_is_allowed_only_once(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)
    provider = ScriptedDecisionProvider()
    service = ApprovalTradingService.paper(
        settings=settings,
        decision_provider=provider,
        audit=repository,
        portfolio=portfolio,
    )
    evidence = ({"evidence_id": "source:add-lifecycle"},)

    opened = service.review_candidate(
        _candidate(now, "candidate-open-before-add"),
        quote=_quote(now),
        evidence=evidence,
        now=now,
    )
    assert opened.status == "FILLED"
    initial_thesis = service.theses["BTCUSDT"]

    below_one_r = initial_thesis.model_copy(update={"pnl_r": 0.99})
    rejected = service.review_candidate(
        _candidate(
            now,
            "candidate-add-too-early",
            max_quantity=0.05,
            max_risk_fraction=settings.add_position_risk,
            action=TradeAction.ADD,
        ),
        quote=_quote(now),
        evidence=evidence,
        position=below_one_r,
        signal_strengthening=True,
        now=now,
    )
    assert rejected.status == "REJECTED"
    assert rejected.intent.reason == "add_requires_one_r_profit"

    at_one_r = initial_thesis.model_copy(update={"pnl_r": 1.0})
    added = service.review_candidate(
        _candidate(
            now,
            "candidate-add-once",
            max_quantity=0.05,
            max_risk_fraction=settings.add_position_risk,
            action=TradeAction.ADD,
        ),
        quote=_quote(now),
        evidence=evidence,
        position=at_one_r,
        signal_strengthening=True,
        now=now,
    )
    assert added.status == "FILLED"
    assert service.theses["BTCUSDT"].add_count == 1

    second_add = service.review_candidate(
        _candidate(
            now,
            "candidate-add-second-attempt",
            max_quantity=0.05,
            max_risk_fraction=settings.add_position_risk,
            action=TradeAction.ADD,
        ),
        quote=_quote(now),
        evidence=evidence,
        position=service.theses["BTCUSDT"],
        signal_strengthening=True,
        now=now,
    )
    assert second_add.status == "REJECTED"
    assert second_add.intent.reason == "add_limit_reached"
    assert portfolio.position("BTCUSDT").quantity == pytest.approx(0.15)


def test_repeated_hold_reviews_advance_append_only_thesis_lineage(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)
    service = ApprovalTradingService.paper(
        settings=settings,
        decision_provider=ScriptedDecisionProvider(),
        audit=repository,
        portfolio=portfolio,
    )
    evidence = ({"evidence_id": "source:thesis-lineage"},)
    opened = service.review_candidate(
        _candidate(now, "candidate-thesis-open"),
        quote=_quote(now),
        evidence=evidence,
        now=now,
    )
    assert opened.status == "FILLED"

    for index in (1, 2):
        reviewed_at = now + timedelta(seconds=index)
        held = service.review_candidate(
            _candidate(
                reviewed_at,
                f"candidate-thesis-hold-{index}",
                action=TradeAction.HOLD,
            ),
            quote=_quote(reviewed_at),
            evidence=evidence,
            position=service.theses["BTCUSDT"],
            now=reviewed_at,
        )
        assert held.status == "REJECTED"
        assert held.intent.reason == "no_execution_required"

    history = repository.position_thesis_history("BTCUSDT:BOTH")
    assert [item["version"] for item in history] == [1, 2, 3]
    assert len({item["thesis_id"] for item in history}) == 3
    assert service.theses["BTCUSDT"].version == 3
    assert service.theses["BTCUSDT"].previous_version_id == history[-2]["thesis_id"]


def test_model_latency_cannot_authorize_an_expired_candidate(tmp_path: Path) -> None:
    started = datetime.now(UTC)
    current = [started]
    settings = replace(_settings(tmp_path), market_data_max_age_seconds=300)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)

    class DelayedProvider(ScriptedDecisionProvider):
        def decide(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            decision = super().decide(*args, **kwargs)
            current[0] = started + timedelta(seconds=121)
            return decision

    service = ApprovalTradingService(
        settings=settings,
        decision_provider=DelayedProvider(),
        account_source=portfolio,
        gateway=AuditCheckingGateway(portfolio, repository),
        audit=repository,
        control=TradingControl(settings),
        clock=lambda: current[0],
    )
    result = service.review_candidate(
        _candidate(started, "candidate-model-delay"),
        quote=_quote(started),
        evidence=({"evidence_id": "source:model-delay"},),
    )

    assert result.status == "REJECTED"
    assert result.intent.reason == "candidate_expired"


@pytest.mark.parametrize(
    ("mark_price", "reason", "permanent"),
    [
        (46_000.0, "daily_loss_limit", False),
        (20_000.0, "total_drawdown_limit", True),
    ],
)
def test_emergency_circuits_cancel_entries_and_close_positions_reduce_only(
    tmp_path: Path,
    mark_price: float,
    reason: str,
    permanent: bool,
) -> None:
    now = datetime.now(UTC)
    settings = _settings(tmp_path)
    repository = _audit(settings)
    portfolio = FuturesPortfolio(settings.initial_cash)
    portfolio.apply_fill(symbol="BTCUSDT", side="BUY", quantity=1, price=50_000, leverage=3)
    portfolio.mark("BTCUSDT", mark_price)
    provider = ScriptedDecisionProvider()
    gateway = AuditCheckingGateway(portfolio, repository)
    control = TradingControl(settings)
    service = ApprovalTradingService(
        settings=settings,
        decision_provider=provider,
        account_source=portfolio,
        gateway=gateway,
        audit=repository,
        control=control,
    )

    results = service.emergency_close_all({"BTCUSDT": _quote(now, mark_price)}, now=now)

    assert len(results) == 1
    assert results[0].status == "FILLED"
    assert results[0].decision.provider_model == "deterministic-hard-risk"
    assert results[0].intent.reduce_only is True
    assert results[0].intent.reason == "risk_reducing_order"
    assert gateway.cancel_calls == gateway.submit_calls == 1
    assert provider.calls == 0
    assert portfolio.position("BTCUSDT").quantity == 0
    snapshot = control.snapshot()
    assert snapshot.kill_switch_active is True
    assert snapshot.reason == reason
    assert snapshot.permanent_risk_lock is permanent
    if reason == "daily_loss_limit":
        assert snapshot.freeze_until == datetime.combine(
            (now + timedelta(days=1)).date(), datetime.min.time(), UTC
        )
