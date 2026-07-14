from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from crypto_event_trader.approval import ApprovalTradingService
from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.config import Settings
from crypto_event_trader.contracts import (
    CandleInterval,
    MarketBar,
    PositionThesis,
    StrategySpec,
    TradeAction,
    TradeCandidate,
    TradeDecision,
    TradeDirection,
)
from crypto_event_trader.domain import MarketQuote
from crypto_event_trader.futures_portfolio import FuturesPortfolio
from crypto_event_trader.market_data import DerivativesRiskSnapshot
from crypto_event_trader.openai_decision import enforce_decision
from crypto_event_trader.trading_cycle import FuturesTradingCycle

NOW = datetime(2026, 7, 14, 12, tzinfo=UTC)


def _settings() -> Settings:
    return replace(
        Settings.from_env(),
        initial_position_risk=0.0075,
        add_position_risk=0.0025,
        risk_per_trade=0.01,
    )


def _bars(interval: CandleInterval, count: int = 20) -> tuple[MarketBar, ...]:
    hours = 1 if interval is CandleInterval.ONE_HOUR else 4
    start = NOW - timedelta(hours=hours * count)
    bars: list[MarketBar] = []
    previous = 100.0
    for index in range(count):
        close = 100 + index * 0.1
        bars.append(
            MarketBar(
                symbol="BTCUSDT",
                interval=interval,
                open_time=start + timedelta(hours=hours * index),
                close_time=start + timedelta(hours=hours * (index + 1)),
                open=previous,
                high=max(previous, close) + 0.2,
                low=min(previous, close) - 0.2,
                close=close,
                volume=100,
            )
        )
        previous = close
    return tuple(bars)


def _candidate() -> TradeCandidate:
    return TradeCandidate(
        candidate_id="cycle-candidate-1",
        strategy_version="test-v1",
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        max_quantity=2,
        max_risk_fraction=0.0075,
        feature_snapshot={"atr_1h": 2.0, "long_votes": 4, "short_votes": 0},
        created_at=NOW,
    )


class MarketData:
    def closed_bars(
        self, symbol: str, interval: CandleInterval, limit: int
    ) -> tuple[MarketBar, ...]:
        del symbol, limit
        return _bars(interval)

    def derivatives_snapshot(
        self, symbol: str, *, expected_order_notional: float
    ) -> DerivativesRiskSnapshot:
        return DerivativesRiskSnapshot(
            symbol=symbol,
            mark_price=100,
            index_price=100,
            funding_rate=0,
            open_interest=1_000_000,
            adl_quantile=0,
            spread_bps=2,
            depth_within_20bps=expected_order_notional * 50,
            expected_order_notional=expected_order_notional,
            observed_at=NOW,
            open_interest_change_24h_fraction=0,
        )


class Strategy:
    spec = StrategySpec(version="cycle-test-v1")

    def __init__(self, candidate: TradeCandidate | None) -> None:
        self.candidate = candidate

    def generate_candidate(self, **_: Any) -> TradeCandidate | None:
        return self.candidate


class OpeningProvider:
    def decide(
        self,
        candidate: TradeCandidate | None,
        *,
        position: PositionThesis | None = None,
        evidence: Any = (),
        signal_strengthening: bool = False,
        now: datetime | None = None,
    ) -> TradeDecision:
        del position, signal_strengthening
        assert candidate is not None and now is not None
        return TradeDecision(
            candidate_id=candidate.candidate_id,
            symbol=candidate.symbol,
            action=TradeAction.OPEN,
            direction=candidate.direction,
            position_multiplier=0.25,
            confidence=0.9,
            evidence_ids=tuple(item["evidence_id"] for item in evidence),
            position_thesis="Closed-bar trend remains aligned.",
            invalidation_conditions=("fewer than three votes remain aligned",),
            next_review_at=now + timedelta(minutes=15),
            reason="approve traced paper entry",
            provider_model="test-exact-model",
            decided_at=now,
        )


class Approvals:
    def __init__(self, thesis: PositionThesis | None) -> None:
        self.thesis = thesis
        self.candidate: TradeCandidate | None = None
        self.position: PositionThesis | None = None
        self.evidence: tuple[dict[str, Any], ...] = ()
        self.raw_evidence: tuple[dict[str, Any], ...] = ()
        self.audit = AuditRepository("sqlite:///:memory:")
        self.audit.initialize()

    def current_thesis(
        self, symbol: str, *, mark_price: float | None = None
    ) -> PositionThesis | None:
        del symbol, mark_price
        return self.thesis

    def review_candidate(
        self, candidate: TradeCandidate, **kwargs: Any
    ) -> SimpleNamespace:
        self.candidate = candidate
        self.position = kwargs["position"]
        self.evidence = tuple(kwargs["evidence"])
        self.raw_evidence = tuple(kwargs["raw_evidence"])
        return SimpleNamespace(status="CAPTURED")


def _run_symbol(
    *, strategy_candidate: TradeCandidate | None, thesis: PositionThesis | None
) -> Approvals:
    approvals = Approvals(thesis)
    cycle = FuturesTradingCycle(
        settings=_settings(),
        market_data=MarketData(),  # type: ignore[arg-type]
        strategy=Strategy(strategy_candidate),  # type: ignore[arg-type]
        approvals=approvals,  # type: ignore[arg-type]
    )
    result = cycle._run_symbol(  # noqa: SLF001
        "BTCUSDT",
        quote=MarketQuote("BTCUSDT", 99.99, 100.01, 100, 1_000_000, NOW),
        equity=100_000,
        raw_position={"symbol": "BTCUSDT", "quantity": 1.0},
        external_evidence=(),
        now=NOW,
    )
    assert result.status == "CAPTURED"
    return approvals


def _attempt_add(
    candidate: TradeCandidate, position: PositionThesis
) -> TradeDecision:
    decision = TradeDecision(
        candidate_id=candidate.candidate_id,
        symbol=candidate.symbol,
        action=TradeAction.ADD,
        direction=candidate.direction,
        position_multiplier=0.25,
        confidence=0.9,
        evidence_ids=("strategy:evidence",),
        position_thesis="Trend remains valid and has strengthened.",
        invalidation_conditions=("trend weakens",),
        next_review_at=NOW + timedelta(minutes=10),
        reason="attempt to add",
        decided_at=NOW,
    )
    return enforce_decision(
        decision,
        candidate=candidate,
        position=position,
        signal_strengthening=True,
        now=NOW,
    )


def test_same_direction_position_candidate_is_capped_to_quarter_percent_risk() -> None:
    thesis = PositionThesis(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_reason="Initial trend entry",
        expected_horizon_minutes=240,
        pnl_r=1.2,
    )

    approvals = _run_symbol(strategy_candidate=_candidate(), thesis=thesis)

    assert approvals.candidate is not None
    assert approvals.candidate.max_risk_fraction == pytest.approx(0.0025)
    assert approvals.candidate.feature_snapshot.get("position_management_only") is not True
    lineage = approvals.candidate.feature_snapshot["market_input_lineage"]
    assert lineage["hourly_bars"]["count"] == 20
    assert lineage["four_hour_bars"]["count"] == 20
    assert lineage["hourly_bars"]["digest_sha256"]
    model_by_id = {item["evidence_id"]: item for item in approvals.evidence}
    raw_by_id = {item["evidence_id"]: item for item in approvals.raw_evidence}
    hourly_id = lineage["hourly_bars"]["evidence_id"]
    assert "bars" not in model_by_id[hourly_id].get("attributes", {})
    assert len(raw_by_id[hourly_id]["payload"]["bars"]) == 20
    stored = approvals.audit.latest_external_evidence(hourly_id)
    assert stored is not None
    assert stored["content_hash"] == lineage["hourly_bars"]["digest_sha256"]


def test_real_approval_trace_links_full_bar_inputs_without_sending_them_to_model() -> None:
    reference = datetime.now(UTC)
    settings = _settings()
    audit = AuditRepository("sqlite:///:memory:")
    service = ApprovalTradingService.paper(
        settings=settings,
        decision_provider=OpeningProvider(),
        audit=audit,
        portfolio=FuturesPortfolio(settings.initial_cash),
    )
    candidate = _candidate().model_copy(
        update={
            "created_at": reference,
            "expires_at": reference + timedelta(seconds=120),
        }
    )
    result = FuturesTradingCycle(
        settings=settings,
        market_data=MarketData(),  # type: ignore[arg-type]
        strategy=Strategy(candidate),  # type: ignore[arg-type]
        approvals=service,
    )._run_symbol(  # noqa: SLF001
        "BTCUSDT",
        quote=MarketQuote("BTCUSDT", 99.99, 100.01, 100, 1_000_000, reference),
        equity=settings.initial_cash,
        raw_position=None,
        external_evidence=(),
        now=reference,
    )

    assert result.status == "FILLED"
    assert result.approval is not None
    trace = audit.get_trace(result.approval.trace_id)
    linked = trace["linked_external_evidence"]
    bar_inputs = [
        item
        for item in linked
        if item["payload"].get("schema") == "binance-usdm-closed-bars-v1"
    ]
    assert {item["payload"]["interval"] for item in bar_inputs} == {"1h", "4h"}
    assert all(len(item["payload"]["bars"]) == 20 for item in bar_inputs)
    feature_lineage = trace["trade_candidates"][0]["feature_snapshot"][
        "market_input_lineage"
    ]
    assert {
        feature_lineage["hourly_bars"]["evidence_record_id"],
        feature_lineage["four_hour_bars"]["evidence_record_id"],
    } <= {item["evidence_record_id"] for item in bar_inputs}


def test_synthetic_position_management_candidate_cannot_add_exposure() -> None:
    thesis = PositionThesis(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_reason="Existing position",
        expected_horizon_minutes=240,
        pnl_r=2,
    )
    approvals = _run_symbol(strategy_candidate=None, thesis=thesis)

    assert approvals.candidate is not None
    assert approvals.position is thesis
    assert approvals.candidate.feature_snapshot["position_management_only"] is True
    checked = _attempt_add(approvals.candidate, thesis)
    assert checked.action is TradeAction.REJECT
    assert "management_candidate_cannot_add_exposure" in checked.reason


def test_recovered_position_has_add_disabled_even_with_same_direction_signal() -> None:
    approvals = _run_symbol(strategy_candidate=_candidate(), thesis=None)

    assert approvals.candidate is not None
    assert approvals.position is not None
    assert approvals.position.add_count == 1
    assert "Recovered position" in approvals.position.entry_reason
    checked = _attempt_add(approvals.candidate, approvals.position)
    assert checked.action is TradeAction.REJECT
    assert "position_already_added_once" in checked.reason


def test_opposing_signal_becomes_original_direction_management_only_candidate() -> None:
    thesis = PositionThesis(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_reason="Existing long",
        expected_horizon_minutes=240,
        pnl_r=-0.5,
    )
    opposing = _candidate().model_copy(
        update={
            "direction": TradeDirection.SHORT,
            "feature_snapshot": {
                "atr_1h": 2.0,
                "long_votes": 1,
                "short_votes": 4,
                "votes": {"momentum_1h_24": -1},
            },
        }
    )

    approvals = _run_symbol(strategy_candidate=opposing, thesis=thesis)

    assert approvals.candidate is not None
    assert approvals.candidate.direction is TradeDirection.LONG
    assert approvals.candidate.feature_snapshot["position_management_only"] is True
    assert approvals.candidate.feature_snapshot["opposing_signal"] is True
    assert approvals.candidate.feature_snapshot["opposing_signal_direction"] == "SHORT"
    assert approvals.candidate.feature_snapshot["opposing_signal_short_votes"] == 4
    checked = _attempt_add(approvals.candidate, thesis)
    assert checked.action is TradeAction.REJECT
    assert "management_candidate_cannot_add_exposure" in checked.reason


def test_approval_identity_is_stable_for_same_evidence_and_changes_with_context() -> None:
    cycle = FuturesTradingCycle(
        settings=_settings(),
        market_data=MarketData(),  # type: ignore[arg-type]
        strategy=Strategy(_candidate()),  # type: ignore[arg-type]
        approvals=Approvals(None),  # type: ignore[arg-type]
    )
    market_data = MarketData()
    derivatives = market_data.derivatives_snapshot(
        "BTCUSDT", expected_order_notional=1_000
    )
    first_evidence = {
        "evidence_id": "x:btc-security",
        "evidence_record_id": "evr_btc_security_v1",
        "content_hash": "content-v1",
        "attributes": {"evidence_version": 1, "event_type": "SECURITY"},
    }
    second_evidence = {
        "evidence_id": "github:release",
        "evidence_record_id": "evr_github_release_v3",
        "content_hash": "release-v3",
        "attributes": {"evidence_version": 3, "event_type": "RELEASE"},
    }

    first = cycle._bind_approval_context(  # noqa: SLF001
        _candidate(),
        derivatives=derivatives,
        overlay_reasons=(),
        external_evidence=(first_evidence, second_evidence),
    )
    reordered = cycle._bind_approval_context(  # noqa: SLF001
        _candidate(),
        derivatives=derivatives,
        overlay_reasons=(),
        external_evidence=(second_evidence, first_evidence),
    )

    assert reordered.candidate_id == first.candidate_id
    assert (
        reordered.feature_snapshot["approval_context_digest"]
        == first.feature_snapshot["approval_context_digest"]
    )

    revised_evidence = dict(first_evidence)
    revised_evidence.update(
        {
            "evidence_record_id": "evr_btc_security_v2",
            "content_hash": "content-v2",
            "attributes": {"evidence_version": 2, "event_type": "SECURITY"},
        }
    )
    revised = cycle._bind_approval_context(  # noqa: SLF001
        _candidate(),
        derivatives=derivatives,
        overlay_reasons=(),
        external_evidence=(revised_evidence, second_evidence),
    )
    funding_changed = cycle._bind_approval_context(  # noqa: SLF001
        _candidate(),
        derivatives=replace(derivatives, funding_rate=0.001),
        overlay_reasons=("funding_elevated",),
        external_evidence=(first_evidence, second_evidence),
    )

    assert revised.candidate_id != first.candidate_id
    assert funding_changed.candidate_id != first.candidate_id
