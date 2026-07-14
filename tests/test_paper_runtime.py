from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, RLock
from typing import Any

import pytest

from crypto_event_trader.audit import (
    IncompleteVenueAccountingError,
    paper_funding_coverage_evidence_id,
)
from crypto_event_trader.config import Settings
from crypto_event_trader.contracts import (
    PositionThesis,
    TradeAction,
    TradeCandidate,
    TradeDecision,
    TradeDirection,
)
from crypto_event_trader.domain import MarketQuote
from crypto_event_trader.paper_runtime import (
    PaperRuntimeBoundaryError,
    build_paper_approval_runtime,
)
from crypto_event_trader.paper_worker import (
    PAPER_PROTECTIVE_POLL_SECONDS,
    _paper_protective_poll_once,
    _poll_paper_protective_stops,
)

NOW = datetime.now(UTC)


def _settings(tmp_path: Path) -> Settings:
    return replace(
        Settings.from_env(),
        trading_stage="paper",
        execution_venue="internal",
        live_trading_enabled=False,
        allow_binance_production=False,
        binance_api_key=None,
        binance_api_secret=None,
        audit_database_url=f"sqlite:///{tmp_path / 'paper-audit.db'}",
        strategy_registry_path=str(tmp_path / "registry.json"),
        initial_cash=100_000,
        decision_cycle_seconds=900,
        market_data_max_age_seconds=10,
    )


class PublicClient:
    api_key = None
    api_secret = None
    mark_price = 50_000.0

    def close(self) -> None:
        return None

    def funding_rate_history(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    def premium_index(
        self, symbol: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        item = {
            "symbol": "BTCUSDT",
            "markPrice": str(self.mark_price),
            "time": int(datetime.now(UTC).timestamp() * 1_000),
        }
        return item if symbol is not None else [item]


class FundingHistoryClient(PublicClient):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def funding_rate_history(
        self,
        symbol: str,
        *,
        start_time: int,
        end_time: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "symbol": symbol,
                "start_time": start_time,
                "end_time": end_time,
                "limit": limit,
            }
        )
        return [
            dict(item)
            for item in self.events
            if start_time <= int(item["fundingTime"]) <= end_time
        ][:limit]


class OpeningDecisionProvider:
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
        assert candidate is not None
        reference = now or NOW
        evidence_ids = tuple(
            str(item["evidence_id"])
            for item in evidence
            if isinstance(item.get("evidence_id"), str)
        )
        return TradeDecision(
            candidate_id=candidate.candidate_id,
            symbol=candidate.symbol,
            action=TradeAction.OPEN,
            direction=candidate.direction,
            position_multiplier=1,
            confidence=0.9,
            evidence_ids=evidence_ids,
            position_thesis="Five-vote paper trend entry.",
            invalidation_conditions=("ATR protective stop crosses",),
            next_review_at=reference + timedelta(minutes=15),
            reason="approved in deterministic test",
            provider_model="test-decision-model",
            decided_at=reference,
        )


class UnavailableExactProvider(OpeningDecisionProvider):
    model = "gpt-5.6-terra"

    def check_model_access(self) -> bool:
        return False


def _candidate(now: datetime | None = None) -> TradeCandidate:
    reference = now or NOW
    return TradeCandidate(
        candidate_id="paper-runtime-candidate",
        strategy_version="trend-breakout-v1",
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        max_quantity=1,
        max_risk_fraction=0.0075,
        feature_snapshot={"atr_1h": 1_000.0, "long_votes": 4, "short_votes": 1},
        created_at=reference,
    )


def _quote(price: float = 50_000.0, *, now: datetime | None = None) -> MarketQuote:
    return MarketQuote(
        symbol="BTCUSDT",
        bid=price * 0.9999,
        ask=price * 1.0001,
        last=price,
        volume_24h=1_000_000_000,
        timestamp=now or NOW,
    )


def _open(runtime: Any) -> Any:
    # Build point-in-time inputs at invocation time.  Full-suite collection may occur
    # several seconds before this test runs, while the account snapshot is created now.
    reference = datetime.now(UTC)
    result = runtime.approvals.review_candidate(
        _candidate(reference),
        quote=_quote(now=reference),
        evidence=({"evidence_id": "strategy:five-vote-paper"},),
        now=reference,
    )
    assert result.status == "FILLED"
    return result


def test_restart_rebuilds_position_cash_and_funding_from_authoritative_audit(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    first = build_paper_approval_runtime(
        settings,
        client=PublicClient(),
        decision_provider=OpeningDecisionProvider(),
    )
    opened = _open(first)
    quantity = first.portfolio.position("BTCUSDT").quantity
    wallet_before_funding = first.portfolio.wallet_balance
    funding_time = datetime.now(UTC) + timedelta(hours=1)
    event_id = first.audit.append_venue_accounting_event(
        venue=first.gateway.venue,
        external_income_id="paper-funding:BTCUSDT:test",
        income_type="FUNDING_FEE",
        asset="USDT",
        amount=12.5,
        transaction_time=funding_time,
        symbol="BTCUSDT",
        raw_response={"source": "test-public-funding"},
    )
    attribution = first.audit.resolve_funding_attribution(
        venue=first.gateway.venue,
        symbol="BTCUSDT",
        transaction_time=funding_time,
    )
    assert attribution.status == "ATTRIBUTED"
    first.audit.append_venue_accounting_attribution(
        accounting_event_id=event_id,
        status=attribution.status,
        reason=attribution.reason,
        trace_id=attribution.trace_id,
        venue_order_id=attribution.venue_order_id,
        resolved_at=funding_time,
    )
    first.close()

    restored = build_paper_approval_runtime(
        settings,
        client=PublicClient(),
        decision_provider=OpeningDecisionProvider(),
    )
    try:
        assert restored.portfolio.position("BTCUSDT").quantity == pytest.approx(quantity)
        assert restored.portfolio.wallet_balance == pytest.approx(wallet_before_funding + 12.5)
        assert restored.portfolio.position("BTCUSDT").funding_pnl == pytest.approx(12.5)
        assert restored.protective_stops["BTCUSDT"].trace_id == opened.trace_id
    finally:
        restored.close()


def test_restarted_audited_atr_stop_executes_traced_reduce_only_close(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    first = build_paper_approval_runtime(
        settings,
        client=PublicClient(),
        decision_provider=OpeningDecisionProvider(),
    )
    opened = _open(first)
    first.close()
    restored = build_paper_approval_runtime(
        settings,
        client=PublicClient(),
        decision_provider=OpeningDecisionProvider(),
    )
    stop = restored.protective_stops["BTCUSDT"]
    trigger_time = datetime.now(UTC)
    trigger = _quote(stop.trigger_price * 0.99, now=trigger_time)

    results = restored.enforce_protective_stops({"BTCUSDT": trigger}, now=trigger_time)

    assert len(results) == 1
    closed = results[0]
    assert closed.status == "FILLED"
    assert closed.intent.reduce_only is True
    assert closed.decision.provider_model == "deterministic-paper-risk"
    assert restored.portfolio.position("BTCUSDT").quantity == pytest.approx(0)
    close_trace = restored.audit.get_trace(closed.trace_id)
    assert close_trace["venue_orders"][0]["reduce_only"] is True
    original = restored.audit.get_trace(opened.trace_id)
    event_types = {item["event_type"] for item in original["venue_order_events"]}
    assert {"PROTECTIVE_CREATED", "PROTECTIVE_TRIGGERING", "PROTECTIVE_CONSUMED"} <= event_types
    restored.close()

    final = build_paper_approval_runtime(
        settings,
        client=PublicClient(),
        decision_provider=OpeningDecisionProvider(),
    )
    try:
        assert final.portfolio.snapshot().positions == ()
        assert final.protective_stops == {}
    finally:
        final.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"trading_stage": "demo"},
        {"execution_venue": "binance_futures_demo"},
        {"binance_api_key": "forbidden", "binance_api_secret": "forbidden"},
        {"live_trading_enabled": True},
        {"allow_binance_production": True},
        {"decision_cycle_seconds": 60},
    ],
)
def test_paper_runtime_rejects_stage_credentials_and_live_switches_before_io(
    tmp_path: Path,
    changes: dict[str, Any],
) -> None:
    with pytest.raises(PaperRuntimeBoundaryError):
        build_paper_approval_runtime(
            replace(_settings(tmp_path), **changes),
            client=PublicClient(),
            decision_provider=OpeningDecisionProvider(),
        )


def test_exact_decision_model_failure_engages_shared_kill_switch(tmp_path: Path) -> None:
    runtime = build_paper_approval_runtime(
        _settings(tmp_path),
        client=PublicClient(),
        decision_provider=UnavailableExactProvider(),
    )
    try:
        check = runtime.startup_check()
        assert check["model_access_verified"] is False
        snapshot = runtime.control.snapshot()
        assert snapshot.kill_switch_active is True
        assert snapshot.new_positions_enabled is False
        assert snapshot.reason == "paper_decision_model_unavailable"
    finally:
        runtime.close()


def test_injected_market_client_with_account_key_is_refused(tmp_path: Path) -> None:
    class UnsafeClient(PublicClient):
        api_key = "must-not-enter-paper-worker"

    with pytest.raises(PaperRuntimeBoundaryError, match="credential-free"):
        build_paper_approval_runtime(
            _settings(tmp_path),
            client=UnsafeClient(),
            decision_provider=OpeningDecisionProvider(),
        )


def test_funding_coverage_tracks_closed_episode_and_performance_fails_until_sealed(
    tmp_path: Path,
) -> None:
    client = FundingHistoryClient()
    runtime = build_paper_approval_runtime(
        _settings(tmp_path),
        client=client,
        decision_provider=OpeningDecisionProvider(),
    )
    try:
        _open(runtime)
        fills = runtime.audit.audited_performance_records(venue=runtime.gateway.venue)[
            "fills"
        ]
        opened_at = datetime.fromisoformat(
            str(fills[0]["filled_at"]).replace("Z", "+00:00")
        ).astimezone(UTC)
        funding_at = opened_at + timedelta(hours=1)
        client.events = [
            {
                "symbol": "BTCUSDT",
                "fundingTime": int(funding_at.timestamp() * 1_000),
                "fundingRate": "0.001",
                "markPrice": "50000",
            },
            {
                "symbol": "BTCUSDT",
                "fundingTime": int((opened_at + timedelta(hours=4)).timestamp() * 1_000),
                "fundingRate": "0.002",
                "markPrice": "50000",
            },
        ]

        assert runtime.sync_funding(now=funding_at + timedelta(minutes=1)) == 1
        runtime.refresh_protective_stops(require_all=True)
        stop = runtime.protective_stops["BTCUSDT"]
        close_at = opened_at + timedelta(hours=2)
        closed = runtime.enforce_protective_stops(
            {"BTCUSDT": _quote(stop.trigger_price * 0.99, now=close_at)},
            now=close_at,
        )
        assert len(closed) == 1

        with pytest.raises(
            IncompleteVenueAccountingError,
            match="INCOMPLETE_FUNDING_COVERAGE",
        ):
            runtime.audit.build_trade_outcomes(venue=runtime.gateway.venue)

        calls_before_seal = len(client.calls)
        assert runtime.sync_funding(now=close_at + timedelta(hours=1)) == 0
        assert len(client.calls) == calls_before_seal + 1
        assert client.calls[-1]["end_time"] == int(close_at.timestamp() * 1_000)
        calls_after_seal = len(client.calls)
        assert runtime.sync_funding(now=close_at + timedelta(hours=2)) == 0
        assert len(client.calls) == calls_after_seal

        outcomes = runtime.audit.build_trade_outcomes(venue=runtime.gateway.venue)
        assert len(outcomes) == 1
        assert outcomes[0].funding_cost > 0
        assert len(
            runtime.audit.list_venue_accounting_events(
                venue=runtime.gateway.venue,
                income_type="FUNDING_FEE",
            )
        ) == 1
        coverage_id = paper_funding_coverage_evidence_id(
            runtime.gateway.venue,
            str(outcomes[0].episode_id),
        )
        coverage = runtime.audit.latest_external_evidence(coverage_id)
        assert coverage is not None
        assert coverage["version"] == 2
        assert coverage["payload"]["episode_closed_at"] == close_at.isoformat().replace(
            "+00:00", "Z"
        )
        assert coverage["payload"]["covered_through"] == close_at.isoformat().replace(
            "+00:00", "Z"
        )
        assert coverage["evidence_record_id"] in outcomes[0].source_record_ids
    finally:
        runtime.close()


def test_one_second_protective_poll_uses_same_mutex_as_cycle_work() -> None:
    assert PAPER_PROTECTIVE_POLL_SECONDS == 1.0

    class Control:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def engage_kill_switch(self, reason: str) -> None:
            self.reasons.append(reason)

    class Runtime:
        def __init__(self) -> None:
            self.lock = RLock()
            self.control = Control()
            self.blocking_started = Event()
            self.release_blocking = Event()
            self.polls = 0
            self.active = 0
            self.max_active = 0

        def run_serialized(self, operation: Any, /, *args: Any, **kwargs: Any) -> Any:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                try:
                    return operation(*args, **kwargs)
                finally:
                    self.active -= 1

        def blocking_cycle(self) -> None:
            self.blocking_started.set()
            assert self.release_blocking.wait(timeout=1)

        def mark_positions(self) -> dict[str, Any]:
            self.polls += 1
            return {}

        def enforce_protective_stops(
            self, _quotes: dict[str, Any], *, now: datetime
        ) -> tuple[Any, ...]:
            del now
            return ()

    async def scenario() -> Runtime:
        runtime = Runtime()
        stop = asyncio.Event()
        cycle_task = asyncio.create_task(
            asyncio.to_thread(runtime.run_serialized, runtime.blocking_cycle)
        )
        assert await asyncio.to_thread(runtime.blocking_started.wait, 1)
        poll_task = asyncio.create_task(
            _poll_paper_protective_stops(runtime, stop, poll_seconds=0.01)
        )
        await asyncio.sleep(0.03)
        assert runtime.polls == 0
        runtime.release_blocking.set()
        await cycle_task
        deadline = time.monotonic() + 1
        while runtime.polls < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        stop.set()
        await poll_task
        return runtime

    runtime = asyncio.run(scenario())

    assert runtime.polls >= 2
    assert runtime.max_active == 1
    assert runtime.control.reasons == []


def test_protective_poll_triggers_on_mark_price_not_last_trade(tmp_path: Path) -> None:
    client = FundingHistoryClient()
    runtime = build_paper_approval_runtime(
        _settings(tmp_path),
        client=client,
        decision_provider=OpeningDecisionProvider(),
    )
    try:
        _open(runtime)
        runtime.refresh_protective_stops(require_all=True)
        stop = runtime.protective_stops["BTCUSDT"]
        client.mark_price = stop.trigger_price * 0.99

        class BookMarket:
            def quote(self, symbol: str) -> MarketQuote:
                assert symbol == "BTCUSDT"
                # The last trade is deliberately above the stop. Only the premium-index
                # mark crosses it; bid/ask remain the conservative simulated fill source.
                return MarketQuote(
                    symbol=symbol,
                    bid=client.mark_price * 0.999,
                    ask=client.mark_price * 1.001,
                    last=stop.trigger_price * 1.1,
                    volume_24h=1_000_000,
                    timestamp=datetime.now(UTC),
                )

        runtime.market_data = BookMarket()  # type: ignore[assignment]

        assert _paper_protective_poll_once(runtime) == 1
        assert runtime.portfolio.position("BTCUSDT").quantity == pytest.approx(0)
        trace = runtime.audit.get_trace(stop.trace_id)
        triggering = next(
            item
            for item in trace["venue_order_events"]
            if item["event_type"] == "PROTECTIVE_TRIGGERING"
        )
        assert triggering["raw_response"]["mark_price"] == pytest.approx(
            client.mark_price
        )
    finally:
        runtime.close()


def test_protective_poll_retries_funding_seal_after_close_failure() -> None:
    class Control:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def engage_kill_switch(self, reason: str) -> None:
            self.reasons.append(reason)

    class Runtime:
        def __init__(self) -> None:
            self.lock = RLock()
            self.control = Control()
            self.enforcements = 0
            self.funding_attempts = 0
            self.snapshots = 0

        def run_serialized(self, operation: Any, /, *args: Any, **kwargs: Any) -> Any:
            with self.lock:
                return operation(*args, **kwargs)

        def mark_positions(self) -> dict[str, Any]:
            return {}

        def enforce_protective_stops(
            self, _quotes: dict[str, Any], *, now: datetime
        ) -> tuple[object, ...]:
            del now
            self.enforcements += 1
            return (object(),) if self.enforcements == 1 else ()

        def sync_funding(self, *, now: datetime) -> int:
            del now
            self.funding_attempts += 1
            if self.funding_attempts == 1:
                raise RuntimeError("temporary public funding failure")
            return 0

        def persist_account_snapshot(self, *, observed_at: datetime) -> None:
            del observed_at
            self.snapshots += 1

    async def scenario() -> Runtime:
        runtime = Runtime()
        stop = asyncio.Event()
        task = asyncio.create_task(
            _poll_paper_protective_stops(runtime, stop, poll_seconds=0.01)
        )
        deadline = time.monotonic() + 1
        while runtime.funding_attempts < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        stop.set()
        await task
        return runtime

    runtime = asyncio.run(scenario())

    assert runtime.funding_attempts == 2
    assert runtime.snapshots == 1
    assert runtime.control.reasons == ["paper_protective_poll_unavailable"]


def test_paper_worker_and_factory_are_registered() -> None:
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert 'crypto-paper-worker = "crypto_event_trader.paper_worker:main"' in project
    assert "crypto_event_trader.paper_api:create_paper_app" in compose
    assert "profiles: [\"paper\"]" in compose
