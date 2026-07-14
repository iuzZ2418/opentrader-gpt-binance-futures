from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crypto_event_trader import worker as worker_module
from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.binance import BinanceApiError
from crypto_event_trader.contracts import (
    CandleInterval,
    MarketBar,
    StrategySpec,
)
from crypto_event_trader.learning import TradeOutcome
from crypto_event_trader.learning_runtime import (
    AuthoritativePerformanceMonitor,
    CounterfactualOutcomeScheduler,
    LearningPromotionScheduler,
    RollbackSignal,
    build_champion_strategy,
)
from crypto_event_trader.openai_research import (
    ResearchRecommendation,
    StrategyResearchResult,
)
from crypto_event_trader.strategy_registry import StrategyRegistry
from crypto_event_trader.worker import binance_rate_limit_delay

NOW = datetime(2026, 7, 14, 12, tzinfo=UTC)


class HistoricalBars:
    def closed_bars_between(
        self,
        symbol: str,
        interval: CandleInterval,
        *,
        start: datetime,
        end: datetime,
        limit: int = 100,
    ) -> tuple[MarketBar, ...]:
        del limit
        close_time = start + timedelta(hours=1)
        assert close_time <= end
        return (
            MarketBar(
                symbol=symbol,
                interval=interval,
                open_time=close_time - timedelta(hours=1),
                close_time=close_time,
                open=100,
                high=112,
                low=99,
                close=110,
                volume=1_000,
            ),
        )


class ProposalResearcher:
    def research(
        self,
        champion: StrategySpec,
        research_context: dict[str, Any],
        *,
        available_evidence_ids: tuple[str, ...] = (),
        now: datetime | None = None,
    ) -> StrategyResearchResult:
        assert research_context["cost_and_shadow_gates_required"] is True
        assert available_evidence_ids
        return StrategyResearchResult(
            recommendation=ResearchRecommendation.PROPOSE,
            spec=StrategySpec(
                version="trend-breakout-v2",
                minimum_directional_votes=4,
            ),
            parent_version=champion.version,
            hypothesis="A four-vote threshold may reduce weak entries.",
            rationale=("Completed point-in-time outcomes justify a challenger test.",),
            evidence_ids=(available_evidence_ids[0],),
            expected_failure_modes=("May miss early trend transitions.",),
            provider_model="research-test-model",
            response_id="resp-research-v2",
            research_prompt_version="strategy-research-v1",
            latency_ms=5,
            created_at=now or NOW,
        )


def _repository(tmp_path: Path) -> AuditRepository:
    audit = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    audit.initialize()
    return audit


def _candidate(audit: AuditRepository) -> tuple[str, str]:
    created_at = NOW - timedelta(hours=24)
    candidate_id = audit.append_trade_candidate(
        trace_id="candidate-trace",
        strategy_version="trend-breakout-v1",
        symbol="BTCUSDT",
        direction="LONG",
        max_quantity=0.01,
        max_risk_fraction=0.0075,
        feature_snapshot={"last_price": 100.0},
        evidence_ids=["closed-bars"],
        valid_until=created_at + timedelta(seconds=120),
        created_at=created_at,
    )
    decision_id = audit.append_llm_decision(
        trace_id="candidate-trace",
        candidate_id=candidate_id,
        action="REJECT",
        position_multiplier=0,
        confidence=0.8,
        evidence_ids=["closed-bars"],
        thesis="Rejected by the approval committee.",
        invalidation_conditions=["candidate expired"],
        model="decision-test-model",
        prompt_version="trade-v1",
        created_at=created_at,
    )
    return candidate_id, decision_id


def _seed_complete_promotion_evidence(
    audit: AuditRepository,
    *,
    champion_spec_id: str,
    challenger_spec_id: str,
) -> None:
    trace_id = "strategy_learning"
    audit.append_backtest_run(
        trace_id=trace_id,
        spec_id=challenger_spec_id,
        started_at="2024-01-01T00:00:00Z",
        ended_at="2026-06-30T00:00:00Z",
        completed=True,
        net_profit=2_500,
        net_return=0.25,
        max_drawdown=0.10,
        total_cost=200,
        stressed_net_return_2x=0.20,
        dsr_significance_probability=0.96,
        pbo_probability=0.08,
        symbol_concentration=0.30,
        month_concentration=0.30,
        trade_count=100,
        holdout_months=12,
        validation={
            "walk_forward_passed": True,
            "holdout_passed": True,
            "parameter_perturbation_passed": True,
            "latency_stress_passed": True,
            "social_placebo_passed": True,
        },
    )
    shadow_start = "2026-04-01T00:00:00Z"
    shadow_end = "2026-07-01T00:00:00Z"
    audit.append_shadow_result(
        trace_id=trace_id,
        spec_id=champion_spec_id,
        started_at=shadow_start,
        ended_at=shadow_end,
        completed=True,
        elapsed_days=91,
        closed_trades=60,
        net_return=0.10,
        max_drawdown=0.12,
        stressed_net_return_2x=0.06,
        symbol_concentration=0.30,
        month_concentration=0.30,
    )
    audit.append_shadow_result(
        trace_id=trace_id,
        spec_id=challenger_spec_id,
        started_at=shadow_start,
        ended_at=shadow_end,
        completed=True,
        elapsed_days=91,
        closed_trades=30,
        net_return=0.12,
        max_drawdown=0.10,
        stressed_net_return_2x=0.08,
        symbol_concentration=0.30,
        month_concentration=0.30,
    )


def test_counterfactuals_settle_only_matured_horizons_from_bounded_bars(
    tmp_path: Path,
) -> None:
    audit = _repository(tmp_path)
    candidate_id, decision_id = _candidate(audit)
    scheduler = CounterfactualOutcomeScheduler(audit, HistoricalBars())

    result = scheduler.settle_due(as_of=NOW)

    assert result.appended == 3
    trace = audit.get_trace("candidate-trace")
    outcomes = trace["counterfactual_outcomes"]
    assert {row["horizon_hours"] for row in outcomes} == {1, 4, 24}
    assert all(row["candidate_id"] == candidate_id for row in outcomes)
    assert all(row["decision_id"] == decision_id for row in outcomes)
    assert all(row["realized_return"] == pytest.approx(0.10) for row in outcomes)
    assert all(row["decision_regret"] == pytest.approx(0.10) for row in outcomes)
    assert scheduler.settle_due(as_of=NOW).appended == 0


def test_weekly_research_registers_only_a_challenger_and_waits_for_real_evidence(
    tmp_path: Path,
) -> None:
    audit = _repository(tmp_path)
    _candidate(audit)
    CounterfactualOutcomeScheduler(audit, HistoricalBars()).settle_due(as_of=NOW)
    registry = StrategyRegistry(tmp_path / "registry.json")
    scheduler = LearningPromotionScheduler(
        audit=audit,
        registry=registry,
        market_data=HistoricalBars(),
        researcher=ProposalResearcher(),
    )

    result = scheduler.tick(now=NOW, force_weekly=True)

    assert result.research_status == "CHALLENGER_REGISTERED:trend-breakout-v2"
    assert registry.champion.version == "trend-breakout-v1"
    assert [item.version for item in registry.challengers] == ["trend-breakout-v2"]
    assert any("WAITING_FOR_COMPLETE_EVIDENCE" in item for item in result.promotion_statuses)
    row = audit.strategy_spec_by_version("trend-breakout-v2")
    assert row is not None
    assert row["status"] == "CHALLENGER"
    assert row["parameters"]["ewma_span_hours"] == 720


def test_complete_persisted_gate_promotes_and_typed_signal_rolls_back(
    tmp_path: Path,
) -> None:
    audit = _repository(tmp_path)
    _candidate(audit)
    CounterfactualOutcomeScheduler(audit, HistoricalBars()).settle_due(as_of=NOW)
    registry = StrategyRegistry(tmp_path / "registry.json")
    scheduler = LearningPromotionScheduler(
        audit=audit,
        registry=registry,
        market_data=HistoricalBars(),
        researcher=ProposalResearcher(),
    )
    scheduler.tick(now=NOW, force_weekly=True)
    champion_row = audit.strategy_spec_by_version("trend-breakout-v1")
    challenger_row = audit.strategy_spec_by_version("trend-breakout-v2")
    assert champion_row is not None and challenger_row is not None
    _seed_complete_promotion_evidence(
        audit,
        champion_spec_id=str(champion_row["spec_id"]),
        challenger_spec_id=str(challenger_row["spec_id"]),
    )

    result = scheduler.tick(now=NOW + timedelta(days=1), force_daily=True)

    assert "trend-breakout-v2:PROMOTED" in result.promotion_statuses
    assert registry.champion.version == "trend-breakout-v2"
    assert scheduler.handle_rollback_signal(
        RollbackSignal.PERFORMANCE_DRIFT,
        observed_at=NOW + timedelta(days=1, minutes=1),
        detail="audited_rolling_window_detector",
    )
    assert registry.champion.version == "trend-breakout-v1"


def test_worker_strategy_uses_registry_champion_and_rate_limit_delay_is_bounded(
    tmp_path: Path,
) -> None:
    champion = StrategySpec(version="custom-champion", minimum_directional_votes=5)
    registry = StrategyRegistry(tmp_path / "registry.json", initial_champion=champion)

    strategy = build_champion_strategy(registry)

    assert strategy.spec == champion
    assert binance_rate_limit_delay(
        BinanceApiError("limited", status_code=429, retry_after_seconds=7)
    ) == 7
    assert binance_rate_limit_delay(BinanceApiError("banned", status_code=418)) == 300
    assert binance_rate_limit_delay(BinanceApiError("server", status_code=503)) is None


def test_startup_rate_limit_retries_in_process_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runtime:
        calls = 0

        def startup_check(self) -> dict[str, bool]:
            self.calls += 1
            if self.calls == 1:
                raise BinanceApiError(
                    "limited", status_code=429, retry_after_seconds=9
                )
            return {"model_access_verified": True}

    observed_delays: list[float] = []

    async def no_wait(_stop: asyncio.Event, delay: float) -> None:
        observed_delays.append(delay)

    monkeypatch.setattr(worker_module, "_rate_limit_wait", no_wait)
    runtime = Runtime()

    async def scenario() -> dict[str, bool]:
        return await worker_module._startup_check_with_backoff(
            runtime, asyncio.Event()
        )

    assert asyncio.run(scenario()) == {"model_access_verified": True}
    assert runtime.calls == 2
    assert observed_delays == [9]


def test_control_watchdog_retries_until_cancel_and_reconcile_barrier_succeeds() -> None:
    class Control:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def snapshot(self) -> Any:
            return type("Snapshot", (), {"kill_switch_active": True})()

        def engage_kill_switch(self, reason: str) -> None:
            self.reasons.append(reason)

    class Runtime:
        def __init__(self, stop: asyncio.Event) -> None:
            self.control = Control()
            self.stop = stop
            self.calls = 0

        def cancel_all_entries(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("unknown cancellation state")
            self.stop.set()

    async def scenario() -> Runtime:
        stop = asyncio.Event()
        runtime = Runtime(stop)
        await asyncio.wait_for(
            worker_module._watch_control(runtime, stop, poll_seconds=0.001),
            timeout=1,
        )
        return runtime

    runtime = asyncio.run(scenario())

    assert runtime.calls == 2
    assert runtime.control.reasons == ["kill_switch_entry_cancel_unresolved"]


def test_committed_promotion_is_replayed_after_registry_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _repository(tmp_path)
    _candidate(audit)
    CounterfactualOutcomeScheduler(audit, HistoricalBars()).settle_due(as_of=NOW)
    registry_path = tmp_path / "registry-crash.json"
    registry = StrategyRegistry(registry_path)
    scheduler = LearningPromotionScheduler(
        audit=audit,
        registry=registry,
        market_data=HistoricalBars(),
        researcher=ProposalResearcher(),
    )
    scheduler.tick(now=NOW, force_weekly=True)
    champion_row = audit.strategy_spec_by_version("trend-breakout-v1")
    challenger_row = audit.strategy_spec_by_version("trend-breakout-v2")
    assert champion_row is not None and challenger_row is not None
    _seed_complete_promotion_evidence(
        audit,
        champion_spec_id=str(champion_row["spec_id"]),
        challenger_spec_id=str(challenger_row["spec_id"]),
    )

    def crash_after_commit(_record: object) -> None:
        raise RuntimeError("simulated registry persistence crash")

    monkeypatch.setattr(registry, "_apply_audited_record", crash_after_commit)
    failed = scheduler.tick(now=NOW + timedelta(days=1), force_daily=True)
    assert any("FAIL_CLOSED:PROMOTION" in item for item in failed.promotion_statuses)
    assert StrategyRegistry(registry_path).champion.version == "trend-breakout-v1"

    recovered_registry = StrategyRegistry(registry_path)
    LearningPromotionScheduler(
        audit=audit,
        registry=recovered_registry,
        market_data=HistoricalBars(),
    )
    assert recovered_registry.champion.version == "trend-breakout-v2"


def test_audited_rollback_is_replayed_after_registry_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _repository(tmp_path)
    _candidate(audit)
    CounterfactualOutcomeScheduler(audit, HistoricalBars()).settle_due(as_of=NOW)
    registry_path = tmp_path / "registry-rollback.json"
    registry = StrategyRegistry(registry_path)
    scheduler = LearningPromotionScheduler(
        audit=audit,
        registry=registry,
        market_data=HistoricalBars(),
        researcher=ProposalResearcher(),
    )
    scheduler.tick(now=NOW, force_weekly=True)
    champion_row = audit.strategy_spec_by_version("trend-breakout-v1")
    challenger_row = audit.strategy_spec_by_version("trend-breakout-v2")
    assert champion_row is not None and challenger_row is not None
    _seed_complete_promotion_evidence(
        audit,
        champion_spec_id=str(champion_row["spec_id"]),
        challenger_spec_id=str(challenger_row["spec_id"]),
    )
    scheduler.tick(now=NOW + timedelta(days=1), force_daily=True)
    assert registry.champion.version == "trend-breakout-v2"

    def crash_after_rollback_audit(_record: object) -> None:
        raise RuntimeError("simulated rollback persistence crash")

    monkeypatch.setattr(registry, "_apply_audited_rollback", crash_after_rollback_audit)
    with pytest.raises(RuntimeError, match="rollback persistence crash"):
        scheduler.handle_rollback_signal(
            RollbackSignal.PERFORMANCE_DRIFT,
            observed_at=NOW + timedelta(days=1, minutes=1),
            detail="audited_detector",
        )
    trace = audit.get_trace("strategy_learning")
    assert len(trace["strategy_registry_events"]) == 1
    assert StrategyRegistry(registry_path).champion.version == "trend-breakout-v2"

    recovered_registry = StrategyRegistry(registry_path)
    LearningPromotionScheduler(
        audit=audit,
        registry=recovered_registry,
        market_data=HistoricalBars(),
    )
    assert recovered_registry.champion.version == "trend-breakout-v1"
    assert recovered_registry.promotion_records[-1].status.value == "rolled_back"


def test_authoritative_monitor_excludes_mixed_lineage_and_detects_mature_loss() -> None:
    exact = tuple(
        TradeOutcome(
            symbol="BTCUSDT",
            closed_at=NOW + timedelta(hours=index),
            gross_pnl=-1,
            episode_id=f"episode-{index}",
            trace_ids=(f"trace-{index}",),
            strategy_versions=("champion-v2",),
            source_record_ids=(f"fill-{index}",),
        )
        for index in range(30)
    )
    mixed = TradeOutcome(
        symbol="ETHUSDT",
        closed_at=NOW,
        gross_pnl=10_000,
        episode_id="mixed",
        trace_ids=("mixed-trace",),
        strategy_versions=("champion-v1", "champion-v2"),
        source_record_ids=("mixed-fill",),
    )

    class OutcomeAudit:
        def build_trade_outcomes(self, *, venue: str) -> tuple[TradeOutcome, ...]:
            assert venue == "binance_futures_demo"
            return (*exact, mixed)

    result = AuthoritativePerformanceMonitor(
        OutcomeAudit(),  # type: ignore[arg-type]
        venue="binance_futures_demo",
        min_trades=30,
    ).evaluate("champion-v2")

    assert result.status == "DRIFT"
    assert result.trade_count == 30
    assert result.excluded_trade_count == 1
    assert result.net_profit == -30
    not_ready = AuthoritativePerformanceMonitor(
        OutcomeAudit(),  # type: ignore[arg-type]
        venue="binance_futures_demo",
        min_trades=31,
    ).evaluate("champion-v2")
    assert not_ready.status == "NOT_READY"
    assert not_ready.net_profit is None
