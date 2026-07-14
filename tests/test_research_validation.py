from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.learning import TradeOutcome
from crypto_event_trader.research_validation import (
    AuditedShadowTrade,
    ExecutedBacktestTrade,
    ExpandingWalkForwardConfig,
    FundingPayment,
    PairedShadowAccumulator,
    ResearchBacktestValidator,
    ResearchValidationError,
    ScenarioRequest,
    StatisticalValidation,
    build_expanding_windows,
)


class _Evaluator:
    def __init__(self) -> None:
        self.requests: list[ScenarioRequest] = []
        self.sealed = False

    def evaluate(self, request: ScenarioRequest) -> tuple[ExecutedBacktestTrade, ...]:
        if request.window.kind == "SEALED_HOLDOUT":
            assert self.sealed is True
            assert request.holdout_seal_id == "sealed-revision-001"
        else:
            assert self.sealed is False
        self.requests.append(request)
        signal_at = request.window.test_started_at + timedelta(days=1)
        opened_at = signal_at + timedelta(minutes=request.scenario.execution_delay_minutes)
        closed_at = opened_at + timedelta(days=1)
        trade_id = f"{request.window.window_id}-{request.scenario.scenario_id}"
        return (
            ExecutedBacktestTrade(
                trade_id=trade_id,
                symbol="BTCUSDT",
                direction=1,
                quantity=1.0,
                signal_at=signal_at,
                information_cutoff_at=signal_at,
                opened_at=opened_at,
                closed_at=closed_at,
                entry_reference_price=100.0,
                entry_fill_price=100.1,
                exit_reference_price=110.0,
                exit_fill_price=109.9,
                entry_reference_available_at=opened_at,
                exit_reference_available_at=closed_at,
                entry_fee=0.5,
                exit_fee=0.5,
                funding_events=(
                    FundingPayment(
                        event_id=f"{trade_id}-funding",
                        effective_at=opened_at + timedelta(hours=4),
                        cost=0.2,
                    ),
                ),
                entry_fee_evidence_ids=(f"{trade_id}-entry-fee",),
                exit_fee_evidence_ids=(f"{trade_id}-exit-fee",),
                funding_coverage_id=f"{trade_id}-funding-window",
                source_ids=(f"{trade_id}-bars",),
                market_data_digest=f"digest-{trade_id}",
                fees_complete=True,
                funding_complete=True,
            ),
        )

    def freeze_for_holdout(self, *, strategy_digest: str, pre_holdout_results_digest: str) -> str:
        assert len(strategy_digest) == 64
        assert len(pre_holdout_results_digest) == 64
        self.sealed = True
        return "sealed-revision-001"


class _InvalidStatistics:
    def calculate(self, request: object) -> StatisticalValidation:
        return StatisticalValidation(
            dsr_significance_probability=0.99,
            pbo_probability=0.01,
            dsr_method="UNVERIFIED_DSR",
            pbo_method="UNVERIFIED_PBO",
            source_digest="stats-digest",
            observation_count=100,
            independent_trial_count=20,
            fold_count=5,
        )


def _config() -> ExpandingWalkForwardConfig:
    return ExpandingWalkForwardConfig(
        research_started_at=datetime(2020, 1, 1, tzinfo=UTC),
        research_ended_at=datetime(2024, 1, 1, tzinfo=UTC),
        initial_training_months=12,
        test_window_months=6,
    )


def _validator(
    evaluator: _Evaluator,
    *,
    statistical_validator: object | None = None,
) -> ResearchBacktestValidator:
    return ResearchBacktestValidator(
        spec_id="challenger-spec",
        trace_id="research-trace",
        strategy_parameters={
            "momentum_windows_1h": [24, 72, 168],
            "minimum_directional_votes": 3,
        },
        config=_config(),
        initial_equity=10_000,
        evaluator=evaluator,
        statistical_validator=statistical_validator,  # type: ignore[arg-type]
    )


def test_expanding_windows_reserve_exact_final_twelve_months() -> None:
    walks, holdout = build_expanding_windows(_config())

    assert len(walks) == 4
    assert {item.training_started_at for item in walks} == {datetime(2020, 1, 1, tzinfo=UTC)}
    assert [item.training_ended_at.year for item in walks] == [2021, 2021, 2022, 2022]
    assert holdout.test_started_at == datetime(2023, 1, 1, tzinfo=UTC)
    assert holdout.test_ended_at == datetime(2024, 1, 1, tzinfo=UTC)

    with pytest.raises(ResearchValidationError, match="INVALID_HOLDOUT_LENGTH"):
        replace(_config(), holdout_months=6)


def test_backtest_runs_full_matrix_then_seals_holdout_and_computes_real_costs() -> None:
    evaluator = _Evaluator()
    report = _validator(evaluator).run()

    assert len(evaluator.requests) == 35
    assert [item.scenario.execution_delay_minutes for item in evaluator.requests[:7]] == [
        0,
        0,
        0,
        1,
        5,
        15,
        0,
    ]
    assert [item.scenario.parameter_scale for item in evaluator.requests[:3]] == [
        1.0,
        0.75,
        1.25,
    ]
    assert evaluator.requests[6].scenario.social_placebo is True
    assert all(item.window.kind == "SEALED_HOLDOUT" for item in evaluator.requests[-7:])
    assert report.evidence.completed is True
    assert report.evidence.holdout_months == 12
    assert report.evidence.walk_forward_passed is True
    assert report.evidence.holdout_passed is True
    assert report.evidence.parameter_perturbation_passed is True
    assert report.evidence.latency_stress_passed is True
    assert report.evidence.social_placebo_passed is True
    assert report.evidence.metrics.net_profit == pytest.approx(5 * 8.6)
    assert report.evidence.metrics.total_cost == pytest.approx(5 * 1.4)
    assert report.evidence.metrics.max_drawdown == 0
    assert report.evidence.metrics.symbol_concentration == 1
    assert report.evidence.metrics.month_concentration == pytest.approx(0.2)
    assert report.raw_metrics["baseline_cost_stress_2x"]["net_profit"] == pytest.approx(5 * 7.2)
    assert report.raw_metrics["baseline_cost_stress_3x"]["net_profit"] == pytest.approx(5 * 5.8)
    assert len(report.audited_input_digest) == 64


def test_missing_or_future_inputs_fail_closed_before_evidence_exists() -> None:
    incomplete = _Evaluator()
    original = incomplete.evaluate

    def missing_cost(request: ScenarioRequest) -> tuple[ExecutedBacktestTrade, ...]:
        return (replace(original(request)[0], funding_complete=False),)

    incomplete.evaluate = missing_cost  # type: ignore[method-assign]
    with pytest.raises(ResearchValidationError, match="INCOMPLETE_COST_ACCOUNTING"):
        _validator(incomplete).run()

    future = _Evaluator()
    original_future = future.evaluate

    def leaked(request: ScenarioRequest) -> tuple[ExecutedBacktestTrade, ...]:
        raw = original_future(request)[0]
        return (replace(raw, information_cutoff_at=raw.signal_at + timedelta(seconds=1)),)

    future.evaluate = leaked  # type: ignore[method-assign]
    with pytest.raises(ResearchValidationError, match="FUTURE_FEATURE_LEAKAGE"):
        _validator(future).run()

    duplicated = _Evaluator()
    original_duplicated = duplicated.evaluate

    def duplicate_across_windows(
        request: ScenarioRequest,
    ) -> tuple[ExecutedBacktestTrade, ...]:
        raw = original_duplicated(request)[0]
        return (replace(raw, trade_id=f"duplicate-{request.scenario.scenario_id}"),)

    duplicated.evaluate = duplicate_across_windows  # type: ignore[method-assign]
    with pytest.raises(ResearchValidationError, match="CROSS_WINDOW_DUPLICATE_TRADE_ID"):
        _validator(duplicated).run()


def test_invalid_or_missing_dsr_and_pbo_are_none_and_persist_fail_closed(
    tmp_path: Path,
) -> None:
    report = _validator(_Evaluator(), statistical_validator=_InvalidStatistics()).run()
    assert report.evidence.dsr_significance_probability is None
    assert report.evidence.pbo_probability is None
    assert report.raw_metrics["statistics"]["status"] == "INVALID_FAIL_CLOSED"

    audit = AuditRepository(tmp_path / "audit.db")
    audit.initialize()
    spec_id = audit.append_strategy_spec(
        spec_id="challenger-spec",
        trace_id="research-trace",
        strategy_version="challenger-v2",
        parent_version="champion-v1",
        status="CHALLENGER",
        parameters={"minimum_directional_votes": 3},
        prompt_version="research-v1",
    )
    assert spec_id == "challenger-spec"
    backtest_id = report.append_backtest_run(audit)
    row = audit.get_trace("research-trace")["backtest_runs"][0]
    assert row["backtest_run_id"] == backtest_id
    assert row["dsr_significance_probability"] is None
    assert row["pbo_probability"] is None
    assert row["raw_metrics"]["input_summary"]["sealed_holdout"]["window_id"] == ("holdout-12m")
    assert report.append_backtest_run(audit) == backtest_id
    assert len(audit.get_trace("research-trace")["backtest_runs"]) == 1


def _create_shadow_ledger(tmp_path: Path) -> tuple[AuditRepository, str, str]:
    audit = AuditRepository(tmp_path / "shadow.db")
    audit.initialize()
    trace_id = "paired-shadow"
    champion_id = audit.append_strategy_spec(
        trace_id=trace_id,
        spec_id="champion-spec",
        strategy_version="champion-v1",
        status="CHAMPION",
        parameters={"minimum_directional_votes": 3},
        prompt_version="research-v1",
    )
    challenger_id = audit.append_strategy_spec(
        trace_id=trace_id,
        spec_id="challenger-spec",
        strategy_version="challenger-v2",
        parent_version="champion-v1",
        status="CHALLENGER",
        parameters={"minimum_directional_votes": 4},
        prompt_version="research-v2",
    )
    return audit, champion_id, challenger_id


def _append_shadow_cost_evidence(
    audit: AuditRepository,
    trade: AuditedShadowTrade,
    *,
    fee_payload_override: dict[str, Any] | None = None,
    ordinary: bool = False,
    record_trace_id: str | None = None,
) -> None:
    trace_id = trade.outcome.trace_ids[0]
    records = (
        (trade.fee_evidence_id, "FEE", trade.outcome.fees),
        (trade.slippage_evidence_id, "SLIPPAGE", trade.outcome.slippage_cost),
        (trade.funding_evidence_id, "FUNDING", trade.outcome.funding_cost),
    )
    for index, (record_id, cost_type, amount) in enumerate(records):
        payload: dict[str, Any] = {
            "schema": "paired-shadow-cost-v1",
            "cost_type": cost_type,
            "trade_id": trade.trade_id,
            "episode_id": trade.outcome.episode_id,
            "trace_id": trace_id,
            "symbol": trade.outcome.symbol,
            "strategy_version": trade.outcome.strategy_versions[0],
            "closed_at": trade.outcome.closed_at.isoformat(),
            "amount": amount,
        }
        if ordinary:
            payload = {"manifest_type": record_id}
        elif index == 0 and fee_payload_override:
            payload.update(fee_payload_override)
        audit.append_external_evidence(
            trace_id=record_trace_id if index == 0 and record_trace_id else trace_id,
            source="signed_shadow_manifest",
            source_id=record_id,
            evidence_id=f"shadow-cost:{record_id}",
            evidence_record_id=record_id,
            occurred_at=trade.outcome.closed_at,
            first_observed_at=trade.outcome.closed_at,
            payload=payload,
            created_at=trade.outcome.closed_at,
        )


def _shadow_trade(prefix: str, number: int, closed_at: datetime) -> AuditedShadowTrade:
    return AuditedShadowTrade(
        trade_id=f"{prefix}-{number}",
        outcome=TradeOutcome(
            symbol="BTCUSDT" if number % 2 else "ETHUSDT",
            closed_at=closed_at,
            gross_pnl=10.0,
            fees=1.0,
            slippage_cost=0.5,
            funding_cost=0.25,
        ),
        fee_evidence_id=f"fee-{prefix}-{number}",
        slippage_evidence_id=f"slippage-{prefix}-{number}",
        funding_evidence_id=f"funding-{prefix}-{number}",
        accounting_complete=True,
    )


def _durable_shadow_trade(
    prefix: str,
    number: int,
    closed_at: datetime,
    *,
    strategy_version: str,
) -> AuditedShadowTrade:
    trade = _shadow_trade(prefix, number, closed_at)
    cost_source_ids = (
        f"{trade.trade_id}-fee-record",
        f"{trade.trade_id}-slippage-record",
        f"{trade.trade_id}-funding-record",
    )
    trade = replace(
        trade,
        fee_evidence_id=cost_source_ids[0],
        slippage_evidence_id=cost_source_ids[1],
        funding_evidence_id=cost_source_ids[2],
    )
    return replace(
        trade,
        outcome=replace(
            trade.outcome,
            episode_id=f"episode-{prefix}-{number}",
            trace_ids=("paired-shadow",),
            strategy_versions=(strategy_version,),
            source_record_ids=cost_source_ids,
        ),
    )


def test_paired_shadow_only_appends_after_90_days_30_trades_and_daily_coverage(
    tmp_path: Path,
) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
    )
    for day in range(91):
        accumulator.record_daily_coverage(
            observed_at=started + timedelta(days=day),
            evidence_id=f"journal-{day}",
        )
    for number in range(30):
        closed_at = started + timedelta(days=number + 1)
        champion_trade = _durable_shadow_trade(
            "champion",
            number,
            closed_at,
            strategy_version="champion-v1",
        )
        challenger_trade = _durable_shadow_trade(
            "challenger",
            number,
            closed_at,
            strategy_version="challenger-v2",
        )
        _append_shadow_cost_evidence(audit, champion_trade)
        _append_shadow_cost_evidence(audit, challenger_trade)
        accumulator.record_trade(
            spec_id=champion_id,
            trade=champion_trade,
        )
        accumulator.record_trade(
            spec_id=challenger_id,
            trade=challenger_trade,
        )

    immature = accumulator.append_if_mature(audit, ended_at=started + timedelta(days=89))
    assert immature.appended is False
    assert "INSUFFICIENT_SHADOW_DAYS" in immature.reason_codes
    assert audit.get_trace("paired-shadow")["shadow_results"] == []

    mature = accumulator.append_if_mature(audit, ended_at=started + timedelta(days=90))
    assert mature.appended is True
    rows = audit.get_trace("paired-shadow")["shadow_results"]
    assert len(rows) == 2
    assert {row["elapsed_days"] for row in rows} == {90}
    assert {row["closed_trades"] for row in rows} == {30}
    assert rows[0]["raw_metrics"]["input_summary"]["coverage_day_count"] == 91

    duplicate = accumulator.append_if_mature(audit, ended_at=started + timedelta(days=90))
    assert duplicate.appended is False
    assert duplicate.reason_codes == ("ALREADY_APPENDED",)
    assert len(audit.get_trace("paired-shadow")["shadow_results"]) == 2


def test_in_memory_accumulator_cannot_bypass_audited_cost_binding(tmp_path: Path) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
    )
    for day in range(91):
        accumulator.record_daily_coverage(
            observed_at=started + timedelta(days=day),
            evidence_id=f"unbound-coverage-{day}",
        )
    for number in range(30):
        closed_at = started + timedelta(days=number + 1)
        accumulator.record_trade(
            spec_id=champion_id,
            trade=_shadow_trade("unbound-champion", number, closed_at),
        )
        accumulator.record_trade(
            spec_id=challenger_id,
            trade=_shadow_trade("unbound-challenger", number, closed_at),
        )

    with pytest.raises(ResearchValidationError, match="INCOMPLETE_SHADOW_LINEAGE"):
        accumulator.append_if_mature(audit, ended_at=started + timedelta(days=90))
    assert audit.get_trace("paired-shadow")["shadow_results"] == []


def test_shadow_missing_daily_coverage_or_exact_cost_evidence_never_appends(
    tmp_path: Path,
) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
    )
    broken = replace(
        _shadow_trade("broken", 1, started + timedelta(days=1)),
        accounting_complete=False,
    )
    with pytest.raises(ResearchValidationError, match="INCOMPLETE_SHADOW_ACCOUNTING"):
        accumulator.record_trade(spec_id=champion_id, trade=broken)

    for number in range(30):
        closed_at = started + timedelta(days=number + 1)
        accumulator.record_trade(
            spec_id=champion_id,
            trade=_shadow_trade("champion", number, closed_at),
        )
        accumulator.record_trade(
            spec_id=challenger_id,
            trade=_shadow_trade("challenger", number, closed_at),
        )
    result = accumulator.append_if_mature(audit, ended_at=started + timedelta(days=90))
    assert result.appended is False
    assert result.reason_codes == ("INCOMPLETE_DAILY_SHADOW_COVERAGE",)
    assert audit.get_trace("paired-shadow")["shadow_results"] == []


def test_paired_shadow_journal_rehydrates_after_restart(tmp_path: Path) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
        audit=audit,
    )
    for day in range(91):
        accumulator.record_daily_coverage(
            observed_at=started + timedelta(days=day),
            evidence_id=f"durable-journal-{day}",
        )
    for number in range(30):
        closed_at = started + timedelta(days=number + 1)
        champion_trade = _durable_shadow_trade(
            "durable-champion",
            number,
            closed_at,
            strategy_version="champion-v1",
        )
        challenger_trade = _durable_shadow_trade(
            "durable-challenger",
            number,
            closed_at,
            strategy_version="challenger-v2",
        )
        _append_shadow_cost_evidence(audit, champion_trade)
        _append_shadow_cost_evidence(audit, challenger_trade)
        accumulator.record_trade(
            spec_id=champion_id,
            trade=champion_trade,
        )
        accumulator.record_trade(
            spec_id=challenger_id,
            trade=challenger_trade,
        )
    assert len(audit.shadow_journal_events(accumulator.pair_id)) == 151

    restarted = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
        audit=audit,
    )
    result = restarted.append_if_mature(
        audit,
        ended_at=started + timedelta(days=90),
    )

    assert result.appended is True
    assert len(audit.get_trace("paired-shadow")["shadow_results"]) == 2


def test_durable_shadow_rejects_unverifiable_lineage_and_cost_ids(tmp_path: Path) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
        audit=audit,
    )
    with pytest.raises(ResearchValidationError, match="INCOMPLETE_SHADOW_LINEAGE"):
        accumulator.record_trade(
            spec_id=champion_id,
            trade=_shadow_trade("missing-lineage", 1, started + timedelta(days=1)),
        )

    missing_cost_source = _durable_shadow_trade(
        "missing-cost-source",
        2,
        started + timedelta(days=2),
        strategy_version="champion-v1",
    )
    missing_cost_source = replace(
        missing_cost_source,
        outcome=replace(
            missing_cost_source.outcome,
            source_record_ids=(
                missing_cost_source.fee_evidence_id,
                missing_cost_source.slippage_evidence_id,
            ),
        ),
    )
    with pytest.raises(ResearchValidationError, match="UNVERIFIED_SHADOW_COST_EVIDENCE"):
        accumulator.record_trade(spec_id=champion_id, trade=missing_cost_source)

    fake_sources = _durable_shadow_trade(
        "fake-sources",
        4,
        started + timedelta(days=4),
        strategy_version="champion-v1",
    )
    fake_sources = replace(
        fake_sources,
        fee_evidence_id="fake-fee",
        slippage_evidence_id="fake-slippage",
        funding_evidence_id="fake-funding",
        outcome=replace(
            fake_sources.outcome,
            source_record_ids=("fake-fee", "fake-slippage", "fake-funding"),
        ),
    )
    with pytest.raises(ResearchValidationError, match="UNVERIFIED_SHADOW_SOURCE_RECORD"):
        accumulator.record_trade(spec_id=champion_id, trade=fake_sources)

    wrong_strategy = _durable_shadow_trade(
        "wrong-strategy",
        3,
        started + timedelta(days=3),
        strategy_version="challenger-v2",
    )
    with pytest.raises(ResearchValidationError, match="SHADOW_STRATEGY_LINEAGE_MISMATCH"):
        accumulator.record_trade(spec_id=champion_id, trade=wrong_strategy)

    assert audit.shadow_journal_events(accumulator.pair_id) == []


def test_durable_shadow_accepts_fully_bound_typed_cost_records(tmp_path: Path) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    trade = _durable_shadow_trade(
        "typed-positive",
        1,
        started + timedelta(days=1),
        strategy_version="champion-v1",
    )
    _append_shadow_cost_evidence(audit, trade)
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
        audit=audit,
    )

    accumulator.record_trade(spec_id=champion_id, trade=trade)

    events = audit.shadow_journal_events(accumulator.pair_id)
    assert len(events) == 1
    assert events[0]["event_key"] == f"{champion_id}:{trade.trade_id}"


def test_durable_shadow_requires_three_distinct_cost_record_ids(tmp_path: Path) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    trade = _durable_shadow_trade(
        "duplicate-cost-id",
        1,
        started + timedelta(days=1),
        strategy_version="champion-v1",
    )
    trade = replace(
        trade,
        slippage_evidence_id=trade.fee_evidence_id,
        outcome=replace(
            trade.outcome,
            source_record_ids=(trade.fee_evidence_id, trade.funding_evidence_id),
        ),
    )
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
        audit=audit,
    )

    with pytest.raises(ResearchValidationError, match="DUPLICATE_SHADOW_COST_EVIDENCE"):
        accumulator.record_trade(spec_id=champion_id, trade=trade)


def test_ordinary_external_evidence_cannot_authorize_shadow_costs(tmp_path: Path) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    trade = _durable_shadow_trade(
        "legacy-ordinary",
        1,
        started + timedelta(days=1),
        strategy_version="champion-v1",
    )
    _append_shadow_cost_evidence(audit, trade, ordinary=True)
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
        audit=audit,
    )

    with pytest.raises(ResearchValidationError, match="UNVERIFIED_SHADOW_COST_EVIDENCE"):
        accumulator.record_trade(spec_id=champion_id, trade=trade)
    assert audit.shadow_journal_events(accumulator.pair_id) == []


@pytest.mark.parametrize(
    ("field", "bad_value", "reason_code"),
    (
        ("schema", "external-evidence-v1", "UNVERIFIED_SHADOW_COST_EVIDENCE"),
        ("cost_type", "FUNDING", "UNVERIFIED_SHADOW_COST_EVIDENCE"),
        ("trade_id", "another-trade", "SHADOW_COST_EVIDENCE_BINDING_MISMATCH"),
        ("episode_id", "another-episode", "SHADOW_COST_EVIDENCE_BINDING_MISMATCH"),
        ("trace_id", "another-trace", "SHADOW_COST_EVIDENCE_BINDING_MISMATCH"),
        ("symbol", "SOLUSDT", "SHADOW_COST_EVIDENCE_BINDING_MISMATCH"),
        ("strategy_version", "another-version", "SHADOW_COST_EVIDENCE_BINDING_MISMATCH"),
        ("closed_at", "2026-01-03T00:00:00+00:00", "SHADOW_COST_EVIDENCE_BINDING_MISMATCH"),
        ("amount", 99.0, "SHADOW_COST_EVIDENCE_BINDING_MISMATCH"),
        ("amount", 1.0000000000005, "SHADOW_COST_EVIDENCE_BINDING_MISMATCH"),
    ),
)
def test_shadow_cost_record_fields_are_semantically_bound(
    tmp_path: Path,
    field: str,
    bad_value: Any,
    reason_code: str,
) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    trade = _durable_shadow_trade(
        f"wrong-{field}",
        1,
        started + timedelta(days=1),
        strategy_version="champion-v1",
    )
    _append_shadow_cost_evidence(
        audit,
        trade,
        fee_payload_override={field: bad_value},
    )
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
        audit=audit,
    )

    with pytest.raises(ResearchValidationError, match=reason_code):
        accumulator.record_trade(spec_id=champion_id, trade=trade)
    assert audit.shadow_journal_events(accumulator.pair_id) == []


def test_shadow_cost_record_requires_matching_audit_trace(tmp_path: Path) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    trade = _durable_shadow_trade(
        "wrong-audit-trace",
        1,
        started + timedelta(days=1),
        strategy_version="champion-v1",
    )
    _append_shadow_cost_evidence(audit, trade, record_trace_id="fabricated-trace")
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
        audit=audit,
    )

    with pytest.raises(
        ResearchValidationError,
        match="SHADOW_COST_EVIDENCE_BINDING_MISMATCH",
    ):
        accumulator.record_trade(spec_id=champion_id, trade=trade)


def test_shadow_trade_trace_must_preexist_in_non_evidence_audit_facts(
    tmp_path: Path,
) -> None:
    audit, champion_id, challenger_id = _create_shadow_ledger(tmp_path)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    trade = _durable_shadow_trade(
        "fabricated-trace",
        1,
        started + timedelta(days=1),
        strategy_version="champion-v1",
    )
    trade = replace(
        trade,
        outcome=replace(trade.outcome, trace_ids=("external-evidence-only-trace",)),
    )
    _append_shadow_cost_evidence(audit, trade)
    accumulator = PairedShadowAccumulator(
        trace_id="paired-shadow",
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        started_at=started,
        initial_equity=10_000,
        audit=audit,
    )

    with pytest.raises(ResearchValidationError, match="UNVERIFIED_SHADOW_TRACE"):
        accumulator.record_trade(spec_id=champion_id, trade=trade)
