from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_event_trader.audit import TRACE_TABLES, AuditRepository
from crypto_event_trader.migrations import MIGRATIONS

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


def _repository(tmp_path: Path) -> AuditRepository:
    path = tmp_path / "audit.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE legacy_records (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO legacy_records(value) VALUES ('preserved')")
    repository = AuditRepository(f"sqlite:///{path}")
    repository.initialize()
    repository.initialize()
    return repository


def _complete_order_trace(
    repository: AuditRepository, trace_id: str = "trace-001"
) -> dict[str, str]:
    evidence_id = "binance:kline:BTCUSDT:2026-07-14T11:00:00Z"
    evidence_record_id = repository.append_external_evidence(
        trace_id=trace_id,
        source="binance",
        source_id="kline:BTCUSDT:2026-07-14T11:00:00Z",
        source_url="https://fapi.binance.com/fapi/v1/klines",
        first_observed_at=NOW,
        occurred_at=NOW,
        payload={"symbol": "BTCUSDT", "interval": "1h", "closed": True},
        created_at=NOW,
    )
    candidate_id = repository.append_trade_candidate(
        trace_id=trace_id,
        strategy_version="champion-1",
        symbol="BTCUSDT",
        direction="LONG",
        max_quantity=0.01,
        max_risk_fraction=0.0075,
        feature_snapshot={"momentum_24h": 0.03, "bar_close": "2026-07-14T11:00:00Z"},
        evidence_ids=[evidence_id],
        evidence_record_ids=[evidence_record_id],
        valid_until=NOW + timedelta(seconds=120),
        created_at=NOW,
    )
    decision_id = repository.append_llm_decision(
        trace_id=trace_id,
        candidate_id=candidate_id,
        action="OPEN",
        direction="LONG",
        position_multiplier=0.5,
        confidence=0.82,
        evidence_ids=[evidence_id],
        thesis="Trend and breakout votes agree.",
        invalidation_conditions=["three votes no longer agree"],
        next_review_at=NOW + timedelta(minutes=15),
        model="decision-model",
        prompt_version="trade-v1",
        response_id="resp-1",
        latency_ms=210,
        raw_response={"action": "OPEN"},
        created_at=NOW,
    )
    thesis_id = repository.append_position_thesis(
        trace_id=trace_id,
        position_id="BTCUSDT:ONE_WAY",
        decision_id=decision_id,
        entry_reason="Three momentum/breakout votes agree.",
        expected_horizon="1d-7d",
        supporting_evidence=[evidence_id],
        opposing_evidence=["funding elevated"],
        add_count=0,
        pnl_r=0.0,
        invalidation_conditions=["close below breakout"],
        created_at=NOW,
    )
    risk_id = repository.append_risk_decision(
        trace_id=trace_id,
        candidate_id=candidate_id,
        decision_id=decision_id,
        outcome="RESIZE",
        approved_quantity=0.005,
        reason_codes=["GPT_SIZE_MULTIPLIER"],
        limits_snapshot={"single_position_risk": 0.0075, "gross_leverage": 0.5},
        created_at=NOW,
    )
    order_id = repository.append_venue_order(
        trace_id=trace_id,
        candidate_id=candidate_id,
        decision_id=decision_id,
        risk_decision_id=risk_id,
        venue="binance-futures-demo",
        client_order_id="bot-trace-001-entry-1",
        external_order_id="10001",
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT",
        quantity=0.005,
        price=65000,
        reduce_only=False,
        status="FILLED",
        raw_response={"status": "FILLED"},
        observed_at=NOW,
        created_at=NOW,
    )
    fill_id = repository.append_venue_fill(
        trace_id=trace_id,
        venue_order_id=order_id,
        external_fill_id="fill-10001-1",
        price=64995,
        quantity=0.005,
        fee=0.13,
        fee_asset="USDT",
        filled_at=NOW,
        raw_response={"maker": True},
        created_at=NOW,
    )
    repository.append_account_snapshot(
        trace_id=trace_id,
        equity=10_000,
        cash=9_675,
        gross_exposure=325,
        net_exposure=325,
        daily_pnl=-0.13,
        drawdown=0.000013,
        positions=[{"symbol": "BTCUSDT", "quantity": 0.005}],
        source="binance-rest-reconciliation",
        observed_at=NOW,
        created_at=NOW,
    )
    outcome_ids = []
    for horizon, realized_return in ((1, 0.002), (4, 0.003), (24, 0.01)):
        outcome_ids.append(
            repository.append_counterfactual_outcome(
                trace_id=trace_id,
                candidate_id=candidate_id,
                decision_id=decision_id,
                horizon_hours=horizon,
                realized_return=realized_return,
                decision_regret=0.0,
                confidence_calibration_error=0.18,
                source_reliability={"binance_market": 1.0},
                observed_at=NOW + timedelta(hours=horizon),
                created_at=NOW + timedelta(hours=horizon),
            )
        )
    return {
        "candidate_id": candidate_id,
        "decision_id": decision_id,
        "thesis_id": thesis_id,
        "risk_id": risk_id,
        "order_id": order_id,
        "fill_id": fill_id,
        "outcome_id": outcome_ids[0],
        "evidence_record_id": evidence_record_id,
    }


def test_protective_event_recovery_limit_keeps_newest_lifecycle_facts(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    ids = _complete_order_trace(repository)
    lifecycle = (
        ("PROTECTIVE_CREATED", "ACTIVE"),
        ("PROTECTIVE_CONSUMED", "FILLED"),
        ("PROTECTIVE_CREATED", "ACTIVE"),
    )
    for index, (event_type, status) in enumerate(lifecycle, start=1):
        timestamp = NOW + timedelta(seconds=index)
        repository.append_venue_order_event(
            trace_id="trace-001",
            venue_order_id=ids["order_id"],
            event_type=event_type,
            status=status,
            source_event_id=f"protective-lifecycle-{index}",
            observed_at=timestamp,
            created_at=timestamp,
        )

    events = repository.list_protective_order_events(
        venue="binance-futures-demo", limit=2
    )

    assert [event["source_event_id"] for event in events] == [
        "protective-lifecycle-2",
        "protective-lifecycle-3",
    ]


def test_versioned_schema_preserves_legacy_tables_and_is_idempotent(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    with repository.connect() as connection:
        existing_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        migration_versions = connection.execute(
            "SELECT version, checksum FROM crypto_schema_migrations ORDER BY version"
        ).fetchall()
        legacy = connection.execute("SELECT value FROM legacy_records").fetchone()

    assert set(TRACE_TABLES) <= existing_tables
    for table in TRACE_TABLES:
        assert f"deny_update_{table}" in triggers
        assert f"deny_delete_{table}" in triggers
    assert legacy[0] == "preserved"
    assert [row[0] for row in migration_versions] == [migration.version for migration in MIGRATIONS]
    assert all(row[1] for row in migration_versions)


def test_ensure_external_evidence_reuses_exact_payload_and_versions_changes(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    common = {
        "source": "binance_futures",
        "source_id": "BTCUSDT:klines:1h",
        "evidence_id": "binance:BTCUSDT:klines:1h:closed-window",
        "occurred_at": NOW,
        "first_observed_at": NOW,
        "created_at": NOW,
    }

    first = repository.ensure_external_evidence(
        **common,
        payload={"bars": [{"close": 100.0}]},
    )
    duplicate = repository.ensure_external_evidence(
        **common,
        payload={"bars": [{"close": 100.0}]},
    )
    changed = repository.ensure_external_evidence(
        **common,
        payload={"bars": [{"close": 101.0}]},
    )

    assert duplicate["evidence_record_id"] == first["evidence_record_id"]
    assert duplicate["version"] == 1
    assert changed["evidence_record_id"] != first["evidence_record_id"]
    assert changed["version"] == 2
    assert changed["prior_evidence_record_id"] == first["evidence_record_id"]


def test_append_only_audit_and_complete_order_trace(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    ids = _complete_order_trace(repository)

    trace = repository.trace_for_order(ids["order_id"])
    complete, missing = repository.validate_order_trace(ids["order_id"], require_outcome=True)

    assert complete is True
    assert missing == ()
    assert trace["trace_id"] == "trace-001"
    assert trace["trade_candidates"][0]["feature_snapshot"]["momentum_24h"] == 0.03
    assert trace["llm_decisions"][0]["raw_response"] == {"action": "OPEN"}
    assert trace["venue_orders"][0]["reduce_only"] is False
    assert trace["venue_order_events"][0]["event_type"] == "ORDER_RECORDED"
    assert {item["horizon_hours"] for item in trace["counterfactual_outcomes"]} == {
        1,
        4,
        24,
    }
    assert trace["linked_external_evidence"][0]["content_hash"]
    assert trace["linked_external_evidence"][0]["payload"]["closed"] is True

    submit_event = repository.append_venue_order_event(
        trace_id="trace-001",
        venue_order_id=ids["order_id"],
        event_type="submit_attempt",
        status="pending_submit",
        source_event_id="local-submit-1",
        external_order_id="10001",
        observed_at=NOW + timedelta(milliseconds=10),
    )
    assert (
        repository.append_venue_order_event(
            trace_id="trace-001",
            venue_order_id=ids["order_id"],
            event_type="submit_attempt",
            status="pending_submit",
            source_event_id="local-submit-1",
            observed_at=NOW + timedelta(milliseconds=10),
        )
        == submit_event
    )
    repository.append_venue_order_event(
        trace_id="trace-001",
        venue_order_id=ids["order_id"],
        event_type="acknowledged",
        status="new",
        source_event_id="binance-order-update-1",
        external_order_id="10001",
        executed_quantity=0,
        observed_at=NOW + timedelta(milliseconds=100),
        raw_response={"X": "NEW"},
    )
    latest = repository.latest_order_event(ids["order_id"])
    assert latest["event_sequence"] == 3
    assert latest["event_type"] == "ACKNOWLEDGED"
    assert latest["raw_response"] == {"X": "NEW"}
    assert repository.list_traces(limit=1)[0]["trace_id"] == "trace-001"

    stable_evidence_id = "binance:kline:BTCUSDT:2026-07-14T11:00:00Z"
    tombstone_id = repository.append_external_evidence(
        trace_id="trace-001",
        evidence_id=stable_evidence_id,
        source="binance",
        source_id="kline:BTCUSDT:2026-07-14T11:00:00Z",
        first_observed_at=NOW,
        occurred_at=NOW,
        payload={},
        deleted_at=NOW + timedelta(hours=2),
        created_at=NOW + timedelta(hours=2),
    )
    latest_evidence = repository.latest_external_evidence(stable_evidence_id)
    assert latest_evidence["evidence_record_id"] == tombstone_id
    assert latest_evidence["version"] == 2
    assert latest_evidence["prior_evidence_record_id"] == ids["evidence_record_id"]
    assert latest_evidence["deleted_at"] == "2026-07-14T14:00:00Z"

    second_thesis = repository.append_position_thesis(
        trace_id="trace-001",
        position_id="BTCUSDT:ONE_WAY",
        decision_id=ids["decision_id"],
        entry_reason="Original thesis remains valid.",
        expected_horizon="1d-7d",
        supporting_evidence=["mark-price:2026-07-14T12:15:00Z"],
        opposing_evidence=[],
        add_count=0,
        pnl_r=0.4,
        invalidation_conditions=["close below breakout"],
        created_at=NOW + timedelta(minutes=15),
    )
    updated_trace = repository.get_trace("trace-001")
    versions = updated_trace["position_theses"]
    assert [item["version"] for item in versions] == [1, 2]
    assert versions[1]["prior_thesis_id"] == ids["thesis_id"]
    assert versions[1]["thesis_id"] == second_thesis
    assert [item["version"] for item in repository.position_thesis_history("BTCUSDT:ONE_WAY")] == [
        1,
        2,
    ]
    assert repository.latest_position_thesis("BTCUSDT:ONE_WAY")["version"] == 2
    repository.append_position_thesis(
        trace_id="trace-001",
        position_id="BTCUSDT:ONE_WAY",
        decision_id=ids["decision_id"],
        entry_reason="One profitable add was approved.",
        expected_horizon="1d-7d",
        supporting_evidence=["mark-price:2026-07-14T12:30:00Z"],
        opposing_evidence=[],
        add_count=1,
        pnl_r=1.1,
        invalidation_conditions=["close below breakout"],
        created_at=NOW + timedelta(minutes=30),
    )
    assert repository.latest_position_thesis("BTCUSDT:ONE_WAY")["add_count"] == 1
    with pytest.raises(ValueError, match="cannot decrease"):
        repository.append_position_thesis(
            trace_id="trace-001",
            position_id="BTCUSDT:ONE_WAY",
            decision_id=ids["decision_id"],
            entry_reason="Invalid state rollback.",
            expected_horizon="1d-7d",
            supporting_evidence=["mark-price:2026-07-14T12:45:00Z"],
            opposing_evidence=[],
            add_count=0,
            pnl_r=1.2,
            invalidation_conditions=["close below breakout"],
        )
    with pytest.raises(ValueError, match="integer 0 or 1"):
        repository.append_position_thesis(
            trace_id="trace-001",
            position_id="BTCUSDT:ONE_WAY",
            decision_id=ids["decision_id"],
            entry_reason="Invalid second add.",
            expected_horizon="1d-7d",
            supporting_evidence=["mark-price:2026-07-14T13:00:00Z"],
            opposing_evidence=[],
            add_count=2,
            pnl_r=1.3,
            invalidation_conditions=["close below breakout"],
        )

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with repository.connect() as connection:
            connection.execute(
                "UPDATE llm_decisions SET confidence=0.99 WHERE decision_id=?",
                (ids["decision_id"],),
            )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with repository.connect() as connection:
            connection.execute("DELETE FROM position_theses WHERE thesis_id=?", (ids["thesis_id"],))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with repository.connect() as connection:
            connection.execute(
                "UPDATE venue_orders SET status='CANCELED' WHERE venue_order_id=?",
                (ids["order_id"],),
            )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with repository.connect() as connection:
            connection.execute(
                "DELETE FROM venue_order_events WHERE order_event_id=?", (submit_event,)
            )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with repository.connect() as connection:
            connection.execute(
                "UPDATE external_evidence SET payload_json='{}' WHERE evidence_record_id=?",
                (ids["evidence_record_id"],),
            )


def test_trace_lineage_and_strategy_parameter_whitelist_fail_closed(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    ids = _complete_order_trace(repository, trace_id="trace-a")
    other_evidence = repository.append_external_evidence(
        trace_id="trace-b",
        source="binance",
        source_id="kline:ETHUSDT:1h",
        occurred_at=NOW,
        first_observed_at=NOW,
        payload={"symbol": "ETHUSDT", "interval": "1h"},
    )
    other_candidate = repository.append_trade_candidate(
        trace_id="trace-b",
        strategy_version="champion-1",
        symbol="ETHUSDT",
        direction="SHORT",
        max_quantity=0.1,
        max_risk_fraction=0.0075,
        feature_snapshot={"momentum_24h": -0.02},
        evidence_ids=["binance:kline:ETHUSDT:1h"],
        evidence_record_ids=[other_evidence],
        valid_until=NOW + timedelta(seconds=120),
    )

    with pytest.raises(ValueError, match="different trace"):
        repository.append_risk_decision(
            trace_id="trace-a",
            candidate_id=other_candidate,
            decision_id=ids["decision_id"],
            outcome="ALLOW",
            approved_quantity=0.01,
            reason_codes=[],
            limits_snapshot={},
        )

    with pytest.raises(ValueError, match="non-approved parameters"):
        repository.append_strategy_spec(
            strategy_version="unsafe-v1",
            status="CHALLENGER",
            parameters={"max_leverage": 100},
            prompt_version="research-v1",
        )

    with pytest.raises(ValueError, match="risk_scale"):
        repository.append_strategy_spec(
            strategy_version="unsafe-scale-v1",
            status="CHALLENGER",
            parameters={"risk_scale": 100},
            prompt_version="research-v1",
        )

    with pytest.raises(ValueError, match="increasing"):
        repository.append_strategy_spec(
            strategy_version="unsafe-window-v1",
            status="CHALLENGER",
            parameters={"momentum_windows_1h": [72, 24, 24]},
            prompt_version="research-v1",
        )

    spec_id = repository.append_strategy_spec(
        strategy_version="safe-v1",
        status="CHALLENGER",
        parameters={
            "momentum_windows_1h": [24, 72, 168],
            "donchian_windows_4h": [42, 126],
            "vote_threshold": 3,
            "risk_scale": 0.5,
        },
        prompt_version="research-v1",
    )
    assert spec_id.startswith("spec_")


def test_order_audit_rejects_semantically_invalid_execution_chain(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    ids = _complete_order_trace(repository)

    with pytest.raises(ValueError, match="ORDER_SYMBOL_MISMATCH") as error:
        repository.append_venue_order(
            trace_id="trace-001",
            candidate_id=ids["candidate_id"],
            decision_id=ids["decision_id"],
            risk_decision_id=ids["risk_id"],
            venue="binance-futures-demo",
            client_order_id="invalid-semantic-order",
            symbol="ETHUSDT",
            side="SELL",
            order_type="LIMIT",
            quantity=0.02,
            status="PREPARED",
            observed_at=NOW + timedelta(seconds=121),
        )
    assert "ENTRY_SIDE_DIRECTION_MISMATCH" in str(error.value)
    assert "ORDER_EXCEEDS_CANDIDATE_QUANTITY" in str(error.value)
    assert "ORDER_EXCEEDS_RISK_QUANTITY" in str(error.value)
    assert "CANDIDATE_EXPIRED" in str(error.value)

    rejected_risk = repository.append_risk_decision(
        trace_id="trace-001",
        candidate_id=ids["candidate_id"],
        decision_id=ids["decision_id"],
        outcome="REJECT",
        approved_quantity=0,
        reason_codes=["daily_loss_lock"],
        limits_snapshot={"trading_enabled": False},
    )
    with pytest.raises(ValueError, match="RISK_DID_NOT_APPROVE_ORDER"):
        repository.append_venue_order(
            trace_id="trace-001",
            candidate_id=ids["candidate_id"],
            decision_id=ids["decision_id"],
            risk_decision_id=rejected_risk,
            venue="binance-futures-demo",
            client_order_id="invalid-rejected-order",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.001,
            status="PREPARED",
            observed_at=NOW,
        )

    attempted_reversal = repository.append_llm_decision(
        trace_id="trace-001",
        candidate_id=ids["candidate_id"],
        action="REJECT",
        direction="SHORT",
        position_multiplier=0,
        confidence=0.9,
        evidence_ids=["binance:kline:BTCUSDT:2026-07-14T11:00:00Z"],
        thesis="Attempted reversal retained for audit.",
        invalidation_conditions=["candidate direction mismatch"],
        model="decision-model",
        prompt_version="trade-v1",
    )
    decision = next(
        item
        for item in repository.get_trace("trace-001")["llm_decisions"]
        if item["decision_id"] == attempted_reversal
    )
    assert decision["direction"] == "SHORT"


def test_relative_sqlite_url_and_promotion_evidence_are_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    repository = AuditRepository("sqlite:///nested/audit.db")
    repository.initialize()
    trace_id = "promotion-trace"
    champion_id = repository.append_strategy_spec(
        trace_id=trace_id,
        strategy_version="champion-v1",
        status="CHAMPION",
        parameters={"vote_threshold": 3},
        prompt_version="research-v1",
    )
    challenger_id = repository.append_strategy_spec(
        trace_id=trace_id,
        strategy_version="challenger-v2",
        status="CHALLENGER",
        parent_version="champion-v1",
        parameters={"vote_threshold": 4},
        prompt_version="research-v2",
    )
    backtest_id = repository.append_backtest_run(
        trace_id=trace_id,
        spec_id=challenger_id,
        started_at="2025-01-01T00:00:00Z",
        ended_at="2026-06-30T00:00:00Z",
        completed=True,
        net_profit=2500,
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
    champion_shadow_id = repository.append_shadow_result(
        trace_id=trace_id,
        spec_id=champion_id,
        started_at="2026-04-01T00:00:00Z",
        ended_at="2026-07-01T00:00:00Z",
        completed=True,
        elapsed_days=91,
        closed_trades=60,
        net_return=0.10,
        max_drawdown=0.12,
        stressed_net_return_2x=0.06,
        symbol_concentration=0.30,
        month_concentration=0.30,
    )
    challenger_shadow_id = repository.append_shadow_result(
        trace_id=trace_id,
        spec_id=challenger_id,
        started_at="2026-04-01T00:00:00Z",
        ended_at="2026-07-01T00:00:00Z",
        completed=True,
        elapsed_days=91,
        closed_trades=30,
        net_return=0.12,
        max_drawdown=0.10,
        stressed_net_return_2x=0.08,
        symbol_concentration=0.30,
        month_concentration=0.30,
    )
    promotion_id = repository.append_promotion_record(
        trace_id=trace_id,
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        backtest_run_id=backtest_id,
        champion_shadow_result_id=champion_shadow_id,
        challenger_shadow_result_id=challenger_shadow_id,
        eligible=True,
        reason_codes=[],
        evaluation={"required_return": 0.11, "observed_return": 0.12},
    )

    assert (tmp_path / "nested" / "audit.db").exists()
    trace = repository.get_trace(trace_id)
    assert trace["promotion_records"][0]["promotion_record_id"] == promotion_id
    assert trace["promotion_records"][0]["eligible"] is True
    assert trace["promotion_records"][0]["evaluation"]["gate"]["eligible"] is True
    assert trace["backtest_runs"][0]["dsr_significance_probability"] == 0.96
    assert trace["shadow_results"][1]["elapsed_days"] == 91

    with pytest.raises(ValueError, match="deterministic gate"):
        repository.append_promotion_record(
            trace_id=trace_id,
            champion_spec_id=champion_id,
            challenger_spec_id=challenger_id,
            backtest_run_id=backtest_id,
            champion_shadow_result_id=champion_shadow_id,
            challenger_shadow_result_id=challenger_shadow_id,
            eligible=False,
            reason_codes=[],
            evaluation={},
        )

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with repository.connect() as connection:
            connection.execute(
                "UPDATE promotion_records SET eligible=0 WHERE promotion_record_id=?",
                (promotion_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with repository.connect() as connection:
            connection.execute(
                "UPDATE backtest_runs SET dsr_significance_probability=0.10 "
                "WHERE backtest_run_id=?",
                (backtest_id,),
            )


def test_account_risk_state_survives_restart_without_resetting_daily_baselines(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    def snapshot(equity: float, observed_at: datetime, source: str = "binance") -> None:
        repository.append_account_snapshot(
            equity=equity,
            cash=equity,
            gross_exposure=0,
            net_exposure=0,
            daily_pnl=0,
            drawdown=0,
            positions=[],
            source=source,
            observed_at=observed_at,
            created_at=observed_at,
        )

    snapshot(12_000, datetime(2026, 7, 13, 23, 0, tzinfo=UTC))
    snapshot(11_000, datetime(2026, 7, 14, 0, 1, tzinfo=UTC))
    snapshot(10_500, datetime(2026, 7, 14, 12, 0, tzinfo=UTC))
    snapshot(50_000, datetime(2026, 7, 14, 11, 0, tzinfo=UTC), source="paper")

    restored = repository.account_risk_state(
        "binance", now=datetime(2026, 7, 14, 18, 0, tzinfo=UTC)
    )

    assert repository.latest_account_snapshot("binance")["equity"] == 10_500
    assert restored["latest"]["equity"] == 10_500
    assert restored["historical_high_water_equity"] == 12_000
    assert restored["utc_day_start_equity"] == 11_000
    assert restored["utc_day_start_snapshot"]["observed_at"] == "2026-07-14T00:01:00Z"

    next_day = repository.account_risk_state("binance", now=datetime(2026, 7, 15, 0, 1, tzinfo=UTC))
    assert next_day["latest"]["equity"] == 10_500
    assert next_day["utc_day_start_equity"] is None


def test_venue_order_trace_lookup_is_not_bounded_by_recent_activity_feed(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _complete_order_trace(repository, "old-position-trace")

    assert repository.list_venue_order_trace_ids(
        venue="binance-futures-demo"
    ) == ("old-position-trace",)
    assert repository.list_venue_order_trace_ids(venue="another-venue") == ()
