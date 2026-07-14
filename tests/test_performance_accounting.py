from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crypto_event_trader.audit import AuditRepository, IncompleteVenueAccountingError
from crypto_event_trader.binance import BinanceSafetyError, FuturesRestSnapshot
from crypto_event_trader.binance_execution import BinanceFuturesAccountSource
from crypto_event_trader.binance_runtime import BinanceApprovalRuntime
from crypto_event_trader.config import Settings
from crypto_event_trader.control import TradingControl
from crypto_event_trader.learning import IncompletePerformanceAccounting

NOW = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


def _repository(tmp_path: Path, name: str = "performance.db") -> AuditRepository:
    repository = AuditRepository(f"sqlite:///{tmp_path / name}")
    repository.initialize()
    return repository


def _append_order_fill(
    repository: AuditRepository,
    *,
    trace_id: str,
    action: str,
    side: str,
    reduce_only: bool,
    filled_at: datetime,
    quantity: float = 0.1,
    price: float = 50_000,
    fee: float = 1,
    fee_asset: str = "USDT",
    realized_pnl: float = 0,
) -> tuple[str, str]:
    evidence_id = f"binance:closed-bar:{trace_id}"
    evidence_record_id = repository.append_external_evidence(
        trace_id=trace_id,
        evidence_id=evidence_id,
        source="binance",
        source_id=f"closed-bar:{trace_id}",
        occurred_at=filled_at - timedelta(minutes=1),
        first_observed_at=filled_at - timedelta(minutes=1),
        payload={"closed": True},
        created_at=filled_at - timedelta(minutes=1),
    )
    candidate_id = repository.append_trade_candidate(
        trace_id=trace_id,
        strategy_version="champion-1",
        symbol="BTCUSDT",
        direction="LONG",
        max_quantity=quantity,
        max_risk_fraction=0.0075,
        feature_snapshot={"atr_1h": 500},
        evidence_ids=[evidence_id],
        evidence_record_ids=[evidence_record_id],
        valid_until=filled_at + timedelta(seconds=119),
        created_at=filled_at - timedelta(seconds=1),
    )
    decision_id = repository.append_llm_decision(
        trace_id=trace_id,
        candidate_id=candidate_id,
        action=action,
        direction="LONG",
        position_multiplier=1 if action == "OPEN" else 0,
        confidence=0.9,
        evidence_ids=[evidence_id],
        thesis=f"audited {action.lower()}",
        invalidation_conditions=["closed-bar reversal"],
        next_review_at=filled_at + timedelta(minutes=15),
        model="decision-model",
        prompt_version="trade-v1",
        raw_response={"action": action},
        created_at=filled_at - timedelta(seconds=1),
    )
    repository.append_position_thesis(
        trace_id=trace_id,
        position_id="BTCUSDT:BOTH",
        decision_id=decision_id,
        entry_reason=f"audited {action.lower()}",
        expected_horizon="240 minutes",
        supporting_evidence=[evidence_id],
        opposing_evidence=[],
        add_count=0,
        pnl_r=0,
        invalidation_conditions=["closed-bar reversal"],
        created_at=filled_at - timedelta(seconds=1),
    )
    risk_id = repository.append_risk_decision(
        trace_id=trace_id,
        candidate_id=candidate_id,
        decision_id=decision_id,
        outcome="EXIT" if reduce_only else "ALLOW",
        approved_quantity=quantity,
        reason_codes=[],
        limits_snapshot={"gross": 0.1},
        created_at=filled_at - timedelta(seconds=1),
    )
    order_id = repository.append_venue_order(
        trace_id=trace_id,
        candidate_id=candidate_id,
        decision_id=decision_id,
        risk_decision_id=risk_id,
        venue="binance_futures_demo",
        client_order_id=f"client-{trace_id}",
        symbol="BTCUSDT",
        side=side,
        order_type="MARKET" if reduce_only else "LIMIT",
        quantity=quantity,
        price=price,
        reduce_only=reduce_only,
        status="FILLED",
        raw_response={"status": "FILLED"},
        observed_at=filled_at,
        created_at=filled_at,
    )
    fill_id = repository.append_venue_fill(
        trace_id=trace_id,
        venue_order_id=order_id,
        external_fill_id=f"fill-{trace_id}",
        price=price,
        quantity=quantity,
        fee=fee,
        fee_asset=fee_asset,
        realized_pnl=realized_pnl,
        raw_response={"authoritative": True},
        filled_at=filled_at,
        created_at=filled_at,
    )
    return order_id, fill_id


def _append_attributed_funding(
    repository: AuditRepository,
    *,
    transaction_time: datetime,
    amount: float,
) -> str:
    attribution = repository.resolve_funding_attribution(
        venue="binance_futures_demo",
        symbol="BTCUSDT",
        transaction_time=transaction_time,
    )
    assert attribution.status == "ATTRIBUTED"
    event_id = repository.append_venue_accounting_event(
        venue="binance_futures_demo",
        external_income_id=f"FUNDING_FEE:{int(transaction_time.timestamp())}",
        symbol="BTCUSDT",
        income_type="FUNDING_FEE",
        asset="USDT",
        amount=amount,
        transaction_time=transaction_time,
        trace_id=attribution.trace_id,
        venue_order_id=attribution.venue_order_id,
        raw_response={"income": amount},
    )
    repository.append_venue_accounting_attribution(
        accounting_event_id=event_id,
        status=attribution.status,
        reason=attribution.reason,
        trace_id=attribution.trace_id,
        venue_order_id=attribution.venue_order_id,
        resolved_at=transaction_time,
    )
    return event_id


def test_funding_is_point_in_time_attributed_and_net_outcome_is_exact(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _append_order_fill(
        repository,
        trace_id="trace-open",
        action="OPEN",
        side="BUY",
        reduce_only=False,
        filled_at=NOW,
        fee=1,
    )
    funding_id = _append_attributed_funding(
        repository, transaction_time=NOW + timedelta(hours=1), amount=-5
    )
    _append_order_fill(
        repository,
        trace_id="trace-close",
        action="CLOSE",
        side="SELL",
        reduce_only=True,
        filled_at=NOW + timedelta(hours=2),
        price=51_000,
        fee=2,
        realized_pnl=100,
    )

    funding = repository.list_venue_accounting_events(income_type="FUNDING_FEE")[0]
    owner = repository.latest_venue_accounting_attribution(funding_id)
    assert funding["trace_id"] == "trace-open"
    assert owner is not None and owner["status"] == "ATTRIBUTED"
    assert repository.authoritative_accounting_totals(
        venue="binance_futures_demo"
    ) == {"realized_pnl": 100.0, "funding_pnl": -5.0}

    outcomes = repository.build_trade_outcomes(venue="binance_futures_demo")
    assert len(outcomes) == 1
    assert outcomes[0].gross_pnl == 100
    assert outcomes[0].fees == 3
    assert outcomes[0].funding_cost == 5
    assert outcomes[0].net_pnl == 92
    assert outcomes[0].episode_id is not None
    assert outcomes[0].strategy_versions == ("champion-1",)
    assert set(outcomes[0].trace_ids) == {"trace-open", "trace-close"}
    assert funding_id in outcomes[0].source_record_ids
    after_close = repository.resolve_funding_attribution(
        venue="binance_futures_demo",
        symbol="BTCUSDT",
        transaction_time=NOW + timedelta(hours=3),
    )
    assert after_close.status == "UNATTRIBUTED"
    assert after_close.reason == "NO_OPEN_POSITION_AT_TRANSACTION_TIME"


def test_non_usdt_fee_requires_exact_persisted_fill_time_conversion(tmp_path: Path) -> None:
    repository = _repository(tmp_path, "conversion.db")
    _order_id, fill_id = _append_order_fill(
        repository,
        trace_id="trace-bnb-open",
        action="OPEN",
        side="BUY",
        reduce_only=False,
        filled_at=NOW,
        fee=0.01,
        fee_asset="BNB",
    )
    _append_order_fill(
        repository,
        trace_id="trace-bnb-close",
        action="CLOSE",
        side="SELL",
        reduce_only=True,
        filled_at=NOW + timedelta(hours=1),
        price=51_000,
        fee=2,
        realized_pnl=100,
    )

    with pytest.raises(IncompletePerformanceAccounting) as exc_info:
        repository.build_trade_outcomes(venue="binance_futures_demo")
    assert exc_info.value.reason_code == "MISSING_POINT_IN_TIME_FEE_CONVERSION"
    assert exc_info.value.record_id == fill_id

    conversion_id = repository.append_venue_fee_conversion(
        trace_id="trace-bnb-open",
        venue_fill_id=fill_id,
        quote_asset="USDT",
        rate=600,
        effective_at=NOW,
        source="binance-agg-trade",
        source_record_id="BNBUSDT:2026-07-14T08:00:00.000Z",
        created_at=NOW,
    )
    assert repository.append_venue_fee_conversion(
        trace_id="trace-bnb-open",
        venue_fill_id=fill_id,
        quote_asset="USDT",
        rate=600,
        effective_at=NOW,
        source="binance-agg-trade",
        source_record_id="BNBUSDT:2026-07-14T08:00:00.000Z",
        created_at=NOW,
    ) == conversion_id
    outcome = repository.build_trade_outcomes(venue="binance_futures_demo")[0]
    assert outcome.fees == pytest.approx(8)
    assert outcome.net_pnl == pytest.approx(92)


class _SnapshotClient:
    is_production = False

    def rest_snapshot(self) -> FuturesRestSnapshot:
        observed = NOW + timedelta(hours=3)
        return FuturesRestSnapshot(
            observed_at_ms=int(observed.timestamp() * 1_000),
            account={
                "totalWalletBalance": "10092",
                "totalUnrealizedProfit": "0",
                "totalMarginBalance": "10092",
                "totalInitialMargin": "0",
                "totalMaintMargin": "0",
            },
            balances=(),
            positions=(),
            open_orders=(),
        )


def test_rest_account_snapshot_uses_authoritative_realized_and_funding_ledger(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path, "snapshot.db")
    _append_order_fill(
        repository,
        trace_id="trace-snapshot-open",
        action="OPEN",
        side="BUY",
        reduce_only=False,
        filled_at=NOW,
    )
    _append_attributed_funding(
        repository, transaction_time=NOW + timedelta(hours=1), amount=-5
    )
    _append_order_fill(
        repository,
        trace_id="trace-snapshot-close",
        action="CLOSE",
        side="SELL",
        reduce_only=True,
        filled_at=NOW + timedelta(hours=2),
        realized_pnl=100,
    )
    source = BinanceFuturesAccountSource(
        _SnapshotClient(),  # type: ignore[arg-type]
        audit=repository,
        source="binance_futures_demo",
    )

    snapshot = source.snapshot()

    assert snapshot.realized_pnl == 100
    assert snapshot.funding_pnl == -5


class _Gateway:
    venue = "binance_futures_demo"


class _UnownedFundingClient:
    def income_history(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "BTCUSDT",
                "incomeType": "FUNDING_FEE",
                "income": "-1",
                "asset": "USDT",
                "time": int(NOW.timestamp() * 1_000),
                "tranId": 999,
                "tradeId": "",
            }
        ]


def test_unattributable_funding_is_globally_audited_then_kills(tmp_path: Path) -> None:
    repository = _repository(tmp_path, "unowned.db")
    settings = replace(
        Settings.from_env(),
        trading_stage="demo",
        execution_venue="binance_futures_demo",
    )
    control = TradingControl(settings)
    unused: Any = object()
    runtime = BinanceApprovalRuntime(
        settings=settings,
        control=control,
        audit=repository,
        client=_UnownedFundingClient(),  # type: ignore[arg-type]
        streams=unused,
        market_data=unused,
        account_source=unused,
        gateway=_Gateway(),  # type: ignore[arg-type]
        decision_provider=unused,
        approvals=unused,
    )

    with pytest.raises(BinanceSafetyError, match="could not be reliably attributed"):
        runtime._reconcile_account_income()

    event = repository.list_venue_accounting_events(income_type="FUNDING_FEE")[0]
    attribution = repository.latest_venue_accounting_attribution(
        event["accounting_event_id"]
    )
    assert event["trace_id"] is None
    assert attribution is not None
    assert attribution["status"] == "UNATTRIBUTED"
    assert attribution["reason"] == "NO_AUDITED_FILL_AT_TRANSACTION_TIME"
    assert control.snapshot().kill_switch_active is True
    assert control.snapshot().reason == "unattributed_funding_accounting"
    with pytest.raises(IncompleteVenueAccountingError) as exc_info:
        repository.authoritative_accounting_totals(venue="binance_futures_demo")
    assert exc_info.value.reason_code == "UNATTRIBUTED_FUNDING"
