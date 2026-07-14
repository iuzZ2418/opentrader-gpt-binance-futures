from __future__ import annotations

import hashlib
import hmac
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.binance import BinanceFuturesDemoClient, BinanceSafetyError
from crypto_event_trader.binance_runtime import BinanceApprovalRuntime
from crypto_event_trader.binance_streams import OrderTradeUpdate, parse_futures_stream_event
from crypto_event_trader.config import Settings
from crypto_event_trader.control import TradingControl

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


def _audited_order(repository: AuditRepository) -> dict[str, str]:
    trace_id = "trace-private-fill"
    evidence_id = "binance:closed-kline:BTCUSDT:2026-07-14T11:00:00Z"
    evidence_record_id = repository.append_external_evidence(
        trace_id=trace_id,
        evidence_id=evidence_id,
        source="binance",
        source_id="closed-kline:BTCUSDT:2026-07-14T11:00:00Z",
        occurred_at=NOW - timedelta(hours=1),
        first_observed_at=NOW - timedelta(hours=1),
        payload={"closed": True},
        created_at=NOW - timedelta(hours=1),
    )
    candidate_id = repository.append_trade_candidate(
        trace_id=trace_id,
        strategy_version="champion-1",
        symbol="BTCUSDT",
        direction="LONG",
        max_quantity=0.1,
        max_risk_fraction=0.0075,
        feature_snapshot={"atr_1h": 500},
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
        position_multiplier=1,
        confidence=0.8,
        evidence_ids=[evidence_id],
        thesis="closed-bar trend remains valid",
        invalidation_conditions=["trend reversal"],
        next_review_at=NOW + timedelta(minutes=15),
        model="decision-model",
        prompt_version="trade-v1",
        raw_response={"action": "OPEN"},
        created_at=NOW,
    )
    repository.append_position_thesis(
        trace_id=trace_id,
        position_id="BTCUSDT:BOTH",
        decision_id=decision_id,
        entry_reason="trend",
        expected_horizon="240 minutes",
        supporting_evidence=[evidence_id],
        opposing_evidence=[],
        add_count=0,
        pnl_r=0,
        invalidation_conditions=["trend reversal"],
        created_at=NOW,
    )
    risk_id = repository.append_risk_decision(
        trace_id=trace_id,
        candidate_id=candidate_id,
        decision_id=decision_id,
        outcome="ALLOW",
        approved_quantity=0.1,
        reason_codes=[],
        limits_snapshot={"gross": 0.1},
        created_at=NOW,
    )
    venue_order_id = repository.append_venue_order(
        trace_id=trace_id,
        candidate_id=candidate_id,
        decision_id=decision_id,
        risk_decision_id=risk_id,
        venue="binance_futures_demo",
        client_order_id="parent-entry-1",
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT",
        quantity=0.1,
        price=65_000,
        reduce_only=False,
        status="PREPARED",
        raw_response={},
        observed_at=NOW,
        created_at=NOW,
    )
    repository.append_venue_order_event(
        trace_id=trace_id,
        venue_order_id=venue_order_id,
        event_type="ENTRY_ATTEMPT_SUBMITTED",
        status="NEW",
        source_event_id="gateway-child-1",
        raw_response={
            "gateway_order_event": {
                "role": "ENTRY_ATTEMPT",
                "child_client_order_id": "child-entry-1",
                "symbol": "BTCUSDT",
            }
        },
        created_at=NOW,
    )
    return {"trace_id": trace_id, "venue_order_id": venue_order_id}


class _Gateway:
    venue = "binance_futures_demo"


class _AccountingClient:
    def __init__(self) -> None:
        self.income_calls: list[dict[str, Any]] = []
        self.incomes = [
            {
                "symbol": "BTCUSDT",
                "incomeType": "FUNDING_FEE",
                "income": "-1.25",
                "asset": "USDT",
                "time": int(NOW.timestamp() * 1_000),
                "tranId": 7001,
                "tradeId": "",
            },
            {
                "symbol": "BTCUSDT",
                "incomeType": "FUNDING_FEE",
                "income": "0.75",
                "asset": "USDT",
                "time": int((NOW + timedelta(hours=8)).timestamp() * 1_000),
                "tranId": 7002,
                "tradeId": "",
            },
        ]
        self.trades = [
            {
                "symbol": "BTCUSDT",
                "id": 88,
                "orderId": 42,
                "side": "BUY",
                "price": "65001",
                "qty": "0.01",
                "quoteQty": "650.01",
                "realizedPnl": "1.25",
                "commission": "0.20",
                "commissionAsset": "USDT",
                "time": int(NOW.timestamp() * 1_000),
            }
        ]

    def income_history(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.income_calls.append(kwargs)
        return list(self.incomes)

    def user_trades(self, symbol: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert symbol == "BTCUSDT"
        assert kwargs["order_id"] == 42
        assert kwargs["limit"] == 1_000
        return list(self.trades)


def _runtime(
    repository: AuditRepository, client: _AccountingClient
) -> BinanceApprovalRuntime:
    settings = replace(
        Settings.from_env(),
        trading_stage="demo",
        execution_venue="binance_futures_demo",
    )
    unused: Any = object()
    return BinanceApprovalRuntime(
        settings=settings,
        control=TradingControl(settings),
        audit=repository,
        client=client,  # type: ignore[arg-type]
        streams=unused,
        market_data=unused,
        account_source=unused,
        gateway=_Gateway(),  # type: ignore[arg-type]
        decision_provider=unused,
        approvals=unused,
    )


def _append_protective_stop(
    repository: AuditRepository, ids: dict[str, str]
) -> None:
    repository.append_venue_order_event(
        trace_id=ids["trace_id"],
        venue_order_id=ids["venue_order_id"],
        event_type="PROTECTIVE_STOP_SUBMITTED",
        status="NEW",
        source_event_id="protective-stop-submitted-701",
        external_order_id="701",
        raw_response={
            "gateway_order_event": {
                "role": "PROTECTIVE_STOP",
                "child_client_order_id": "gpt-protective-1",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "order_type": "STOP_MARKET_ALGO",
                "quantity": 0.1,
                "reduce_only": True,
                "trigger_price": 49_000,
            }
        },
        created_at=NOW,
    )


def _algo_update(*, status: str) -> Any:
    return parse_futures_stream_event(
        {
            "e": "ALGO_UPDATE",
            "E": int(NOW.timestamp() * 1_000) + 2,
            "T": int(NOW.timestamp() * 1_000) + 1,
            "o": {
                "caid": "gpt-protective-1",
                "aid": 701,
                "at": "CONDITIONAL",
                "o": "STOP_MARKET",
                "s": "BTCUSDT",
                "S": "SELL",
                "q": "0.1",
                "X": status,
                "tp": "49000",
                "R": True,
                "rm": "exchange rejected trigger" if status == "REJECTED" else "",
            },
        }
    )


def _trade_event(client_order_id: str = "child-entry-1") -> OrderTradeUpdate:
    parsed = parse_futures_stream_event(
        {
            "e": "ORDER_TRADE_UPDATE",
            "E": int(NOW.timestamp() * 1_000) + 2,
            "T": int(NOW.timestamp() * 1_000) + 1,
            "o": {
                "s": "BTCUSDT",
                "c": client_order_id,
                "S": "BUY",
                "o": "LIMIT",
                "f": "GTC",
                "q": "0.1",
                "p": "65000",
                "ap": "65001",
                "x": "TRADE",
                "X": "PARTIALLY_FILLED",
                "i": 42,
                "l": "0.01",
                "z": "0.01",
                "L": "65001",
                "N": "USDT",
                "n": "0.20",
                "T": int(NOW.timestamp() * 1_000),
                "t": 88,
                "R": False,
                "rp": "1.25",
            },
        }
    )
    assert isinstance(parsed, OrderTradeUpdate)
    return parsed


def test_private_trade_update_resolves_child_and_is_idempotent(tmp_path: Path) -> None:
    repository = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    repository.initialize()
    ids = _audited_order(repository)
    runtime = _runtime(repository, _AccountingClient())

    assert repository.resolve_venue_order_client_id(
        "parent-entry-1", venue="binance_futures_demo"
    )["client_id_role"] == "PARENT"
    assert repository.resolve_venue_order_client_id(
        "child-entry-1", venue="binance_futures_demo"
    )["client_id_role"] == "ENTRY_ATTEMPT"

    runtime.handle_stream_event(_trade_event())
    runtime.handle_stream_event(_trade_event())

    trace = repository.get_trace(ids["trace_id"])
    private_events = [
        item
        for item in trace["venue_order_events"]
        if item["event_type"] == "PRIVATE_WS_TRADE"
    ]
    assert len(private_events) == 1
    assert private_events[0]["executed_quantity"] == pytest.approx(0.01)
    assert len(trace["venue_fills"]) == 1
    fill = trace["venue_fills"][0]
    assert fill["external_fill_id"] == "binance-trade:BTCUSDT:88"
    assert fill["fee"] == pytest.approx(0.20)
    assert fill["fee_asset"] == "USDT"
    assert fill["realized_pnl"] == pytest.approx(1.25)
    accounting = trace["venue_accounting_events"]
    assert len(accounting) == 1
    assert accounting[0]["income_type"] == "COMMISSION"
    assert accounting[0]["amount"] == pytest.approx(-0.20)


def test_unknown_private_client_id_kills_and_forces_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    repository.initialize()
    runtime = _runtime(repository, _AccountingClient())
    reconciliations: list[str] = []

    def reconcile(_self: BinanceApprovalRuntime) -> Any:
        reconciliations.append("called")
        return object()

    monkeypatch.setattr(BinanceApprovalRuntime, "reconcile", reconcile)

    with pytest.raises(BinanceSafetyError, match="no audited parent/child"):
        runtime.handle_stream_event(_trade_event("external-manual-order"))

    assert reconciliations == ["called"]
    assert runtime.control.snapshot().kill_switch_active is True
    assert runtime.control.snapshot().reason == "unknown_private_order_client_id"


def test_active_protective_algo_update_is_append_only_and_keeps_trading_safe(
    tmp_path: Path,
) -> None:
    repository = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    repository.initialize()
    ids = _audited_order(repository)
    _append_protective_stop(repository, ids)
    runtime = _runtime(repository, _AccountingClient())

    runtime.handle_stream_event(_algo_update(status="WORKING"))

    latest = repository.latest_order_event(ids["venue_order_id"])
    assert latest["event_type"] == "PROTECTIVE_STOP_PRIVATE_WS_ALGO_UPDATE"
    assert latest["status"] == "WORKING"
    assert runtime.control.snapshot().kill_switch_active is False


@pytest.mark.parametrize(
    ("payload", "reason", "event_type"),
    [
        (
            _algo_update(status="REJECTED"),
            "protective_algo_became_unsafe",
            "PROTECTIVE_STOP_PRIVATE_WS_ALGO_UPDATE",
        ),
        (
            parse_futures_stream_event(
                {
                    "e": "CONDITIONAL_ORDER_TRIGGER_REJECT",
                    "E": int(NOW.timestamp() * 1_000) + 2,
                    "T": int(NOW.timestamp() * 1_000) + 1,
                    "or": {
                        "s": "BTCUSDT",
                        "i": 701,
                        "r": "would immediately trigger",
                    },
                }
            ),
            "protective_algo_trigger_rejected",
            "PROTECTIVE_STOP_CONDITIONAL_TRIGGER_REJECT",
        ),
    ],
)
def test_protective_algo_rejection_kills_and_forces_remote_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
    reason: str,
    event_type: str,
) -> None:
    repository = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    repository.initialize()
    ids = _audited_order(repository)
    _append_protective_stop(repository, ids)
    runtime = _runtime(repository, _AccountingClient())
    reconciliations: list[str] = []

    def reconcile(_self: BinanceApprovalRuntime) -> Any:
        reconciliations.append("called")
        return object()

    monkeypatch.setattr(BinanceApprovalRuntime, "reconcile", reconcile)

    with pytest.raises(BinanceSafetyError):
        runtime.handle_stream_event(payload)

    assert reconciliations == ["called"]
    assert runtime.control.snapshot().kill_switch_active is True
    assert runtime.control.snapshot().reason == reason
    latest = repository.latest_order_event(ids["venue_order_id"])
    assert latest["event_type"] == event_type
    assert latest["status"] == "REJECTED"


def test_periodic_reconciliation_persists_signed_funding_and_exact_trade_fees(
    tmp_path: Path,
) -> None:
    repository = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    repository.initialize()
    ids = _audited_order(repository)
    client = _AccountingClient()
    runtime = _runtime(repository, client)
    trace = repository.get_trace(ids["trace_id"])
    order = trace["venue_orders"][0]
    response = {"orderId": 42, "executedQty": "0.01", "avgPrice": "65001"}

    runtime._append_reconciled_fill_delta(trace, order, response, child_id="child-entry-1")
    runtime._append_reconciled_fill_delta(trace, order, response, child_id="child-entry-1")
    runtime._reconcile_account_income()
    runtime._reconcile_account_income()

    updated = repository.get_trace(ids["trace_id"])
    assert len(updated["venue_fills"]) == 1
    assert updated["venue_fills"][0]["fee"] == pytest.approx(0.20)
    global_income = repository.list_venue_accounting_events(
        venue="binance_futures_demo"
    )
    assert len(global_income) == 3
    funding = [item for item in global_income if item["income_type"] == "FUNDING_FEE"]
    assert sorted(item["amount"] for item in funding) == [-1.25, 0.75]
    assert client.income_calls[0]["start_time"] is None
    assert client.income_calls[1]["start_time"] == int(
        (NOW + timedelta(hours=8)).timestamp() * 1_000
    )


@pytest.mark.parametrize(
    ("response", "trade_changes"),
    [
        ({"executedQty": "0.01"}, {}),
        (
            {"orderId": 42, "executedQty": "0.01"},
            {"orderId": 43},
        ),
        (
            {"orderId": 42, "executedQty": "0.01"},
            {"symbol": "ETHUSDT"},
        ),
        (
            {"orderId": 42, "executedQty": "0.01"},
            {"qty": "0.02"},
        ),
    ],
)
def test_rest_fill_reconciliation_requires_exact_order_bound_trades(
    tmp_path: Path,
    response: dict[str, Any],
    trade_changes: dict[str, Any],
) -> None:
    repository = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    repository.initialize()
    ids = _audited_order(repository)
    client = _AccountingClient()
    client.trades = [{**client.trades[0], **trade_changes}]
    runtime = _runtime(repository, client)
    trace = repository.get_trace(ids["trace_id"])
    order = trace["venue_orders"][0]

    with pytest.raises(BinanceSafetyError):
        runtime._append_reconciled_fill_delta(trace, order, response)

    assert runtime.control.snapshot().kill_switch_active is True
    assert repository.get_trace(ids["trace_id"])["venue_fills"] == []


def test_income_history_is_signed_and_keeps_signed_amounts() -> None:
    secret = "test-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1_000_000})
        assert request.url.path == "/fapi/v1/income"
        assert request.headers["X-MBX-APIKEY"] == "test-key"
        payload, signature = request.url.query.decode("ascii").rsplit("&signature=", 1)
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        assert signature == expected
        assert request.url.params["incomeType"] == "FUNDING_FEE"
        assert request.url.params["limit"] == "1000"
        return httpx.Response(
            200,
            json=[
                {
                    "symbol": "BTCUSDT",
                    "incomeType": "FUNDING_FEE",
                    "income": "-1.25",
                    "asset": "USDT",
                    "time": 1_000_000,
                    "tranId": 7,
                }
            ],
        )

    client = BinanceFuturesDemoClient(
        "test-key",
        secret,
        transport=httpx.MockTransport(handler),
        time_provider=lambda: 1000,
    )
    records = client.income_history(income_type="funding_fee", limit=1_000)

    assert records[0]["income"] == "-1.25"
