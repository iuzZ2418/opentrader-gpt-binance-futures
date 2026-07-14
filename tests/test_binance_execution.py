from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.binance import (
    BinanceSafetyError,
    FuturesRestSnapshot,
    PositionRiskSnapshot,
)
from crypto_event_trader.binance_execution import (
    BinanceFuturesAccountSource,
    BinanceFuturesExecutionGateway,
    ProtectiveOrderState,
)
from crypto_event_trader.binance_execution_gateway import (
    BinanceEntryCancellationUnresolved,
)
from crypto_event_trader.config import Settings
from crypto_event_trader.contracts import TradeAction
from crypto_event_trader.control import TradingControl
from crypto_event_trader.domain import MarketQuote
from crypto_event_trader.futures_risk import ExecutionIntent


def _settings(*, production: bool = False, statically_unlocked: bool = False) -> Settings:
    base = Settings.from_env()
    return replace(
        base,
        trading_stage="live" if production else "demo",
        live_trading_enabled=statically_unlocked if production else False,
        allow_binance_production=statically_unlocked if production else False,
        control_api_token="control-secret" if production else None,
        execution_venue=("binance_futures_live" if production else "binance_futures_demo"),
        entry_order_wait_seconds=5,
        entry_price_protection_bps=20,
    )


def _quote(price: float = 50_000) -> MarketQuote:
    return MarketQuote(
        symbol="BTCUSDT",
        bid=price - 1,
        ask=price + 1,
        last=price,
        volume_24h=2_000_000_000,
        timestamp=datetime.now(UTC),
    )


def _entry_intent() -> ExecutionIntent:
    return ExecutionIntent(
        approved=True,
        reason="approved",
        action=TradeAction.OPEN,
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.2,
        notional=10_000,
        reduce_only=False,
        protective_stop_price=49_000,
    )


def _exit_intent() -> ExecutionIntent:
    return ExecutionIntent(
        approved=True,
        reason="risk_reducing_order",
        action=TradeAction.CLOSE,
        symbol="BTCUSDT",
        side="SELL",
        quantity=0.2,
        notional=10_000,
        reduce_only=True,
    )


def _protective_response(
    client_id: str,
    *,
    quantity: float = 0.2,
    trigger_price: float = 49_000,
    side: str = "SELL",
    status: str = "NEW",
    algo_id: int = 701,
    order_type: str = "STOP_MARKET",
    reduce_only: bool = True,
) -> dict[str, Any]:
    return {
        "algoId": algo_id,
        "clientAlgoId": client_id,
        "algoType": "CONDITIONAL",
        "orderType": order_type,
        "algoStatus": status,
        "symbol": "BTCUSDT",
        "side": side,
        "positionSide": "BOTH",
        "quantity": str(quantity),
        "triggerPrice": str(trigger_price),
        "workingType": "MARK_PRICE",
        "reduceOnly": reduce_only,
        "closePosition": False,
        "priceProtect": True,
    }


class FakeBinanceClient:
    def __init__(
        self,
        *,
        production: bool = False,
        observations: list[dict[str, Any]] | None = None,
        algo_observations: list[dict[str, Any]] | None = None,
        open_order_values: list[Any] | None = None,
        open_algo_values: list[dict[str, Any]] | None = None,
        stop_error: Exception | None = None,
        rest_snapshot: FuturesRestSnapshot | None = None,
    ) -> None:
        self.is_production = production
        self.allow_production_trading = False
        self.observations = list(observations or [])
        self.algo_observations = list(algo_observations or [])
        self.open_order_values = list(open_order_values or [])
        self.open_algo_values = list(open_algo_values or [])
        self.stop_error = stop_error
        self.snapshot_value = rest_snapshot
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._order_id = 100

    def _record(self, operation: str, **kwargs: Any) -> None:
        self.calls.append((operation, kwargs))

    def set_position_mode(self, **kwargs: Any) -> dict[str, Any]:
        self._record("set_position_mode", **kwargs)
        return {"code": 200}

    def set_margin_type(self, **kwargs: Any) -> dict[str, Any]:
        self._record("set_margin_type", **kwargs)
        return {"code": 200}

    def set_leverage(self, **kwargs: Any) -> dict[str, Any]:
        self._record("set_leverage", **kwargs)
        return {"leverage": kwargs["leverage"]}

    def place_limit_order(self, **kwargs: Any) -> dict[str, Any]:
        self._record("place_limit_order", **kwargs)
        self._order_id += 1
        return {
            "orderId": self._order_id,
            "clientOrderId": kwargs["client_order_id"],
            "status": "NEW",
            "executedQty": "0",
        }

    def query_order(self, **kwargs: Any) -> dict[str, Any]:
        self._record("query_order", **kwargs)
        if not self.observations:
            raise AssertionError("test did not provide an order observation")
        return self.observations.pop(0)

    def cancel_order(self, **kwargs: Any) -> dict[str, Any]:
        self._record("cancel_order", **kwargs)
        return {"status": "CANCELED"}

    def place_stop_market_algo_order(self, **kwargs: Any) -> dict[str, Any]:
        self._record("place_stop_market_algo_order", **kwargs)
        if self.stop_error is not None:
            raise self.stop_error
        return {"algoId": 701, "algoStatus": "NEW"}

    def query_algo_order(self, **kwargs: Any) -> dict[str, Any]:
        self._record("query_algo_order", **kwargs)
        if self.algo_observations:
            return self.algo_observations.pop(0)
        if self.stop_error is not None:
            raise self.stop_error
        stops = [payload for name, payload in self.calls if name == "place_stop_market_algo_order"]
        latest = stops[-1] if stops else {}
        return _protective_response(
            str(kwargs.get("client_algo_id") or latest.get("client_algo_id") or "unknown"),
            quantity=float(latest.get("quantity", 0.2)),
            trigger_price=float(latest.get("trigger_price", 49_000)),
            side=str(latest.get("side", "SELL")),
        )

    def cancel_algo_order(self, **kwargs: Any) -> dict[str, Any]:
        self._record("cancel_algo_order", **kwargs)
        return {"algoId": 701, "algoStatus": "CANCELED"}

    def open_orders(self, symbol: str) -> list[Any]:
        self._record("open_orders", symbol=symbol)
        return list(self.open_order_values)

    def open_algo_orders(self) -> list[dict[str, Any]]:
        self._record("open_algo_orders")
        return list(self.open_algo_values)

    def place_reduce_only_market_order(self, **kwargs: Any) -> dict[str, Any]:
        self._record("place_reduce_only_market_order", **kwargs)
        return {
            "orderId": 801,
            "clientOrderId": kwargs["client_order_id"],
            "status": "FILLED",
            "executedQty": str(kwargs["quantity"]),
            "avgPrice": "50000",
        }

    def rest_snapshot(self, **kwargs: Any) -> FuturesRestSnapshot:
        self._record("rest_snapshot", **kwargs)
        assert self.snapshot_value is not None
        return self.snapshot_value


def _calls(client: FakeBinanceClient, operation: str) -> list[dict[str, Any]]:
    return [payload for name, payload in client.calls if name == operation]


def test_entry_reprices_at_most_once_then_installs_protective_algo_stop() -> None:
    observations = [
        {
            "orderId": 101,
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
        },
        {
            "orderId": 101,
            "status": "CANCELED",
            "executedQty": "0",
            "avgPrice": "0",
        },
        {
            "orderId": 102,
            "status": "FILLED",
            "executedQty": "0.2",
            "avgPrice": "50002",
        },
    ]
    client = FakeBinanceClient(observations=observations)
    sleeps: list[float] = []
    refreshed_quotes: list[str] = []

    def refreshed(symbol: str) -> MarketQuote:
        refreshed_quotes.append(symbol)
        return _quote(50_005)

    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=_settings(),
        control=TradingControl(_settings()),
        quote_provider=refreshed,
        sleeper=sleeps.append,
    )

    submission = gateway.submit(
        intent=_entry_intent(), quote=_quote(), client_order_id="gpt-open-trace-123"
    )

    assert submission.status == "FILLED"
    assert submission.protective_order_id == "701"
    assert len(_calls(client, "place_limit_order")) == 2
    assert len(_calls(client, "cancel_order")) == 1
    assert refreshed_quotes == ["BTCUSDT"]
    assert sleeps == [5, 5]
    second_order = _calls(client, "place_limit_order")[1]
    assert second_order["client_order_id"].endswith("-r")
    assert second_order["quantity"] == pytest.approx(0.2)
    stops = _calls(client, "place_stop_market_algo_order")
    assert len(stops) == 1
    assert stops[0] == {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "trigger_price": 49_000,
        "quantity": pytest.approx(0.2),
        "client_algo_id": "gpt-open-trace-123-s",
        "price_protect": True,
    }


def test_entry_abandons_reprice_when_market_moves_beyond_price_protection() -> None:
    client = FakeBinanceClient(
        observations=[
            {
                "orderId": 101,
                "status": "NEW",
                "executedQty": "0",
                "avgPrice": "0",
            },
            {
                "orderId": 101,
                "status": "CANCELED",
                "executedQty": "0",
                "avgPrice": "0",
            },
        ]
    )
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=_settings(),
        control=TradingControl(_settings()),
        quote_provider=lambda _symbol: _quote(50_200),
        sleeper=lambda _seconds: None,
    )

    submission = gateway.submit(
        intent=_entry_intent(), quote=_quote(), client_order_id="gpt-open-price-move"
    )

    assert submission.status == "CANCELED"
    assert submission.fills == ()
    assert len(_calls(client, "place_limit_order")) == 1
    assert len(_calls(client, "cancel_order")) == 1
    assert _calls(client, "place_stop_market_algo_order") == []


def test_reduce_only_exit_survives_production_kill_switch() -> None:
    settings = _settings(production=True, statically_unlocked=True)
    control = TradingControl(settings)
    control.unlock_live("control-secret")
    control.engage_kill_switch("manual-emergency")
    client = FakeBinanceClient(production=True)
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=settings,
        control=control,
        stream_ready=lambda: False,
        sleeper=lambda _seconds: None,
    )

    submission = gateway.submit(
        intent=_exit_intent(), quote=_quote(), client_order_id="gpt-close-emergency"
    )

    assert submission.status == "FILLED"
    # A reduce-only emergency grant is scoped to that one REST mutation.
    assert client.allow_production_trading is False
    exits = _calls(client, "place_reduce_only_market_order")
    assert len(exits) == 1
    assert exits[0]["side"] == "SELL"
    assert _calls(client, "place_limit_order") == []


def test_production_requires_static_gates_and_authenticated_runtime_unlock() -> None:
    locked_settings = _settings(production=True, statically_unlocked=False)
    locked_client = FakeBinanceClient(production=True)
    locked_gateway = BinanceFuturesExecutionGateway(
        locked_client,  # type: ignore[arg-type]
        settings=locked_settings,
        control=TradingControl(locked_settings),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(BinanceSafetyError, match="static safety gates"):
        locked_gateway.submit(
            intent=_entry_intent(), quote=_quote(), client_order_id="gpt-live-static"
        )

    eligible_settings = _settings(production=True, statically_unlocked=True)
    eligible_client = FakeBinanceClient(production=True)
    eligible_gateway = BinanceFuturesExecutionGateway(
        eligible_client,  # type: ignore[arg-type]
        settings=eligible_settings,
        control=TradingControl(eligible_settings),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(BinanceSafetyError, match="runtime unlock"):
        eligible_gateway.submit(
            intent=_entry_intent(), quote=_quote(), client_order_id="gpt-live-runtime"
        )

    assert _calls(locked_client, "place_limit_order") == []
    assert _calls(eligible_client, "place_limit_order") == []


def test_protective_stop_failure_submits_emergency_reduce_only_close() -> None:
    client = FakeBinanceClient(
        observations=[
            {
                "orderId": 101,
                "status": "FILLED",
                "executedQty": "0.2",
                "avgPrice": "50001",
            }
        ],
        stop_error=RuntimeError("algo endpoint unavailable"),
    )
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=_settings(),
        control=TradingControl(_settings()),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(
        BinanceSafetyError,
        match="protective stop failed; emergency reduce-only close submitted",
    ):
        gateway.submit(intent=_entry_intent(), quote=_quote(), client_order_id="gpt-stop-failure")

    emergency_exits = _calls(client, "place_reduce_only_market_order")
    assert len(emergency_exits) == 1
    assert emergency_exits[0]["side"] == "SELL"
    assert emergency_exits[0]["quantity"] == pytest.approx(0.2)
    assert emergency_exits[0]["client_order_id"].endswith("-x")


def test_cancel_barrier_captures_late_partial_fill_before_repricing() -> None:
    client = FakeBinanceClient(
        observations=[
            {"orderId": 101, "status": "NEW", "executedQty": "0"},
            {
                "orderId": 101,
                "status": "CANCELED",
                "executedQty": "0.05",
                "avgPrice": "50001",
            },
            {
                "orderId": 102,
                "status": "FILLED",
                "executedQty": "0.15",
                "avgPrice": "50002",
            },
        ]
    )
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=_settings(),
        control=TradingControl(_settings()),
        quote_provider=lambda _symbol: _quote(50_001),
        sleeper=lambda _seconds: None,
    )

    submission = gateway.submit(
        intent=_entry_intent(), quote=_quote(), client_order_id="late-partial"
    )

    children = _calls(client, "place_limit_order")
    assert len(children) == 2
    assert children[1]["quantity"] == pytest.approx(0.15)
    assert sum(fill.quantity for fill in submission.fills) == pytest.approx(0.2)
    assert all(fill.authoritative is False for fill in submission.fills)
    assert _calls(client, "place_stop_market_algo_order")[0]["quantity"] == pytest.approx(0.2)
    assert any(
        event.event_type == "POST_CANCEL_REQUERY" and event.executed_quantity == pytest.approx(0.05)
        for event in submission.order_events
    )


def test_unknown_child_blocks_reprice_cancels_remote_entries_and_emergency_closes() -> None:
    settings = _settings()
    control = TradingControl(settings)
    client = FakeBinanceClient(
        observations=[
            {
                "orderId": 101,
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.05",
                "avgPrice": "50001",
            },
            {
                "orderId": 101,
                "status": "NEW",
                "executedQty": "0.05",
                "avgPrice": "50001",
            },
        ],
        open_order_values=[
            {
                "clientOrderId": "late-unknown",
                "side": "BUY",
                "origQty": "0.2",
                "price": "50001",
                "reduceOnly": False,
            },
            {
                "clientOrderId": "existing-exit",
                "side": "SELL",
                "origQty": "0.1",
                "price": "0",
                "reduceOnly": True,
            },
        ],
    )
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=settings,
        control=control,
        quote_provider=lambda _symbol: _quote(),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(BinanceSafetyError) as caught:
        gateway.submit(intent=_entry_intent(), quote=_quote(), client_order_id="late-unknown")

    assert len(_calls(client, "place_limit_order")) == 1
    assert _calls(client, "place_stop_market_algo_order") == []
    assert _calls(client, "place_reduce_only_market_order")[0]["quantity"] == pytest.approx(0.05)
    canceled_ids = {item["client_order_id"] for item in _calls(client, "cancel_order")}
    assert "late-unknown" in canceled_ids
    assert "existing-exit" not in canceled_ids
    assert control.snapshot().kill_switch_active is True
    submission = caught.value.submission  # type: ignore[attr-defined]
    assert any(event.role == "EMERGENCY_EXIT" for event in submission.order_events)


def test_second_child_rechecks_shared_control_and_protects_partial_fill() -> None:
    settings = _settings()
    control = TradingControl(settings)
    client = FakeBinanceClient(
        observations=[
            {
                "orderId": 101,
                "status": "CANCELED",
                "executedQty": "0.05",
                "avgPrice": "50001",
            }
        ]
    )

    def flip_control(_seconds: float) -> None:
        control.engage_kill_switch("operator-kill-during-entry")

    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=settings,
        control=control,
        quote_provider=lambda _symbol: _quote(),
        sleeper=flip_control,
    )

    submission = gateway.submit(
        intent=_entry_intent(), quote=_quote(), client_order_id="control-recheck"
    )

    assert len(_calls(client, "place_limit_order")) == 1
    assert submission.status == "PARTIALLY_FILLED"
    assert _calls(client, "place_stop_market_algo_order")[0]["quantity"] == pytest.approx(0.05)


def test_kill_switch_cancel_all_proves_terminal_state_and_reenumerates() -> None:
    class RemoteOrderClient(FakeBinanceClient):
        def __init__(self) -> None:
            super().__init__()
            self.remote_open = True

        def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
            self._record("open_orders", symbol=symbol)
            if not self.remote_open:
                return []
            return [
                {
                    "symbol": "BTCUSDT",
                    "clientOrderId": "gpt-pending-entry",
                    "side": "BUY",
                    "origQty": "0.2",
                    "price": "50001",
                    "reduceOnly": False,
                }
            ]

        def query_order(self, **kwargs: Any) -> dict[str, Any]:
            self._record("query_order", **kwargs)
            self.remote_open = False
            return {
                "orderId": 101,
                "status": "CANCELED",
                "executedQty": "0.05",
                "avgPrice": "50001",
            }

    client = RemoteOrderClient()
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=_settings(),
        control=TradingControl(_settings()),
        sleeper=lambda _seconds: None,
    )

    events = gateway.cancel_all()

    assert len(_calls(client, "cancel_order")) == 1
    assert len(_calls(client, "query_order")) == 1
    assert len(_calls(client, "open_orders")) == 2
    assert any(
        event.event_type == "KILL_SWITCH_POST_CANCEL_REQUERY"
        and event.status == "CANCELED"
        and event.executed_quantity == pytest.approx(0.05)
        for event in events
    )


def test_kill_switch_cancel_all_unknown_state_fails_without_blind_retry() -> None:
    class UnknownRemoteOrderClient(FakeBinanceClient):
        def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
            self._record("open_orders", symbol=symbol)
            return [
                {
                    "symbol": "BTCUSDT",
                    "clientOrderId": "gpt-unknown-entry",
                    "side": "BUY",
                    "origQty": "0.2",
                    "price": "50001",
                    "reduceOnly": False,
                }
            ]

        def query_order(self, **kwargs: Any) -> dict[str, Any]:
            self._record("query_order", **kwargs)
            return {"orderId": 101, "status": "NEW", "executedQty": "0"}

    settings = _settings()
    control = TradingControl(settings)
    client = UnknownRemoteOrderClient()
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=settings,
        control=control,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(BinanceEntryCancellationUnresolved, match="unresolved"):
        gateway.cancel_all()

    assert len(_calls(client, "cancel_order")) == 1
    assert len(_calls(client, "query_order")) == 1
    assert control.snapshot().kill_switch_active is True


def test_kill_switch_cancel_all_enumeration_failure_is_not_silenced() -> None:
    class UnavailableOrderClient(FakeBinanceClient):
        def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
            self._record("open_orders", symbol=symbol)
            raise OSError("venue unavailable")

    settings = _settings()
    control = TradingControl(settings)
    gateway = BinanceFuturesExecutionGateway(
        UnavailableOrderClient(),  # type: ignore[arg-type]
        settings=settings,
        control=control,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(BinanceEntryCancellationUnresolved, match="enumeration"):
        gateway.cancel_all()

    assert control.snapshot().kill_switch_active is True


def test_close_cancels_and_verifies_stale_protective_algo() -> None:
    client = FakeBinanceClient(algo_observations=[{"algoId": 701, "algoStatus": "CANCELED"}])
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=_settings(),
        control=TradingControl(_settings()),
        sleeper=lambda _seconds: None,
    )
    gateway.register_protective_order(
        ProtectiveOrderState("BTCUSDT", "old-stop", "SELL", 49_000, 0.2, "701")
    )

    submission = gateway.submit(
        intent=_exit_intent(), quote=_quote(), client_order_id="close-with-stop"
    )

    assert submission.status == "FILLED"
    assert _calls(client, "cancel_algo_order") == [{"client_algo_id": "old-stop"}]
    assert _calls(client, "query_algo_order") == [{"client_algo_id": "old-stop"}]
    assert any(
        event.role == "PROTECTIVE_STOP"
        and event.event_type == "POST_CANCEL_REQUERY"
        and event.status == "CANCELED"
        for event in submission.order_events
    )


def test_restart_reconciliation_resizes_protective_algo_to_actual_position() -> None:
    client = FakeBinanceClient(
        algo_observations=[
            _protective_response("restart-stop", quantity=0.2),
            {"algoId": 701, "algoStatus": "CANCELED"},
        ]
    )
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=_settings(),
        control=TradingControl(_settings()),
        sleeper=lambda _seconds: None,
    )

    events = gateway.reconcile_protective_orders(
        actual_positions={"BTCUSDT": 0.1},
        known_orders=(ProtectiveOrderState("BTCUSDT", "restart-stop", "SELL", 49_000, 0.2, "701"),),
    )

    assert _calls(client, "cancel_algo_order") == [{"client_algo_id": "restart-stop"}]
    replacement = _calls(client, "place_stop_market_algo_order")
    assert len(replacement) == 1
    assert replacement[0]["quantity"] == pytest.approx(0.1)
    assert replacement[0]["client_algo_id"] != "restart-stop"
    assert any(event.event_type == "REST_RECONCILIATION" for event in events)


def test_restart_without_auditable_stop_kills_and_flattens_in_position_direction() -> None:
    settings = _settings()
    control = TradingControl(settings)
    client = FakeBinanceClient()
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=settings,
        control=control,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(BinanceSafetyError):
        gateway.reconcile_protective_orders(actual_positions={"BTCUSDT": 0.1}, known_orders=())

    emergency = _calls(client, "place_reduce_only_market_order")
    assert len(emergency) == 1
    assert emergency[0]["side"] == "SELL"
    assert emergency[0]["quantity"] == pytest.approx(0.1)
    assert control.snapshot().kill_switch_active is True


def test_restart_rejects_active_algo_that_is_not_a_reduce_only_mark_price_stop() -> None:
    settings = _settings()
    control = TradingControl(settings)
    invalid = _protective_response(
        "gpt-invalid-stop",
        quantity=0.1,
        order_type="TAKE_PROFIT_MARKET",
        reduce_only=False,
    )
    client = FakeBinanceClient(
        open_algo_values=[invalid],
        algo_observations=[invalid],
    )
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=settings,
        control=control,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(BinanceSafetyError, match="algo order reconciliation"):
        gateway.reconcile_protective_orders(
            actual_positions={"BTCUSDT": 0.1}, known_orders=()
        )

    assert _calls(client, "cancel_algo_order") == [{"client_algo_id": "gpt-invalid-stop"}]
    emergency = _calls(client, "place_reduce_only_market_order")
    assert len(emergency) == 1
    assert emergency[0]["side"] == "SELL"
    assert control.snapshot().kill_switch_active is True


def test_restart_never_adopts_unrecorded_bot_algo_from_client_id_prefix() -> None:
    settings = _settings()
    control = TradingControl(settings)
    client = FakeBinanceClient(
        open_algo_values=[
            _protective_response(
                "gpt-open-unrecorded-s", quantity=0.1, algo_id=702
            )
        ],
        algo_observations=[{"algoId": 702, "algoStatus": "CANCELED"}],
    )
    gateway = BinanceFuturesExecutionGateway(
        client,  # type: ignore[arg-type]
        settings=settings,
        control=control,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(BinanceSafetyError, match="algo order reconciliation"):
        gateway.reconcile_protective_orders(
            actual_positions={"BTCUSDT": 0.1}, known_orders=()
        )

    assert _calls(client, "cancel_algo_order") == [
        {"client_algo_id": "gpt-open-unrecorded-s"}
    ]
    assert _calls(client, "place_stop_market_algo_order") == []
    assert len(_calls(client, "place_reduce_only_market_order")) == 1
    assert control.snapshot().kill_switch_active is True


def _live_snapshot(now: datetime) -> FuturesRestSnapshot:
    return FuturesRestSnapshot(
        observed_at_ms=int(now.timestamp() * 1_000),
        account={
            "totalWalletBalance": "100000",
            "totalUnrealizedProfit": "1000",
            "totalMarginBalance": "101000",
            "totalInitialMargin": "5000",
            "totalMaintMargin": "100",
        },
        balances=(),
        positions=(
            PositionRiskSnapshot(
                symbol="BTCUSDT",
                position_side="BOTH",
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                break_even_price=Decimal("50010"),
                mark_price=Decimal("51000"),
                unrealized_pnl=Decimal("100"),
                liquidation_price=Decimal("20000"),
                leverage=3,
                margin_type="isolated",
                update_time=int(now.timestamp() * 1_000),
                raw={},
            ),
        ),
        open_orders=(),
    )


def test_live_account_source_fails_closed_until_risk_baseline_is_persisted(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    repository = AuditRepository(f"sqlite:///{tmp_path / 'live-risk.db'}")
    repository.initialize()
    client = FakeBinanceClient(production=True, rest_snapshot=_live_snapshot(now))
    source = BinanceFuturesAccountSource(
        client,  # type: ignore[arg-type]
        audit=repository,
        source="binance_futures_live",
    )

    assert source.ready_for_new_orders is False
    snapshot = source.snapshot()
    assert snapshot.equity == 101_000
    assert source.ready_for_new_orders is False

    confirmed = source.confirm_risk_baseline()
    assert confirmed.equity == 101_000
    assert source.ready_for_new_orders is True
    state = repository.account_risk_state(source="binance_futures_live", now=now)
    assert state["historical_high_water_equity"] == 101_000
    assert state["utc_day_start_equity"] == 101_000

    restarted = BinanceFuturesAccountSource(
        client,  # type: ignore[arg-type]
        audit=repository,
        source="binance_futures_live",
    )
    assert restarted.ready_for_new_orders is True

    private_stream_stale = BinanceFuturesAccountSource(
        client,  # type: ignore[arg-type]
        audit=repository,
        source="binance_futures_live",
        private_stream_ready=lambda: False,
    )
    assert private_stream_stale.ready_for_new_orders is False
