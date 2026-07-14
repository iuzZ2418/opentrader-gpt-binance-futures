from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest

from crypto_event_trader.approval import GatewayOrderEvent
from crypto_event_trader.binance import (
    BinanceSafetyError,
    FuturesRestSnapshot,
    PositionRiskSnapshot,
    ReconciliationResult,
)
from crypto_event_trader.binance_runtime import (
    BinanceApprovalRuntime,
    build_binance_approval_runtime,
)
from crypto_event_trader.binance_streams import FuturesStreamState
from crypto_event_trader.binance_ws_runtime import FuturesWebSocketRuntime
from crypto_event_trader.config import Settings
from crypto_event_trader.control import TradingControl


def test_demo_runtime_factory_is_explicit_and_does_not_fall_back(tmp_path) -> None:
    settings = replace(
        Settings.from_env(),
        trading_stage="demo",
        execution_venue="binance_futures_demo",
        audit_database_url=f"sqlite:///{tmp_path / 'audit.db'}",
        binance_api_key="key",
        binance_api_secret="secret",
        openai_api_key="openai-key",
        futures_universe=("BTCUSDT",),
    )
    runtime = build_binance_approval_runtime(settings)
    try:
        assert runtime.gateway.venue == "binance_futures_demo"
        assert runtime.client.is_production is False
        assert runtime.approvals.gateway is runtime.gateway
    finally:
        runtime.close()


def test_live_runtime_rejects_sqlite_even_when_static_flags_are_set(tmp_path) -> None:
    settings = replace(
        Settings.from_env(),
        trading_stage="live",
        execution_venue="binance_futures_live",
        live_trading_enabled=True,
        allow_binance_production=True,
        control_api_token="token",
        openai_project="dedicated-live-project",
        audit_database_url=f"sqlite:///{tmp_path / 'audit.db'}",
        binance_api_key="key",
        binance_api_secret="secret",
        openai_api_key="openai-key",
    )
    with pytest.raises(BinanceSafetyError, match="PostgreSQL"):
        build_binance_approval_runtime(settings)


def test_ws_runtime_uses_futures_streams_for_each_symbol() -> None:
    class Client:
        pass

    runtime = FuturesWebSocketRuntime(
        client=Client(),  # type: ignore[arg-type]
        ws_base_url="wss://fstream.example",
        symbols=("BTCUSDT",),
        state=FuturesStreamState(required_market_symbols=("BTCUSDT",)),
    )
    assert "btcusdt@markPrice@1s" in runtime.public_streams
    assert "btcusdt@depth@100ms" in runtime.public_streams
    assert "btcusdt@kline_1h" in runtime.public_streams
    assert "btcusdt@kline_4h" in runtime.public_streams


def _remote_position(
    *,
    side: str = "BOTH",
    margin_type: str = "isolated",
    leverage: int = 3,
) -> PositionRiskSnapshot:
    return PositionRiskSnapshot(
        symbol="BTCUSDT",
        position_side=side,
        quantity=Decimal("0.1"),
        entry_price=Decimal("65000"),
        break_even_price=Decimal("65001"),
        mark_price=Decimal("65100"),
        unrealized_pnl=Decimal("10"),
        liquidation_price=Decimal("50000"),
        leverage=leverage,
        margin_type=margin_type,
        update_time=1,
        raw={},
    )


class _UnsafeStartupClient:
    def __init__(
        self,
        *,
        dual_side_position: bool = False,
        positions: tuple[PositionRiskSnapshot, ...] = (),
        mode_error: Exception | None = None,
    ) -> None:
        self.dual_side_position = dual_side_position
        self.positions = positions
        self.mode_error = mode_error

    def sync_time(self) -> int:
        return 0

    def get_position_mode(self) -> bool:
        if self.mode_error is not None:
            raise self.mode_error
        return self.dual_side_position

    def rest_snapshot(self) -> FuturesRestSnapshot:
        return FuturesRestSnapshot(
            observed_at_ms=1,
            account={},
            balances=(),
            positions=self.positions,
            open_orders=(),
        )


class _ModelMustNotRun:
    def check_model_access(self) -> bool:
        raise AssertionError("model access must follow account safety validation")


def _unsafe_runtime(client: _UnsafeStartupClient) -> BinanceApprovalRuntime:
    settings = replace(
        Settings.from_env(),
        trading_stage="demo",
        execution_venue="binance_futures_demo",
        max_leverage=3,
    )
    control = TradingControl(settings)
    unused: Any = object()
    return BinanceApprovalRuntime(
        settings=settings,
        control=control,
        audit=unused,
        client=client,  # type: ignore[arg-type]
        streams=unused,
        market_data=unused,
        account_source=unused,
        gateway=unused,
        decision_provider=_ModelMustNotRun(),  # type: ignore[arg-type]
        approvals=unused,
    )


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (_UnsafeStartupClient(dual_side_position=True), "hedge_mode_enabled"),
        (
            _UnsafeStartupClient(positions=(_remote_position(side="LONG"),)),
            "position_side=LONG",
        ),
        (
            _UnsafeStartupClient(positions=(_remote_position(margin_type="cross"),)),
            "margin_type=cross",
        ),
        (
            _UnsafeStartupClient(positions=(_remote_position(leverage=4),)),
            "leverage=4",
        ),
    ],
)
def test_startup_fails_closed_for_unsafe_remote_account_configuration(
    client: _UnsafeStartupClient, message: str
) -> None:
    runtime = _unsafe_runtime(client)

    with pytest.raises(BinanceSafetyError, match=message):
        runtime.startup_check()

    snapshot = runtime.control.snapshot()
    assert snapshot.kill_switch_active is True
    assert snapshot.reason == "unsafe_binance_account_configuration"


def test_startup_fails_closed_when_position_mode_cannot_be_verified() -> None:
    runtime = _unsafe_runtime(_UnsafeStartupClient(mode_error=RuntimeError("endpoint unavailable")))

    with pytest.raises(BinanceSafetyError, match="could not be verified"):
        runtime.startup_check()

    snapshot = runtime.control.snapshot()
    assert snapshot.kill_switch_active is True
    assert snapshot.reason == "position_mode_verification_failed"


def test_restart_wires_audited_protective_stop_into_remote_resize_reconciliation() -> None:
    position = _remote_position()
    remote = FuturesRestSnapshot(
        observed_at_ms=1,
        account={},
        balances=(),
        positions=(position,),
        open_orders=(),
    )

    class Client:
        def get_position_mode(self) -> bool:
            return False

    class Audit:
        appended: list[dict[str, Any]] = []

        def latest_account_snapshot(self, *, source: str) -> dict[str, Any]:
            assert source == "binance_futures_demo"
            return {"positions": ({"symbol": "BTCUSDT", "quantity": 0.1},)}

        def list_venue_order_trace_ids(self, *, venue: str) -> tuple[str, ...]:
            assert venue == "binance_futures_demo"
            return ("trace-1",)

        def get_trace(self, trace_id: str) -> dict[str, Any]:
            assert trace_id == "trace-1"
            return {
                "venue_orders": [
                    {
                        "venue_order_id": "parent-1",
                        "trace_id": "trace-1",
                        "venue": "binance_futures_demo",
                        "symbol": "BTCUSDT",
                        "client_order_id": "entry-parent",
                        "status": "FILLED",
                    }
                ],
                "venue_order_events": [
                    {
                        "venue_order_id": "parent-1",
                        "event_sequence": 1,
                        "status": "FILLED",
                        "raw_response": {},
                    },
                    {
                        "venue_order_id": "parent-1",
                        "event_sequence": 2,
                        "status": "NEW",
                        "external_order_id": "701",
                        "raw_response": {
                            "gateway_order_event": {
                                "role": "PROTECTIVE_STOP",
                                "child_client_order_id": "old-stop",
                                "symbol": "BTCUSDT",
                                "side": "SELL",
                                "order_type": "STOP_MARKET_ALGO",
                                "quantity": 0.2,
                                "reduce_only": True,
                                "trigger_price": 49_000,
                            }
                        },
                    },
                ],
                "venue_fills": [],
            }

        def append_venue_order_event(self, **kwargs: Any) -> str:
            self.appended.append(kwargs)
            return "event-new"

    class Gateway:
        venue = "binance_futures_demo"
        captured_positions: dict[str, float] | None = None
        captured_stops: tuple[Any, ...] = ()

        def reconcile(self, **kwargs: Any) -> ReconciliationResult:
            assert kwargs["expected_positions"] == {"BTCUSDT": 0.1}
            return ReconciliationResult(snapshot=remote, issues=())

        def reconcile_protective_orders(
            self, *, actual_positions: dict[str, float], known_orders: list[Any]
        ) -> tuple[GatewayOrderEvent, ...]:
            self.captured_positions = actual_positions
            self.captured_stops = tuple(known_orders)
            return (
                GatewayOrderEvent(
                    role="PROTECTIVE_STOP",
                    event_type="SUBMITTED",
                    status="NEW",
                    client_order_id="new-stop",
                    symbol="BTCUSDT",
                    side="SELL",
                    order_type="STOP_MARKET_ALGO",
                    quantity=0.1,
                    reduce_only=True,
                    external_order_id="702",
                    trigger_price=49_000,
                    source_event_id="restart:new-stop",
                ),
            )

    settings = replace(
        Settings.from_env(),
        trading_stage="demo",
        execution_venue="binance_futures_demo",
    )
    audit = Audit()
    gateway = Gateway()
    unused: Any = object()
    runtime = BinanceApprovalRuntime(
        settings=settings,
        control=TradingControl(settings),
        audit=audit,  # type: ignore[arg-type]
        client=Client(),  # type: ignore[arg-type]
        streams=unused,
        market_data=unused,
        account_source=unused,
        gateway=gateway,  # type: ignore[arg-type]
        decision_provider=unused,
        approvals=unused,
    )

    result = runtime.reconcile()

    assert result.consistent is True
    assert gateway.captured_positions == {"BTCUSDT": 0.1}
    assert gateway.captured_stops[0].client_algo_id == "old-stop"
    assert gateway.captured_stops[0].quantity == pytest.approx(0.2)
    assert audit.appended[0]["event_type"] == "PROTECTIVE_STOP_SUBMITTED"
    assert audit.appended[0]["venue_order_id"] == "parent-1"


@pytest.mark.parametrize(
    ("field", "url", "reason"),
    [
        (
            "binance_futures_demo_ws_url",
            "wss://attacker.invalid",
            "binance_ws_url_not_allowlisted",
        ),
        (
            "binance_futures_demo_url",
            "https://demo-fapi.binance.com.attacker.invalid",
            "binance_rest_url_not_allowlisted",
        ),
        (
            "openai_base_url",
            "https://api.openai.com.attacker.invalid/v1",
            "openai_url_not_allowlisted",
        ),
    ],
)
def test_external_runtime_rejects_unallowlisted_destinations_before_startup(
    tmp_path, field: str, url: str, reason: str
) -> None:
    settings = replace(
        Settings.from_env(),
        trading_stage="demo",
        execution_venue="binance_futures_demo",
        audit_database_url=f"sqlite:///{tmp_path / 'audit.db'}",
        binance_api_key="key",
        binance_api_secret="secret",
        openai_api_key="openai-key",
        **{field: url},
    )
    with pytest.raises(BinanceSafetyError, match=reason):
        build_binance_approval_runtime(settings)
