from __future__ import annotations

from decimal import Decimal

from crypto_event_trader.binance_streams import ForceOrderUpdate, MarkPriceUpdate
from crypto_event_trader.worker import MarketReviewWakeupGate


class MonotonicClock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _mark_price(funding_rate: str, *, symbol: str = "BTCUSDT") -> MarkPriceUpdate:
    return MarkPriceUpdate(
        event_time=1,
        symbol=symbol,
        mark_price=Decimal("100"),
        index_price=Decimal("100"),
        estimated_settle_price=Decimal("100"),
        funding_rate=Decimal(funding_rate),
        next_funding_time=2,
        raw={},
    )


def _force_order(*, symbol: str = "BTCUSDT") -> ForceOrderUpdate:
    return ForceOrderUpdate(
        event_time=1,
        symbol=symbol,
        side="SELL",
        order_type="LIMIT",
        status="FILLED",
        original_quantity=Decimal("1"),
        average_price=Decimal("100"),
        accumulated_quantity=Decimal("1"),
        trade_time=1,
        raw={},
    )


def test_funding_wakeup_only_on_entry_or_escalation_and_rearms_at_normal() -> None:
    gate = MarketReviewWakeupGate()

    assert gate.should_wake(_mark_price("0.0009")) is False
    assert gate.should_wake(_mark_price("0.0010")) is True
    assert gate.should_wake(_mark_price("0.0029")) is False
    assert gate.should_wake(_mark_price("0.0030")) is True
    assert gate.should_wake(_mark_price("0.0040")) is False
    assert gate.should_wake(_mark_price("0.0015")) is False
    assert gate.should_wake(_mark_price("0.0001")) is False
    assert gate.should_wake(_mark_price("-0.0010")) is True

    # Funding state is tracked per symbol rather than suppressing another market's transition.
    assert gate.should_wake(_mark_price("0.0031", symbol="ETHUSDT")) is True


def test_force_order_wakeup_is_globally_debounced() -> None:
    clock = MonotonicClock()
    gate = MarketReviewWakeupGate(force_order_debounce_seconds=60, clock=clock)

    assert gate.should_wake(_force_order()) is True
    clock.advance(59.9)
    assert gate.should_wake(_force_order(symbol="ETHUSDT")) is False
    clock.advance(0.1)
    assert gate.should_wake(_force_order(symbol="ETHUSDT")) is True
