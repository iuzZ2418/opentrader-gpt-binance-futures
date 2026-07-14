from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from crypto_event_trader.contracts import CandleInterval, RiskRegime
from crypto_event_trader.domain import MarketQuote
from crypto_event_trader.market_data import (
    BinanceFuturesMarketDataProvider,
    DerivativesRiskOverlay,
    DerivativesRiskSnapshot,
)

NOW = datetime(2026, 7, 14, 12, tzinfo=UTC)


class CandleClient:
    def __init__(self, rows: list[list[object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str, int]] = []

    def server_time(self) -> int:
        return int(NOW.timestamp() * 1_000)

    def klines(self, symbol: str, interval: str, *, limit: int) -> list[list[object]]:
        self.calls.append((symbol, interval, limit))
        return self.rows


def _raw_kline(open_time_ms: int, close_time_ms: int) -> list[object]:
    return [open_time_ms, "100", "102", "99", "101", "25", close_time_ms]


def test_closed_bars_excludes_equal_and_future_close_times() -> None:
    now_ms = int(NOW.timestamp() * 1_000)
    client = CandleClient(
        [
            _raw_kline(now_ms - 3_600_000, now_ms - 1),
            _raw_kline(now_ms - 1_800_000, now_ms),
            _raw_kline(now_ms, now_ms + 3_600_000),
        ]
    )
    provider = BinanceFuturesMarketDataProvider(client)  # type: ignore[arg-type]

    bars = provider.closed_bars("btcusdt", CandleInterval.ONE_HOUR, 3)

    assert client.calls == [("BTCUSDT", "1h", 3)]
    assert len(bars) == 1
    assert bars[0].symbol == "BTCUSDT"
    assert bars[0].is_closed is True
    assert bars[0].close_time < NOW


class SnapshotClient:
    api_key = ""
    api_secret = ""

    def __init__(self) -> None:
        self.oi_history_calls: list[tuple[str, str, int]] = []

    def premium_index(self, symbol: str) -> dict[str, object]:
        return {
            "symbol": symbol,
            "markPrice": "100",
            "indexPrice": "100",
            "lastFundingRate": "0.0001",
            "time": int(NOW.timestamp() * 1_000),
        }

    def open_interest(self, symbol: str) -> dict[str, object]:
        return {"symbol": symbol, "openInterest": "115"}

    def open_interest_history(
        self, symbol: str, *, period: str, limit: int
    ) -> list[dict[str, str]]:
        self.oi_history_calls.append((symbol, period, limit))
        return [
            {"sumOpenInterest": "100"},
            {"sumOpenInterest": "not-a-number"},
            {"sumOpenInterest": "115"},
        ]

    def depth(self, symbol: str, *, limit: int) -> dict[str, object]:
        del symbol, limit
        return {
            "bids": [["99.9", "100"]],
            "asks": [["100.1", "1"]],
        }

    def fetch_quotes(self, symbols: dict[str, str]) -> dict[str, MarketQuote]:
        symbol = next(iter(symbols))
        return {
            symbol: MarketQuote(symbol, 99.99, 100.01, 100, 1_000_000, NOW)
        }


def test_derivatives_snapshot_computes_24h_oi_change_and_thinner_side_depth() -> None:
    client = SnapshotClient()
    provider = BinanceFuturesMarketDataProvider(client)  # type: ignore[arg-type]

    snapshot = provider.derivatives_snapshot(
        "btcusdt", expected_order_notional=10
    )

    assert client.oi_history_calls == [("BTCUSDT", "1h", 25)]
    assert snapshot.open_interest_change_24h_fraction == pytest.approx(0.15)
    assert snapshot.depth_within_20bps == pytest.approx(100.1)
    assert snapshot.depth_multiple == pytest.approx(10.01)


def test_depth_gate_uses_zero_when_either_book_side_is_absent() -> None:
    depth = {"bids": [["99.9", "100"]], "asks": []}

    result = BinanceFuturesMarketDataProvider._depth_within(  # noqa: SLF001
        depth, 100, fraction=0.002
    )

    assert result == 0


def _snapshot(**updates: object) -> DerivativesRiskSnapshot:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "mark_price": 100.0,
        "index_price": 100.0,
        "funding_rate": 0.0,
        "open_interest": 1_000_000.0,
        "adl_quantile": 0,
        "spread_bps": 2.0,
        "depth_within_20bps": 500_000.0,
        "expected_order_notional": 10_000.0,
        "observed_at": NOW,
        "open_interest_change_24h_fraction": 0.0,
    }
    values.update(updates)
    return DerivativesRiskSnapshot(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("snapshot", "reason"),
    [
        (_snapshot(funding_rate=0.001), "funding_elevated"),
        (_snapshot(funding_rate=-0.001), "funding_elevated"),
        (_snapshot(mark_price=102.0), "basis_elevated"),
        (_snapshot(adl_quantile=3), "adl_elevated"),
        (_snapshot(spread_bps=6.0), "spread_elevated"),
        (_snapshot(depth_within_20bps=300_000.0), "depth_thin"),
        (_snapshot(open_interest_change_24h_fraction=0.15), "oi_change_elevated"),
        (_snapshot(open_interest_change_24h_fraction=-0.15), "oi_change_elevated"),
        (_snapshot(open_interest_change_24h_fraction=None), "oi_change_unavailable"),
    ],
)
def test_derivatives_overlay_caution_inputs_only_halve_size(
    snapshot: DerivativesRiskSnapshot, reason: str
) -> None:
    result = DerivativesRiskOverlay().classify(snapshot)

    assert result.regime is RiskRegime.CAUTION
    assert result.multiplier == 0.5
    assert reason in result.reason_codes


@pytest.mark.parametrize(
    ("snapshot", "reason"),
    [
        (_snapshot(funding_rate=0.003), "funding_extreme"),
        (_snapshot(funding_rate=-0.003), "funding_extreme"),
        (_snapshot(mark_price=105.0), "basis_extreme"),
        (_snapshot(adl_quantile=4), "adl_highest_quantile"),
        (_snapshot(spread_bps=10.01), "spread_above_10bps"),
        (_snapshot(depth_within_20bps=199_999.0), "depth_below_20x_order"),
        (_snapshot(open_interest_change_24h_fraction=0.30), "oi_change_extreme"),
        (_snapshot(open_interest_change_24h_fraction=-0.30), "oi_change_extreme"),
    ],
)
def test_derivatives_overlay_extreme_inputs_only_block(
    snapshot: DerivativesRiskSnapshot, reason: str
) -> None:
    result = DerivativesRiskOverlay().classify(snapshot)

    assert result.regime is RiskRegime.BLOCKED
    assert result.multiplier == 0
    assert reason in result.reason_codes


def test_derivatives_overlay_normal_is_one_and_never_amplifies() -> None:
    overlay = DerivativesRiskOverlay()
    snapshots = (
        _snapshot(),
        _snapshot(funding_rate=0.001),
        _snapshot(open_interest_change_24h_fraction=0.30),
        replace(_snapshot(), spread_bps=6.0, adl_quantile=4),
    )

    results = tuple(overlay.classify(snapshot) for snapshot in snapshots)

    assert results[0].regime is RiskRegime.NORMAL
    assert results[0].multiplier == 1
    assert {result.multiplier for result in results} <= {0, 0.5, 1}
