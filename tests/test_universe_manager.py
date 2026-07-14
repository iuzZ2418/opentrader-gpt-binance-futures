from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.binance import BinanceFuturesDemoClient
from crypto_event_trader.strategy import UniverseSelector
from crypto_event_trader.universe import (
    DynamicUniverseManager,
    LiquidityObservation,
    LiquidityObservationStore,
    UniverseSelection,
)

NOW = datetime(2026, 7, 13, 12, tzinfo=UTC)


class FakeUniverseClient:
    def __init__(self) -> None:
        old = int((NOW - timedelta(days=400)).timestamp() * 1_000)
        new = int((NOW - timedelta(days=100)).timestamp() * 1_000)
        self.symbols = [
            {
                "symbol": f"S{rank:02d}USDT",
                "quoteAsset": "USDT",
                "contractType": "PERPETUAL",
                "onboardDate": old,
                "status": "TRADING",
            }
            for rank in range(1, 14)
        ]
        self.symbols.extend(
            [
                {
                    "symbol": "WRONGQUOTE",
                    "quoteAsset": "USDC",
                    "contractType": "PERPETUAL",
                    "onboardDate": old,
                    "status": "TRADING",
                },
                {
                    "symbol": "QUARTERUSDT",
                    "quoteAsset": "USDT",
                    "contractType": "CURRENT_QUARTER",
                    "onboardDate": old,
                    "status": "TRADING",
                },
                {
                    "symbol": "NEWUSDT",
                    "quoteAsset": "USDT",
                    "contractType": "PERPETUAL",
                    "onboardDate": new,
                    "status": "TRADING",
                },
            ]
        )
        self.depth_calls: list[str] = []
        self.ticker_calls = 0
        self.book_calls = 0

    def exchange_info(self, *, refresh: bool = False) -> dict[str, Any]:
        assert refresh
        return {"symbols": self.symbols}

    def ticker_24h(self, symbol: str | None = None) -> list[dict[str, str]]:
        assert symbol is None
        self.ticker_calls += 1
        return [
            {
                "symbol": item["symbol"],
                "quoteVolume": str(1_000_000 - index * 10_000),
            }
            for index, item in enumerate(self.symbols)
        ]

    def book_ticker(self, symbol: str | None = None) -> list[dict[str, str]]:
        assert symbol is None
        self.book_calls += 1
        return [
            {"symbol": item["symbol"], "bidPrice": "99.99", "askPrice": "100.01"}
            for item in self.symbols
        ]

    def depth(self, symbol: str, *, limit: int = 100) -> dict[str, Any]:
        assert limit == 100
        self.depth_calls.append(symbol)
        return {
            "bids": [["99.99", "300"], ["99.00", "10000"]],
            "asks": [["100.01", "250"], ["101.00", "10000"]],
        }


@pytest.fixture
def store() -> LiquidityObservationStore:
    repository = AuditRepository("sqlite:///:memory:")
    result = LiquidityObservationStore(repository)
    result.initialize()
    yield result
    repository.close()


def _observation(
    symbol: str,
    observed_at: datetime,
    *,
    turnover: float,
    spread: float = 2.0,
    depth: float | None = 30_000,
) -> LiquidityObservation:
    return LiquidityObservation(
        observation_id=f"{symbol}_{observed_at.timestamp()}_{turnover}",
        symbol=symbol,
        quote_asset="USDT",
        contract_type="PERPETUAL",
        onboarded_at=NOW - timedelta(days=400),
        observed_at=observed_at,
        turnover_24h=turnover,
        bid_price=99.99,
        ask_price=100.01,
        spread_bps=spread,
        depth_within_20bps=depth,
        expected_order_notional=1_000,
    )


def _seed_complete_history(store: LiquidityObservationStore, *, missing: str | None = None) -> None:
    for rank in range(1, 14):
        symbol = f"S{rank:02d}USDT"
        for offset in range(29, -1, -1):
            observed_at = NOW - timedelta(days=offset)
            if missing == symbol and offset == 7:
                continue
            store.append_observation(
                _observation(symbol, observed_at, turnover=1_000_000 - rank * 1_000)
            )


def test_collection_filters_contracts_and_ranks_before_depth(
    store: LiquidityObservationStore,
) -> None:
    client = FakeUniverseClient()
    manager = DynamicUniverseManager(client=client, store=store)

    observations = manager.collect_daily(as_of=NOW, expected_order_notional=1_000)

    assert len(observations) == 13
    assert client.ticker_calls == 1
    assert client.book_calls == 1
    assert client.depth_calls == [f"S{rank:02d}USDT" for rank in range(1, 13)]
    assert observations[-1].symbol == "S13USDT"
    assert observations[-1].depth_within_20bps is None
    assert all(item.quote_asset == "USDT" for item in observations)
    assert all(item.contract_type == "PERPETUAL" for item in observations)


def test_binance_client_exposes_bulk_ticker_endpoints() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "symbol" not in request.url.params
        if request.url.path == "/fapi/v1/ticker/24hr":
            return httpx.Response(200, json=[{"symbol": "BTCUSDT", "quoteVolume": "10"}])
        if request.url.path == "/fapi/v1/ticker/bookTicker":
            return httpx.Response(
                200,
                json=[{"symbol": "BTCUSDT", "bidPrice": "9", "askPrice": "11"}],
            )
        return httpx.Response(404)

    client = BinanceFuturesDemoClient(
        None,
        None,
        transport=httpx.MockTransport(handler),
    )
    assert client.ticker_24h() == [{"symbol": "BTCUSDT", "quoteVolume": "10"}]
    assert client.book_ticker() == [
        {"symbol": "BTCUSDT", "bidPrice": "9", "askPrice": "11"}
    ]


def test_depth_is_conservative_smaller_side_within_twenty_bps() -> None:
    snapshot = {
        "bids": [["99.90", "10"], ["99.79", "1000"]],
        "asks": [["100.10", "7"], ["100.21", "1000"]],
    }
    result = DynamicUniverseManager.depth_within_20bps(snapshot, bid=99.9, ask=100.1)
    assert result == pytest.approx(100.10 * 7)


def test_complete_30_day_history_selects_top_10_with_rank_12_retention(
    store: LiquidityObservationStore,
) -> None:
    _seed_complete_history(store)
    prior_symbols = tuple(f"S{rank:02d}USDT" for rank in range(1, 10)) + ("S11USDT",)
    store.append_selection(
        UniverseSelection(
            selection_id="prior",
            week_start=(NOW - timedelta(days=7)).date(),
            as_of=NOW - timedelta(days=7),
            symbols=prior_symbols,
            fallback_used=False,
            reason="SELECTED",
            eligible_count=13,
        )
    )
    manager = DynamicUniverseManager(client=FakeUniverseClient(), store=store)

    selected = manager.select_weekly(as_of=NOW)

    assert selected.reason == "SELECTED"
    assert len(selected.symbols) == 10
    assert "S11USDT" in selected.symbols
    assert "S10USDT" not in selected.symbols


def test_missing_daily_coverage_fails_closed_and_fallback_is_explicit(
    store: LiquidityObservationStore,
) -> None:
    _seed_complete_history(store, missing="S13USDT")
    manager = DynamicUniverseManager(
        client=FakeUniverseClient(),
        store=store,
        selector=UniverseSelector(size=13, retention_rank=13),
        fallback_symbols=("BTCUSDT", "ETHUSDT"),
    )

    closed = manager.select_weekly(as_of=NOW)
    fallback = manager.select_weekly(as_of=NOW, allow_fallback=True)
    closed_again = manager.select_weekly(as_of=NOW)

    assert closed.symbols == ()
    assert closed.reason == "INSUFFICIENT_POINT_IN_TIME_COVERAGE"
    assert fallback.symbols == ("BTCUSDT", "ETHUSDT")
    assert fallback.fallback_used
    assert closed_again.symbols == ()
    assert closed_again.reason == "FALLBACK_REQUIRES_EXPLICIT_OPT_IN"


def test_future_rows_and_future_same_day_versions_are_not_used(
    store: LiquidityObservationStore,
) -> None:
    for offset in range(29, -1, -1):
        observed_at = NOW - timedelta(days=offset)
        store.append_observation(_observation("BTCUSDT", observed_at, turnover=100))
    store.append_observation(
        _observation("BTCUSDT", NOW + timedelta(minutes=1), turnover=1_000_000_000)
    )
    manager = DynamicUniverseManager(
        client=FakeUniverseClient(),
        store=store,
        selector=UniverseSelector(size=1, retention_rank=1),
    )

    markets = manager.point_in_time_markets(as_of=NOW)

    assert len(markets) == 1
    assert markets[0].median_turnover_30d == 100
    assert markets[0].as_of == NOW


def test_observations_are_append_only_by_primary_key(
    store: LiquidityObservationStore,
) -> None:
    observation = _observation("BTCUSDT", NOW, turnover=100)
    store.append_observation(observation)

    with pytest.raises(sqlite3.IntegrityError):
        store.append_observation(observation)

    rows = store.observations(as_of=NOW)
    assert len(rows) == 1
    assert rows[0] == observation
