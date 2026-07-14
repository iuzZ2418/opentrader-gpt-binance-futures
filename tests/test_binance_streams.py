from __future__ import annotations

from decimal import Decimal

from crypto_event_trader.binance_streams import (
    AccountUpdate,
    AlgoOrderUpdate,
    BookTickerUpdate,
    ConditionalOrderTriggerReject,
    FuturesStreamState,
    KlineUpdate,
    ListenKeyExpired,
    MarkPriceUpdate,
    OrderTradeUpdate,
    StreamIngestStatus,
    parse_futures_stream_event,
)


def test_parses_mark_kline_and_book_ticker_market_events() -> None:
    mark = parse_futures_stream_event(
        {
            "e": "markPriceUpdate",
            "E": 1000,
            "s": "BTCUSDT",
            "p": "65000",
            "i": "64990",
            "P": "65010",
            "r": "0.0001",
            "T": 2000,
        }
    )
    assert isinstance(mark, MarkPriceUpdate)
    assert mark.funding_rate == Decimal("0.0001")

    kline = parse_futures_stream_event(
        {
            "e": "kline",
            "E": 1001,
            "s": "BTCUSDT",
            "k": {
                "s": "BTCUSDT",
                "i": "1h",
                "t": 0,
                "T": 3599999,
                "o": "10",
                "h": "12",
                "l": "9",
                "c": "11",
                "v": "5",
                "x": True,
            },
        }
    )
    assert isinstance(kline, KlineUpdate)
    assert kline.is_closed is True

    book = parse_futures_stream_event(
        {
            "e": "bookTicker",
            "E": 1002,
            "T": 1001,
            "u": 7,
            "s": "BTCUSDT",
            "b": "10",
            "B": "2",
            "a": "11",
            "A": "3",
        }
    )
    assert isinstance(book, BookTickerUpdate)
    assert book.ask_price == Decimal("11")


def test_every_required_market_channel_must_be_fresh() -> None:
    state = FuturesStreamState(
        required_market_symbols=("BTCUSDT",),
        market_stale_after_seconds=2,
        time_provider=lambda: 1.0,
    )
    result = state.ingest(_mark_event(), received_at=1.0)
    assert result.status is StreamIngestStatus.ACCEPTED
    assert state.is_market_stream_stale("BTCUSDT", now=2.0) is True
    assert state.health(now=2.0).stale_market_channels == (
        "BTCUSDT:book_ticker",
        "BTCUSDT:depth",
    )

    state.ingest(_book_event(), received_at=1.0)
    state.initialize_depth("BTCUSDT", 100, received_at=1.0)
    # A snapshot alone is not a continuous depth stream and cannot enable orders.
    assert state.is_market_stream_stale("BTCUSDT", now=2.0) is True
    state.ingest(_depth_event(first=100, final=101, previous=99), received_at=1.0)
    assert state.is_market_stream_stale("BTCUSDT", now=2.0) is False
    assert state.is_market_stream_stale("BTCUSDT", now=4.0) is True


def _mark_event(*, event_time: int = 1000) -> dict:
    return {
        "e": "markPriceUpdate",
        "E": event_time,
        "s": "BTCUSDT",
        "p": "65000",
        "i": "64990",
        "P": "65010",
        "r": "0.0001",
        "T": event_time + 1000,
    }


def _book_event(*, event_time: int = 1000) -> dict:
    return {
        "e": "bookTicker",
        "E": event_time,
        "T": event_time - 1,
        "u": 7,
        "s": "BTCUSDT",
        "b": "64999",
        "B": "2",
        "a": "65001",
        "A": "3",
    }


def _order_event(*, event_time: int = 1000) -> dict:
    return {
        "e": "ORDER_TRADE_UPDATE",
        "E": event_time,
        "T": event_time - 1,
        "o": {
            "s": "BTCUSDT",
            "c": "cet-1",
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
            "n": "0.2",
            "T": event_time - 2,
            "t": 7,
            "R": False,
            "rp": "1.25",
        },
    }


def _account_event(*, event_time: int = 1001) -> dict:
    return {
        "e": "ACCOUNT_UPDATE",
        "E": event_time,
        "T": event_time - 1,
        "a": {
            "m": "ORDER",
            "B": [{"a": "USDT", "wb": "101", "cw": "100", "bc": "1"}],
            "P": [
                {
                    "s": "BTCUSDT",
                    "pa": "0.1",
                    "ep": "65000",
                    "bep": "65002",
                    "cr": "0",
                    "up": "2.5",
                    "mt": "isolated",
                    "iw": "10",
                    "ps": "BOTH",
                }
            ],
        },
    }


def _depth_event(
    *, first: int, final: int, previous: int, event_time: int = 1000
) -> dict:
    return {
        "e": "depthUpdate",
        "E": event_time,
        "T": event_time - 1,
        "s": "BTCUSDT",
        "U": first,
        "u": final,
        "pu": previous,
        "b": [["65000", "1.2"]],
        "a": [["65001", "0.5"]],
    }


def test_user_stream_events_are_parsed_into_typed_values() -> None:
    order = parse_futures_stream_event({"stream": "x", "data": _order_event()})
    account = parse_futures_stream_event(_account_event())
    expired = parse_futures_stream_event(
        {"e": "listenKeyExpired", "E": 1002, "listenKey": "listen-1"}
    )

    assert isinstance(order, OrderTradeUpdate)
    assert order.order_id == 42
    assert order.accumulated_filled_quantity == Decimal("0.01")
    assert order.realized_profit == Decimal("1.25")
    assert isinstance(account, AccountUpdate)
    assert account.balances[0].wallet_balance == Decimal("101")
    assert account.positions[0].unrealized_pnl == Decimal("2.5")
    assert isinstance(expired, ListenKeyExpired)
    assert expired.listen_key == "listen-1"


def test_protective_algo_updates_and_trigger_rejections_are_typed_user_events() -> None:
    update = parse_futures_stream_event(
        {
            "e": "ALGO_UPDATE",
            "E": 1003,
            "T": 1002,
            "o": {
                "caid": "gpt-protective-1",
                "aid": 701,
                "at": "CONDITIONAL",
                "o": "STOP_MARKET",
                "s": "BTCUSDT",
                "S": "SELL",
                "q": "0.1",
                "X": "REJECTED",
                "tp": "49000",
                "R": True,
                "rm": "trigger rejected",
            },
        }
    )
    rejected = parse_futures_stream_event(
        {
            "e": "CONDITIONAL_ORDER_TRIGGER_REJECT",
            "E": 1004,
            "T": 1003,
            "or": {"s": "BTCUSDT", "i": 701, "r": "would immediately trigger"},
        }
    )

    assert isinstance(update, AlgoOrderUpdate)
    assert update.client_algo_id == "gpt-protective-1"
    assert update.reduce_only is True
    assert update.reject_reason == "trigger rejected"
    assert isinstance(rejected, ConditionalOrderTriggerReject)
    assert rejected.algo_id == 701
    assert rejected.reason == "would immediately trigger"


def test_user_stream_rejects_duplicates_and_out_of_order_events() -> None:
    state = FuturesStreamState(time_provider=lambda: 1.0)
    first = state.ingest(_order_event(), received_at=1.0)
    duplicate = state.ingest(_order_event(), received_at=2.0)
    out_of_order = state.ingest(_account_event(event_time=999), received_at=2.0)
    next_event = state.ingest(_account_event(), received_at=2.0)

    assert first.status == StreamIngestStatus.ACCEPTED
    assert duplicate.status == StreamIngestStatus.DUPLICATE
    assert out_of_order.status == StreamIngestStatus.OUT_OF_ORDER
    assert next_event.status == StreamIngestStatus.ACCEPTED
    assert state.health(now=2.0).user_reconciliation_required is True
    assert state.health(now=2.0).ready_for_new_orders is False
    state.mark_user_stream_reconciled()
    assert state.health(now=2.0).user_reconciliation_required is False


def test_disconnect_requires_both_reconciliation_and_a_fresh_heartbeat() -> None:
    state = FuturesStreamState(time_provider=lambda: 1.0)
    state.mark_user_stream_heartbeat(received_at=1.0)
    assert state.health(now=1.0).user_stream_stale is False

    state.mark_user_stream_disconnected()
    disconnected = state.health(now=1.0)
    assert disconnected.user_stream_stale is True
    assert disconnected.user_reconciliation_required is True
    assert disconnected.ready_for_new_orders is False

    state.mark_user_stream_heartbeat(received_at=2.0)
    assert state.health(now=2.0).ready_for_new_orders is False
    state.mark_user_stream_reconciled()
    assert state.health(now=2.0).ready_for_new_orders is True


def test_depth_requires_snapshot_and_detects_pu_gap() -> None:
    state = FuturesStreamState(time_provider=lambda: 1.0)
    before_snapshot = state.ingest(
        _depth_event(first=100, final=102, previous=99), received_at=1.0
    )
    state.initialize_depth("BTCUSDT", 100, received_at=1.0)
    first = state.ingest(_depth_event(first=100, final=102, previous=99), received_at=1.1)
    next_event = state.ingest(
        _depth_event(first=103, final=105, previous=102), received_at=1.2
    )
    gap = state.ingest(_depth_event(first=106, final=107, previous=104), received_at=1.3)
    after_gap = state.ingest(
        _depth_event(first=108, final=109, previous=107), received_at=1.4
    )

    assert before_snapshot.status == StreamIngestStatus.NEEDS_SNAPSHOT
    assert first.status == StreamIngestStatus.ACCEPTED
    assert next_event.status == StreamIngestStatus.ACCEPTED
    assert state.depth_last_update_id("BTCUSDT") == 105
    assert gap.status == StreamIngestStatus.GAP
    assert after_gap.status == StreamIngestStatus.NEEDS_SNAPSHOT
    assert state.health(now=1.4).depth_resync_symbols == ("BTCUSDT",)


def test_stream_staleness_and_listen_key_expiry_block_new_orders() -> None:
    state = FuturesStreamState(
        user_stale_after_seconds=10,
        market_stale_after_seconds=5,
        required_market_symbols={"BTCUSDT"},
        time_provider=lambda: 0.0,
    )
    state.initialize_depth("BTCUSDT", 100, received_at=1.0)
    state.ingest(_depth_event(first=100, final=101, previous=99), received_at=1.0)
    state.ingest(_mark_event(), received_at=1.0)
    state.ingest(_book_event(), received_at=1.0)
    state.ingest(_order_event(), received_at=1.0)

    assert state.health(now=4.0).ready_for_new_orders is True
    assert state.health(now=7.0).stale_market_symbols == ("BTCUSDT",)
    expired = state.ingest(
        {"e": "listenKeyExpired", "E": 2000, "listenKey": "listen-1"},
        received_at=7.0,
    )
    assert expired.status == StreamIngestStatus.LISTEN_KEY_EXPIRED
    assert state.health(now=7.0).listen_key_expired is True
    assert state.health(now=7.0).ready_for_new_orders is False

    state.mark_listen_key_refreshed()
    assert state.health(now=7.0).user_stream_stale is True
