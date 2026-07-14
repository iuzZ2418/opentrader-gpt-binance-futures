from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any


class StreamParseError(ValueError):
    pass


def _decimal(value: Any, *, field_name: str) -> Decimal:
    try:
        result = Decimal(str("0" if value in (None, "") else value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise StreamParseError(f"Invalid decimal in {field_name}: {value!r}") from error
    if not result.is_finite():
        raise StreamParseError(f"Non-finite decimal in {field_name}: {value!r}")
    return result


def _integer(value: Any, *, field_name: str, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise StreamParseError(f"Invalid integer in {field_name}: {value!r}") from error


def _boolean(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


@dataclass(frozen=True, slots=True)
class OrderTradeUpdate:
    event_time: int
    transaction_time: int
    symbol: str
    client_order_id: str
    side: str
    order_type: str
    time_in_force: str
    execution_type: str
    order_status: str
    order_id: int
    original_quantity: Decimal
    original_price: Decimal
    average_price: Decimal
    last_filled_quantity: Decimal
    accumulated_filled_quantity: Decimal
    last_filled_price: Decimal
    commission_asset: str | None
    commission: Decimal
    trade_time: int
    trade_id: int
    reduce_only: bool
    realized_profit: Decimal
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AccountBalanceUpdate:
    asset: str
    wallet_balance: Decimal
    cross_wallet_balance: Decimal
    balance_change: Decimal


@dataclass(frozen=True, slots=True)
class AccountPositionUpdate:
    symbol: str
    position_amount: Decimal
    entry_price: Decimal
    break_even_price: Decimal
    accumulated_realized: Decimal
    unrealized_pnl: Decimal
    margin_type: str
    isolated_wallet: Decimal
    position_side: str


@dataclass(frozen=True, slots=True)
class AccountUpdate:
    event_time: int
    transaction_time: int
    reason: str
    balances: tuple[AccountBalanceUpdate, ...]
    positions: tuple[AccountPositionUpdate, ...]
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ListenKeyExpired:
    event_time: int
    listen_key: str | None
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DepthUpdate:
    event_time: int
    transaction_time: int
    symbol: str
    first_update_id: int
    final_update_id: int
    previous_final_update_id: int
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class MarkPriceUpdate:
    event_time: int
    symbol: str
    mark_price: Decimal
    index_price: Decimal
    estimated_settle_price: Decimal
    funding_rate: Decimal
    next_funding_time: int
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class KlineUpdate:
    event_time: int
    symbol: str
    interval: str
    open_time: int
    close_time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    is_closed: bool
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class BookTickerUpdate:
    event_time: int
    transaction_time: int
    symbol: str
    update_id: int
    bid_price: Decimal
    bid_quantity: Decimal
    ask_price: Decimal
    ask_quantity: Decimal
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ForceOrderUpdate:
    event_time: int
    symbol: str
    side: str
    order_type: str
    status: str
    original_quantity: Decimal
    average_price: Decimal
    accumulated_quantity: Decimal
    trade_time: int
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AlgoOrderUpdate:
    event_time: int
    transaction_time: int
    symbol: str
    client_algo_id: str
    algo_id: int
    side: str
    order_type: str
    status: str
    quantity: Decimal
    trigger_price: Decimal
    reduce_only: bool
    reject_reason: str | None
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ConditionalOrderTriggerReject:
    event_time: int
    transaction_time: int
    symbol: str
    algo_id: int
    reason: str
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class UnknownStreamEvent:
    event_type: str
    event_time: int
    raw: dict[str, Any] = field(repr=False, compare=False)


FuturesStreamEvent = (
    OrderTradeUpdate
    | AccountUpdate
    | ListenKeyExpired
    | DepthUpdate
    | MarkPriceUpdate
    | KlineUpdate
    | BookTickerUpdate
    | ForceOrderUpdate
    | AlgoOrderUpdate
    | ConditionalOrderTriggerReject
    | UnknownStreamEvent
)


def parse_futures_stream_event(
    payload: str | bytes | bytearray | Mapping[str, Any],
) -> FuturesStreamEvent:
    """Parse either a raw stream event or a combined-stream envelope."""

    if isinstance(payload, (str, bytes, bytearray)):
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError) as error:
            raise StreamParseError("WebSocket payload is not valid JSON") from error
    else:
        decoded = dict(payload)
    if not isinstance(decoded, dict):
        raise StreamParseError("WebSocket payload must be an object")
    if "stream" in decoded and "data" in decoded:
        decoded = decoded["data"]
        if not isinstance(decoded, dict):
            raise StreamParseError("Combined stream data must be an object")

    event_type = str(decoded.get("e", ""))
    event_time = _integer(decoded.get("E"), field_name="E")
    if event_type == "ORDER_TRADE_UPDATE":
        order = decoded.get("o")
        if not isinstance(order, dict):
            raise StreamParseError("ORDER_TRADE_UPDATE is missing order data")
        return OrderTradeUpdate(
            event_time=event_time,
            transaction_time=_integer(decoded.get("T"), field_name="T"),
            symbol=str(order.get("s", "")),
            client_order_id=str(order.get("c", "")),
            side=str(order.get("S", "")),
            order_type=str(order.get("o", "")),
            time_in_force=str(order.get("f", "")),
            execution_type=str(order.get("x", "")),
            order_status=str(order.get("X", "")),
            order_id=_integer(order.get("i"), field_name="o.i"),
            original_quantity=_decimal(order.get("q"), field_name="o.q"),
            original_price=_decimal(order.get("p"), field_name="o.p"),
            average_price=_decimal(order.get("ap"), field_name="o.ap"),
            last_filled_quantity=_decimal(order.get("l"), field_name="o.l"),
            accumulated_filled_quantity=_decimal(order.get("z"), field_name="o.z"),
            last_filled_price=_decimal(order.get("L"), field_name="o.L"),
            commission_asset=(str(order["N"]) if order.get("N") is not None else None),
            commission=_decimal(order.get("n"), field_name="o.n"),
            trade_time=_integer(order.get("T"), field_name="o.T"),
            trade_id=_integer(order.get("t"), field_name="o.t", default=-1),
            reduce_only=_boolean(order.get("R")),
            realized_profit=_decimal(order.get("rp"), field_name="o.rp"),
            raw=dict(decoded),
        )
    if event_type == "ACCOUNT_UPDATE":
        account = decoded.get("a")
        if not isinstance(account, dict):
            raise StreamParseError("ACCOUNT_UPDATE is missing account data")
        raw_balances = account.get("B", [])
        raw_positions = account.get("P", [])
        if not isinstance(raw_balances, list) or not isinstance(raw_positions, list):
            raise StreamParseError("ACCOUNT_UPDATE balance and position data must be arrays")
        balances = tuple(
            AccountBalanceUpdate(
                asset=str(item.get("a", "")),
                wallet_balance=_decimal(item.get("wb"), field_name="a.B.wb"),
                cross_wallet_balance=_decimal(item.get("cw"), field_name="a.B.cw"),
                balance_change=_decimal(item.get("bc"), field_name="a.B.bc"),
            )
            for item in raw_balances
            if isinstance(item, dict)
        )
        positions = tuple(
            AccountPositionUpdate(
                symbol=str(item.get("s", "")),
                position_amount=_decimal(item.get("pa"), field_name="a.P.pa"),
                entry_price=_decimal(item.get("ep"), field_name="a.P.ep"),
                break_even_price=_decimal(item.get("bep"), field_name="a.P.bep"),
                accumulated_realized=_decimal(item.get("cr"), field_name="a.P.cr"),
                unrealized_pnl=_decimal(item.get("up"), field_name="a.P.up"),
                margin_type=str(item.get("mt", "")),
                isolated_wallet=_decimal(item.get("iw"), field_name="a.P.iw"),
                position_side=str(item.get("ps", "BOTH")),
            )
            for item in raw_positions
            if isinstance(item, dict)
        )
        return AccountUpdate(
            event_time=event_time,
            transaction_time=_integer(decoded.get("T"), field_name="T"),
            reason=str(account.get("m", "")),
            balances=balances,
            positions=positions,
            raw=dict(decoded),
        )
    if event_type == "listenKeyExpired":
        listen_key = decoded.get("listenKey")
        return ListenKeyExpired(
            event_time=event_time,
            listen_key=str(listen_key) if listen_key is not None else None,
            raw=dict(decoded),
        )
    if event_type == "ALGO_UPDATE":
        order = decoded.get("o")
        if not isinstance(order, dict):
            raise StreamParseError("ALGO_UPDATE is missing algo order data")
        raw_reason = order.get("rm")
        return AlgoOrderUpdate(
            event_time=event_time,
            transaction_time=_integer(decoded.get("T"), field_name="T"),
            symbol=str(order.get("s", "")),
            client_algo_id=str(order.get("caid", "")),
            algo_id=_integer(order.get("aid"), field_name="o.aid"),
            side=str(order.get("S", "")),
            order_type=str(order.get("o", "")),
            status=str(order.get("X", "")),
            quantity=_decimal(order.get("q"), field_name="o.q"),
            trigger_price=_decimal(order.get("tp"), field_name="o.tp"),
            reduce_only=_boolean(order.get("R")),
            reject_reason=(str(raw_reason) if raw_reason not in (None, "") else None),
            raw=dict(decoded),
        )
    if event_type == "CONDITIONAL_ORDER_TRIGGER_REJECT":
        rejection = decoded.get("or")
        if not isinstance(rejection, dict):
            raise StreamParseError(
                "CONDITIONAL_ORDER_TRIGGER_REJECT is missing rejection data"
            )
        return ConditionalOrderTriggerReject(
            event_time=event_time,
            transaction_time=_integer(decoded.get("T"), field_name="T"),
            symbol=str(rejection.get("s", "")),
            algo_id=_integer(rejection.get("i"), field_name="or.i"),
            reason=str(rejection.get("r", "")),
            raw=dict(decoded),
        )
    if event_type == "depthUpdate":
        return DepthUpdate(
            event_time=event_time,
            transaction_time=_integer(decoded.get("T"), field_name="T"),
            symbol=str(decoded.get("s", "")),
            first_update_id=_integer(decoded.get("U"), field_name="U"),
            final_update_id=_integer(decoded.get("u"), field_name="u"),
            previous_final_update_id=_integer(decoded.get("pu"), field_name="pu"),
            bids=_parse_depth_levels(decoded.get("b"), "b"),
            asks=_parse_depth_levels(decoded.get("a"), "a"),
            raw=dict(decoded),
        )
    if event_type == "markPriceUpdate":
        return MarkPriceUpdate(
            event_time=event_time,
            symbol=str(decoded.get("s", "")),
            mark_price=_decimal(decoded.get("p"), field_name="p"),
            index_price=_decimal(decoded.get("i"), field_name="i"),
            estimated_settle_price=_decimal(decoded.get("P"), field_name="P"),
            funding_rate=_decimal(decoded.get("r"), field_name="r"),
            next_funding_time=_integer(decoded.get("T"), field_name="T"),
            raw=dict(decoded),
        )
    if event_type == "kline":
        kline = decoded.get("k")
        if not isinstance(kline, dict):
            raise StreamParseError("kline event is missing kline data")
        return KlineUpdate(
            event_time=event_time,
            symbol=str(kline.get("s", decoded.get("s", ""))),
            interval=str(kline.get("i", "")),
            open_time=_integer(kline.get("t"), field_name="k.t"),
            close_time=_integer(kline.get("T"), field_name="k.T"),
            open=_decimal(kline.get("o"), field_name="k.o"),
            high=_decimal(kline.get("h"), field_name="k.h"),
            low=_decimal(kline.get("l"), field_name="k.l"),
            close=_decimal(kline.get("c"), field_name="k.c"),
            volume=_decimal(kline.get("v"), field_name="k.v"),
            is_closed=_boolean(kline.get("x")),
            raw=dict(decoded),
        )
    if event_type == "bookTicker":
        return BookTickerUpdate(
            event_time=event_time,
            transaction_time=_integer(decoded.get("T"), field_name="T"),
            symbol=str(decoded.get("s", "")),
            update_id=_integer(decoded.get("u"), field_name="u"),
            bid_price=_decimal(decoded.get("b"), field_name="b"),
            bid_quantity=_decimal(decoded.get("B"), field_name="B"),
            ask_price=_decimal(decoded.get("a"), field_name="a"),
            ask_quantity=_decimal(decoded.get("A"), field_name="A"),
            raw=dict(decoded),
        )
    if event_type == "forceOrder":
        order = decoded.get("o")
        if not isinstance(order, dict):
            raise StreamParseError("forceOrder event is missing order data")
        return ForceOrderUpdate(
            event_time=event_time,
            symbol=str(order.get("s", "")),
            side=str(order.get("S", "")),
            order_type=str(order.get("o", "")),
            status=str(order.get("X", "")),
            original_quantity=_decimal(order.get("q"), field_name="o.q"),
            average_price=_decimal(order.get("ap"), field_name="o.ap"),
            accumulated_quantity=_decimal(order.get("z"), field_name="o.z"),
            trade_time=_integer(order.get("T"), field_name="o.T"),
            raw=dict(decoded),
        )
    return UnknownStreamEvent(event_type=event_type, event_time=event_time, raw=dict(decoded))


def _parse_depth_levels(value: Any, field_name: str) -> tuple[tuple[Decimal, Decimal], ...]:
    if not isinstance(value, list):
        raise StreamParseError(f"Depth field {field_name} must be an array")
    levels: list[tuple[Decimal, Decimal]] = []
    for index, level in enumerate(value):
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            raise StreamParseError(f"Invalid depth level in {field_name}[{index}]")
        levels.append(
            (
                _decimal(level[0], field_name=f"{field_name}[{index}].price"),
                _decimal(level[1], field_name=f"{field_name}[{index}].quantity"),
            )
        )
    return tuple(levels)


class StreamIngestStatus(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    NEEDS_SNAPSHOT = "needs_snapshot"
    GAP = "gap"
    LISTEN_KEY_EXPIRED = "listen_key_expired"
    IGNORED = "ignored"


@dataclass(frozen=True, slots=True)
class StreamIngestResult:
    event: FuturesStreamEvent
    status: StreamIngestStatus
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == StreamIngestStatus.ACCEPTED


@dataclass(slots=True)
class _DepthSequence:
    last_update_id: int | None = None
    awaiting_first_update: bool = True
    needs_resync: bool = True
    last_received_at: float | None = None


@dataclass(frozen=True, slots=True)
class StreamHealth:
    user_stream_stale: bool
    stale_market_symbols: tuple[str, ...]
    depth_resync_symbols: tuple[str, ...]
    listen_key_expired: bool
    stale_market_channels: tuple[str, ...] = ()
    user_reconciliation_required: bool = False

    @property
    def ready_for_new_orders(self) -> bool:
        return not (
            self.user_stream_stale
            or self.stale_market_symbols
            or self.depth_resync_symbols
            or self.listen_key_expired
            or self.user_reconciliation_required
        )


class FuturesStreamState:
    """Pure state machine for ordering, depth continuity and receipt-time staleness."""

    REQUIRED_MARKET_CHANNELS = frozenset({"mark_price", "book_ticker", "depth"})

    def __init__(
        self,
        *,
        user_stale_after_seconds: float = 60.0,
        market_stale_after_seconds: float = 5.0,
        required_market_symbols: Iterable[str] = (),
        time_provider: Callable[[], float] = time.monotonic,
        duplicate_window: int = 2_048,
    ) -> None:
        if user_stale_after_seconds <= 0 or market_stale_after_seconds <= 0:
            raise ValueError("Staleness thresholds must be positive")
        self.user_stale_after_seconds = user_stale_after_seconds
        self.market_stale_after_seconds = market_stale_after_seconds
        self.required_market_symbols = {symbol.upper() for symbol in required_market_symbols}
        self.time_provider = time_provider
        self._depth: dict[str, _DepthSequence] = {}
        self._last_market_channel_received: dict[tuple[str, str], float] = {}
        self._last_user_received: float | None = None
        self._last_user_event_time: int | None = None
        self._seen_user_keys: set[str] = set()
        self._seen_user_queue: deque[str] = deque(maxlen=max(1, duplicate_window))
        self.listen_key_expired = False
        self._user_reconciliation_required = False

    def set_required_market_symbols(self, symbols: Iterable[str]) -> None:
        """Atomically replace the subscribed universe and forget removed stream state."""

        required = {symbol.upper() for symbol in symbols}
        removed = self.required_market_symbols - required
        self.required_market_symbols = required
        for symbol in removed:
            self._depth.pop(symbol, None)
            for channel in self.REQUIRED_MARKET_CHANNELS:
                self._last_market_channel_received.pop((symbol, channel), None)

    def initialize_depth(
        self,
        symbol: str,
        last_update_id: int,
        *,
        received_at: float | None = None,
    ) -> None:
        if last_update_id < 0:
            raise ValueError("last_update_id must be non-negative")
        normalized = symbol.upper()
        now = self.time_provider() if received_at is None else received_at
        self._depth[normalized] = _DepthSequence(
            last_update_id=last_update_id,
            awaiting_first_update=True,
            needs_resync=False,
            last_received_at=now,
        )
        # A REST snapshot is only the synchronization anchor. It must not make
        # the depth channel healthy until a continuous update overlaps it.
        self._last_market_channel_received.pop((normalized, "depth"), None)

    initialize_depth_snapshot = initialize_depth

    def mark_listen_key_refreshed(self) -> None:
        self.listen_key_expired = False
        self._last_user_received = None
        self._last_user_event_time = None
        self._seen_user_keys.clear()
        self._seen_user_queue.clear()
        self._user_reconciliation_required = True

    def mark_user_stream_disconnected(self) -> None:
        """Fail closed until a REST reconciliation and a fresh socket heartbeat."""

        self._last_user_received = None
        self._user_reconciliation_required = True

    def mark_user_stream_reconciled(self) -> None:
        """A successful authoritative REST reconciliation clears only that latch."""

        self._user_reconciliation_required = False

    def mark_user_stream_heartbeat(self, *, received_at: float | None = None) -> None:
        """Record that the private socket is connected even when the account is idle."""

        self._last_user_received = (
            self.time_provider() if received_at is None else received_at
        )
        self.listen_key_expired = False

    def ingest(
        self,
        payload: str | bytes | bytearray | Mapping[str, Any],
        *,
        received_at: float | None = None,
    ) -> StreamIngestResult:
        event = parse_futures_stream_event(payload)
        now = self.time_provider() if received_at is None else received_at
        if isinstance(event, DepthUpdate):
            return self._ingest_depth(event, now)
        if isinstance(event, ListenKeyExpired):
            self.listen_key_expired = True
            self._last_user_received = None
            self._user_reconciliation_required = True
            return StreamIngestResult(
                event, StreamIngestStatus.LISTEN_KEY_EXPIRED, "renew listen key and reconcile"
            )
        if isinstance(
            event,
            (
                OrderTradeUpdate,
                AccountUpdate,
                AlgoOrderUpdate,
                ConditionalOrderTriggerReject,
            ),
        ):
            return self._ingest_user_event(event, now)
        if isinstance(event, MarkPriceUpdate):
            symbol = event.symbol.upper()
            self._last_market_channel_received[(symbol, "mark_price")] = now
            return StreamIngestResult(event, StreamIngestStatus.ACCEPTED)
        if isinstance(event, BookTickerUpdate):
            symbol = event.symbol.upper()
            self._last_market_channel_received[(symbol, "book_ticker")] = now
            return StreamIngestResult(event, StreamIngestStatus.ACCEPTED)
        if isinstance(event, (KlineUpdate, ForceOrderUpdate)):
            return StreamIngestResult(event, StreamIngestStatus.ACCEPTED)
        return StreamIngestResult(event, StreamIngestStatus.IGNORED, "unsupported event type")

    def health(self, *, now: float | None = None) -> StreamHealth:
        checked_at = self.time_provider() if now is None else now
        user_stale = self._last_user_received is None or (
            checked_at - self._last_user_received > self.user_stale_after_seconds
        )
        observed_symbols = {
            symbol for symbol, _channel in self._last_market_channel_received
        }
        symbols = self.required_market_symbols | observed_symbols | set(self._depth)
        stale_channels = tuple(
            sorted(
                f"{symbol}:{channel}"
                for symbol in symbols
                for channel in self.REQUIRED_MARKET_CHANNELS
                if (last_received := self._last_market_channel_received.get((symbol, channel)))
                is None
                or checked_at - last_received > self.market_stale_after_seconds
            )
        )
        stale_symbols = tuple(
            sorted(
                {
                    entry.split(":", 1)[0]
                    for entry in stale_channels
                }
            )
        )
        resync_symbols = tuple(
            sorted(symbol for symbol, state in self._depth.items() if state.needs_resync)
        )
        return StreamHealth(
            user_stream_stale=user_stale,
            stale_market_symbols=stale_symbols,
            depth_resync_symbols=resync_symbols,
            listen_key_expired=self.listen_key_expired,
            stale_market_channels=stale_channels,
            user_reconciliation_required=self._user_reconciliation_required,
        )

    def is_user_stream_stale(self, *, now: float | None = None) -> bool:
        return self.health(now=now).user_stream_stale

    def is_market_stream_stale(self, symbol: str, *, now: float | None = None) -> bool:
        return symbol.upper() in self.health(now=now).stale_market_symbols

    def depth_last_update_id(self, symbol: str) -> int | None:
        state = self._depth.get(symbol.upper())
        return state.last_update_id if state else None

    def _ingest_depth(self, event: DepthUpdate, now: float) -> StreamIngestResult:
        symbol = event.symbol.upper()
        state = self._depth.get(symbol)
        if state is None or state.needs_resync or state.last_update_id is None:
            if state is None:
                self._depth[symbol] = _DepthSequence()
            return StreamIngestResult(
                event,
                StreamIngestStatus.NEEDS_SNAPSHOT,
                "load a REST depth snapshot before applying updates",
            )

        if event.final_update_id <= state.last_update_id:
            return StreamIngestResult(
                event, StreamIngestStatus.DUPLICATE, "depth update is already reflected"
            )

        if state.awaiting_first_update:
            target = state.last_update_id + 1
            if not event.first_update_id <= target <= event.final_update_id:
                state.needs_resync = True
                return StreamIngestResult(
                    event,
                    StreamIngestStatus.GAP,
                    "first update does not overlap the REST snapshot",
                )
            state.awaiting_first_update = False
        elif event.previous_final_update_id != state.last_update_id:
            state.needs_resync = True
            return StreamIngestResult(
                event,
                StreamIngestStatus.GAP,
                "depth pu does not equal the previous u",
            )

        state.last_update_id = event.final_update_id
        state.last_received_at = now
        self._last_market_channel_received[(symbol, "depth")] = now
        return StreamIngestResult(event, StreamIngestStatus.ACCEPTED)

    def _ingest_user_event(
        self,
        event: (
            OrderTradeUpdate
            | AccountUpdate
            | AlgoOrderUpdate
            | ConditionalOrderTriggerReject
        ),
        now: float,
    ) -> StreamIngestResult:
        if self._last_user_event_time is not None and event.event_time < self._last_user_event_time:
            self._last_user_received = None
            self._user_reconciliation_required = True
            return StreamIngestResult(
                event,
                StreamIngestStatus.OUT_OF_ORDER,
                "user event time moved backwards; reconcile before applying it",
            )
        key = json.dumps(event.raw, sort_keys=True, separators=(",", ":"))
        if key in self._seen_user_keys:
            return StreamIngestResult(event, StreamIngestStatus.DUPLICATE, "duplicate user event")
        if len(self._seen_user_queue) == self._seen_user_queue.maxlen:
            oldest = self._seen_user_queue.popleft()
            self._seen_user_keys.discard(oldest)
        self._seen_user_queue.append(key)
        self._seen_user_keys.add(key)
        self._last_user_event_time = event.event_time
        self._last_user_received = now
        self.listen_key_expired = False
        return StreamIngestResult(event, StreamIngestStatus.ACCEPTED)
