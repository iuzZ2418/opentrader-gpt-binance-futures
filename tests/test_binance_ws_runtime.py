from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from crypto_event_trader.binance_streams import FuturesStreamState
from crypto_event_trader.binance_ws_runtime import FuturesWebSocketRuntime


class DepthClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def depth(self, symbol: str, *, limit: int) -> dict[str, int]:
        self.calls.append((symbol, limit))
        return {"lastUpdateId": 100 + len(self.calls)}


def test_websocket_runtime_rejects_non_tls_base_before_listen_key_use() -> None:
    with pytest.raises(ValueError, match="binance_ws_url_not_allowlisted"):
        FuturesWebSocketRuntime(
            client=DepthClient(),  # type: ignore[arg-type]
            ws_base_url="ws://fstream.binance.com",
            symbols=("BTCUSDT",),
            state=FuturesStreamState(required_market_symbols=("BTCUSDT",)),
        )


def test_depth_snapshots_are_loaded_for_every_symbol_before_streaming() -> None:
    client = DepthClient()
    state = FuturesStreamState(required_market_symbols=("BTCUSDT", "ETHUSDT"))
    runtime = FuturesWebSocketRuntime(
        client=client,  # type: ignore[arg-type]
        ws_base_url="wss://fstream.example",
        symbols=("BTCUSDT", "ETHUSDT"),
        state=state,
    )

    asyncio.run(runtime._bootstrap_depth())  # noqa: SLF001

    assert client.calls == [("BTCUSDT", 1_000), ("ETHUSDT", 1_000)]
    assert state.depth_last_update_id("BTCUSDT") == 101
    assert state.depth_last_update_id("ETHUSDT") == 102


def test_user_socket_heartbeat_has_a_bounded_freshness_window() -> None:
    state = FuturesStreamState(
        user_stale_after_seconds=10,
        market_stale_after_seconds=5,
        time_provider=lambda: 0,
    )

    assert state.is_user_stream_stale(now=1) is True
    state.mark_user_stream_heartbeat(received_at=5)
    assert state.is_user_stream_stale(now=15) is False
    assert state.is_user_stream_stale(now=15.001) is True


def test_public_stream_list_is_complete_deduplicated_and_non_directional() -> None:
    runtime = FuturesWebSocketRuntime(
        client=DepthClient(),  # type: ignore[arg-type]
        ws_base_url="wss://fstream.example",
        symbols=("btcusdt", "BTCUSDT", "ethusdt"),
        state=FuturesStreamState(),
    )

    assert len(runtime.public_streams) == 12
    assert runtime.high_frequency_streams == (
        "btcusdt@bookTicker",
        "btcusdt@depth@100ms",
        "ethusdt@bookTicker",
        "ethusdt@depth@100ms",
    )
    assert runtime.market_streams[:4] == (
        "btcusdt@markPrice@1s",
        "btcusdt@kline_1h",
        "btcusdt@kline_4h",
        "btcusdt@forceOrder",
    )
    assert runtime.market_streams[4] == "ethusdt@markPrice@1s"


def test_runtime_bootstraps_and_reconciles_before_consumers_start() -> None:
    events: list[str] = []

    class Client:
        def start_user_stream(self) -> str:
            events.append("listen_key")
            return "listen-key"

        def close_user_stream(self, listen_key: str) -> None:
            events.append(f"close:{listen_key}")

    class OrderedRuntime(FuturesWebSocketRuntime):
        async def _bootstrap_depth(self, symbols: Sequence[str] | None = None) -> None:
            del symbols
            events.append("snapshot")

        async def _notify_reconciliation(self, reason: str) -> None:
            events.append(f"reconcile:{reason}")

        async def _consume_public(self, stop_event: asyncio.Event) -> None:
            events.append("public")
            stop_event.set()

        async def _consume_market(self, stop_event: asyncio.Event) -> None:
            del stop_event
            events.append("market")

        async def _consume_user(
            self, listen_key: str, stop_event: asyncio.Event
        ) -> None:
            del stop_event
            events.append(f"user:{listen_key}")

        async def _keepalive_user_stream(
            self, listen_key: str, stop_event: asyncio.Event
        ) -> None:
            del stop_event
            events.append(f"keepalive:{listen_key}")

    runtime = OrderedRuntime(
        client=Client(),  # type: ignore[arg-type]
        ws_base_url="wss://fstream.example",
        symbols=("BTCUSDT",),
        state=FuturesStreamState(),
    )

    asyncio.run(runtime.run(asyncio.Event()))

    assert events[:3] == [
        "snapshot",
        "listen_key",
        "reconcile:startup_or_reconnect",
    ]
    assert set(events[3:7]) == {
        "public",
        "market",
        "user:listen-key",
        "keepalive:listen-key",
    }
    assert events[-1] == "close:listen-key"


class _SocketContext:
    def __init__(self, socket: object) -> None:
        self.socket = socket

    async def __aenter__(self) -> object:
        return self.socket

    async def __aexit__(self, *_: object) -> None:
        return None


class _SocketModule:
    def __init__(self, socket: object) -> None:
        self.socket = socket
        self.urls: list[str] = []

    def connect(self, url: str, **__: object) -> _SocketContext:
        self.urls.append(url)
        return _SocketContext(self.socket)


class _EmptySocket:
    def __aiter__(self) -> _EmptySocket:
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration


def test_runtime_uses_mandatory_public_market_private_routes() -> None:
    runtime = FuturesWebSocketRuntime(
        client=DepthClient(),  # type: ignore[arg-type]
        ws_base_url="wss://fstream.example",
        symbols=("BTCUSDT",),
        state=FuturesStreamState(),
    )
    module = _SocketModule(_EmptySocket())
    runtime._websockets = lambda: module  # type: ignore[method-assign]

    asyncio.run(runtime._consume_public(asyncio.Event()))  # noqa: SLF001
    asyncio.run(runtime._consume_market(asyncio.Event()))  # noqa: SLF001
    stopped = asyncio.Event()
    stopped.set()
    asyncio.run(runtime._consume_user("listen-key", stopped))  # noqa: SLF001

    assert module.urls == [
        (
            "wss://fstream.example/public/stream?streams="
            "btcusdt@bookTicker/btcusdt@depth@100ms"
        ),
        (
            "wss://fstream.example/market/stream?streams="
            "btcusdt@markPrice@1s/btcusdt@kline_1h/"
            "btcusdt@kline_4h/btcusdt@forceOrder"
        ),
        "wss://fstream.example/private/ws/listen-key",
    ]
    assert runtime.state.is_user_stream_stale() is True


class _IdleUserSocket:
    def __init__(
        self,
        *,
        clock: list[float],
        stop_event: asyncio.Event,
        pong_succeeds: bool,
    ) -> None:
        self.clock = clock
        self.stop_event = stop_event
        self.pong_succeeds = pong_succeeds
        self.ping_calls = 0

    async def recv(self) -> object:
        self.clock[0] = 20
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def ping(self) -> asyncio.Future[None]:
        self.ping_calls += 1
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        if self.pong_succeeds:
            waiter.set_result(None)
            self.stop_event.set()
        return waiter


def _idle_user_runtime(
    *, pong_succeeds: bool
) -> tuple[FuturesWebSocketRuntime, FuturesStreamState, _IdleUserSocket, asyncio.Event]:
    clock = [0.0]
    stop_event = asyncio.Event()
    socket = _IdleUserSocket(
        clock=clock,
        stop_event=stop_event,
        pong_succeeds=pong_succeeds,
    )
    state = FuturesStreamState(
        user_stale_after_seconds=10,
        time_provider=lambda: clock[0],
    )
    runtime = FuturesWebSocketRuntime(
        client=DepthClient(),  # type: ignore[arg-type]
        ws_base_url="wss://fstream.example",
        symbols=("BTCUSDT",),
        state=state,
        user_idle_ping_seconds=0.001,
        ping_timeout_seconds=0.001,
    )
    runtime._websockets = lambda: _SocketModule(socket)  # type: ignore[method-assign]
    return runtime, state, socket, stop_event


def test_idle_user_stream_refreshes_heartbeat_only_after_verified_pong() -> None:
    runtime, state, socket, stop_event = _idle_user_runtime(pong_succeeds=True)

    asyncio.run(runtime._consume_user("listen-key", stop_event))  # noqa: SLF001

    assert socket.ping_calls == 1
    assert state.is_user_stream_stale(now=20) is False


def test_failed_user_stream_ping_does_not_refresh_heartbeat() -> None:
    runtime, state, socket, stop_event = _idle_user_runtime(pong_succeeds=False)

    with pytest.raises(TimeoutError):
        asyncio.run(runtime._consume_user("listen-key", stop_event))  # noqa: SLF001

    assert socket.ping_calls == 1
    assert state.is_user_stream_stale(now=20) is True


def test_clean_consumer_exit_triggers_disconnect_reconciliation() -> None:
    events: list[str] = []
    stop_event = asyncio.Event()

    class Client:
        def start_user_stream(self) -> str:
            return "listen-key"

        def close_user_stream(self, listen_key: str) -> None:
            events.append(f"close:{listen_key}")

    class CleanExitRuntime(FuturesWebSocketRuntime):
        async def _bootstrap_depth(self, symbols: Sequence[str] | None = None) -> None:
            del symbols

        async def _notify_reconciliation(self, reason: str) -> None:
            events.append(reason)
            if reason == "websocket_disconnect":
                stop_event.set()

        async def _consume_public(self, event: asyncio.Event) -> None:
            del event
            return None

        async def _consume_market(self, event: asyncio.Event) -> None:
            await event.wait()

        async def _consume_user(
            self, listen_key: str, event: asyncio.Event
        ) -> None:
            del listen_key
            await event.wait()

        async def _keepalive_user_stream(
            self, listen_key: str, event: asyncio.Event
        ) -> None:
            del listen_key
            await event.wait()

    state = FuturesStreamState()
    runtime = CleanExitRuntime(
        client=Client(),  # type: ignore[arg-type]
        ws_base_url="wss://fstream.example",
        symbols=("BTCUSDT",),
        state=state,
    )

    asyncio.run(asyncio.wait_for(runtime.run(stop_event), timeout=0.2))

    assert events[:2] == ["startup_or_reconnect", "websocket_disconnect"]
    assert events[-1] == "close:listen-key"
    health = state.health()
    assert health.user_stream_stale is True
    assert health.user_reconciliation_required is True


def _user_order_event(event_time: int) -> dict[str, object]:
    return {
        "e": "ORDER_TRADE_UPDATE",
        "E": event_time,
        "T": event_time - 1,
        "o": {"s": "BTCUSDT", "i": 1, "c": f"cet-{event_time}"},
    }


class _SequenceUserSocket:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads

    async def recv(self) -> dict[str, object]:
        return self.payloads.pop(0)


def test_out_of_order_user_event_forces_rest_reconciliation_and_stays_unready() -> None:
    stop_event = asyncio.Event()
    reasons: list[str] = []
    state = FuturesStreamState(time_provider=lambda: 1.0)
    socket = _SequenceUserSocket([_user_order_event(1000), _user_order_event(999)])

    def reconcile(reason: str) -> None:
        reasons.append(reason)
        stop_event.set()

    runtime = FuturesWebSocketRuntime(
        client=DepthClient(),  # type: ignore[arg-type]
        ws_base_url="wss://fstream.example",
        symbols=("BTCUSDT",),
        state=state,
        on_reconcile_required=reconcile,
    )
    runtime._websockets = lambda: _SocketModule(socket)  # type: ignore[method-assign]

    asyncio.run(runtime._consume_user("listen-key", stop_event))  # noqa: SLF001

    assert reasons == ["user_event_out_of_order"]
    # REST reconciliation clears its latch, but a backwards event invalidates
    # socket freshness until a subsequent verified heartbeat is observed.
    health = state.health(now=1.0)
    assert health.user_reconciliation_required is False
    assert health.user_stream_stale is True
    assert health.ready_for_new_orders is False
