from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

from .binance import BinanceFuturesClient
from .binance_streams import (
    DepthUpdate,
    FuturesStreamEvent,
    FuturesStreamState,
    StreamIngestStatus,
)
from .security import validate_service_base_url

EventHandler = Callable[[FuturesStreamEvent], Awaitable[None] | None]
ReconcileHandler = Callable[[str], Awaitable[None] | None]


class FuturesWebSocketRuntime:
    """Supervise Binance public/private streams with snapshot-first recovery."""

    def __init__(
        self,
        *,
        client: BinanceFuturesClient,
        ws_base_url: str,
        symbols: Sequence[str],
        state: FuturesStreamState,
        on_event: EventHandler | None = None,
        on_reconcile_required: ReconcileHandler | None = None,
        keepalive_seconds: float = 30 * 60,
        reconnect_max_seconds: float = 30,
        user_idle_ping_seconds: float = 30,
        ping_timeout_seconds: float = 10,
    ) -> None:
        if (
            keepalive_seconds <= 0
            or reconnect_max_seconds <= 0
            or user_idle_ping_seconds <= 0
            or ping_timeout_seconds <= 0
        ):
            raise ValueError("WebSocket timing values must be positive")
        self.client = client
        candidate_url = ws_base_url.rstrip("/")
        host = (urlsplit(candidate_url).hostname or "").lower()
        self.ws_base_url = validate_service_base_url(
            candidate_url,
            service="binance_ws",
            scheme="wss",
            allowed_hosts=frozenset({host}),
            allowed_paths=frozenset({"", "/"}),
        )
        self.symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        self.state = state
        self.on_event = on_event
        self.on_reconcile_required = on_reconcile_required
        self.keepalive_seconds = keepalive_seconds
        self.reconnect_max_seconds = reconnect_max_seconds
        self.user_idle_ping_seconds = user_idle_ping_seconds
        self.ping_timeout_seconds = ping_timeout_seconds

    @property
    def high_frequency_streams(self) -> tuple[str, ...]:
        streams: list[str] = []
        for symbol in self.symbols:
            prefix = symbol.lower()
            streams.extend(
                (
                    f"{prefix}@bookTicker",
                    f"{prefix}@depth@100ms",
                )
            )
        return tuple(streams)

    @property
    def market_streams(self) -> tuple[str, ...]:
        streams: list[str] = []
        for symbol in self.symbols:
            prefix = symbol.lower()
            streams.extend(
                (
                    f"{prefix}@markPrice@1s",
                    f"{prefix}@kline_1h",
                    f"{prefix}@kline_4h",
                    f"{prefix}@forceOrder",
                )
            )
        return tuple(streams)

    @property
    def public_streams(self) -> tuple[str, ...]:
        """Compatibility/introspection view of all non-private subscriptions."""

        return self.high_frequency_streams + self.market_streams

    async def run(self, stop_event: asyncio.Event) -> None:
        backoff = 1.0
        while not stop_event.is_set():
            listen_key: str | None = None
            tasks: list[asyncio.Task[None]] = []
            try:
                await self._bootstrap_depth()
                listen_key = await asyncio.to_thread(self.client.start_user_stream)
                self.state.mark_listen_key_refreshed()
                await self._reconcile_user_state("startup_or_reconnect")
                tasks = [
                    asyncio.create_task(self._consume_public(stop_event)),
                    asyncio.create_task(self._consume_market(stop_event)),
                    asyncio.create_task(self._consume_user(listen_key, stop_event)),
                    asyncio.create_task(
                        self._keepalive_user_stream(listen_key, stop_event)
                    ),
                ]
                done, _ = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if task.cancelled():
                        continue
                    exception = task.exception()
                    if exception is not None:
                        raise exception
                if not stop_event.is_set():
                    raise RuntimeError("Binance WebSocket consumer exited unexpectedly")
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception:
                self.state.mark_user_stream_disconnected()
                await self._reconcile_user_state("websocket_disconnect")
            finally:
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                if listen_key:
                    try:
                        await asyncio.to_thread(self.client.close_user_stream, listen_key)
                    except Exception:
                        pass
                self.state.mark_user_stream_disconnected()
            if not stop_event.is_set():
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                except TimeoutError:
                    pass
                backoff = min(self.reconnect_max_seconds, backoff * 2)

    async def _consume_public(self, stop_event: asyncio.Event) -> None:
        await self._consume_market_data_route(
            route="public",
            streams=self.high_frequency_streams,
            stop_event=stop_event,
            depth_route=True,
        )

    async def _consume_market(self, stop_event: asyncio.Event) -> None:
        await self._consume_market_data_route(
            route="market",
            streams=self.market_streams,
            stop_event=stop_event,
            depth_route=False,
        )

    async def _consume_market_data_route(
        self,
        *,
        route: str,
        streams: Sequence[str],
        stop_event: asyncio.Event,
        depth_route: bool,
    ) -> None:
        websockets = self._websockets()
        stream_path = "/".join(streams)
        url = f"{self.ws_base_url}/{route}/stream?streams={stream_path}"
        async with websockets.connect(url, ping_interval=None, max_queue=10_000) as socket:
            async for payload in socket:
                if stop_event.is_set():
                    return
                result = self.state.ingest(payload)
                if result.status in {
                    StreamIngestStatus.GAP,
                    StreamIngestStatus.NEEDS_SNAPSHOT,
                } and depth_route and isinstance(result.event, DepthUpdate):
                    await self._bootstrap_depth((result.event.symbol,))
                    await self._notify_reconciliation(
                        f"depth_resync:{result.event.symbol}"
                    )
                    continue
                if result.accepted:
                    await self._dispatch(result.event)

    async def _consume_user(
        self, listen_key: str, stop_event: asyncio.Event
    ) -> None:
        websockets = self._websockets()
        url = f"{self.ws_base_url}/private/ws/{listen_key}"
        async with websockets.connect(url, ping_interval=None, max_queue=10_000) as socket:
            while not stop_event.is_set():
                try:
                    payload = await asyncio.wait_for(
                        socket.recv(), timeout=self.user_idle_ping_seconds
                    )
                except TimeoutError:
                    # Account streams may legitimately be idle. Only a verified pong proves
                    # the private socket is still usable; elapsed wall time alone is not a
                    # heartbeat and must never enable new exposure.
                    pong_waiter = await socket.ping()
                    await asyncio.wait_for(
                        pong_waiter, timeout=self.ping_timeout_seconds
                    )
                    self.state.mark_user_stream_heartbeat()
                    continue
                if stop_event.is_set():
                    return
                result = self.state.ingest(payload)
                if result.status is StreamIngestStatus.LISTEN_KEY_EXPIRED:
                    await self._reconcile_user_state("listen_key_expired")
                    raise RuntimeError("Binance listen key expired")
                if result.status is StreamIngestStatus.OUT_OF_ORDER:
                    await self._reconcile_user_state("user_event_out_of_order")
                    continue
                if result.accepted:
                    await self._dispatch(result.event)

    async def _keepalive_user_stream(
        self, listen_key: str, stop_event: asyncio.Event
    ) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=self.keepalive_seconds
                )
            except TimeoutError:
                await asyncio.to_thread(self.client.keepalive_user_stream, listen_key)

    async def _bootstrap_depth(self, symbols: Sequence[str] | None = None) -> None:
        for symbol in symbols or self.symbols:
            snapshot = await asyncio.to_thread(self.client.depth, symbol, limit=1_000)
            last_update_id = int(snapshot.get("lastUpdateId", -1))
            if last_update_id < 0:
                raise ValueError(f"depth snapshot missing lastUpdateId for {symbol}")
            self.state.initialize_depth_snapshot(symbol, last_update_id)

    async def _dispatch(self, event: FuturesStreamEvent) -> None:
        if self.on_event is None:
            return
        result = self.on_event(event)
        if inspect.isawaitable(result):
            await result

    async def _notify_reconciliation(self, reason: str) -> None:
        if self.on_reconcile_required is None:
            return
        result = self.on_reconcile_required(reason)
        if inspect.isawaitable(result):
            await result

    async def _reconcile_user_state(self, reason: str) -> None:
        """Clear the private-stream latch only after the configured REST callback succeeds."""

        await self._notify_reconciliation(reason)
        if self.on_reconcile_required is not None:
            self.state.mark_user_stream_reconciled()

    @staticmethod
    def _websockets() -> Any:
        try:
            import websockets
        except ImportError as error:  # pragma: no cover - optional production dependency
            raise RuntimeError("WebSocket runtime requires the trader extra") from error
        return websockets
