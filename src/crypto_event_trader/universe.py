from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

from .audit import AuditRepository
from .strategy import UniverseMarket, UniverseSelector


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return _as_utc(value, "stored datetime")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return _as_utc(parsed, "stored datetime")


def _parse_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value))


def _row_value(row: object, key: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    try:
        return row[key]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return row[index]  # type: ignore[index]


@dataclass(frozen=True, slots=True)
class LiquidityObservation:
    """A receipt-time market observation; existing rows are never updated in place."""

    observation_id: str
    symbol: str
    quote_asset: str
    contract_type: str
    onboarded_at: datetime
    observed_at: datetime
    turnover_24h: float
    bid_price: float
    ask_price: float
    spread_bps: float
    expected_order_notional: float
    depth_within_20bps: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "quote_asset", self.quote_asset.upper())
        object.__setattr__(self, "contract_type", self.contract_type.upper())
        object.__setattr__(self, "onboarded_at", _as_utc(self.onboarded_at, "onboarded_at"))
        observed_at = _as_utc(self.observed_at, "observed_at")
        object.__setattr__(self, "observed_at", observed_at)
        values = (
            self.turnover_24h,
            self.bid_price,
            self.ask_price,
            self.spread_bps,
            self.expected_order_notional,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("liquidity metrics must be finite")
        if self.turnover_24h < 0 or self.spread_bps < 0:
            raise ValueError("turnover and spread cannot be negative")
        if self.bid_price <= 0 or self.ask_price <= 0 or self.ask_price < self.bid_price:
            raise ValueError("bid and ask must form a positive, non-crossed market")
        if self.expected_order_notional <= 0:
            raise ValueError("expected_order_notional must be positive")
        if self.depth_within_20bps is not None and (
            not math.isfinite(self.depth_within_20bps) or self.depth_within_20bps < 0
        ):
            raise ValueError("depth_within_20bps must be finite and non-negative")

    @property
    def observation_date(self) -> date:
        return self.observed_at.date()


@dataclass(frozen=True, slots=True)
class UniverseSelection:
    selection_id: str
    week_start: date
    as_of: datetime
    symbols: tuple[str, ...]
    fallback_used: bool
    reason: str
    eligible_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _as_utc(self.as_of, "as_of"))
        normalized = tuple(
            dict.fromkeys(symbol.strip().upper() for symbol in self.symbols if symbol.strip())
        )
        object.__setattr__(self, "symbols", normalized)
        if not self.reason.strip():
            raise ValueError("reason cannot be empty")
        if self.eligible_count < 0:
            raise ValueError("eligible_count cannot be negative")


@dataclass(frozen=True, slots=True)
class _ExchangeMarket:
    symbol: str
    quote_asset: str
    contract_type: str
    onboarded_at: datetime


class UniverseMarketDataClient(Protocol):
    def exchange_info(self, *, refresh: bool = False) -> dict[str, Any]: ...

    def ticker_24h(self, symbol: str | None = None) -> Any: ...

    def book_ticker(self, symbol: str | None = None) -> Any: ...

    def depth(self, symbol: str, *, limit: int = 100) -> dict[str, Any]: ...


SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS liquidity_observations (
    observation_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    quote_asset TEXT NOT NULL,
    contract_type TEXT NOT NULL,
    onboarded_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    turnover_24h REAL NOT NULL CHECK(turnover_24h >= 0),
    bid_price REAL NOT NULL CHECK(bid_price > 0),
    ask_price REAL NOT NULL CHECK(ask_price > 0),
    spread_bps REAL NOT NULL CHECK(spread_bps >= 0),
    depth_within_20bps REAL CHECK(depth_within_20bps >= 0),
    expected_order_notional REAL NOT NULL CHECK(expected_order_notional > 0),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_liquidity_point_in_time
    ON liquidity_observations(symbol, observation_date, observed_at);

CREATE TABLE IF NOT EXISTS universe_selections (
    selection_id TEXT PRIMARY KEY,
    week_start TEXT NOT NULL,
    as_of TEXT NOT NULL,
    symbols_json TEXT NOT NULL,
    fallback_used INTEGER NOT NULL CHECK(fallback_used IN (0, 1)),
    reason TEXT NOT NULL,
    eligible_count INTEGER NOT NULL CHECK(eligible_count >= 0),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_universe_selection_week
    ON universe_selections(week_start, as_of);
"""


POSTGRES_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS liquidity_observations (
        observation_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        quote_asset TEXT NOT NULL,
        contract_type TEXT NOT NULL,
        onboarded_at TIMESTAMPTZ NOT NULL,
        observed_at TIMESTAMPTZ NOT NULL,
        observation_date DATE NOT NULL,
        turnover_24h DOUBLE PRECISION NOT NULL CHECK(turnover_24h >= 0),
        bid_price DOUBLE PRECISION NOT NULL CHECK(bid_price > 0),
        ask_price DOUBLE PRECISION NOT NULL CHECK(ask_price > 0),
        spread_bps DOUBLE PRECISION NOT NULL CHECK(spread_bps >= 0),
        depth_within_20bps DOUBLE PRECISION CHECK(depth_within_20bps >= 0),
        expected_order_notional DOUBLE PRECISION NOT NULL
            CHECK(expected_order_notional > 0),
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_liquidity_point_in_time
        ON liquidity_observations(symbol, observation_date, observed_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS universe_selections (
        selection_id TEXT PRIMARY KEY,
        week_start DATE NOT NULL,
        as_of TIMESTAMPTZ NOT NULL,
        symbols_json TEXT NOT NULL,
        fallback_used BOOLEAN NOT NULL,
        reason TEXT NOT NULL,
        eligible_count INTEGER NOT NULL CHECK(eligible_count >= 0),
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_universe_selection_week
        ON universe_selections(week_start, as_of)
    """,
)


class LiquidityObservationStore:
    """Append-only point-in-time storage using the audit repository's connection."""

    def __init__(self, repository: AuditRepository) -> None:
        self.repository = repository

    def initialize(self) -> None:
        self.repository.initialize()
        with self.repository.connect() as connection:
            if self.repository.dialect == "sqlite":
                connection.executescript(SQLITE_SCHEMA)
            else:  # pragma: no cover - exercised against an external PostgreSQL service
                for statement in POSTGRES_SCHEMA:
                    connection.execute(statement)

    def _sql(self, query: str) -> str:
        return query if self.repository.dialect == "sqlite" else query.replace("?", "%s")

    def append_observation(self, observation: LiquidityObservation) -> str:
        query = self._sql(
            """
            INSERT INTO liquidity_observations (
                observation_id, symbol, quote_asset, contract_type, onboarded_at,
                observed_at, observation_date, turnover_24h, bid_price, ask_price,
                spread_bps, depth_within_20bps, expected_order_notional, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        )
        timestamp = observation.observed_at.isoformat()
        values = (
            observation.observation_id,
            observation.symbol,
            observation.quote_asset,
            observation.contract_type,
            observation.onboarded_at.isoformat(),
            timestamp,
            observation.observation_date.isoformat(),
            observation.turnover_24h,
            observation.bid_price,
            observation.ask_price,
            observation.spread_bps,
            observation.depth_within_20bps,
            observation.expected_order_notional,
            datetime.now(UTC).isoformat(),
        )
        with self.repository.connect() as connection:
            connection.execute(query, values)
        return observation.observation_id

    def observations(self, *, as_of: datetime, days: int = 30) -> tuple[LiquidityObservation, ...]:
        cutoff = _as_utc(as_of, "as_of")
        if days <= 0:
            raise ValueError("days must be positive")
        start_date = cutoff.date() - timedelta(days=days - 1)
        query = self._sql(
            """
            SELECT observation_id, symbol, quote_asset, contract_type, onboarded_at,
                   observed_at, turnover_24h, bid_price, ask_price, spread_bps,
                   expected_order_notional, depth_within_20bps
            FROM liquidity_observations
            WHERE observed_at <= ? AND observation_date BETWEEN ? AND ?
            ORDER BY symbol, observation_date, observed_at
            """
        )
        with self.repository.connect() as connection:
            rows = connection.execute(
                query,
                (cutoff.isoformat(), start_date.isoformat(), cutoff.date().isoformat()),
            ).fetchall()
        return tuple(
            LiquidityObservation(
                observation_id=str(_row_value(row, "observation_id", 0)),
                symbol=str(_row_value(row, "symbol", 1)),
                quote_asset=str(_row_value(row, "quote_asset", 2)),
                contract_type=str(_row_value(row, "contract_type", 3)),
                onboarded_at=_parse_datetime(_row_value(row, "onboarded_at", 4)),
                observed_at=_parse_datetime(_row_value(row, "observed_at", 5)),
                turnover_24h=float(_row_value(row, "turnover_24h", 6)),
                bid_price=float(_row_value(row, "bid_price", 7)),
                ask_price=float(_row_value(row, "ask_price", 8)),
                spread_bps=float(_row_value(row, "spread_bps", 9)),
                expected_order_notional=float(
                    _row_value(row, "expected_order_notional", 10)
                ),
                depth_within_20bps=(
                    None
                    if _row_value(row, "depth_within_20bps", 11) is None
                    else float(_row_value(row, "depth_within_20bps", 11))
                ),
            )
            for row in rows
        )

    def append_selection(self, selection: UniverseSelection) -> str:
        query = self._sql(
            """
            INSERT INTO universe_selections (
                selection_id, week_start, as_of, symbols_json, fallback_used,
                reason, eligible_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
        )
        values = (
            selection.selection_id,
            selection.week_start.isoformat(),
            selection.as_of.isoformat(),
            json.dumps(selection.symbols, separators=(",", ":")),
            selection.fallback_used,
            selection.reason,
            selection.eligible_count,
            datetime.now(UTC).isoformat(),
        )
        with self.repository.connect() as connection:
            connection.execute(query, values)
        return selection.selection_id

    def selection_for_week(
        self, *, week_start: date, as_of: datetime
    ) -> UniverseSelection | None:
        return self._selection(
            """
            SELECT selection_id, week_start, as_of, symbols_json, fallback_used,
                   reason, eligible_count
            FROM universe_selections
            WHERE week_start = ? AND as_of <= ?
            ORDER BY as_of DESC, created_at DESC LIMIT 1
            """,
            (week_start.isoformat(), _as_utc(as_of, "as_of").isoformat()),
        )

    def latest_verified_selection_before(
        self, *, week_start: date, as_of: datetime
    ) -> UniverseSelection | None:
        return self._selection(
            """
            SELECT selection_id, week_start, as_of, symbols_json, fallback_used,
                   reason, eligible_count
            FROM universe_selections
            WHERE week_start < ? AND as_of <= ? AND fallback_used = ?
              AND symbols_json <> '[]'
            ORDER BY week_start DESC, as_of DESC, created_at DESC LIMIT 1
            """,
            (week_start.isoformat(), _as_utc(as_of, "as_of").isoformat(), False),
        )

    def _selection(self, query: str, params: tuple[object, ...]) -> UniverseSelection | None:
        with self.repository.connect() as connection:
            row = connection.execute(self._sql(query), params).fetchone()
        if row is None:
            return None
        return UniverseSelection(
            selection_id=str(_row_value(row, "selection_id", 0)),
            week_start=_parse_date(_row_value(row, "week_start", 1)),
            as_of=_parse_datetime(_row_value(row, "as_of", 2)),
            symbols=tuple(json.loads(str(_row_value(row, "symbols_json", 3)))),
            fallback_used=bool(_row_value(row, "fallback_used", 4)),
            reason=str(_row_value(row, "reason", 5)),
            eligible_count=int(_row_value(row, "eligible_count", 6)),
        )


class DynamicUniverseManager:
    """Daily collector and weekly point-in-time USDT perpetual universe selector."""

    def __init__(
        self,
        *,
        client: UniverseMarketDataClient,
        store: LiquidityObservationStore,
        selector: UniverseSelector | None = None,
        fallback_symbols: Sequence[str] = (),
        coverage_days: int = 30,
        depth_limit: int = 100,
    ) -> None:
        if coverage_days < 1:
            raise ValueError("coverage_days must be positive")
        self.client = client
        self.store = store
        self.selector = selector or UniverseSelector()
        self.fallback_symbols = tuple(
            dict.fromkeys(symbol.strip().upper() for symbol in fallback_symbols if symbol.strip())
        )
        self.coverage_days = coverage_days
        self.depth_limit = depth_limit

    def collect_daily(
        self,
        *,
        as_of: datetime,
        expected_order_notional: float | Mapping[str, float],
    ) -> tuple[LiquidityObservation, ...]:
        observed_at = _as_utc(as_of, "as_of")
        markets = self._eligible_exchange_markets(observed_at)
        turnover = self._index_payload(self.client.ticker_24h(), "symbol")
        books = self._index_payload(self.client.book_ticker(), "symbol")

        ranked: list[tuple[_ExchangeMarket, float, float, float, float, float]] = []
        for market in markets:
            ticker = turnover.get(market.symbol)
            book = books.get(market.symbol)
            notional = self._expected_notional(expected_order_notional, market.symbol)
            if ticker is None or book is None or notional is None:
                continue
            try:
                volume = float(ticker["quoteVolume"])
                bid = float(book["bidPrice"])
                ask = float(book["askPrice"])
                spread = self._spread_bps(bid, ask)
            except (KeyError, TypeError, ValueError):
                continue
            if not self._valid_metrics(volume, bid, ask, spread):
                continue
            ranked.append((market, volume, bid, ask, spread, notional))

        ranked.sort(key=lambda item: (-item[1], item[0].symbol))
        depth_symbols = {
            item[0].symbol for item in ranked[: self.selector.retention_rank]
        }
        observations: list[LiquidityObservation] = []
        for market, volume, bid, ask, spread, notional in ranked:
            depth_notional: float | None = None
            if market.symbol in depth_symbols:
                try:
                    snapshot = self.client.depth(market.symbol, limit=self.depth_limit)
                    depth_notional = self.depth_within_20bps(snapshot, bid=bid, ask=ask)
                except (KeyError, TypeError, ValueError, RuntimeError):
                    # The observation remains useful for turnover/spread history. A missing
                    # current depth always makes selection fail closed for this symbol.
                    depth_notional = None
            observation = LiquidityObservation(
                observation_id=f"liq_{uuid4().hex}",
                symbol=market.symbol,
                quote_asset=market.quote_asset,
                contract_type=market.contract_type,
                onboarded_at=market.onboarded_at,
                observed_at=observed_at,
                turnover_24h=volume,
                bid_price=bid,
                ask_price=ask,
                spread_bps=spread,
                expected_order_notional=notional,
                depth_within_20bps=depth_notional,
            )
            self.store.append_observation(observation)
            observations.append(observation)
        return tuple(observations)

    def select_weekly(
        self,
        *,
        as_of: datetime,
        allow_fallback: bool = False,
    ) -> UniverseSelection:
        cutoff = _as_utc(as_of, "as_of")
        week_start = cutoff.date() - timedelta(days=cutoff.weekday())
        existing = self.store.selection_for_week(week_start=week_start, as_of=cutoff)
        if existing is not None and existing.symbols:
            if not existing.fallback_used or allow_fallback:
                return existing
            return UniverseSelection(
                selection_id=f"universe_{uuid4().hex}",
                week_start=week_start,
                as_of=cutoff,
                symbols=(),
                fallback_used=False,
                reason="FALLBACK_REQUIRES_EXPLICIT_OPT_IN",
                eligible_count=existing.eligible_count,
            )

        previous = self.store.latest_verified_selection_before(
            week_start=week_start, as_of=cutoff
        )
        markets = self.point_in_time_markets(as_of=cutoff)
        proposed = self.selector.select(
            markets,
            current_symbols=previous.symbols if previous else (),
            as_of=cutoff,
        )
        complete = len(proposed) == self.selector.size
        if complete:
            symbols = proposed
            fallback_used = False
            reason = "SELECTED"
        elif allow_fallback and self.fallback_symbols:
            symbols = self.fallback_symbols
            fallback_used = True
            reason = "EXPLICIT_FALLBACK"
        else:
            symbols = ()
            fallback_used = False
            reason = "INSUFFICIENT_POINT_IN_TIME_COVERAGE"
        selection = UniverseSelection(
            selection_id=f"universe_{uuid4().hex}",
            week_start=week_start,
            as_of=cutoff,
            symbols=symbols,
            fallback_used=fallback_used,
            reason=reason,
            eligible_count=len(markets),
        )
        self.store.append_selection(selection)
        return selection

    def refresh(
        self,
        *,
        as_of: datetime,
        expected_order_notional: float | Mapping[str, float],
        allow_fallback: bool = False,
    ) -> UniverseSelection:
        self.collect_daily(
            as_of=as_of,
            expected_order_notional=expected_order_notional,
        )
        return self.select_weekly(as_of=as_of, allow_fallback=allow_fallback)

    def point_in_time_markets(self, *, as_of: datetime) -> tuple[UniverseMarket, ...]:
        cutoff = _as_utc(as_of, "as_of")
        observations = self.store.observations(as_of=cutoff, days=self.coverage_days)
        required_dates = {
            cutoff.date() - timedelta(days=offset) for offset in range(self.coverage_days)
        }
        by_symbol: dict[str, dict[date, LiquidityObservation]] = {}
        for observation in observations:
            per_day = by_symbol.setdefault(observation.symbol, {})
            current = per_day.get(observation.observation_date)
            if current is None or observation.observed_at > current.observed_at:
                per_day[observation.observation_date] = observation

        markets: list[UniverseMarket] = []
        for symbol, daily in by_symbol.items():
            if set(daily) != required_dates:
                continue
            ordered = [daily[day] for day in sorted(required_dates)]
            latest = daily[cutoff.date()]
            if latest.depth_within_20bps is None:
                continue
            markets.append(
                UniverseMarket(
                    symbol=symbol,
                    quote_asset=latest.quote_asset,
                    contract_type=latest.contract_type,
                    listed_at=latest.onboarded_at,
                    as_of=latest.observed_at,
                    median_turnover_30d=statistics.median(
                        item.turnover_24h for item in ordered
                    ),
                    median_spread_bps_30d=statistics.median(
                        item.spread_bps for item in ordered
                    ),
                    depth_within_20bps=latest.depth_within_20bps,
                    expected_order_notional=latest.expected_order_notional,
                )
            )
        markets.sort(key=lambda item: (-item.median_turnover_30d, item.symbol))
        return tuple(markets)

    def _eligible_exchange_markets(self, as_of: datetime) -> tuple[_ExchangeMarket, ...]:
        payload = self.client.exchange_info(refresh=True)
        raw_symbols = payload.get("symbols")
        if not isinstance(raw_symbols, list):
            raise ValueError("exchangeInfo symbols must be a list")
        minimum_date = as_of - timedelta(days=self.selector.minimum_listing_days)
        markets: list[_ExchangeMarket] = []
        for raw in raw_symbols:
            if not isinstance(raw, Mapping):
                continue
            try:
                symbol = str(raw["symbol"]).upper()
                quote_asset = str(raw["quoteAsset"]).upper()
                contract_type = str(raw["contractType"]).upper()
                onboarded_at = datetime.fromtimestamp(float(raw["onboardDate"]) / 1000, UTC)
            except (KeyError, TypeError, ValueError, OSError):
                continue
            if (
                quote_asset != "USDT"
                or contract_type != "PERPETUAL"
                or str(raw.get("status", "TRADING")).upper() != "TRADING"
                or onboarded_at > minimum_date
            ):
                continue
            markets.append(
                _ExchangeMarket(symbol, quote_asset, contract_type, onboarded_at)
            )
        return tuple(markets)

    @staticmethod
    def _index_payload(payload: Any, key: str) -> dict[str, Mapping[str, Any]]:
        items = payload if isinstance(payload, list) else [payload]
        indexed: dict[str, Mapping[str, Any]] = {}
        for item in items:
            if isinstance(item, Mapping) and item.get(key):
                indexed[str(item[key]).upper()] = item
        return indexed

    @staticmethod
    def _expected_notional(
        expected: float | Mapping[str, float], symbol: str
    ) -> float | None:
        raw = expected.get(symbol) if isinstance(expected, Mapping) else expected
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    @staticmethod
    def _spread_bps(bid: float, ask: float) -> float:
        mid = (bid + ask) / 2
        if mid <= 0:
            raise ValueError("mid price must be positive")
        return (ask - bid) / mid * 10_000

    @staticmethod
    def _valid_metrics(volume: float, bid: float, ask: float, spread: float) -> bool:
        return (
            all(math.isfinite(value) for value in (volume, bid, ask, spread))
            and volume >= 0
            and bid > 0
            and ask >= bid
            and spread >= 0
        )

    @staticmethod
    def depth_within_20bps(
        snapshot: Mapping[str, Any], *, bid: float, ask: float
    ) -> float:
        """Return conservative two-sided depth: the smaller executable side."""

        mid = (bid + ask) / 2
        lower = mid * (1 - 0.002)
        upper = mid * (1 + 0.002)

        def side_notional(levels: object, *, bids: bool) -> float:
            if not isinstance(levels, list):
                raise ValueError("depth levels must be a list")
            total = 0.0
            for level in levels:
                if not isinstance(level, (list, tuple)) or len(level) < 2:
                    raise ValueError("invalid depth level")
                price = float(level[0])
                quantity = float(level[1])
                if (
                    not math.isfinite(price)
                    or price <= 0
                    or not math.isfinite(quantity)
                    or quantity < 0
                ):
                    raise ValueError("invalid depth price or quantity")
                if (bids and price >= lower) or (not bids and price <= upper):
                    total += price * quantity
            return total

        bid_depth = side_notional(snapshot.get("bids"), bids=True)
        ask_depth = side_notional(snapshot.get("asks"), bids=False)
        return min(bid_depth, ask_depth)
