from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, Literal
from urllib.parse import urlencode, urlparse

import httpx

from .domain import MarketQuote
from .security import (
    BINANCE_DEMO_REST_HOSTS,
    BINANCE_LIVE_REST_HOSTS,
    validate_service_base_url,
)

DEMO_FUTURES_HOSTS = BINANCE_DEMO_REST_HOSTS
PRODUCTION_FUTURES_HOSTS = BINANCE_LIVE_REST_HOSTS
CLIENT_ID_PATTERN = re.compile(r"^[.A-Z:/a-z0-9_-]{1,36}$")


def _decimal(value: Any, default: str = "0") -> Decimal:
    try:
        result = Decimal(str(default if value in (None, "") else value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise BinanceApiError(f"Invalid decimal value from Binance: {value!r}") from error
    if not result.is_finite():
        raise BinanceApiError(f"Non-finite decimal value from Binance: {value!r}")
    return result


def _wire_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return format(value, "f")
    return value


def _boolean(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _clean_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        key: _wire_value(value)
        for key, value in (parameters or {}).items()
        if value is not None
    }


@dataclass(frozen=True, slots=True)
class RateLimitSnapshot:
    """Rate-limit information returned with the most recent REST response."""

    used_weight: dict[str, int] = field(default_factory=dict)
    order_count: dict[str, int] = field(default_factory=dict)
    retry_after_seconds: float | None = None
    observed_at_ms: int = 0


class BinanceApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: int | None = None,
        execution_unknown: bool = False,
        retry_after_seconds: float | None = None,
        rate_limits: RateLimitSnapshot | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.execution_unknown = execution_unknown
        self.retry_after_seconds = retry_after_seconds
        self.rate_limits = rate_limits


class BinanceSafetyError(BinanceApiError):
    """Raised before an unsafe account or production mutation can be sent."""


@dataclass(frozen=True, slots=True)
class SymbolRules:
    symbol: str
    status: str
    contract_type: str
    min_price: Decimal | None
    max_price: Decimal | None
    tick_size: Decimal | None
    min_quantity: Decimal
    max_quantity: Decimal | None
    quantity_step: Decimal
    market_min_quantity: Decimal
    market_max_quantity: Decimal | None
    market_quantity_step: Decimal
    min_notional: Decimal | None
    max_notional: Decimal | None
    min_notional_applies_to_market: bool
    max_notional_applies_to_market: bool

    @classmethod
    def from_exchange_symbol(cls, item: Mapping[str, Any]) -> SymbolRules:
        symbol = str(item.get("symbol", ""))
        if not symbol:
            raise BinanceApiError("Exchange symbol is missing its symbol name")
        filters = {
            str(entry.get("filterType")): entry
            for entry in item.get("filters", [])
            if isinstance(entry, Mapping)
        }
        price_filter = filters.get("PRICE_FILTER", {})
        lot_filter = filters.get("LOT_SIZE", {})
        market_filter = filters.get("MARKET_LOT_SIZE") or lot_filter
        if not lot_filter and not market_filter:
            raise BinanceApiError(f"No lot-size filters for {symbol}")

        lot_step = _decimal(lot_filter.get("stepSize") or market_filter.get("stepSize"))
        market_step = _decimal(market_filter.get("stepSize") or lot_step)
        if market_step <= 0:
            market_filter = lot_filter
            market_step = lot_step
        if lot_step <= 0 or market_step <= 0:
            raise BinanceApiError(f"Invalid quantity step for {symbol}")

        min_filter = filters.get("MIN_NOTIONAL", {})
        notional_filter = filters.get("NOTIONAL", {})
        minimums = [
            parsed
            for value in (
                min_filter.get("notional"),
                min_filter.get("minNotional"),
                notional_filter.get("notional"),
                notional_filter.get("minNotional"),
            )
            if value not in (None, "")
            if (parsed := _decimal(value)) > 0
        ]
        maximums = [
            parsed
            for value in (notional_filter.get("maxNotional"), min_filter.get("maxNotional"))
            if value not in (None, "")
            if (parsed := _decimal(value)) > 0
        ]

        tick = _decimal(price_filter.get("tickSize")) if price_filter else None
        if tick is not None and tick <= 0:
            tick = None

        def optional_positive(value: Any) -> Decimal | None:
            if value in (None, ""):
                return None
            parsed = _decimal(value)
            return parsed if parsed > 0 else None

        return cls(
            symbol=symbol,
            status=str(item.get("status", "")),
            contract_type=str(item.get("contractType", "")),
            min_price=optional_positive(price_filter.get("minPrice")),
            max_price=optional_positive(price_filter.get("maxPrice")),
            tick_size=tick,
            min_quantity=_decimal(lot_filter.get("minQty") or market_filter.get("minQty")),
            max_quantity=optional_positive(
                lot_filter.get("maxQty") or market_filter.get("maxQty")
            ),
            quantity_step=lot_step,
            market_min_quantity=_decimal(
                market_filter.get("minQty") or lot_filter.get("minQty")
            ),
            market_max_quantity=optional_positive(
                market_filter.get("maxQty") or lot_filter.get("maxQty")
            ),
            market_quantity_step=market_step,
            min_notional=max(minimums) if minimums else None,
            max_notional=min(maximums) if maximums else None,
            min_notional_applies_to_market=_boolean(
                notional_filter.get(
                    "applyMinToMarket", min_filter.get("applyToMarket", True)
                ),
                default=True,
            ),
            max_notional_applies_to_market=_boolean(
                notional_filter.get("applyMaxToMarket", True), default=True
            ),
        )

    def normalize_price(self, price: float | Decimal) -> str:
        value = _decimal(price)
        if value <= 0:
            raise BinanceApiError(f"Price must be positive for {self.symbol}")
        if self.tick_size:
            value = (value / self.tick_size).to_integral_value(rounding=ROUND_DOWN)
            value *= self.tick_size
        if self.min_price is not None and value < self.min_price:
            raise BinanceApiError(
                f"Price {value} is below Binance minimum {self.min_price} for {self.symbol}"
            )
        if self.max_price is not None and value > self.max_price:
            raise BinanceApiError(
                f"Price {value} is above Binance maximum {self.max_price} for {self.symbol}"
            )
        return format(value, "f")

    def normalize_quantity(self, quantity: float | Decimal, *, market: bool) -> str:
        value = _decimal(quantity)
        if value <= 0:
            raise BinanceApiError(f"Quantity must be positive for {self.symbol}")
        step = self.market_quantity_step if market else self.quantity_step
        minimum = self.market_min_quantity if market else self.min_quantity
        maximum = self.market_max_quantity if market else self.max_quantity
        value = (value / step).to_integral_value(rounding=ROUND_DOWN) * step
        if value < minimum:
            raise BinanceApiError(
                f"Quantity {value} is below Binance minimum {minimum} for {self.symbol}"
            )
        if maximum is not None and value > maximum:
            raise BinanceApiError(
                f"Quantity {value} is above Binance maximum {maximum} for {self.symbol}"
            )
        return format(value, "f")

    def validate_notional(
        self,
        quantity: float | Decimal,
        price: float | Decimal,
        *,
        market: bool,
    ) -> Decimal:
        notional = _decimal(quantity) * _decimal(price)
        enforce_minimum = not market or self.min_notional_applies_to_market
        enforce_maximum = not market or self.max_notional_applies_to_market
        if enforce_minimum and self.min_notional is not None and notional < self.min_notional:
            raise BinanceApiError(
                f"Notional {notional} is below Binance minimum "
                f"{self.min_notional} for {self.symbol}"
            )
        if enforce_maximum and self.max_notional is not None and notional > self.max_notional:
            raise BinanceApiError(
                f"Notional {notional} is above Binance maximum "
                f"{self.max_notional} for {self.symbol}"
            )
        return notional


@dataclass(frozen=True, slots=True)
class BalanceSnapshot:
    asset: str
    balance: Decimal
    available_balance: Decimal
    cross_wallet_balance: Decimal
    cross_unrealized_pnl: Decimal
    update_time: int
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class PositionRiskSnapshot:
    symbol: str
    position_side: str
    quantity: Decimal
    entry_price: Decimal
    break_even_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    liquidation_price: Decimal
    leverage: int
    margin_type: str
    update_time: int
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class OpenOrderSnapshot:
    symbol: str
    order_id: int
    client_order_id: str
    side: str
    order_type: str
    status: str
    price: Decimal
    original_quantity: Decimal
    executed_quantity: Decimal
    reduce_only: bool
    update_time: int
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class UserTradeSnapshot:
    symbol: str
    trade_id: int
    order_id: int
    side: str
    price: Decimal
    quantity: Decimal
    quote_quantity: Decimal
    realized_pnl: Decimal
    commission: Decimal
    commission_asset: str
    time: int
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class FuturesRestSnapshot:
    observed_at_ms: int
    account: dict[str, Any]
    balances: tuple[BalanceSnapshot, ...]
    positions: tuple[PositionRiskSnapshot, ...]
    open_orders: tuple[OpenOrderSnapshot, ...]
    user_trades: tuple[UserTradeSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    category: str
    key: str
    expected: str | None
    actual: str | None


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    snapshot: FuturesRestSnapshot
    issues: tuple[ReconciliationIssue, ...]

    @property
    def consistent(self) -> bool:
        return not self.issues


def reconcile_rest_snapshot(
    snapshot: FuturesRestSnapshot,
    *,
    expected_open_client_ids: Iterable[str] | None = None,
    expected_positions: Mapping[str, float | Decimal] | None = None,
) -> ReconciliationResult:
    """Compare a remote snapshot with a small, deterministic local ledger projection."""

    issues: list[ReconciliationIssue] = []
    if expected_open_client_ids is not None:
        expected_ids = set(expected_open_client_ids)
        actual_ids = {order.client_order_id for order in snapshot.open_orders}
        for missing in sorted(expected_ids - actual_ids):
            issues.append(ReconciliationIssue("missing_open_order", missing, "open", None))
        for unexpected in sorted(actual_ids - expected_ids):
            issues.append(
                ReconciliationIssue("unexpected_open_order", unexpected, None, "open")
            )

    if expected_positions is not None:
        actual_positions: dict[str, Decimal] = {}
        for position in snapshot.positions:
            actual_positions[position.symbol] = (
                actual_positions.get(position.symbol, Decimal("0")) + position.quantity
            )
        symbols = set(expected_positions) | set(actual_positions)
        for symbol in sorted(symbols):
            expected = _decimal(expected_positions.get(symbol, 0))
            actual = actual_positions.get(symbol, Decimal("0"))
            if expected != actual:
                issues.append(
                    ReconciliationIssue(
                        "position_quantity_mismatch",
                        symbol,
                        format(expected, "f"),
                        format(actual, "f"),
                    )
                )
    return ReconciliationResult(snapshot=snapshot, issues=tuple(issues))


class BinanceFuturesDemoClient:
    """USDⓈ-M Futures REST client with demo-by-default execution safeguards."""

    def __init__(
        self,
        api_key: str | None,
        api_secret: str | None,
        *,
        base_url: str = "https://demo-fapi.binance.com",
        environment: Literal["demo", "production"] = "demo",
        allow_production_trading: bool = False,
        allow_custom_host: bool = False,
        allow_cross_margin: bool = False,
        allow_hedge_mode: bool = False,
        max_leverage: int = 3,
        recv_window_ms: int = 5_000,
        timeout: float = 10,
        transport: httpx.BaseTransport | None = None,
        time_provider: Callable[[], float] = time.time,
    ) -> None:
        if environment not in {"demo", "production"}:
            raise BinanceSafetyError("environment must be 'demo' or 'production'")
        if not 1 <= recv_window_ms <= 60_000:
            raise BinanceSafetyError("recv_window_ms must be between 1 and 60000")
        if timeout <= 0:
            raise BinanceSafetyError("timeout must be positive")
        if not 1 <= max_leverage <= 3:
            raise BinanceSafetyError("The configured leverage safety cap must be between 1 and 3")
        candidate_url = base_url.rstrip("/")
        host = (urlparse(candidate_url).hostname or "").lower()
        is_production = host in PRODUCTION_FUTURES_HOSTS
        is_demo = host in DEMO_FUTURES_HOSTS
        if is_production and environment != "production":
            raise BinanceSafetyError(
                "A Binance production host requires environment='production'"
            )
        if environment == "production" and not is_production:
            raise BinanceSafetyError("Production mode requires an approved Binance Futures host")
        if not (is_production or is_demo) and transport is None and not allow_custom_host:
            raise BinanceSafetyError(
                "Unrecognized Binance host; explicitly set allow_custom_host for a trusted proxy"
            )
        try:
            normalized_url = validate_service_base_url(
                candidate_url,
                service="binance_rest",
                scheme="https",
                allowed_hosts=(
                    DEMO_FUTURES_HOSTS | PRODUCTION_FUTURES_HOSTS
                    if is_production or is_demo
                    else frozenset({host})
                ),
                allowed_paths=frozenset({"", "/"}),
            )
        except ValueError as error:
            raise BinanceSafetyError(str(error)) from error

        self.api_key = api_key
        self.api_secret = api_secret
        self.environment = environment
        self.is_production = is_production
        self.allow_production_trading = allow_production_trading
        self.allow_cross_margin = allow_cross_margin
        self.allow_hedge_mode = allow_hedge_mode
        self.max_leverage = max_leverage
        self.recv_window_ms = recv_window_ms
        self.time_provider = time_provider
        self.time_offset_ms = 0
        self.time_synced = False
        self.client = httpx.Client(
            base_url=normalized_url,
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "crypto-event-trader/0.2"},
        )
        self._symbols: dict[str, dict[str, Any]] | None = None
        self._rules: dict[str, SymbolRules] = {}
        self._submitted_client_ids: set[str] = set()
        self._submitted_algo_client_ids: set[str] = set()
        self._mutation_authorization_lock = threading.RLock()
        self.rate_limits = RateLimitSnapshot()
        self._rate_limit_blocked_until = 0.0

    def __enter__(self) -> BinanceFuturesDemoClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def ping(self) -> bool:
        self._request("GET", "/fapi/v1/ping")
        return True

    def server_time(self) -> int:
        payload = self._request("GET", "/fapi/v1/time")
        return int(payload["serverTime"])

    def sync_time(self) -> int:
        local_before = int(self.time_provider() * 1000)
        server = self.server_time()
        local_after = int(self.time_provider() * 1000)
        midpoint = (local_before + local_after) // 2
        self.time_offset_ms = server - midpoint
        self.time_synced = True
        return self.time_offset_ms

    def fetch_quotes(self, symbols: dict[str, str]) -> dict[str, MarketQuote]:
        quotes: dict[str, MarketQuote] = {}
        for local_symbol, exchange_symbol in symbols.items():
            book = self._request(
                "GET", "/fapi/v1/ticker/bookTicker", {"symbol": exchange_symbol}
            )
            stats = self._request("GET", "/fapi/v1/ticker/24hr", {"symbol": exchange_symbol})
            bid = float(book["bidPrice"])
            ask = float(book["askPrice"])
            last = float(stats.get("lastPrice") or (bid + ask) / 2)
            quotes[local_symbol] = MarketQuote(
                symbol=local_symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume_24h=float(stats.get("quoteVolume", 0)),
            )
        return quotes

    def ticker_24h(self, symbol: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        """Return one or all USD-M Futures 24-hour ticker statistics."""

        payload = self._request("GET", "/fapi/v1/ticker/24hr", {"symbol": symbol})
        if isinstance(payload, dict):
            return payload
        return self._expect_list(payload, "24-hour ticker statistics")

    def book_ticker(self, symbol: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        """Return one or all USD-M Futures best bid/ask snapshots."""

        payload = self._request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        if isinstance(payload, dict):
            return payload
        return self._expect_list(payload, "book ticker")

    def klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[list[Any]]:
        if not 1 <= limit <= 1_500:
            raise BinanceApiError("Futures kline limit must be between 1 and 1500")
        payload = self._request(
            "GET",
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "startTime": start_time,
                "endTime": end_time,
            },
        )
        if not isinstance(payload, list) or not all(isinstance(item, list) for item in payload):
            raise BinanceApiError("Unexpected kline response")
        return payload

    def premium_index(self, symbol: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        payload = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        if isinstance(payload, dict):
            return payload
        return self._expect_list(payload, "premium index")

    def open_interest(self, symbol: str) -> dict[str, Any]:
        payload = self._request("GET", "/fapi/v1/openInterest", {"symbol": symbol})
        return self._expect_dict(payload, "open interest")

    def open_interest_history(
        self,
        symbol: str,
        *,
        period: str = "1h",
        limit: int = 25,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise BinanceApiError("Open-interest history limit must be between 1 and 500")
        payload = self._request(
            "GET",
            "/futures/data/openInterestHist",
            {
                "symbol": symbol,
                "period": period,
                "limit": limit,
                "startTime": start_time,
                "endTime": end_time,
            },
        )
        return self._expect_list(payload, "open interest history")

    def depth(self, symbol: str, *, limit: int = 100) -> dict[str, Any]:
        if limit not in {5, 10, 20, 50, 100, 500, 1_000}:
            raise BinanceApiError("Unsupported Futures depth limit")
        payload = self._request(
            "GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit}
        )
        return self._expect_dict(payload, "depth snapshot")

    def funding_rate_history(
        self,
        symbol: str | None = None,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1_000:
            raise BinanceApiError("Funding history limit must be between 1 and 1000")
        payload = self._request(
            "GET",
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol,
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit,
            },
        )
        return self._expect_list(payload, "funding history")

    def adl_quantile(self, symbol: str | None = None) -> list[dict[str, Any]]:
        payload = self._request(
            "GET", "/fapi/v1/adlQuantile", {"symbol": symbol}, signed=True
        )
        return self._expect_list(payload, "ADL quantile")

    def account_information(self) -> dict[str, Any]:
        payload = self._request("GET", "/fapi/v3/account", signed=True)
        return self._expect_dict(payload, "account information")

    def account_balance(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/fapi/v3/balance", signed=True)
        return self._expect_list(payload, "account balance")

    def position_risk(self, symbol: str | None = None) -> list[dict[str, Any]]:
        payload = self._request(
            "GET", "/fapi/v3/positionRisk", {"symbol": symbol}, signed=True
        )
        return self._expect_list(payload, "position risk")

    def get_position_mode(self) -> bool:
        """Return ``True`` only when the account is in hedge (dual-side) mode."""

        payload = self._request(
            "GET", "/fapi/v1/positionSide/dual", signed=True
        )
        value = self._expect_dict(payload, "position mode").get("dualSidePosition")
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise BinanceApiError("Binance position mode response is invalid")

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        payload = self._request(
            "GET", "/fapi/v1/openOrders", {"symbol": symbol}, signed=True
        )
        return self._expect_list(payload, "open orders")

    def user_trades(
        self,
        symbol: str,
        *,
        order_id: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        from_id: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/fapi/v1/userTrades",
            {
                "symbol": symbol,
                "orderId": order_id,
                "startTime": start_time,
                "endTime": end_time,
                "fromId": from_id,
                "limit": limit,
            },
            signed=True,
        )
        return self._expect_list(payload, "user trades")

    def income_history(
        self,
        *,
        symbol: str | None = None,
        income_type: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return signed USD-M account income records (for example ``FUNDING_FEE``)."""

        if not 1 <= limit <= 1_000:
            raise BinanceApiError("Income history limit must be between 1 and 1000")
        normalized_type = income_type.strip().upper() if income_type else None
        payload = self._request(
            "GET",
            "/fapi/v1/income",
            {
                "symbol": symbol,
                "incomeType": normalized_type,
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit,
            },
            signed=True,
        )
        return self._expect_list(payload, "income history")

    get_account_information = account_information
    get_account_balance = account_balance
    get_position_risk = position_risk
    is_hedge_mode = get_position_mode
    get_open_orders = open_orders
    get_user_trades = user_trades
    get_income_history = income_history

    def rest_snapshot(
        self,
        *,
        symbol: str | None = None,
        trade_symbols: Iterable[str] = (),
    ) -> FuturesRestSnapshot:
        account = self.account_information()
        balances = tuple(self._parse_balance(item) for item in self.account_balance())
        positions = tuple(self._parse_position(item) for item in self.position_risk(symbol))
        orders = tuple(self._parse_open_order(item) for item in self.open_orders(symbol))
        trades = tuple(
            self._parse_trade(item)
            for trade_symbol in trade_symbols
            for item in self.user_trades(trade_symbol)
        )
        return FuturesRestSnapshot(
            observed_at_ms=int(self.time_provider() * 1000),
            account=account,
            balances=balances,
            positions=positions,
            open_orders=orders,
            user_trades=trades,
        )

    def reconcile(
        self,
        *,
        symbol: str | None = None,
        expected_open_client_ids: Iterable[str] | None = None,
        expected_positions: Mapping[str, float | Decimal] | None = None,
    ) -> ReconciliationResult:
        return reconcile_rest_snapshot(
            self.rest_snapshot(symbol=symbol),
            expected_open_client_ids=expected_open_client_ids,
            expected_positions=expected_positions,
        )

    def exchange_info(self, *, refresh: bool = False) -> dict[str, Any]:
        if refresh:
            self._symbols = None
            self._rules.clear()
        payload = self._request("GET", "/fapi/v1/exchangeInfo")
        return self._expect_dict(payload, "exchange info")

    def symbol_rules(self, symbol: str, *, refresh: bool = False) -> SymbolRules:
        if refresh:
            self._rules.pop(symbol, None)
            self._symbols = None
        if symbol in self._rules:
            return self._rules[symbol]
        if self._symbols is None:
            exchange_info = self.exchange_info()
            self._symbols = {
                str(item["symbol"]): item
                for item in exchange_info.get("symbols", [])
                if isinstance(item, dict) and item.get("symbol")
            }
        details = self._symbols.get(symbol)
        if not details:
            raise BinanceApiError(f"Unknown Binance Futures symbol: {symbol}")
        rules = SymbolRules.from_exchange_symbol(details)
        self._rules[symbol] = rules
        return rules

    def normalize_quantity(
        self,
        symbol: str,
        quantity: float | Decimal,
        *,
        market: bool = True,
    ) -> str:
        return self.symbol_rules(symbol).normalize_quantity(quantity, market=market)

    def normalize_price(self, symbol: str, price: float | Decimal) -> str:
        return self.symbol_rules(symbol).normalize_price(price)

    @staticmethod
    def generate_client_order_id(prefix: str = "cet") -> str:
        safe_prefix = re.sub(r"[^.A-Z:/a-z0-9_-]", "-", prefix).strip("-") or "cet"
        identifier = f"{safe_prefix[:11]}-{uuid.uuid4().hex[:24]}"
        return identifier[:36]

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str,
        reference_price: float | Decimal | None = None,
    ) -> dict[str, Any]:
        normalized = self.normalize_quantity(symbol, quantity, market=True)
        if reference_price is not None:
            self.symbol_rules(symbol).validate_notional(
                normalized, reference_price, market=True
            )
        return self._place_standard_order(
            {
                "symbol": symbol,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": normalized,
                "newClientOrderId": client_order_id,
                "newOrderRespType": "RESULT",
            }
        )

    def place_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float | Decimal,
        price: float | Decimal,
        client_order_id: str,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        normalized_quantity = self.normalize_quantity(symbol, quantity, market=False)
        normalized_price = self.normalize_price(symbol, price)
        self.symbol_rules(symbol).validate_notional(
            normalized_quantity, normalized_price, market=False
        )
        return self._place_standard_order(
            {
                "symbol": symbol,
                "side": side.upper(),
                "type": "LIMIT",
                "timeInForce": time_in_force.upper(),
                "quantity": normalized_quantity,
                "price": normalized_price,
                "newClientOrderId": client_order_id,
                "newOrderRespType": "ACK",
            }
        )

    def place_reduce_only_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float | Decimal,
        client_order_id: str,
    ) -> dict[str, Any]:
        normalized = self.normalize_quantity(symbol, quantity, market=True)
        return self._place_standard_order(
            {
                "symbol": symbol,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": normalized,
                "reduceOnly": True,
                "newClientOrderId": client_order_id,
                "newOrderRespType": "RESULT",
            }
        )

    def place_stop_market_algo_order(
        self,
        *,
        symbol: str,
        side: str,
        trigger_price: float | Decimal,
        client_algo_id: str,
        quantity: float | Decimal | None = None,
        close_position: bool = False,
        working_type: str = "MARK_PRICE",
        price_protect: bool = True,
    ) -> dict[str, Any]:
        if close_position == (quantity is not None):
            raise BinanceApiError(
                "STOP_MARKET requires either quantity or close_position=true, but not both"
            )
        parameters: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side.upper(),
            "type": "STOP_MARKET",
            "triggerPrice": self.normalize_price(symbol, trigger_price),
            "workingType": working_type.upper(),
            "priceProtect": price_protect,
            "clientAlgoId": client_algo_id,
            "newOrderRespType": "ACK",
        }
        if close_position:
            parameters["closePosition"] = True
        else:
            parameters["quantity"] = self.normalize_quantity(symbol, quantity, market=True)
            parameters["reduceOnly"] = True
        return self._place_algo_order(parameters)

    def query_order(self, *, symbol: str, client_order_id: str) -> dict[str, Any]:
        self._validate_client_id(client_order_id)
        payload = self._request(
            "GET",
            "/fapi/v1/order",
            {"symbol": symbol, "origClientOrderId": client_order_id},
            signed=True,
        )
        return self._expect_dict(payload, "order query")

    def cancel_order(self, *, symbol: str, client_order_id: str) -> dict[str, Any]:
        """Cancel by deterministic client id; unknown outcomes must be reconciled by caller."""

        self._validate_client_id(client_order_id)
        payload = self._request(
            "DELETE",
            "/fapi/v1/order",
            {"symbol": symbol, "origClientOrderId": client_order_id},
            signed=True,
        )
        return self._expect_dict(payload, "order cancellation")

    def cancel_all_open_orders(self, *, symbol: str) -> dict[str, Any]:
        payload = self._request(
            "DELETE",
            "/fapi/v1/allOpenOrders",
            {"symbol": symbol},
            signed=True,
        )
        return self._expect_dict(payload, "open order cancellation")

    def query_algo_order(self, *, client_algo_id: str) -> dict[str, Any]:
        self._validate_client_id(client_algo_id)
        payload = self._request(
            "GET",
            "/fapi/v1/algoOrder",
            {"clientAlgoId": client_algo_id},
            signed=True,
        )
        return self._expect_dict(payload, "algo order query")

    def cancel_algo_order(self, *, client_algo_id: str) -> dict[str, Any]:
        """Cancel a conditional order by its deterministic client id.

        As with standard orders, callers must query the algo order after this mutation before
        assuming it is terminal; the cancellation response alone is not a fill barrier.
        """

        self._validate_client_id(client_algo_id)
        payload = self._request(
            "DELETE",
            "/fapi/v1/algoOrder",
            {"clientAlgoId": client_algo_id},
            signed=True,
        )
        return self._expect_dict(payload, "algo order cancellation")

    def open_algo_orders(self, *, symbol: str | None = None) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/fapi/v1/openAlgoOrders",
            {"symbol": symbol},
            signed=True,
        )
        return self._expect_list(payload, "open algo orders")

    def set_leverage(self, *, symbol: str, leverage: int) -> dict[str, Any]:
        if not 1 <= leverage <= self.max_leverage:
            raise BinanceSafetyError(
                f"Leverage {leverage} exceeds configured safety cap {self.max_leverage}"
            )
        self._assert_account_mutation_allowed()
        payload = self._request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": leverage},
            signed=True,
        )
        return self._expect_dict(payload, "leverage change")

    change_initial_leverage = set_leverage
    change_leverage = set_leverage

    def set_margin_type(self, *, symbol: str, margin_type: str = "ISOLATED") -> dict[str, Any]:
        normalized = margin_type.upper()
        if normalized not in {"ISOLATED", "CROSSED"}:
            raise BinanceApiError("margin_type must be ISOLATED or CROSSED")
        if normalized == "CROSSED" and not self.allow_cross_margin:
            raise BinanceSafetyError("Cross margin is disabled by the gateway safety policy")
        self._assert_account_mutation_allowed()
        payload = self._request(
            "POST",
            "/fapi/v1/marginType",
            {"symbol": symbol, "marginType": normalized},
            signed=True,
        )
        return self._expect_dict(payload, "margin type change")

    change_margin_type = set_margin_type

    def set_position_mode(self, *, dual_side_position: bool = False) -> dict[str, Any]:
        if dual_side_position and not self.allow_hedge_mode:
            raise BinanceSafetyError("Hedge mode is disabled by the gateway safety policy")
        self._assert_account_mutation_allowed()
        payload = self._request(
            "POST",
            "/fapi/v1/positionSide/dual",
            {"dualSidePosition": dual_side_position},
            signed=True,
        )
        return self._expect_dict(payload, "position mode change")

    change_position_mode = set_position_mode

    place_algo_stop_market_order = place_stop_market_algo_order

    def start_user_stream(self) -> str:
        payload = self._request(
            "POST", "/fapi/v1/listenKey", api_key_required=True
        )
        listen_key = self._expect_dict(payload, "user stream").get("listenKey")
        if not isinstance(listen_key, str) or not listen_key:
            raise BinanceApiError("Binance user stream response is missing listenKey")
        return listen_key

    def keepalive_user_stream(self, listen_key: str) -> None:
        self._request(
            "PUT",
            "/fapi/v1/listenKey",
            {"listenKey": listen_key},
            api_key_required=True,
        )

    def close_user_stream(self, listen_key: str) -> None:
        self._request(
            "DELETE",
            "/fapi/v1/listenKey",
            {"listenKey": listen_key},
            api_key_required=True,
        )

    def _place_standard_order(self, parameters: dict[str, Any]) -> dict[str, Any]:
        self._assert_account_mutation_allowed()
        client_order_id = str(parameters["newClientOrderId"])
        symbol = str(parameters["symbol"])
        self._validate_client_id(client_order_id)
        if client_order_id in self._submitted_client_ids:
            return self.query_order(symbol=symbol, client_order_id=client_order_id)
        self._submitted_client_ids.add(client_order_id)
        try:
            payload = self._request("POST", "/fapi/v1/order", parameters, signed=True)
        except BinanceApiError as error:
            if not error.execution_unknown and not self._is_duplicate_client_id_error(error):
                raise
            try:
                return self.query_order(symbol=symbol, client_order_id=client_order_id)
            except BinanceApiError:
                raise error from None
        return self._expect_dict(payload, "order")

    def _place_algo_order(self, parameters: dict[str, Any]) -> dict[str, Any]:
        self._assert_account_mutation_allowed()
        client_algo_id = str(parameters["clientAlgoId"])
        self._validate_client_id(client_algo_id)
        if client_algo_id in self._submitted_algo_client_ids:
            return self.query_algo_order(client_algo_id=client_algo_id)
        self._submitted_algo_client_ids.add(client_algo_id)
        try:
            payload = self._request("POST", "/fapi/v1/algoOrder", parameters, signed=True)
        except BinanceApiError as error:
            if not error.execution_unknown and not self._is_duplicate_client_id_error(error):
                raise
            try:
                return self.query_algo_order(client_algo_id=client_algo_id)
            except BinanceApiError:
                raise error from None
        return self._expect_dict(payload, "algo order")

    def _assert_account_mutation_allowed(self) -> None:
        if self.is_production and not self.allow_production_trading:
            raise BinanceSafetyError(
                "Production account mutations are locked; explicit dual-control unlock is required"
            )

    @contextmanager
    def mutation_authorization(self) -> Iterator[None]:
        """Grant one serialized gateway mutation and restore the previous lock state.

        The production permission is deliberately scoped to the duration of one REST call.  A
        reduce-only emergency order therefore cannot leave the client unlocked for a later entry
        mutation, including when an exception interrupts the call.
        """

        with self._mutation_authorization_lock:
            previous = self.allow_production_trading
            self.allow_production_trading = True
            try:
                yield
            finally:
                self.allow_production_trading = previous

    @staticmethod
    def _validate_client_id(client_id: str) -> None:
        if not CLIENT_ID_PATTERN.fullmatch(client_id):
            raise BinanceApiError(
                "Client order id must match ^[.A-Z:/a-z0-9_-]{1,36}$"
            )

    @staticmethod
    def _is_duplicate_client_id_error(error: BinanceApiError) -> bool:
        if error.code in {-2010, -4116}:
            return True
        message = str(error).lower()
        return any(
            marker in message
            for marker in ("duplicate", "not unique", "client order id is not")
        )

    def _request(
        self,
        method: str,
        path: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        signed: bool = False,
        api_key_required: bool = False,
    ) -> Any:
        method = method.upper()
        now = self.time_provider()
        if now < self._rate_limit_blocked_until:
            retry_after = self._rate_limit_blocked_until - now
            raise BinanceApiError(
                "Binance REST rate-limit embargo is active",
                status_code=429,
                retry_after_seconds=retry_after,
                rate_limits=self.rate_limits,
            )
        params = _clean_parameters(parameters)
        headers: dict[str, str] = {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise BinanceApiError("Binance Futures API credentials are not configured")
            if not self.time_synced:
                self.sync_time()
            params["recvWindow"] = self.recv_window_ms
            params["timestamp"] = int(self.time_provider() * 1000) + self.time_offset_ms
            payload = urlencode(params)
            params["signature"] = hmac.new(
                self.api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            headers["X-MBX-APIKEY"] = self.api_key
        elif api_key_required:
            if not self.api_key:
                raise BinanceApiError("Binance Futures API key is not configured")
            headers["X-MBX-APIKEY"] = self.api_key
        try:
            response = self.client.request(method, path, params=params, headers=headers)
        except httpx.HTTPError as error:
            execution_unknown = method in {"POST", "PUT", "DELETE"} and not isinstance(
                error, httpx.ConnectError
            )
            raise BinanceApiError(
                f"Binance network error: {error}", execution_unknown=execution_unknown
            ) from error

        self.rate_limits = self._parse_rate_limits(response.headers)
        if response.is_success:
            try:
                return response.json()
            except ValueError as error:
                raise BinanceApiError(
                    "Binance returned invalid JSON", rate_limits=self.rate_limits
                ) from error

        try:
            body = response.json()
        except ValueError:
            body = {"msg": response.text}
        raw_code = body.get("code") if isinstance(body, dict) else None
        try:
            code = int(raw_code) if raw_code is not None else None
        except (TypeError, ValueError):
            code = None
        message = body.get("msg", response.text) if isinstance(body, dict) else response.text
        execution_unknown = response.status_code == 503 and method in {
            "POST",
            "PUT",
            "DELETE",
        }
        if response.status_code in {418, 429}:
            # Binance returns Retry-After for both request throttles and IP bans.  Never spin on
            # the endpoint when it is absent: use a conservative local embargo and let the
            # supervisor retry only after it expires.
            fallback = 300.0 if response.status_code == 418 else 60.0
            retry_after = self.rate_limits.retry_after_seconds or fallback
            self._rate_limit_blocked_until = max(
                self._rate_limit_blocked_until,
                self.time_provider() + retry_after,
            )
        raise BinanceApiError(
            f"Binance API error {raw_code}: {message}",
            status_code=response.status_code,
            code=code,
            execution_unknown=execution_unknown,
            retry_after_seconds=self.rate_limits.retry_after_seconds,
            rate_limits=self.rate_limits,
        )

    def _parse_rate_limits(self, headers: httpx.Headers) -> RateLimitSnapshot:
        used_weight = dict(self.rate_limits.used_weight)
        order_count = dict(self.rate_limits.order_count)
        for key, value in headers.items():
            normalized = key.lower()
            try:
                count = int(value)
            except ValueError:
                continue
            if normalized.startswith("x-mbx-used-weight-"):
                used_weight[normalized.removeprefix("x-mbx-used-weight-")] = count
            elif normalized.startswith("x-mbx-order-count-"):
                order_count[normalized.removeprefix("x-mbx-order-count-")] = count
        return RateLimitSnapshot(
            used_weight=used_weight,
            order_count=order_count,
            retry_after_seconds=self._retry_after_seconds(headers.get("Retry-After")),
            observed_at_ms=int(self.time_provider() * 1000),
        )

    def _retry_after_seconds(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                now = datetime.fromtimestamp(self.time_provider(), UTC)
                return max(0.0, (retry_at - now).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _expect_dict(payload: Any, description: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise BinanceApiError(f"Unexpected {description} response")
        return payload

    @staticmethod
    def _expect_list(payload: Any, description: str) -> list[dict[str, Any]]:
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise BinanceApiError(f"Unexpected {description} response")
        return payload

    @staticmethod
    def _parse_balance(item: dict[str, Any]) -> BalanceSnapshot:
        return BalanceSnapshot(
            asset=str(item.get("asset", "")),
            balance=_decimal(item.get("balance")),
            available_balance=_decimal(item.get("availableBalance")),
            cross_wallet_balance=_decimal(item.get("crossWalletBalance")),
            cross_unrealized_pnl=_decimal(item.get("crossUnPnl")),
            update_time=int(item.get("updateTime", 0)),
            raw=dict(item),
        )

    @staticmethod
    def _parse_position(item: dict[str, Any]) -> PositionRiskSnapshot:
        isolated = item.get("isolated")
        margin_type = str(item.get("marginType", ""))
        if not margin_type and isolated is not None:
            margin_type = "isolated" if isolated else "cross"
        return PositionRiskSnapshot(
            symbol=str(item.get("symbol", "")),
            position_side=str(item.get("positionSide", "BOTH")),
            quantity=_decimal(item.get("positionAmt")),
            entry_price=_decimal(item.get("entryPrice")),
            break_even_price=_decimal(item.get("breakEvenPrice")),
            mark_price=_decimal(item.get("markPrice")),
            unrealized_pnl=_decimal(
                item.get("unRealizedProfit", item.get("unrealizedProfit"))
            ),
            liquidation_price=_decimal(item.get("liquidationPrice")),
            leverage=int(item.get("leverage", 0)),
            margin_type=margin_type,
            update_time=int(item.get("updateTime", 0)),
            raw=dict(item),
        )

    @staticmethod
    def _parse_open_order(item: dict[str, Any]) -> OpenOrderSnapshot:
        return OpenOrderSnapshot(
            symbol=str(item.get("symbol", "")),
            order_id=int(item.get("orderId", 0)),
            client_order_id=str(item.get("clientOrderId", "")),
            side=str(item.get("side", "")),
            order_type=str(item.get("type", item.get("origType", ""))),
            status=str(item.get("status", "")),
            price=_decimal(item.get("price")),
            original_quantity=_decimal(item.get("origQty")),
            executed_quantity=_decimal(item.get("executedQty")),
            reduce_only=_boolean(item.get("reduceOnly", False)),
            update_time=int(item.get("updateTime", item.get("time", 0))),
            raw=dict(item),
        )

    @staticmethod
    def _parse_trade(item: dict[str, Any]) -> UserTradeSnapshot:
        return UserTradeSnapshot(
            symbol=str(item.get("symbol", "")),
            trade_id=int(item.get("id", 0)),
            order_id=int(item.get("orderId", 0)),
            side=str(item.get("side", "")),
            price=_decimal(item.get("price")),
            quantity=_decimal(item.get("qty")),
            quote_quantity=_decimal(item.get("quoteQty")),
            realized_pnl=_decimal(item.get("realizedPnl")),
            commission=_decimal(item.get("commission")),
            commission_asset=str(item.get("commissionAsset", "")),
            time=int(item.get("time", 0)),
            raw=dict(item),
        )


# A neutral alias for new code; the historical name remains for backwards compatibility.
BinanceFuturesClient = BinanceFuturesDemoClient
