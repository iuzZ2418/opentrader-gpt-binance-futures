from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from .binance import BinanceFuturesClient
from .contracts import CandleInterval, MarketBar, RiskRegime
from .domain import MarketQuote

FUNDING_ELEVATED_THRESHOLD = 0.001
FUNDING_EXTREME_THRESHOLD = 0.003


@dataclass(frozen=True, slots=True)
class DerivativesRiskSnapshot:
    symbol: str
    mark_price: float
    index_price: float
    funding_rate: float
    open_interest: float
    adl_quantile: int | None
    spread_bps: float
    depth_within_20bps: float
    expected_order_notional: float
    observed_at: datetime
    open_interest_change_24h_fraction: float | None = None

    @property
    def basis_fraction(self) -> float:
        if self.index_price <= 0:
            return 0
        return self.mark_price / self.index_price - 1

    @property
    def depth_multiple(self) -> float:
        return self.depth_within_20bps / max(self.expected_order_notional, 1e-12)


@dataclass(frozen=True, slots=True)
class RiskOverlayResult:
    regime: RiskRegime
    multiplier: float
    reason_codes: tuple[str, ...]


class DerivativesRiskOverlay:
    """Non-directional funding/basis/ADL/book overlay restricted to 1/0.5/0."""

    def classify(self, snapshot: DerivativesRiskSnapshot) -> RiskOverlayResult:
        caution: list[str] = []
        blocked: list[str] = []
        funding = abs(snapshot.funding_rate)
        basis = abs(snapshot.basis_fraction)
        if funding >= FUNDING_EXTREME_THRESHOLD:
            blocked.append("funding_extreme")
        elif funding >= FUNDING_ELEVATED_THRESHOLD:
            caution.append("funding_elevated")
        if basis >= 0.05:
            blocked.append("basis_extreme")
        elif basis >= 0.02:
            caution.append("basis_elevated")
        if snapshot.adl_quantile is not None:
            if snapshot.adl_quantile >= 4:
                blocked.append("adl_highest_quantile")
            elif snapshot.adl_quantile >= 3:
                caution.append("adl_elevated")
        oi_change = snapshot.open_interest_change_24h_fraction
        if oi_change is None:
            caution.append("oi_change_unavailable")
        elif abs(oi_change) >= 0.30:
            blocked.append("oi_change_extreme")
        elif abs(oi_change) >= 0.15:
            caution.append("oi_change_elevated")
        if snapshot.spread_bps > 10:
            blocked.append("spread_above_10bps")
        elif snapshot.spread_bps > 5:
            caution.append("spread_elevated")
        if snapshot.depth_multiple < 20:
            blocked.append("depth_below_20x_order")
        elif snapshot.depth_multiple < 40:
            caution.append("depth_thin")
        if blocked:
            return RiskOverlayResult(RiskRegime.BLOCKED, 0, tuple(blocked + caution))
        if caution:
            return RiskOverlayResult(RiskRegime.CAUTION, 0.5, tuple(caution))
        return RiskOverlayResult(RiskRegime.NORMAL, 1, ())


class BinanceFuturesMarketDataProvider:
    """REST snapshot adapter used on startup/reconnect and for closed-candle decisions."""

    def __init__(self, client: BinanceFuturesClient) -> None:
        self.client = client

    def closed_bars(
        self, symbol: str, interval: CandleInterval, limit: int
    ) -> tuple[MarketBar, ...]:
        if limit > 1_500:
            raise ValueError("closed_bars currently supports at most 1500 Binance candles")
        now_ms = self.client.server_time()
        raw = self.client.klines(symbol.upper(), interval.value, limit=limit)
        return self._parse_closed_bars(symbol, interval, raw, now_ms=now_ms)

    def closed_bars_between(
        self,
        symbol: str,
        interval: CandleInterval,
        *,
        start: datetime,
        end: datetime,
        limit: int = 100,
    ) -> tuple[MarketBar, ...]:
        """Fetch a bounded historical window without using candles after ``end``.

        This is the point-in-time price source for counterfactual settlement.  ``end`` may
        not be in the exchange's future, and every returned candle must have closed by it.
        """

        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("counterfactual candle bounds must be timezone-aware")
        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        if end_utc <= start_utc:
            raise ValueError("counterfactual candle end must be after start")
        if not 1 <= limit <= 1_500:
            raise ValueError("closed_bars_between limit must be between 1 and 1500")
        server_ms = self.client.server_time()
        end_ms = int(end_utc.timestamp() * 1_000)
        if end_ms > server_ms + 1_000:
            raise ValueError("counterfactual candle end cannot be in the exchange future")
        raw = self.client.klines(
            symbol.upper(),
            interval.value,
            limit=limit,
            start_time=int(start_utc.timestamp() * 1_000),
            end_time=end_ms,
        )
        return tuple(
            bar
            for bar in self._parse_closed_bars(
                symbol, interval, raw, now_ms=min(server_ms, end_ms + 1)
            )
            if start_utc <= bar.close_time <= end_utc
        )

    @staticmethod
    def _parse_closed_bars(
        symbol: str,
        interval: CandleInterval,
        raw: list[list[Any]],
        *,
        now_ms: int,
    ) -> tuple[MarketBar, ...]:
        bars: list[MarketBar] = []
        for item in raw:
            if len(item) < 7:
                continue
            close_time_ms = int(item[6])
            is_closed = close_time_ms < now_ms
            if not is_closed:
                continue
            bars.append(
                MarketBar(
                    symbol=symbol,
                    interval=interval,
                    open_time=datetime.fromtimestamp(int(item[0]) / 1_000, UTC),
                    close_time=datetime.fromtimestamp(close_time_ms / 1_000, UTC),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    is_closed=True,
                )
            )
        return tuple(bars)

    def quote(self, symbol: str) -> MarketQuote:
        normalized = symbol.upper()
        return self.client.fetch_quotes({normalized: normalized})[normalized]

    def derivatives_snapshot(
        self,
        symbol: str,
        *,
        expected_order_notional: float,
    ) -> DerivativesRiskSnapshot:
        normalized = symbol.upper()
        premium = self.client.premium_index(normalized)
        if not isinstance(premium, dict):
            raise ValueError("single-symbol premiumIndex returned an array")
        interest = self.client.open_interest(normalized)
        interest_history = self.client.open_interest_history(
            normalized, period="1h", limit=25
        )
        depth = self.client.depth(normalized, limit=100)
        quote = self.quote(normalized)
        adl: int | None = None
        if self.client.api_key and self.client.api_secret:
            rows = self.client.adl_quantile(normalized)
            if rows:
                quantiles = rows[0].get("adlQuantile", {})
                if isinstance(quantiles, dict):
                    adl = max((int(value) for value in quantiles.values()), default=None)
        depth_notional = self._depth_within(depth, quote.last, fraction=0.002)
        observed_ms = int(
            premium.get("time") or interest.get("time") or self.client.server_time()
        )
        oi_values: list[float] = []
        for item in interest_history:
            try:
                value = float(item.get("sumOpenInterest", 0) or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                oi_values.append(value)
        oi_change = (
            oi_values[-1] / oi_values[0] - 1 if len(oi_values) >= 2 else None
        )
        return DerivativesRiskSnapshot(
            symbol=normalized,
            mark_price=float(premium.get("markPrice", quote.last)),
            index_price=float(premium.get("indexPrice", quote.last)),
            funding_rate=float(premium.get("lastFundingRate", 0)),
            open_interest=float(interest.get("openInterest", 0)),
            adl_quantile=adl,
            spread_bps=(quote.ask - quote.bid) / quote.last * 10_000,
            depth_within_20bps=depth_notional,
            expected_order_notional=expected_order_notional,
            observed_at=datetime.fromtimestamp(observed_ms / 1_000, UTC),
            open_interest_change_24h_fraction=oi_change,
        )

    @staticmethod
    def _depth_within(depth: dict[str, Any], mid: float, *, fraction: float) -> float:
        """Return conservative executable depth: the thinner of bids and asks.

        Summing both sides can hide an empty ask book behind deep bids (or vice versa), so a
        candidate must satisfy the liquidity gate on either possible execution side.
        """

        if mid <= 0:
            return 0
        lower = Decimal(str(mid * (1 - fraction)))
        upper = Decimal(str(mid * (1 + fraction)))
        side_totals: list[Decimal] = []
        for side in (depth.get("bids", []), depth.get("asks", [])):
            total = Decimal("0")
            if not isinstance(side, list):
                side_totals.append(total)
                continue
            for level in side:
                if not isinstance(level, (list, tuple)) or len(level) < 2:
                    continue
                price = Decimal(str(level[0]))
                quantity = Decimal(str(level[1]))
                if lower <= price <= upper:
                    total += price * quantity
            side_totals.append(total)
        return float(min(side_totals, default=Decimal("0")))
