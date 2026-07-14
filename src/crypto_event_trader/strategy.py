from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from pydantic import Field, model_validator

from .contracts import (
    CandleInterval,
    MarketBar,
    RiskRegime,
    StrategySpec,
    StrictContract,
    TradeCandidate,
    TradeDirection,
    utc_now,
)

INITIAL_RISK_FRACTION = 0.0075
HOURS_PER_YEAR = 24 * 365


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class UniverseMarket(StrictContract):
    """Point-in-time 30-day liquidity snapshot for a futures market."""

    symbol: str = Field(min_length=1, max_length=30)
    quote_asset: str = Field(min_length=1, max_length=12)
    contract_type: str = Field(min_length=1, max_length=30)
    listed_at: datetime
    as_of: datetime
    median_turnover_30d: float = Field(ge=0)
    median_spread_bps_30d: float = Field(ge=0)
    depth_within_20bps: float = Field(ge=0)
    expected_order_notional: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_snapshot(self) -> UniverseMarket:
        listed_at = _as_utc(self.listed_at, "listed_at")
        as_of = _as_utc(self.as_of, "as_of")
        if listed_at > as_of:
            raise ValueError("listed_at cannot be after as_of")
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "quote_asset", self.quote_asset.upper())
        object.__setattr__(self, "contract_type", self.contract_type.upper())
        object.__setattr__(self, "listed_at", listed_at)
        object.__setattr__(self, "as_of", as_of)
        return self


class UniverseSelector:
    """Weekly Top-10 selector with a rank-12 retention buffer."""

    def __init__(
        self,
        *,
        size: int = 10,
        retention_rank: int = 12,
        minimum_listing_days: int = 180,
        maximum_spread_bps: float = 10,
        minimum_depth_multiple: float = 20,
    ) -> None:
        if size <= 0 or retention_rank < size:
            raise ValueError("retention_rank must be at least the universe size")
        self.size = size
        self.retention_rank = retention_rank
        self.minimum_listing_days = minimum_listing_days
        self.maximum_spread_bps = maximum_spread_bps
        self.minimum_depth_multiple = minimum_depth_multiple

    def select(
        self,
        snapshots: Sequence[UniverseMarket],
        *,
        current_symbols: Sequence[str] = (),
        as_of: datetime | None = None,
    ) -> tuple[str, ...]:
        reference = _as_utc(as_of or utc_now(), "as_of")
        # Multiple snapshots may be supplied during replay; never look beyond the cutoff.
        latest: dict[str, UniverseMarket] = {}
        for item in snapshots:
            if item.as_of > reference:
                continue
            prior = latest.get(item.symbol)
            if prior is None or item.as_of > prior.as_of:
                latest[item.symbol] = item
        eligible = [item for item in latest.values() if self._eligible(item, reference)]
        eligible.sort(key=lambda item: (-item.median_turnover_30d, item.symbol))
        ranks = {item.symbol: rank for rank, item in enumerate(eligible, start=1)}
        current = {symbol.upper() for symbol in current_symbols}
        retained = [
            item.symbol
            for item in eligible
            if item.symbol in current and ranks[item.symbol] <= self.retention_rank
        ][: self.size]
        selected = set(retained)
        for item in eligible:
            if len(selected) >= self.size:
                break
            selected.add(item.symbol)
        return tuple(item.symbol for item in eligible if item.symbol in selected)

    def _eligible(self, item: UniverseMarket, as_of: datetime) -> bool:
        return (
            item.quote_asset == "USDT"
            and item.contract_type == "PERPETUAL"
            and item.listed_at <= as_of - timedelta(days=self.minimum_listing_days)
            and item.median_spread_bps_30d <= self.maximum_spread_bps
            and item.depth_within_20bps
            >= self.minimum_depth_multiple * item.expected_order_notional
        )


def default_champion_spec() -> StrategySpec:
    return StrategySpec(version="trend-breakout-v1")


def ewma_annualized_volatility(closes: Sequence[float], span_hours: int = 720) -> float:
    """Calculate a point-in-time EWMA volatility from at most 30 days of hourly returns."""

    if span_hours < 2:
        raise ValueError("span_hours must be at least 2")
    if len(closes) < span_hours + 1:
        raise ValueError(f"at least {span_hours + 1} closes are required")
    selected = [float(value) for value in closes[-(span_hours + 1) :]]
    if any(value <= 0 or not math.isfinite(value) for value in selected):
        raise ValueError("closes must be finite and positive")
    returns = [
        math.log(current / previous)
        for previous, current in zip(selected, selected[1:], strict=False)
    ]
    alpha = 2 / (span_hours + 1)
    variance = returns[0] ** 2
    for value in returns[1:]:
        variance = alpha * value**2 + (1 - alpha) * variance
    return math.sqrt(max(0.0, variance) * HOURS_PER_YEAR)


def volatility_position_scale(realized_volatility: float, target_volatility: float) -> float:
    if realized_volatility < 0 or not math.isfinite(realized_volatility):
        raise ValueError("realized_volatility must be finite and non-negative")
    if target_volatility <= 0 or not math.isfinite(target_volatility):
        raise ValueError("target_volatility must be finite and positive")
    if realized_volatility == 0:
        return 1.0
    return min(1.0, target_volatility / realized_volatility)


def momentum_vote(closes: Sequence[float], lookback: int) -> int:
    if lookback <= 0 or len(closes) < lookback + 1:
        raise ValueError("insufficient closes for momentum vote")
    delta = float(closes[-1]) - float(closes[-lookback - 1])
    return 1 if delta > 0 else -1 if delta < 0 else 0


def donchian_vote(bars: Sequence[MarketBar], lookback: int) -> int:
    if lookback <= 0 or len(bars) < lookback + 1:
        raise ValueError("insufficient bars for Donchian vote")
    prior = bars[-lookback - 1 : -1]
    last_close = bars[-1].close
    if last_close > max(bar.high for bar in prior):
        return 1
    if last_close < min(bar.low for bar in prior):
        return -1
    return 0


def average_true_range(bars: Sequence[MarketBar], period: int = 14) -> float:
    """Wilder ATR calculated only from the supplied, already-closed bars."""

    if period <= 0 or len(bars) < period + 1:
        raise ValueError("insufficient bars for ATR")
    true_ranges = [
        max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )
        for previous, current in zip(bars, bars[1:], strict=False)
    ]
    atr = sum(true_ranges[:period]) / period
    for true_range in true_ranges[period:]:
        atr = ((period - 1) * atr + true_range) / period
    return atr


class TrendBreakoutStrategy:
    """Five-vote trend champion; it proposes bounded candidates but cannot place orders."""

    def __init__(self, spec: StrategySpec | None = None) -> None:
        self.spec = spec or default_champion_spec()

    def generate_candidate(
        self,
        *,
        symbol: str,
        hourly_bars: Sequence[MarketBar],
        four_hour_bars: Sequence[MarketBar],
        quantity_cap: float,
        risk_regime: RiskRegime = RiskRegime.NORMAL,
        now: datetime | None = None,
    ) -> TradeCandidate | None:
        if quantity_cap <= 0 or not math.isfinite(quantity_cap):
            raise ValueError("quantity_cap must be finite and positive")
        reference = _as_utc(now or utc_now(), "now")
        normalized_symbol = symbol.upper()
        hourly = self._closed_bars(
            hourly_bars, normalized_symbol, CandleInterval.ONE_HOUR, reference
        )
        four_hour = self._closed_bars(
            four_hour_bars, normalized_symbol, CandleInterval.FOUR_HOURS, reference
        )
        required_hourly = max(self.spec.ewma_span_hours + 1, max(self.spec.momentum_windows_1h) + 1)
        required_four_hour = max(self.spec.donchian_windows_4h) + 1
        if len(hourly) < required_hourly or len(four_hour) < required_four_hour:
            return None

        closes = [bar.close for bar in hourly]
        momentum_votes = [momentum_vote(closes, window) for window in self.spec.momentum_windows_1h]
        donchian_votes = [
            donchian_vote(four_hour, window) for window in self.spec.donchian_windows_4h
        ]
        votes = [*momentum_votes, *donchian_votes]
        long_votes = votes.count(1)
        short_votes = votes.count(-1)
        if max(long_votes, short_votes) < self.spec.minimum_directional_votes:
            return None
        if long_votes == short_votes:
            return None
        direction = TradeDirection.LONG if long_votes > short_votes else TradeDirection.SHORT

        risk_scale = self.spec.risk_scale(risk_regime)
        if risk_scale == 0:
            return None
        realized_volatility = ewma_annualized_volatility(closes, self.spec.ewma_span_hours)
        vol_scale = volatility_position_scale(
            realized_volatility, self.spec.target_annualized_volatility
        )
        combined_scale = risk_scale * vol_scale
        max_quantity = quantity_cap * combined_scale
        if max_quantity <= 0:
            return None

        labels = [
            *(f"momentum_1h_{window}" for window in self.spec.momentum_windows_1h),
            *(f"donchian_4h_{window}" for window in self.spec.donchian_windows_4h),
        ]
        atr_14 = average_true_range(hourly, 14)
        stop_distance = 2 * atr_14
        last_price = hourly[-1].close
        suggested_stop_price = (
            max(0.0, last_price - stop_distance)
            if direction is TradeDirection.LONG
            else last_price + stop_distance
        )
        snapshot = {
            "bar_cutoff": reference.isoformat(),
            "latest_hourly_close_time": hourly[-1].close_time.isoformat(),
            "latest_four_hour_close_time": four_hour[-1].close_time.isoformat(),
            "last_price": last_price,
            "votes": {label: vote for label, vote in zip(labels, votes, strict=True)},
            "long_votes": long_votes,
            "short_votes": short_votes,
            "ewma_annualized_volatility": realized_volatility,
            "volatility_scale": vol_scale,
            "risk_regime": risk_regime.value,
            "risk_scale": risk_scale,
            "atr_1h": atr_14,
            "atr_14_1h": atr_14,
            "suggested_stop_distance": stop_distance,
            "suggested_stop_distance_fraction": stop_distance / last_price,
            "suggested_stop_price": suggested_stop_price,
        }
        cycle_bucket = int(reference.timestamp()) // (15 * 60)
        identity = "|".join(
            (
                self.spec.version,
                normalized_symbol,
                direction.value,
                hourly[-1].close_time.isoformat(),
                four_hour[-1].close_time.isoformat(),
                str(cycle_bucket),
            )
        )
        return TradeCandidate(
            candidate_id=f"cand_{sha256(identity.encode('utf-8')).hexdigest()[:40]}",
            strategy_version=self.spec.version,
            symbol=normalized_symbol,
            direction=direction,
            max_quantity=max_quantity,
            max_risk_fraction=INITIAL_RISK_FRACTION * combined_scale,
            feature_snapshot=snapshot,
            created_at=reference,
        )

    @staticmethod
    def _closed_bars(
        bars: Sequence[MarketBar],
        symbol: str,
        interval: CandleInterval,
        cutoff: datetime,
    ) -> list[MarketBar]:
        selected = [
            bar
            for bar in bars
            if bar.symbol == symbol
            and bar.interval is interval
            and bar.is_closed
            and bar.close_time <= cutoff
        ]
        selected.sort(key=lambda bar: bar.close_time)
        # A reconnect can replay a candle. Last-write-wins keeps one point per close time.
        deduplicated = {bar.close_time: bar for bar in selected}
        return [deduplicated[key] for key in sorted(deduplicated)]
