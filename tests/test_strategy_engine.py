from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_event_trader.contracts import (
    CandleInterval,
    MarketBar,
    RiskRegime,
    StrategySpec,
    TradeDirection,
)
from crypto_event_trader.strategy import (
    TrendBreakoutStrategy,
    UniverseMarket,
    UniverseSelector,
    ewma_annualized_volatility,
    volatility_position_scale,
)
from crypto_event_trader.strategy_registry import ChallengerMetrics, StrategyRegistry

NOW = datetime(2026, 7, 14, 12, tzinfo=UTC)


def _bars(
    interval: CandleInterval,
    count: int,
    *,
    direction: int = 1,
    symbol: str = "BTCUSDT",
) -> list[MarketBar]:
    hours = 1 if interval is CandleInterval.ONE_HOUR else 4
    start = NOW - timedelta(hours=hours * count)
    bars: list[MarketBar] = []
    previous = 100.0
    for index in range(count):
        close = 100 + direction * index * 0.2
        if close <= 1:
            close = 1 + (count - index) * 0.2
        opened = previous
        bars.append(
            MarketBar(
                symbol=symbol,
                interval=interval,
                open_time=start + timedelta(hours=hours * index),
                close_time=start + timedelta(hours=hours * (index + 1)),
                open=opened,
                high=max(opened, close) + 0.05,
                low=min(opened, close) - 0.05,
                close=close,
                volume=1_000,
            )
        )
        previous = close
    return bars


def test_five_vote_champion_creates_120_second_candidate_with_atr_stop() -> None:
    strategy = TrendBreakoutStrategy()
    candidate = strategy.generate_candidate(
        symbol="btcusdt",
        hourly_bars=_bars(CandleInterval.ONE_HOUR, 721),
        four_hour_bars=_bars(CandleInterval.FOUR_HOURS, 127),
        quantity_cap=4,
        now=NOW,
    )

    assert candidate is not None
    assert candidate.direction is TradeDirection.LONG
    assert candidate.expires_at - candidate.created_at == timedelta(seconds=120)
    assert candidate.feature_snapshot["long_votes"] == 5
    assert candidate.feature_snapshot["atr_14_1h"] > 0
    assert candidate.feature_snapshot["suggested_stop_distance"] == pytest.approx(
        2 * candidate.feature_snapshot["atr_14_1h"]
    )
    assert candidate.feature_snapshot["suggested_stop_price"] < candidate.feature_snapshot[
        "last_price"
    ]
    assert 0 < candidate.max_quantity <= 4
    assert candidate.max_risk_fraction <= 0.0075
    assert candidate.is_valid(NOW + timedelta(seconds=119))
    assert not candidate.is_valid(NOW + timedelta(seconds=120))


def test_only_closed_point_in_time_bars_are_used() -> None:
    hourly = _bars(CandleInterval.ONE_HOUR, 721)
    future = hourly[-1].model_copy(
        update={
            "open_time": NOW + timedelta(hours=1),
            "close_time": NOW + timedelta(hours=2),
            "close": 1.0,
            "low": 0.9,
            "is_closed": False,
        }
    )
    candidate = TrendBreakoutStrategy().generate_candidate(
        symbol="BTCUSDT",
        hourly_bars=[*hourly, future],
        four_hour_bars=_bars(CandleInterval.FOUR_HOURS, 127),
        quantity_cap=1,
        now=NOW,
    )
    assert candidate is not None
    assert candidate.direction is TradeDirection.LONG
    assert candidate.feature_snapshot["latest_hourly_close_time"] == NOW.isoformat()


def test_volatility_and_market_regime_can_only_shrink_or_block_candidate() -> None:
    calm = [100 * math.exp(index * 0.0001) for index in range(721)]
    volatile = [100 * math.exp(((-1) ** index) * 0.08) for index in range(721)]
    calm_vol = ewma_annualized_volatility(calm)
    volatile_vol = ewma_annualized_volatility(volatile)
    assert volatile_vol > calm_vol
    assert volatility_position_scale(volatile_vol, 0.4) < volatility_position_scale(
        calm_vol, 0.4
    )

    strategy = TrendBreakoutStrategy()
    normal = strategy.generate_candidate(
        symbol="BTCUSDT",
        hourly_bars=_bars(CandleInterval.ONE_HOUR, 721),
        four_hour_bars=_bars(CandleInterval.FOUR_HOURS, 127),
        quantity_cap=2,
        risk_regime=RiskRegime.NORMAL,
        now=NOW,
    )
    caution = strategy.generate_candidate(
        symbol="BTCUSDT",
        hourly_bars=_bars(CandleInterval.ONE_HOUR, 721),
        four_hour_bars=_bars(CandleInterval.FOUR_HOURS, 127),
        quantity_cap=2,
        risk_regime=RiskRegime.CAUTION,
        now=NOW,
    )
    blocked = strategy.generate_candidate(
        symbol="BTCUSDT",
        hourly_bars=_bars(CandleInterval.ONE_HOUR, 721),
        four_hour_bars=_bars(CandleInterval.FOUR_HOURS, 127),
        quantity_cap=2,
        risk_regime=RiskRegime.BLOCKED,
        now=NOW,
    )
    assert normal is not None and caution is not None
    assert caution.max_quantity == pytest.approx(normal.max_quantity * 0.5)
    assert blocked is None


def _market(rank: int, **updates: object) -> UniverseMarket:
    values = {
        "symbol": f"ASSET{rank}USDT",
        "quote_asset": "USDT",
        "contract_type": "PERPETUAL",
        "listed_at": NOW - timedelta(days=365),
        "as_of": NOW,
        "median_turnover_30d": 1_000_000 - rank * 1_000,
        "median_spread_bps_30d": 5,
        "depth_within_20bps": 200_000,
        "expected_order_notional": 10_000,
    }
    values.update(updates)
    return UniverseMarket(**values)


def test_universe_selector_enforces_liquidity_filters_and_rank_12_buffer() -> None:
    snapshots = [_market(rank) for rank in range(1, 14)]
    snapshots.extend(
        [
            _market(20, quote_asset="BUSD"),
            _market(21, contract_type="CURRENT_QUARTER"),
            _market(22, listed_at=NOW - timedelta(days=100)),
            _market(23, median_spread_bps_30d=10.1),
            _market(24, depth_within_20bps=199_999),
        ]
    )
    current = [f"ASSET{rank}USDT" for rank in range(1, 10)] + ["ASSET11USDT"]
    selected = UniverseSelector().select(snapshots, current_symbols=current, as_of=NOW)

    assert len(selected) == 10
    assert "ASSET11USDT" in selected
    assert "ASSET10USDT" not in selected
    assert all(not symbol.startswith("ASSET2") or symbol == "ASSET2USDT" for symbol in selected)


def _passing_metrics(**updates: object) -> ChallengerMetrics:
    values = {
        "champion_net_return": 0.20,
        "challenger_net_return": 0.23,
        "champion_max_drawdown": 0.12,
        "challenger_max_drawdown": 0.10,
        "double_cost_net_return": 0.05,
        "dsr_probability": 0.96,
        "pbo_probability": 0.08,
        "max_symbol_contribution": 0.30,
        "max_month_contribution": 0.30,
        "shadow_days": 90,
        "shadow_closed_trades": 30,
        "walk_forward_passed": True,
        "sealed_holdout_passed": True,
        "parameter_perturbation_passed": True,
        "latency_stress_passed": True,
        "social_placebo_passed": True,
    }
    values.update(updates)
    return ChallengerMetrics(**values)


def test_registry_disables_metrics_only_promotion_path(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = StrategyRegistry(path)
    challenger = StrategySpec(
        version="trend-breakout-v2", momentum_windows_1h=(24, 80, 168)
    )
    registry.register_challenger(challenger)

    with pytest.raises(PermissionError, match="GovernedPromotionCoordinator"):
        registry.evaluate_and_promote(challenger.version, _passing_metrics())
    assert registry.champion.version == "trend-breakout-v1"

    reloaded = StrategyRegistry(path)
    assert reloaded.champion.version == "trend-breakout-v1"
    assert [item.version for item in reloaded.challengers] == [challenger.version]
