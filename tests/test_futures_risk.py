from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from crypto_event_trader.config import Settings
from crypto_event_trader.contracts import (
    PositionThesis,
    TradeAction,
    TradeCandidate,
    TradeDecision,
    TradeDirection,
)
from crypto_event_trader.control import TradingControl
from crypto_event_trader.domain import MarketQuote
from crypto_event_trader.futures_portfolio import FuturesPortfolio
from crypto_event_trader.futures_risk import FuturesHardRisk, emergency_exit_reason


def _candidate(now: datetime) -> TradeCandidate:
    return TradeCandidate(
        candidate_id="candidate-123",
        strategy_version="trend-v1",
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        max_quantity=1,
        max_risk_fraction=0.0075,
        feature_snapshot={"atr_1h": 500, "last_price": 50_000},
        created_at=now,
        expires_at=now + timedelta(seconds=120),
    )


def _decision(now: datetime, action: TradeAction = TradeAction.OPEN) -> TradeDecision:
    return TradeDecision(
        decision_id="decision-123",
        candidate_id="candidate-123",
        symbol="BTCUSDT",
        action=action,
        direction=TradeDirection.LONG,
        position_multiplier=1,
        confidence=0.9,
        next_review_at=now + timedelta(minutes=15),
        reason="approved evidence",
    )


def test_open_is_sized_by_atr_risk_and_has_protective_stop() -> None:
    now = datetime.now(UTC)
    settings = Settings.from_env()
    control = TradingControl(settings).snapshot()
    account = FuturesPortfolio(100_000).snapshot(timestamp=now)
    quote = MarketQuote("BTCUSDT", 49_990, 50_010, 50_000, 10**9, now)
    intent = FuturesHardRisk(settings).evaluate(
        decision=_decision(now),
        candidate=_candidate(now),
        quote=quote,
        account=account,
        control=control,
        now=now,
    )

    assert intent.approved is True
    assert intent.quantity == 0.75
    assert intent.protective_stop_price == 49_000


def test_capital_stage_fraction_is_a_hard_sizing_cap() -> None:
    now = datetime.now(UTC)
    settings = replace(Settings.from_env(), capital_allocation_fraction=0.10)
    intent = FuturesHardRisk(settings).evaluate(
        decision=_decision(now),
        candidate=_candidate(now),
        quote=MarketQuote("BTCUSDT", 49_990, 50_010, 50_000, 10**9, now),
        account=FuturesPortfolio(100_000).snapshot(timestamp=now),
        control=TradingControl(settings).snapshot(),
        now=now,
    )

    assert intent.approved is True
    assert intent.quantity == 0.075


def test_add_requires_existing_profitable_thesis_and_only_once() -> None:
    now = datetime.now(UTC)
    settings = Settings.from_env()
    portfolio = FuturesPortfolio(100_000)
    portfolio.apply_fill(symbol="BTCUSDT", side="BUY", quantity=0.1, price=50_000)
    account = portfolio.snapshot(timestamp=now)
    quote = MarketQuote("BTCUSDT", 49_990, 50_010, 50_000, 10**9, now)
    thesis = PositionThesis(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_reason="trend",
        expected_horizon_minutes=1_440,
        pnl_r=0.5,
    )
    intent = FuturesHardRisk(settings).evaluate(
        decision=_decision(now, TradeAction.ADD),
        candidate=_candidate(now),
        quote=quote,
        account=account,
        control=TradingControl(settings).snapshot(),
        thesis=thesis,
        signal_strengthening=True,
        now=now,
    )
    assert intent.reason == "add_requires_one_r_profit"


def test_model_failure_policy_still_allows_reduce_only_close() -> None:
    now = datetime.now(UTC)
    settings = replace(Settings.from_env(), trading_stage="live")
    portfolio = FuturesPortfolio(100_000)
    portfolio.apply_fill(symbol="BTCUSDT", side="BUY", quantity=0.2, price=50_000)
    decision = TradeDecision(
        decision_id="decision-close",
        symbol="BTCUSDT",
        action=TradeAction.CLOSE,
        direction=TradeDirection.LONG,
        position_multiplier=0,
        confidence=0,
        next_review_at=now + timedelta(minutes=15),
        reason="provider unavailable",
    )
    intent = FuturesHardRisk(settings).evaluate(
        decision=decision,
        candidate=None,
        quote=MarketQuote("BTCUSDT", 49_990, 50_010, 50_000, 10**9, now),
        account=portfolio.snapshot(timestamp=now),
        control=TradingControl(settings).snapshot(),
        now=now,
    )
    assert intent.approved is True
    assert intent.reduce_only is True
    assert intent.quantity == 0.2


def test_stale_quote_never_blocks_reduce_only_close() -> None:
    now = datetime.now(UTC)
    settings = Settings.from_env()
    portfolio = FuturesPortfolio(100_000)
    portfolio.apply_fill(symbol="BTCUSDT", side="BUY", quantity=0.2, price=50_000)
    decision = TradeDecision(
        decision_id="decision-stale-close",
        symbol="BTCUSDT",
        action=TradeAction.CLOSE,
        direction=TradeDirection.LONG,
        position_multiplier=0,
        confidence=0,
        next_review_at=now + timedelta(minutes=15),
        reason="deterministic exit",
    )

    intent = FuturesHardRisk(settings).evaluate(
        decision=decision,
        candidate=None,
        quote=MarketQuote(
            "BTCUSDT", 49_990, 50_010, 50_000, 10**9, now - timedelta(hours=1)
        ),
        account=portfolio.snapshot(timestamp=now),
        control=TradingControl(settings).snapshot(),
        now=now,
    )

    assert intent.approved is True
    assert intent.reduce_only is True


def test_hard_risk_rejects_open_over_existing_position() -> None:
    now = datetime.now(UTC)
    settings = Settings.from_env()
    portfolio = FuturesPortfolio(100_000)
    portfolio.apply_fill(symbol="BTCUSDT", side="BUY", quantity=0.1, price=50_000)

    intent = FuturesHardRisk(settings).evaluate(
        decision=_decision(now),
        candidate=_candidate(now),
        quote=MarketQuote("BTCUSDT", 49_990, 50_010, 50_000, 10**9, now),
        account=portfolio.snapshot(timestamp=now),
        control=TradingControl(settings).snapshot(),
        now=now,
    )

    assert intent.reason == "existing_position_requires_add"


def test_hard_risk_independently_enforces_add_confidence_and_strengthening() -> None:
    now = datetime.now(UTC)
    settings = Settings.from_env()
    portfolio = FuturesPortfolio(100_000)
    portfolio.apply_fill(symbol="BTCUSDT", side="BUY", quantity=0.1, price=50_000)
    thesis = PositionThesis(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_reason="trend",
        expected_horizon_minutes=1_440,
        pnl_r=1.2,
    )
    weak_decision = _decision(now, TradeAction.ADD).model_copy(
        update={"confidence": 0.79}
    )
    kwargs = {
        "candidate": _candidate(now),
        "quote": MarketQuote("BTCUSDT", 49_990, 50_010, 50_000, 10**9, now),
        "account": portfolio.snapshot(timestamp=now),
        "control": TradingControl(settings).snapshot(),
        "thesis": thesis,
        "now": now,
    }

    weak = FuturesHardRisk(settings).evaluate(decision=weak_decision, **kwargs)
    assert weak.reason == "add_confidence_below_threshold"

    not_strengthening = FuturesHardRisk(settings).evaluate(
        decision=_decision(now, TradeAction.ADD), **kwargs
    )
    assert not_strengthening.reason == "add_requires_strengthening_signal"


def test_add_preserves_old_stop_and_caps_combined_position_risk_at_one_percent() -> None:
    now = datetime.now(UTC)
    settings = Settings.from_env()
    portfolio = FuturesPortfolio(100_000)
    portfolio.apply_fill(symbol="BTCUSDT", side="BUY", quantity=0.9, price=50_000)
    thesis = PositionThesis(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_reason="profitable trend",
        expected_horizon_minutes=1_440,
        pnl_r=1.2,
    )
    quote = MarketQuote("BTCUSDT", 49_990, 50_010, 50_000, 10**9, now)

    intent = FuturesHardRisk(settings).evaluate(
        decision=_decision(now, TradeAction.ADD),
        candidate=_candidate(now),
        quote=quote,
        account=portfolio.snapshot(timestamp=now),
        control=TradingControl(settings).snapshot(),
        thesis=thesis,
        signal_strengthening=True,
        existing_protective_stop=49_000,
        now=now,
    )

    assert intent.approved is True
    assert intent.protective_stop_price == 49_000
    estimated_fill = quote.ask * (1 + settings.base_slippage_bps / 10_000)
    post_quantity = 0.9 + intent.quantity
    post_entry = (0.9 * 50_000 + intent.quantity * estimated_fill) / post_quantity
    assert post_quantity * max(0, post_entry - 49_000) <= 1_000 + 1e-8
    assert intent.quantity < 0.1


def test_add_never_widens_existing_stop_and_rejects_unprotected_or_overrisk_position() -> None:
    now = datetime.now(UTC)
    settings = Settings.from_env()
    portfolio = FuturesPortfolio(100_000)
    portfolio.apply_fill(symbol="BTCUSDT", side="BUY", quantity=0.1, price=50_000)
    thesis = PositionThesis(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_reason="profitable trend",
        expected_horizon_minutes=1_440,
        pnl_r=1.2,
    )
    kwargs = {
        "decision": _decision(now, TradeAction.ADD),
        "candidate": _candidate(now),
        "quote": MarketQuote("BTCUSDT", 49_990, 50_010, 50_000, 10**9, now),
        "account": portfolio.snapshot(timestamp=now),
        "control": TradingControl(settings).snapshot(),
        "thesis": thesis,
        "signal_strengthening": True,
        "now": now,
    }
    missing = FuturesHardRisk(settings).evaluate(**kwargs)
    assert missing.reason == "existing_protective_stop_missing"

    preserved = FuturesHardRisk(settings).evaluate(
        **kwargs,
        existing_protective_stop=50_000,
    )
    assert preserved.approved is True
    assert preserved.protective_stop_price == 50_000

    over_risk = FuturesPortfolio(100_000)
    over_risk.apply_fill(symbol="BTCUSDT", side="BUY", quantity=1.1, price=50_000)
    rejected = FuturesHardRisk(settings).evaluate(
        **(kwargs | {"account": over_risk.snapshot(timestamp=now)}),
        existing_protective_stop=49_000,
    )
    assert rejected.reason == "existing_position_risk_above_one_percent"


@pytest.mark.parametrize(
    "changes",
    [
        {"risk_per_trade": 0.0101},
        {"daily_drawdown_limit": 0.031},
        {"total_drawdown_limit": 0.201},
        {"max_gross_exposure": 3.01},
        {"max_net_exposure": 1.51},
        {"max_asset_exposure": 0.51},
        {"max_correlation_cluster_exposure": 1.01},
        {"max_leverage": 4},
        {"initial_position_risk": 0.0076},
        {"add_position_risk": 0.0026},
        {"decision_open_confidence": 0.699},
        {"decision_add_confidence": 0.799},
        {"max_open_positions": 11},
        {"max_spread_bps": 10.01},
        {"entry_order_wait_seconds": 5.01},
        {"entry_price_protection_bps": 20.01},
        {"taker_fee_bps": -0.01},
        {"base_slippage_bps": -0.01},
    ],
)
def test_settings_cannot_relax_hard_risk_caps(changes: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        replace(Settings.from_env(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"decision_cycle_seconds": 1},
        {"decision_cycle_seconds": 901},
        {"market_data_max_age_seconds": 0},
        {"market_data_max_age_seconds": 11},
        {"market_data_max_age_seconds": 86_400},
    ],
)
def test_external_trading_stages_cannot_relax_cycle_or_market_freshness(
    changes: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        replace(Settings.from_env(), trading_stage="demo", **changes)


def test_daily_loss_blocks_open_and_requests_emergency_exit() -> None:
    now = datetime.now(UTC)
    settings = Settings.from_env()
    portfolio = FuturesPortfolio(100_000)
    portfolio.apply_fill(symbol="BTCUSDT", side="BUY", quantity=1, price=50_000)
    portfolio.mark("BTCUSDT", 46_000)
    account = portfolio.snapshot(timestamp=now)
    intent = FuturesHardRisk(settings).evaluate(
        decision=_decision(now),
        candidate=_candidate(now),
        quote=MarketQuote("BTCUSDT", 45_990, 46_010, 46_000, 10**9, now),
        account=account,
        control=TradingControl(settings).snapshot(),
        now=now,
    )
    assert intent.reason == "daily_loss_limit"
    assert emergency_exit_reason(account, settings) == "daily_loss_limit"


def test_hard_risk_rejects_non_usdt_contract_even_if_candidate_is_well_formed() -> None:
    now = datetime.now(UTC)
    candidate = _candidate(now).model_copy(update={"symbol": "BTCUSDC"})
    decision = _decision(now).model_copy(update={"symbol": "BTCUSDC"})
    intent = FuturesHardRisk(Settings.from_env()).evaluate(
        decision=decision,
        candidate=candidate,
        quote=MarketQuote("BTCUSDC", 49_990, 50_010, 50_000, 10**9, now),
        account=FuturesPortfolio(100_000).snapshot(timestamp=now),
        control=TradingControl(Settings.from_env()).snapshot(),
        now=now,
    )

    assert intent.approved is False
    assert intent.reason == "only_usdt_margined_perpetuals_are_allowed"
