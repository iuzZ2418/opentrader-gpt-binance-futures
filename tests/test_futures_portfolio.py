from datetime import date

import pytest

from crypto_event_trader.futures_portfolio import FuturesPortfolio


def test_long_position_uses_margin_and_mark_to_market() -> None:
    portfolio = FuturesPortfolio(10_000, default_leverage=3)
    portfolio.apply_fill(symbol="BTCUSDT", side="BUY", quantity=0.1, price=50_000, fee=2)
    portfolio.mark("BTCUSDT", 51_000)
    snapshot = portfolio.snapshot()

    assert snapshot.wallet_balance == 9_998
    assert snapshot.unrealized_pnl == 100
    assert snapshot.equity == 10_098
    assert snapshot.gross_notional == 5_100
    assert snapshot.initial_margin == pytest.approx(5_000 / 3)


def test_reduce_and_flip_realizes_correct_pnl() -> None:
    portfolio = FuturesPortfolio(10_000)
    portfolio.apply_fill(symbol="ETHUSDT", side="BUY", quantity=2, price=2_000)
    position = portfolio.apply_fill(
        symbol="ETHUSDT", side="SELL", quantity=3, price=2_100, fee=1
    )

    assert position.quantity == -1
    assert position.entry_price == 2_100
    assert position.realized_pnl == 200
    assert portfolio.snapshot().wallet_balance == 10_199


def test_short_funding_and_daily_drawdown() -> None:
    portfolio = FuturesPortfolio(10_000)
    portfolio.apply_fill(symbol="SOLUSDT", side="SELL", quantity=10, price=100)
    portfolio.apply_funding("SOLUSDT", 5)
    portfolio.mark("SOLUSDT", 90)
    assert portfolio.snapshot().equity == 10_105

    portfolio.roll_day(date(2026, 7, 15))
    portfolio.mark("SOLUSDT", 110)
    snapshot = portfolio.snapshot()
    assert snapshot.daily_pnl_fraction < 0
    assert snapshot.drawdown > 0


def test_invalid_fill_is_rejected() -> None:
    portfolio = FuturesPortfolio(10_000)
    with pytest.raises(ValueError):
        portfolio.apply_fill(symbol="BTCUSDT", side="WAIT", quantity=1, price=1)

