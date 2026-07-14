from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from crypto_event_trader.learning import (
    BacktestEvidence,
    PerformanceMetrics,
    ShadowEvidence,
    TradeOutcome,
    compute_performance_metrics,
    evaluate_promotion,
)


def _metrics(
    *,
    net_return: float,
    max_drawdown: float,
    stressed_net_return_2x: float,
    concentration: float = 0.30,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        net_profit=net_return * 10_000,
        net_return=net_return,
        max_drawdown=max_drawdown,
        total_cost=100,
        stressed_net_profit_2x=stressed_net_return_2x * 10_000,
        stressed_net_return_2x=stressed_net_return_2x,
        symbol_concentration=concentration,
        month_concentration=concentration,
        trade_count=100,
        period_days=365,
    )


def _passing_inputs() -> tuple[ShadowEvidence, BacktestEvidence, ShadowEvidence]:
    champion = ShadowEvidence(
        metrics=_metrics(net_return=0.10, max_drawdown=0.12, stressed_net_return_2x=0.06),
        completed=True,
        elapsed_days=90,
        closed_trades=80,
    )
    backtest = BacktestEvidence(
        metrics=_metrics(net_return=0.25, max_drawdown=0.10, stressed_net_return_2x=0.15),
        completed=True,
        dsr_significance_probability=0.96,
        pbo_probability=0.08,
        holdout_months=12,
        walk_forward_passed=True,
        holdout_passed=True,
        parameter_perturbation_passed=True,
        latency_stress_passed=True,
        social_placebo_passed=True,
    )
    challenger = ShadowEvidence(
        metrics=_metrics(net_return=0.11, max_drawdown=0.10, stressed_net_return_2x=0.07),
        completed=True,
        elapsed_days=90,
        closed_trades=30,
    )
    return champion, backtest, challenger


def test_performance_metrics_include_cost_stress_drawdown_and_concentration() -> None:
    trades = [
        TradeOutcome(
            "BTCUSDT",
            datetime(2026, 1, 1, tzinfo=UTC),
            gross_pnl=120,
            fees=10,
            slippage_cost=5,
            funding_cost=5,
        ),
        TradeOutcome(
            "ETHUSDT",
            datetime(2026, 2, 1, tzinfo=UTC),
            gross_pnl=80,
            fees=5,
            slippage_cost=5,
        ),
        TradeOutcome(
            "BTCUSDT",
            datetime(2026, 3, 2, tzinfo=UTC),
            gross_pnl=-30,
        ),
    ]

    result = compute_performance_metrics(
        trades,
        initial_equity=1000,
        equity_curve=[1100, 1050, 1200],
    )

    assert result.net_profit == pytest.approx(140)
    assert result.net_return == pytest.approx(0.14)
    assert result.total_cost == pytest.approx(30)
    assert result.stressed_net_profit_2x == pytest.approx(110)
    assert result.stressed_net_return_2x == pytest.approx(0.11)
    assert result.max_drawdown == pytest.approx(50 / 1100)
    assert result.symbol_concentration == pytest.approx(0.5)
    assert result.month_concentration == pytest.approx(100 / 170)
    assert result.trade_count == 3
    assert result.period_days == pytest.approx(60)

    with pytest.raises(ValueError, match="non-negative"):
        compute_performance_metrics(
            [TradeOutcome("BTCUSDT", datetime(2026, 1, 1, tzinfo=UTC), 10, fees=-1)],
            initial_equity=1000,
        )

    signed_funding = compute_performance_metrics(
        [
            TradeOutcome(
                "ETHUSDT",
                datetime(2026, 1, 2),
                gross_pnl=100,
                fees=10,
                funding_cost=-5,
            ),
            TradeOutcome(
                "BTCUSDT",
                datetime(2026, 1, 1, tzinfo=UTC),
                gross_pnl=0,
            ),
        ],
        initial_equity=1000,
    )
    assert signed_funding.total_cost == 5
    assert signed_funding.net_profit == 95
    assert signed_funding.stressed_net_profit_2x == 85


def test_promotion_gate_accepts_exact_90_day_30_trade_and_10_percent_boundary() -> None:
    champion, backtest, challenger = _passing_inputs()

    result = evaluate_promotion(
        champion_shadow=champion,
        challenger_backtest=backtest,
        challenger_shadow=challenger,
        evaluated_at=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result.eligible is True
    assert result.inputs_complete is True
    assert result.reason_codes == ()
    assert result.required_challenger_net_return == pytest.approx(0.11)
    assert result.observed_relative_improvement == pytest.approx(0.10)


def test_promotion_gate_rejects_missing_statistical_inputs() -> None:
    champion, backtest, challenger = _passing_inputs()
    backtest = replace(
        backtest,
        dsr_significance_probability=None,
        pbo_probability=None,
        social_placebo_passed=None,
    )

    result = evaluate_promotion(
        champion_shadow=champion,
        challenger_backtest=backtest,
        challenger_shadow=challenger,
    )

    assert result.eligible is False
    assert result.inputs_complete is False
    assert "MISSING_DSR_SIGNIFICANCE" in result.reason_codes
    assert "MISSING_PBO" in result.reason_codes
    assert "MISSING_SOCIAL_PLACEBO_RESULT" in result.reason_codes


def test_promotion_gate_rejects_cost_duration_count_improvement_and_risk_failures() -> None:
    champion, backtest, challenger = _passing_inputs()
    backtest = replace(
        backtest,
        metrics=replace(backtest.metrics, stressed_net_return_2x=-0.01),
    )
    challenger = replace(
        challenger,
        metrics=replace(
            challenger.metrics,
            net_return=0.105,
            max_drawdown=0.13,
            stressed_net_return_2x=-0.01,
        ),
        elapsed_days=89,
        closed_trades=29,
    )

    result = evaluate_promotion(
        champion_shadow=champion,
        challenger_backtest=backtest,
        challenger_shadow=challenger,
    )

    assert result.eligible is False
    assert "BACKTEST_2X_COST_NOT_PROFITABLE" in result.reason_codes
    assert "SHADOW_2X_COST_NOT_PROFITABLE" in result.reason_codes
    assert "INSUFFICIENT_SHADOW_DAYS" in result.reason_codes
    assert "INSUFFICIENT_SHADOW_TRADES" in result.reason_codes
    assert "RELATIVE_RETURN_IMPROVEMENT_TOO_LOW" in result.reason_codes
    assert "RISK_WORSENED_VS_CHAMPION" in result.reason_codes


def test_empty_evidence_fails_closed_instead_of_fabricating_metrics() -> None:
    result = evaluate_promotion(
        champion_shadow=ShadowEvidence(metrics=PerformanceMetrics()),
        challenger_backtest=BacktestEvidence(metrics=PerformanceMetrics()),
        challenger_shadow=ShadowEvidence(metrics=PerformanceMetrics()),
    )

    assert result.eligible is False
    assert result.inputs_complete is False
    assert len(result.reason_codes) >= 10
    assert all(
        result.details[key]
        for key in ("champion_shadow", "challenger_backtest", "challenger_shadow")
    )


def test_promotion_requires_comparable_windows_and_no_concentration_or_cost_regression() -> None:
    champion, backtest, challenger = _passing_inputs()
    challenger = replace(
        challenger,
        elapsed_days=100,
        metrics=replace(
            challenger.metrics,
            stressed_net_return_2x=0.05,
            symbol_concentration=0.31,
            month_concentration=0.31,
        ),
    )

    result = evaluate_promotion(
        champion_shadow=champion,
        challenger_backtest=backtest,
        challenger_shadow=challenger,
    )

    assert result.eligible is False
    assert "SHADOW_COMPARISON_WINDOWS_MISMATCH" in result.reason_codes
    assert "COST_ROBUSTNESS_WORSENED_VS_CHAMPION" in result.reason_codes
    assert "SYMBOL_CONCENTRATION_WORSENED_VS_CHAMPION" in result.reason_codes
    assert "MONTH_CONCENTRATION_WORSENED_VS_CHAMPION" in result.reason_codes
