from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import fmean, stdev


@dataclass(frozen=True, slots=True)
class ReturnStatistics:
    observations: int
    mean: float
    standard_error: float
    t_statistic: float
    hit_rate: float
    confidence_interval_95: tuple[float, float]


def return_statistics(values: list[float], *, bootstrap_samples: int = 2000) -> ReturnStatistics:
    if not values:
        return ReturnStatistics(0, 0, 0, 0, 0, (0, 0))
    mean = fmean(values)
    standard_error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    t_statistic = mean / standard_error if standard_error else 0.0
    hit_rate = sum(value > 0 for value in values) / len(values)
    interval = bootstrap_confidence_interval(values, samples=bootstrap_samples)
    return ReturnStatistics(len(values), mean, standard_error, t_statistic, hit_rate, interval)


def bootstrap_confidence_interval(
    values: list[float], *, samples: int = 2000, seed: int = 42
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    randomizer = random.Random(seed)
    means = sorted(
        fmean(randomizer.choices(values, k=len(values))) for _ in range(max(100, samples))
    )
    lower = means[int(len(means) * 0.025)]
    upper = means[min(len(means) - 1, int(len(means) * 0.975))]
    return (lower, upper)


def performance_summary(
    signals: list[dict], orders: list[dict], equity_curve: list[dict], initial_cash: float
) -> dict:
    equity = equity_curve[-1]["equity"] if equity_curve else initial_cash
    returns = []
    for previous, current in zip(equity_curve, equity_curve[1:], strict=False):
        if previous["equity"]:
            returns.append(current["equity"] / previous["equity"] - 1)
    maximum_drawdown = max((point["drawdown"] for point in equity_curve), default=0.0)
    trade_candidates = sum(signal["threshold_bucket"] == "paper_trade" for signal in signals)
    return {
        "signals": len(signals),
        "trade_candidates": trade_candidates,
        "orders": len(orders),
        "equity": round(equity, 4),
        "total_return": round(equity / initial_cash - 1, 8),
        "maximum_drawdown": round(maximum_drawdown, 8),
        "equity_change_statistics": asdict(return_statistics(returns)),
        "evidence_status": "insufficient_sample" if len(orders) < 30 else "analysis_ready",
    }
