from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    symbol: str
    closed_at: datetime
    gross_pnl: float
    fees: float = 0.0
    slippage_cost: float = 0.0
    funding_cost: float = 0.0
    episode_id: str | None = None
    trace_ids: tuple[str, ...] = ()
    strategy_versions: tuple[str, ...] = ()
    source_record_ids: tuple[str, ...] = ()

    @property
    def total_cost(self) -> float:
        return self.fees + self.slippage_cost + self.funding_cost

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_cost

    @property
    def stressed_total_cost_2x(self) -> float:
        # Double unavoidable costs and adverse funding; do not magnify funding credits.
        return 2 * (self.fees + self.slippage_cost + max(self.funding_cost, 0.0)) + min(
            self.funding_cost, 0.0
        )


class IncompletePerformanceAccounting(ValueError):
    """Exact venue costs are missing, so the sample cannot enter an automatic gate."""

    def __init__(self, reason_code: str, record_id: str, detail: str) -> None:
        self.reason_code = reason_code
        self.record_id = record_id
        super().__init__(f"{reason_code}: {record_id}: {detail}")


def _audit_timestamp(value: Any, *, record_id: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise IncompletePerformanceAccounting(
                "INVALID_TIMESTAMP", record_id, str(value)
            ) from error
    else:
        raise IncompletePerformanceAccounting("MISSING_TIMESTAMP", record_id, str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise IncompletePerformanceAccounting(
            "NAIVE_TIMESTAMP", record_id, "audit accounting timestamps must be UTC-aware"
        )
    return parsed.astimezone(UTC)


def build_trade_outcomes_from_audit_records(
    *,
    fills: Sequence[Mapping[str, Any]],
    funding_events: Sequence[Mapping[str, Any]],
    fee_conversions: Sequence[Mapping[str, Any]] = (),
    quote_asset: str = "USDT",
) -> tuple[TradeOutcome, ...]:
    """Reconstruct exact closed one-way position episodes from immutable venue records.

    Binance realized PnL already uses actual execution prices, so no synthetic slippage is
    subtracted a second time. Fees are charged separately. A non-quote fee is accepted only
    with a persisted conversion effective at that exact fill timestamp. Funding is a signed
    cash flow: payments become a positive cost and receipts a negative cost.

    Any missing/non-finite cost, unattributed funding, invalid position transition, or
    conflicting conversion raises :class:`IncompletePerformanceAccounting`. Callers must not
    substitute estimates when feeding an automatic champion/challenger promotion gate.
    """

    quote_asset = quote_asset.strip().upper()
    if not quote_asset:
        raise ValueError("quote_asset must be non-empty")

    conversions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in fee_conversions:
        conversion = dict(raw)
        conversion_id = str(conversion.get("conversion_id") or "unknown-conversion")
        fill_id = str(conversion.get("venue_fill_id") or "")
        target = str(conversion.get("quote_asset") or "").upper()
        key = (fill_id, target)
        if not fill_id or not target or key in conversions:
            raise IncompletePerformanceAccounting(
                "CONFLICTING_FEE_CONVERSION", conversion_id, "duplicate or incomplete key"
            )
        rate = float(conversion.get("rate", math.nan))
        if not math.isfinite(rate) or rate <= 0:
            raise IncompletePerformanceAccounting(
                "INVALID_FEE_CONVERSION", conversion_id, "rate must be finite and positive"
            )
        conversions[key] = conversion

    timeline: list[tuple[datetime, int, str, dict[str, Any]]] = []
    seen_fills: set[str] = set()
    for raw in fills:
        fill = dict(raw)
        fill_id = str(fill.get("venue_fill_id") or "")
        if not fill_id or fill_id in seen_fills:
            raise IncompletePerformanceAccounting(
                "DUPLICATE_OR_MISSING_FILL_ID", fill_id or "missing", "fill IDs must be unique"
            )
        seen_fills.add(fill_id)
        timeline.append(
            (_audit_timestamp(fill.get("filled_at"), record_id=fill_id), 0, fill_id, fill)
        )

    seen_funding: set[str] = set()
    for raw in funding_events:
        funding = dict(raw)
        event_id = str(funding.get("accounting_event_id") or "")
        if not event_id or event_id in seen_funding:
            raise IncompletePerformanceAccounting(
                "DUPLICATE_OR_MISSING_FUNDING_ID",
                event_id or "missing",
                "funding event IDs must be unique",
            )
        seen_funding.add(event_id)
        attribution = funding.get("attribution")
        if not isinstance(attribution, Mapping) or attribution.get("status") != "ATTRIBUTED":
            reason = attribution.get("reason") if isinstance(attribution, Mapping) else "missing"
            raise IncompletePerformanceAccounting(
                "UNATTRIBUTED_FUNDING", event_id, str(reason)
            )
        if not attribution.get("trace_id") or not attribution.get("venue_order_id"):
            raise IncompletePerformanceAccounting(
                "INCOMPLETE_FUNDING_OWNER", event_id, "trace/order owner is missing"
            )
        timeline.append(
            (
                _audit_timestamp(funding.get("transaction_time"), record_id=event_id),
                1,
                event_id,
                funding,
            )
        )

    timeline.sort(key=lambda item: (item[0], item[1], item[2]))
    states: dict[str, dict[str, Any]] = {}
    outcomes: list[TradeOutcome] = []

    for observed_at, event_kind, record_id, event in timeline:
        symbol = str(event.get("symbol") or "").upper()
        if not symbol:
            raise IncompletePerformanceAccounting(
                "MISSING_SYMBOL", record_id, "venue accounting record has no symbol"
            )
        state = states.setdefault(
            symbol,
            {
                "quantity": 0.0,
                "gross_pnl": 0.0,
                "fees": 0.0,
                "funding_pnl": 0.0,
                "total_quantity": 0.0,
                "first_fill_id": None,
                "trace_ids": set(),
                "strategy_versions": set(),
                "source_record_ids": [],
            },
        )
        if event_kind == 1:
            if str(event.get("asset") or "").upper() != quote_asset:
                raise IncompletePerformanceAccounting(
                    "UNCONVERTED_FUNDING_ASSET",
                    record_id,
                    f"{event.get('asset')} cannot be assumed 1:1 with {quote_asset}",
                )
            amount = float(event.get("amount", math.nan))
            if not math.isfinite(amount):
                raise IncompletePerformanceAccounting(
                    "NON_FINITE_FUNDING", record_id, "funding amount is not finite"
                )
            tolerance = max(1e-12, float(state["total_quantity"]) * 1e-10)
            if abs(float(state["quantity"])) <= tolerance:
                raise IncompletePerformanceAccounting(
                    "FUNDING_WITHOUT_OPEN_POSITION",
                    record_id,
                    "attributed funding timestamp is outside an audited position episode",
                )
            attribution = event["attribution"]
            state["trace_ids"].add(str(attribution["trace_id"]))
            state["source_record_ids"].append(record_id)
            state["funding_pnl"] = float(state["funding_pnl"]) + amount
            continue

        quantity = float(event.get("quantity", math.nan))
        price = float(event.get("price", math.nan))
        fee = float(event.get("fee", math.nan))
        realized = event.get("realized_pnl")
        if (
            not math.isfinite(quantity)
            or quantity <= 0
            or not math.isfinite(price)
            or price <= 0
            or not math.isfinite(fee)
            or fee < 0
            or realized is None
            or not math.isfinite(float(realized))
        ):
            raise IncompletePerformanceAccounting(
                "INCOMPLETE_FILL_ACCOUNTING",
                record_id,
                "quantity, price, fee, and realized PnL must be exact finite values",
            )
        fee_asset = str(event.get("fee_asset") or "").upper()
        if not fee_asset:
            raise IncompletePerformanceAccounting(
                "MISSING_FEE_ASSET", record_id, "fee asset is required even for a zero fee"
            )
        quote_fee = fee
        conversion_source_id: str | None = None
        if fee > 0 and fee_asset != quote_asset:
            conversion = conversions.get((record_id, quote_asset))
            if conversion is None:
                raise IncompletePerformanceAccounting(
                    "MISSING_POINT_IN_TIME_FEE_CONVERSION",
                    record_id,
                    f"{fee_asset} fee cannot be assumed 1:1 with {quote_asset}",
                )
            if str(conversion.get("from_asset") or "").upper() != fee_asset:
                raise IncompletePerformanceAccounting(
                    "FEE_CONVERSION_ASSET_MISMATCH", record_id, fee_asset
                )
            effective_at = _audit_timestamp(
                conversion.get("effective_at"),
                record_id=str(conversion.get("conversion_id") or record_id),
            )
            if effective_at != observed_at:
                raise IncompletePerformanceAccounting(
                    "STALE_FEE_CONVERSION", record_id, "conversion is not fill-time exact"
                )
            quote_fee = fee * float(conversion["rate"])
            conversion_source_id = str(conversion.get("conversion_id") or "")
        if not math.isfinite(quote_fee):
            raise IncompletePerformanceAccounting(
                "NON_FINITE_CONVERTED_FEE", record_id, "converted fee is not finite"
            )

        side = str(event.get("side") or "").upper()
        if side not in {"BUY", "SELL"}:
            raise IncompletePerformanceAccounting("INVALID_FILL_SIDE", record_id, side)
        signed = quantity if side == "BUY" else -quantity
        previous = float(state["quantity"])
        next_quantity = previous + signed
        trace_id = str(event.get("trace_id") or "")
        strategy_version = str(event.get("strategy_version") or "")
        if not trace_id or not strategy_version:
            raise IncompletePerformanceAccounting(
                "MISSING_STRATEGY_LINEAGE",
                record_id,
                "authoritative fill is not linked to a trace and strategy version",
            )
        if state["first_fill_id"] is None:
            state["first_fill_id"] = record_id
        state["trace_ids"].add(trace_id)
        state["strategy_versions"].add(strategy_version)
        state["source_record_ids"].append(record_id)
        if conversion_source_id:
            state["source_record_ids"].append(conversion_source_id)
        state["total_quantity"] = float(state["total_quantity"]) + quantity
        tolerance = max(1e-12, float(state["total_quantity"]) * 1e-10)
        reduce_only = bool(event.get("reduce_only"))
        if reduce_only:
            if (
                abs(previous) <= tolerance
                or previous * signed >= 0
                or abs(next_quantity) > abs(previous) + tolerance
                or previous * next_quantity < -tolerance
            ):
                raise IncompletePerformanceAccounting(
                    "INVALID_REDUCE_ONLY_POSITION_TRANSITION", record_id, symbol
                )
        elif abs(previous) > tolerance and previous * signed < 0:
            raise IncompletePerformanceAccounting(
                "NON_REDUCE_ORDER_CHANGED_POSITION_DIRECTION", record_id, symbol
            )
        state["fees"] = float(state["fees"]) + quote_fee
        state["gross_pnl"] = float(state["gross_pnl"]) + float(realized)
        state["quantity"] = 0.0 if abs(next_quantity) <= tolerance else next_quantity
        if state["quantity"] == 0.0:
            outcomes.append(
                TradeOutcome(
                    symbol=symbol,
                    closed_at=observed_at,
                    gross_pnl=float(state["gross_pnl"]),
                    fees=float(state["fees"]),
                    slippage_cost=0.0,
                    funding_cost=-float(state["funding_pnl"]),
                    episode_id=f"{symbol}:{state['first_fill_id']}",
                    trace_ids=tuple(sorted(state["trace_ids"])),
                    strategy_versions=tuple(sorted(state["strategy_versions"])),
                    source_record_ids=tuple(state["source_record_ids"]),
                )
            )
            states[symbol] = {
                "quantity": 0.0,
                "gross_pnl": 0.0,
                "fees": 0.0,
                "funding_pnl": 0.0,
                "total_quantity": 0.0,
                "first_fill_id": None,
                "trace_ids": set(),
                "strategy_versions": set(),
                "source_record_ids": [],
            }

    return tuple(outcomes)


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    net_profit: float | None = None
    net_return: float | None = None
    max_drawdown: float | None = None
    total_cost: float | None = None
    stressed_net_profit_2x: float | None = None
    stressed_net_return_2x: float | None = None
    symbol_concentration: float | None = None
    month_concentration: float | None = None
    trade_count: int | None = None
    period_days: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BacktestEvidence:
    metrics: PerformanceMetrics
    completed: bool | None = None
    dsr_significance_probability: float | None = None
    pbo_probability: float | None = None
    holdout_months: int | None = None
    walk_forward_passed: bool | None = None
    holdout_passed: bool | None = None
    parameter_perturbation_passed: bool | None = None
    latency_stress_passed: bool | None = None
    social_placebo_passed: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["metrics"] = self.metrics.as_dict()
        return result


@dataclass(frozen=True, slots=True)
class ShadowEvidence:
    metrics: PerformanceMetrics
    completed: bool | None = None
    elapsed_days: int | None = None
    closed_trades: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["metrics"] = self.metrics.as_dict()
        return result


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    max_drawdown: float = 0.20
    min_dsr_significance_probability: float = 0.95
    max_pbo_probability: float = 0.10
    max_contribution_concentration: float = 0.35
    min_holdout_months: int = 12
    min_shadow_days: int = 90
    min_shadow_closed_trades: int = 30
    min_relative_net_return_improvement: float = 0.10

    def __post_init__(self) -> None:
        fractions = {
            "max_drawdown": self.max_drawdown,
            "min_dsr_significance_probability": self.min_dsr_significance_probability,
            "max_pbo_probability": self.max_pbo_probability,
            "max_contribution_concentration": self.max_contribution_concentration,
            "min_relative_net_return_improvement": self.min_relative_net_return_improvement,
        }
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions.values()):
            raise ValueError("Promotion policy fractions must be finite values in [0, 1]")
        if self.min_holdout_months < 1:
            raise ValueError("min_holdout_months must be positive")
        if self.min_shadow_days < 1 or self.min_shadow_closed_trades < 1:
            raise ValueError("shadow duration and trade count thresholds must be positive")


@dataclass(frozen=True, slots=True)
class PromotionEvaluation:
    eligible: bool
    reason_codes: tuple[str, ...]
    inputs_complete: bool
    required_challenger_net_return: float | None
    observed_relative_improvement: float | None
    evaluated_at: datetime
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evaluated_at"] = (
            self.evaluated_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        )
        result["reason_codes"] = list(self.reason_codes)
        return result


def _finite_number(value: float | int | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _positive_concentration(contributions: dict[str, float]) -> float | None:
    positive = [max(value, 0.0) for value in contributions.values()]
    total = sum(positive)
    if total <= 0:
        return None
    return max(positive) / total


def _drawdown(equity_values: Sequence[float]) -> float:
    if not equity_values:
        return 0.0
    peak = float(equity_values[0])
    maximum = 0.0
    for raw_value in equity_values:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("equity_curve values must be finite")
        peak = max(peak, value)
        if peak > 0:
            maximum = max(maximum, (peak - value) / peak)
        elif value < peak:
            maximum = math.inf
    return maximum


def compute_performance_metrics(
    trades: Sequence[TradeOutcome],
    *,
    initial_equity: float,
    equity_curve: Sequence[float] | None = None,
) -> PerformanceMetrics:
    """Compute transparent realized metrics; no DSR or PBO is synthesized here.

    `gross_pnl` is price PnL before explicit costs. Fees/slippage must be non-negative;
    `funding_cost` is signed (positive payment, negative credit). The 2x stress doubles fees,
    slippage and funding payments while leaving funding credits unchanged.
    Concentration is the largest positive symbol/month contribution divided by all positive
    contributions. An all-loss sample has no meaningful contribution concentration (`None`),
    which the promotion gate treats as missing rather than inventing a favorable value.
    """

    initial_equity = float(initial_equity)
    if not math.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive")

    def normalized_close(trade: TradeOutcome) -> datetime:
        value = trade.closed_at
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    ordered = sorted(trades, key=normalized_close)
    symbol_contributions: dict[str, float] = defaultdict(float)
    month_contributions: dict[str, float] = defaultdict(float)
    total_gross = 0.0
    total_cost = 0.0
    stressed_total_cost = 0.0
    realized_equity = [initial_equity]

    for trade in ordered:
        values = {
            "gross_pnl": trade.gross_pnl,
            "fees": trade.fees,
            "slippage_cost": trade.slippage_cost,
            "funding_cost": trade.funding_cost,
        }
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("trade PnL and costs must be finite")
        if float(trade.fees) < 0 or float(trade.slippage_cost) < 0:
            raise ValueError("fees and slippage_cost must be non-negative")
        closed_at = normalized_close(trade)
        net = trade.net_pnl
        total_gross += trade.gross_pnl
        total_cost += trade.total_cost
        stressed_total_cost += trade.stressed_total_cost_2x
        symbol_contributions[trade.symbol.upper()] += net
        month_contributions[closed_at.strftime("%Y-%m")] += net
        realized_equity.append(realized_equity[-1] + net)

    net_profit = total_gross - total_cost
    stressed_net_profit = total_gross - stressed_total_cost
    if ordered:
        first = normalized_close(ordered[0])
        last = normalized_close(ordered[-1])
        period_days = max(
            0.0, (last.astimezone(UTC) - first.astimezone(UTC)).total_seconds() / 86400
        )
    else:
        period_days = 0.0

    drawdown_curve = [initial_equity, *(equity_curve or realized_equity[1:])]
    return PerformanceMetrics(
        net_profit=net_profit,
        net_return=net_profit / initial_equity,
        max_drawdown=_drawdown(drawdown_curve),
        total_cost=total_cost,
        stressed_net_profit_2x=stressed_net_profit,
        stressed_net_return_2x=stressed_net_profit / initial_equity,
        symbol_concentration=_positive_concentration(dict(symbol_contributions)),
        month_concentration=_positive_concentration(dict(month_contributions)),
        trade_count=len(ordered),
        period_days=period_days,
    )


def evaluate_promotion(
    *,
    champion_shadow: ShadowEvidence,
    challenger_backtest: BacktestEvidence,
    challenger_shadow: ShadowEvidence,
    policy: PromotionPolicy | None = None,
    evaluated_at: datetime | None = None,
) -> PromotionEvaluation:
    """Apply a deterministic, fail-closed champion/challenger promotion gate.

    DSR significance and PBO are inputs from a separate statistically valid research process.
    This function only validates their presence/range and applies the configured thresholds.
    """

    policy = policy or PromotionPolicy()
    reasons: list[str] = []

    def reject(code: str) -> None:
        if code not in reasons:
            reasons.append(code)

    if challenger_backtest.completed is not True:
        reject(
            "MISSING_BACKTEST_COMPLETION"
            if challenger_backtest.completed is None
            else "BACKTEST_INCOMPLETE"
        )
    if champion_shadow.completed is not True:
        reject(
            "MISSING_CHAMPION_SHADOW_COMPLETION"
            if champion_shadow.completed is None
            else "CHAMPION_SHADOW_INCOMPLETE"
        )
    if challenger_shadow.completed is not True:
        reject(
            "MISSING_CHALLENGER_SHADOW_COMPLETION"
            if challenger_shadow.completed is None
            else "CHALLENGER_SHADOW_INCOMPLETE"
        )

    required_backtest_metrics = (
        "net_return",
        "max_drawdown",
        "stressed_net_return_2x",
        "symbol_concentration",
        "month_concentration",
    )
    required_challenger_shadow_metrics = required_backtest_metrics
    required_champion_metrics = (
        "net_return",
        "max_drawdown",
        "stressed_net_return_2x",
        "symbol_concentration",
        "month_concentration",
    )

    def require_metrics(prefix: str, metrics: PerformanceMetrics, names: Sequence[str]) -> None:
        for name in names:
            if not _finite_number(getattr(metrics, name)):
                reject(f"MISSING_{prefix}_{name.upper()}")

    require_metrics("BACKTEST", challenger_backtest.metrics, required_backtest_metrics)
    require_metrics(
        "CHALLENGER_SHADOW",
        challenger_shadow.metrics,
        required_challenger_shadow_metrics,
    )
    require_metrics("CHAMPION_SHADOW", champion_shadow.metrics, required_champion_metrics)

    if not _finite_number(challenger_backtest.dsr_significance_probability):
        reject("MISSING_DSR_SIGNIFICANCE")
    if not _finite_number(challenger_backtest.pbo_probability):
        reject("MISSING_PBO")
    for name, value in (
        ("DSR_SIGNIFICANCE", challenger_backtest.dsr_significance_probability),
        ("PBO", challenger_backtest.pbo_probability),
    ):
        if _finite_number(value) and not 0 <= float(value) <= 1:
            reject(f"INVALID_{name}")

    if challenger_backtest.holdout_months is None:
        reject("MISSING_HOLDOUT_MONTHS")
    elif challenger_backtest.holdout_months < policy.min_holdout_months:
        reject("INSUFFICIENT_HOLDOUT_MONTHS")

    validation_flags = {
        "WALK_FORWARD": challenger_backtest.walk_forward_passed,
        "SEALED_HOLDOUT": challenger_backtest.holdout_passed,
        "PARAMETER_PERTURBATION": challenger_backtest.parameter_perturbation_passed,
        "LATENCY_STRESS": challenger_backtest.latency_stress_passed,
        "SOCIAL_PLACEBO": challenger_backtest.social_placebo_passed,
    }
    for name, value in validation_flags.items():
        if value is None:
            reject(f"MISSING_{name}_RESULT")
        elif value is not True:
            reject(f"{name}_FAILED")

    if challenger_shadow.elapsed_days is None:
        reject("MISSING_SHADOW_DAYS")
    elif challenger_shadow.elapsed_days < policy.min_shadow_days:
        reject("INSUFFICIENT_SHADOW_DAYS")
    if challenger_shadow.closed_trades is None:
        reject("MISSING_SHADOW_TRADES")
    elif challenger_shadow.closed_trades < policy.min_shadow_closed_trades:
        reject("INSUFFICIENT_SHADOW_TRADES")
    if champion_shadow.elapsed_days is None:
        reject("MISSING_CHAMPION_SHADOW_DAYS")
    elif champion_shadow.elapsed_days < policy.min_shadow_days:
        reject("INSUFFICIENT_CHAMPION_SHADOW_DAYS")
    if champion_shadow.closed_trades is None:
        reject("MISSING_CHAMPION_SHADOW_TRADES")
    elif champion_shadow.closed_trades < policy.min_shadow_closed_trades:
        reject("INSUFFICIENT_CHAMPION_SHADOW_TRADES")
    if (
        champion_shadow.elapsed_days is not None
        and challenger_shadow.elapsed_days is not None
        and abs(champion_shadow.elapsed_days - challenger_shadow.elapsed_days) > 1
    ):
        reject("SHADOW_COMPARISON_WINDOWS_MISMATCH")

    backtest = challenger_backtest.metrics
    challenger = challenger_shadow.metrics
    champion = champion_shadow.metrics

    for prefix, metrics in (("BACKTEST", backtest), ("SHADOW", challenger)):
        if _finite_number(metrics.net_return) and float(metrics.net_return) <= 0:
            reject(f"{prefix}_NET_RETURN_NOT_POSITIVE")
        if _finite_number(metrics.max_drawdown) and float(metrics.max_drawdown) < 0:
            reject(f"INVALID_{prefix}_MAX_DRAWDOWN")
        if (
            _finite_number(metrics.net_return)
            and _finite_number(metrics.stressed_net_return_2x)
            and float(metrics.stressed_net_return_2x) > float(metrics.net_return) + 1e-15
        ):
            reject(f"INVALID_{prefix}_COST_STRESS")
    if _finite_number(champion.max_drawdown) and float(champion.max_drawdown) < 0:
        reject("INVALID_CHAMPION_SHADOW_MAX_DRAWDOWN")

    if _finite_number(backtest.max_drawdown) and float(backtest.max_drawdown) > policy.max_drawdown:
        reject("BACKTEST_DRAWDOWN_EXCEEDED")
    if (
        _finite_number(challenger.max_drawdown)
        and float(challenger.max_drawdown) > policy.max_drawdown
    ):
        reject("SHADOW_DRAWDOWN_EXCEEDED")
    if (
        _finite_number(backtest.stressed_net_return_2x)
        and float(backtest.stressed_net_return_2x) <= 0
    ):
        reject("BACKTEST_2X_COST_NOT_PROFITABLE")
    if (
        _finite_number(challenger.stressed_net_return_2x)
        and float(challenger.stressed_net_return_2x) <= 0
    ):
        reject("SHADOW_2X_COST_NOT_PROFITABLE")
    if (
        _finite_number(challenger_backtest.dsr_significance_probability)
        and 0 <= float(challenger_backtest.dsr_significance_probability) <= 1
        and float(challenger_backtest.dsr_significance_probability)
        < policy.min_dsr_significance_probability
    ):
        reject("DSR_SIGNIFICANCE_TOO_LOW")
    if (
        _finite_number(challenger_backtest.pbo_probability)
        and 0 <= float(challenger_backtest.pbo_probability) <= 1
        and float(challenger_backtest.pbo_probability) > policy.max_pbo_probability
    ):
        reject("PBO_TOO_HIGH")

    for prefix, metrics in (("BACKTEST", backtest), ("SHADOW", challenger)):
        for dimension in ("symbol_concentration", "month_concentration"):
            value = getattr(metrics, dimension)
            if _finite_number(value) and not 0 <= float(value) <= 1:
                reject(f"INVALID_{prefix}_{dimension.upper()}")
            elif _finite_number(value) and float(value) > policy.max_contribution_concentration:
                reject(f"{prefix}_{dimension.upper()}_EXCEEDED")

    required_return: float | None = None
    observed_improvement: float | None = None
    if _finite_number(champion.net_return) and _finite_number(challenger.net_return):
        champion_return = float(champion.net_return)
        challenger_return = float(challenger.net_return)
        minimum_gain = max(
            abs(champion_return) * policy.min_relative_net_return_improvement,
            1e-12,
        )
        required_return = champion_return + minimum_gain
        if challenger_return + 1e-15 < required_return:
            reject("RELATIVE_RETURN_IMPROVEMENT_TOO_LOW")
        if abs(champion_return) > 1e-12:
            observed_improvement = (challenger_return - champion_return) / abs(champion_return)

    if _finite_number(champion.max_drawdown) and _finite_number(challenger.max_drawdown):
        if float(challenger.max_drawdown) > float(champion.max_drawdown) + 1e-15:
            reject("RISK_WORSENED_VS_CHAMPION")
    if _finite_number(champion.stressed_net_return_2x) and _finite_number(
        challenger.stressed_net_return_2x
    ):
        if float(challenger.stressed_net_return_2x) + 1e-15 < float(
            champion.stressed_net_return_2x
        ):
            reject("COST_ROBUSTNESS_WORSENED_VS_CHAMPION")
    for dimension in ("symbol_concentration", "month_concentration"):
        champion_value = getattr(champion, dimension)
        challenger_value = getattr(challenger, dimension)
        if _finite_number(champion_value) and _finite_number(challenger_value):
            if float(challenger_value) > float(champion_value) + 1e-15:
                reject(f"{dimension.upper()}_WORSENED_VS_CHAMPION")

    missing_inputs = any(code.startswith("MISSING_") for code in reasons)
    now = evaluated_at or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    else:
        now = now.astimezone(UTC)
    return PromotionEvaluation(
        eligible=not reasons,
        reason_codes=tuple(reasons),
        inputs_complete=not missing_inputs,
        required_challenger_net_return=required_return,
        observed_relative_improvement=observed_improvement,
        evaluated_at=now,
        details={
            "policy": asdict(policy),
            "champion_shadow": champion_shadow.as_dict(),
            "challenger_backtest": challenger_backtest.as_dict(),
            "challenger_shadow": challenger_shadow.as_dict(),
        },
    )
