from __future__ import annotations

import calendar
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from .audit import AuditRepository
from .learning import (
    BacktestEvidence,
    PerformanceMetrics,
    TradeOutcome,
    compute_performance_metrics,
)


class ResearchValidationError(ValueError):
    """A validation input is incomplete or could introduce look-ahead bias."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}")


PAIRED_SHADOW_COST_SCHEMA = "paired-shadow-cost-v1"


def _utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ResearchValidationError("NAIVE_TIMESTAMP", field_name)
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _shift_months(value: datetime, months: int) -> datetime:
    zero_based = value.year * 12 + value.month - 1 + months
    year, month_index = divmod(zero_based, 12)
    month = month_index + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


@dataclass(frozen=True, slots=True)
class ExpandingWalkForwardConfig:
    research_started_at: datetime
    research_ended_at: datetime
    initial_training_months: int = 12
    test_window_months: int = 3
    holdout_months: int = 12

    def __post_init__(self) -> None:
        started = _utc(self.research_started_at, "research_started_at")
        ended = _utc(self.research_ended_at, "research_ended_at")
        object.__setattr__(self, "research_started_at", started)
        object.__setattr__(self, "research_ended_at", ended)
        if ended <= started:
            raise ResearchValidationError("INVALID_RESEARCH_RANGE", "end must follow start")
        if (
            isinstance(self.initial_training_months, bool)
            or isinstance(self.test_window_months, bool)
            or not isinstance(self.initial_training_months, int)
            or not isinstance(self.test_window_months, int)
            or self.initial_training_months < 1
            or self.test_window_months < 1
        ):
            raise ResearchValidationError(
                "INVALID_WALK_FORWARD_CONFIG", "training and test months must be positive"
            )
        # The production gate specifically requires the final twelve calendar months to be
        # sealed.  A caller cannot silently weaken or reinterpret this interval.
        if self.holdout_months != 12:
            raise ResearchValidationError(
                "INVALID_HOLDOUT_LENGTH", "the sealed holdout must be exactly 12 months"
            )


@dataclass(frozen=True, slots=True)
class EvaluationWindow:
    window_id: str
    kind: str
    training_started_at: datetime
    training_ended_at: datetime
    test_started_at: datetime
    test_ended_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "kind": self.kind,
            "training_started_at": _iso(self.training_started_at),
            "training_ended_at": _iso(self.training_ended_at),
            "test_started_at": _iso(self.test_started_at),
            "test_ended_at": _iso(self.test_ended_at),
        }


def build_expanding_windows(
    config: ExpandingWalkForwardConfig,
) -> tuple[tuple[EvaluationWindow, ...], EvaluationWindow]:
    """Build deterministic expanding windows plus the untouched final 12 months."""

    holdout_start = _shift_months(config.research_ended_at, -config.holdout_months)
    first_test_start = _shift_months(config.research_started_at, config.initial_training_months)
    if first_test_start >= holdout_start:
        raise ResearchValidationError(
            "INSUFFICIENT_PRE_HOLDOUT_HISTORY",
            "initial training must leave at least one pre-holdout test window",
        )

    windows: list[EvaluationWindow] = []
    test_start = first_test_start
    while test_start < holdout_start:
        test_end = min(_shift_months(test_start, config.test_window_months), holdout_start)
        windows.append(
            EvaluationWindow(
                window_id=f"wf-{len(windows) + 1:03d}",
                kind="WALK_FORWARD",
                training_started_at=config.research_started_at,
                training_ended_at=test_start,
                test_started_at=test_start,
                test_ended_at=test_end,
            )
        )
        test_start = test_end

    holdout = EvaluationWindow(
        window_id="holdout-12m",
        kind="SEALED_HOLDOUT",
        training_started_at=config.research_started_at,
        training_ended_at=holdout_start,
        test_started_at=holdout_start,
        test_ended_at=config.research_ended_at,
    )
    return tuple(windows), holdout


@dataclass(frozen=True, slots=True)
class ValidationScenario:
    scenario_id: str
    parameter_scale: float
    execution_delay_minutes: int
    social_placebo: bool
    placebo_seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


STANDARD_SCENARIOS: tuple[ValidationScenario, ...] = (
    ValidationScenario("baseline", 1.0, 0, False),
    ValidationScenario("parameter-075", 0.75, 0, False),
    ValidationScenario("parameter-125", 1.25, 0, False),
    ValidationScenario("latency-1m", 1.0, 1, False),
    ValidationScenario("latency-5m", 1.0, 5, False),
    ValidationScenario("latency-15m", 1.0, 15, False),
    # The fixed seed is an instruction to the evaluator to permute only point-in-time social
    # features.  The resulting fills remain subject to all normal accounting checks.
    ValidationScenario("social-placebo", 1.0, 0, True, 20_260_714),
)


@dataclass(frozen=True, slots=True)
class ScenarioRequest:
    strategy_digest: str
    window: EvaluationWindow
    scenario: ValidationScenario
    holdout_seal_id: str | None = None


@dataclass(frozen=True, slots=True)
class FundingPayment:
    event_id: str
    effective_at: datetime
    cost: float


@dataclass(frozen=True, slots=True)
class ExecutedBacktestTrade:
    """Raw point-in-time execution evidence; no cost field has a permissive default."""

    trade_id: str
    symbol: str
    direction: int
    quantity: float
    signal_at: datetime
    information_cutoff_at: datetime
    opened_at: datetime
    closed_at: datetime
    entry_reference_price: float
    entry_fill_price: float
    exit_reference_price: float
    exit_fill_price: float
    entry_reference_available_at: datetime
    exit_reference_available_at: datetime
    entry_fee: float
    exit_fee: float
    funding_events: tuple[FundingPayment, ...]
    entry_fee_evidence_ids: tuple[str, ...]
    exit_fee_evidence_ids: tuple[str, ...]
    funding_coverage_id: str
    source_ids: tuple[str, ...]
    market_data_digest: str
    fees_complete: bool
    funding_complete: bool


class PointInTimeScenarioEvaluator(Protocol):
    """Adapter boundary for a real historical simulator.

    `evaluate` receives exactly one training/test window and one scenario.  `freeze_for_holdout`
    must durably bind the strategy and all pre-holdout results before any holdout request is
    made.  The validator invokes no pre-holdout method after that call.
    """

    def evaluate(self, request: ScenarioRequest) -> Sequence[ExecutedBacktestTrade]: ...

    def freeze_for_holdout(
        self, *, strategy_digest: str, pre_holdout_results_digest: str
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class StatisticalValidation:
    """Output from an independently audited DSR/CSCV-PBO implementation."""

    dsr_significance_probability: float
    pbo_probability: float
    dsr_method: str
    pbo_method: str
    source_digest: str
    observation_count: int
    independent_trial_count: int
    fold_count: int


@dataclass(frozen=True, slots=True)
class StatisticalValidationRequest:
    strategy_digest: str
    audited_input_digest: str
    fold_net_returns: tuple[float, ...]
    trade_net_returns: tuple[float, ...]
    trade_count: int


class StatisticalValidator(Protocol):
    def calculate(self, request: StatisticalValidationRequest) -> StatisticalValidation: ...


@dataclass(frozen=True, slots=True)
class _ScenarioResult:
    window_id: str
    scenario_id: str
    trade_ids: tuple[str, ...]
    outcomes: tuple[TradeOutcome, ...]
    raw_digest: str
    source_ids: tuple[str, ...]
    metrics: PerformanceMetrics

    def summary(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "scenario_id": self.scenario_id,
            "trade_ids_digest": _digest(list(self.trade_ids)),
            "trade_count": len(self.outcomes),
            "raw_digest": self.raw_digest,
            "source_ids_digest": _digest(list(self.source_ids)),
            "source_id_count": len(self.source_ids),
            "metrics": self.metrics.as_dict(),
        }


def _require_finite(name: str, value: float, *, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ResearchValidationError("INVALID_TRADE_VALUE", f"{name}={value}") from error
    if not math.isfinite(parsed) or (positive and parsed <= 0):
        raise ResearchValidationError("INVALID_TRADE_VALUE", f"{name}={value}")
    return parsed


def _trade_to_outcome(
    raw: ExecutedBacktestTrade,
    *,
    request: ScenarioRequest,
) -> tuple[TradeOutcome, dict[str, Any]]:
    if (
        not isinstance(raw.trade_id, str)
        or not raw.trade_id.strip()
        or not isinstance(raw.symbol, str)
        or not raw.symbol.strip()
    ):
        raise ResearchValidationError("MISSING_TRADE_IDENTITY", request.window.window_id)
    if isinstance(raw.direction, bool) or raw.direction not in {-1, 1}:
        raise ResearchValidationError("INVALID_TRADE_DIRECTION", raw.trade_id)
    if raw.fees_complete is not True or raw.funding_complete is not True:
        raise ResearchValidationError("INCOMPLETE_COST_ACCOUNTING", raw.trade_id)
    fee_evidence_ids = (*raw.entry_fee_evidence_ids, *raw.exit_fee_evidence_ids)
    if (
        not raw.entry_fee_evidence_ids
        or not raw.exit_fee_evidence_ids
        or not all(isinstance(item, str) and item.strip() for item in fee_evidence_ids)
    ):
        raise ResearchValidationError("MISSING_FEE_EVIDENCE", raw.trade_id)
    if not isinstance(raw.funding_coverage_id, str) or not raw.funding_coverage_id.strip():
        raise ResearchValidationError("MISSING_FUNDING_COVERAGE", raw.trade_id)
    if not raw.source_ids or not all(
        isinstance(item, str) and item.strip() for item in raw.source_ids
    ):
        raise ResearchValidationError("MISSING_POINT_IN_TIME_SOURCE", raw.trade_id)
    if not isinstance(raw.market_data_digest, str) or not raw.market_data_digest.strip():
        raise ResearchValidationError("MISSING_MARKET_DATA_DIGEST", raw.trade_id)

    signal_at = _utc(raw.signal_at, "signal_at")
    information_cutoff = _utc(raw.information_cutoff_at, "information_cutoff_at")
    opened_at = _utc(raw.opened_at, "opened_at")
    closed_at = _utc(raw.closed_at, "closed_at")
    entry_available = _utc(raw.entry_reference_available_at, "entry_reference_available_at")
    exit_available = _utc(raw.exit_reference_available_at, "exit_reference_available_at")
    if information_cutoff > signal_at:
        raise ResearchValidationError("FUTURE_FEATURE_LEAKAGE", raw.trade_id)
    if entry_available > opened_at or exit_available > closed_at:
        raise ResearchValidationError("FUTURE_PRICE_LEAKAGE", raw.trade_id)
    if not request.window.test_started_at <= signal_at < request.window.test_ended_at:
        raise ResearchValidationError("SIGNAL_OUTSIDE_TEST_WINDOW", raw.trade_id)
    minimum_open = signal_at + timedelta(minutes=request.scenario.execution_delay_minutes)
    if opened_at < minimum_open:
        raise ResearchValidationError("LATENCY_SCENARIO_NOT_APPLIED", raw.trade_id)
    if closed_at <= opened_at or closed_at >= request.window.test_ended_at:
        raise ResearchValidationError("TRADE_CLOSE_OUTSIDE_TEST_WINDOW", raw.trade_id)

    quantity = _require_finite("quantity", raw.quantity, positive=True)
    entry_reference = _require_finite(
        "entry_reference_price", raw.entry_reference_price, positive=True
    )
    entry_fill = _require_finite("entry_fill_price", raw.entry_fill_price, positive=True)
    exit_reference = _require_finite(
        "exit_reference_price", raw.exit_reference_price, positive=True
    )
    exit_fill = _require_finite("exit_fill_price", raw.exit_fill_price, positive=True)
    entry_fee = _require_finite("entry_fee", raw.entry_fee)
    exit_fee = _require_finite("exit_fee", raw.exit_fee)
    if entry_fee < 0 or exit_fee < 0:
        raise ResearchValidationError("NEGATIVE_FEE", raw.trade_id)

    funding_cost = 0.0
    funding_records: list[dict[str, Any]] = []
    funding_event_ids: set[str] = set()
    for event in raw.funding_events:
        if (
            not isinstance(event.event_id, str)
            or not event.event_id.strip()
            or event.event_id in funding_event_ids
        ):
            raise ResearchValidationError("INVALID_FUNDING_EVENT_ID", raw.trade_id)
        effective_at = _utc(event.effective_at, "funding_effective_at")
        if not opened_at <= effective_at <= closed_at:
            raise ResearchValidationError("FUNDING_OUTSIDE_POSITION_WINDOW", event.event_id)
        cost = _require_finite("funding_event_cost", event.cost)
        funding_event_ids.add(event.event_id)
        funding_cost += cost
        funding_records.append(
            {
                "event_id": event.event_id,
                "effective_at": _iso(effective_at),
                "cost": cost,
            }
        )
    if not math.isfinite(funding_cost):
        raise ResearchValidationError("NON_FINITE_FUNDING_TOTAL", raw.trade_id)

    reference_gross = raw.direction * quantity * (exit_reference - entry_reference)
    execution_gross = raw.direction * quantity * (exit_fill - entry_fill)
    slippage_cost = reference_gross - execution_gross
    tolerance = max(1e-10, abs(reference_gross) * 1e-10)
    if slippage_cost < -tolerance:
        # Promotion accounting is deliberately conservative.  A simulator claiming favorable
        # slippage needs a different signed-cost ledger rather than silently becoming alpha.
        raise ResearchValidationError("FAVORABLE_SLIPPAGE_UNSUPPORTED", raw.trade_id)
    slippage_cost = max(0.0, slippage_cost)

    normalized = {
        "trade_id": raw.trade_id,
        "symbol": raw.symbol.upper(),
        "direction": raw.direction,
        "quantity": quantity,
        "signal_at": _iso(signal_at),
        "information_cutoff_at": _iso(information_cutoff),
        "opened_at": _iso(opened_at),
        "closed_at": _iso(closed_at),
        "entry_reference_price": entry_reference,
        "entry_fill_price": entry_fill,
        "exit_reference_price": exit_reference,
        "exit_fill_price": exit_fill,
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "funding_events": funding_records,
        "funding_cost": funding_cost,
        "slippage_cost": slippage_cost,
        "entry_fee_evidence_ids": list(raw.entry_fee_evidence_ids),
        "exit_fee_evidence_ids": list(raw.exit_fee_evidence_ids),
        "funding_coverage_id": raw.funding_coverage_id,
        "source_ids": sorted(set(raw.source_ids)),
        "market_data_digest": raw.market_data_digest,
    }
    return (
        TradeOutcome(
            symbol=raw.symbol.upper(),
            closed_at=closed_at,
            gross_pnl=reference_gross,
            fees=entry_fee + exit_fee,
            slippage_cost=slippage_cost,
            funding_cost=funding_cost,
        ),
        normalized,
    )


def _evaluate_request(
    evaluator: PointInTimeScenarioEvaluator,
    request: ScenarioRequest,
    *,
    initial_equity: float,
) -> _ScenarioResult:
    raw_trades = tuple(evaluator.evaluate(request))
    seen: set[str] = set()
    outcomes: list[TradeOutcome] = []
    normalized: list[dict[str, Any]] = []
    sources: set[str] = set()
    for raw in raw_trades:
        if raw.trade_id in seen:
            raise ResearchValidationError("DUPLICATE_TRADE_ID", raw.trade_id)
        seen.add(raw.trade_id)
        outcome, audit_record = _trade_to_outcome(raw, request=request)
        outcomes.append(outcome)
        normalized.append(audit_record)
        sources.update(raw.source_ids)
        sources.update(raw.entry_fee_evidence_ids)
        sources.update(raw.exit_fee_evidence_ids)
        sources.update(item.event_id for item in raw.funding_events)
        sources.add(raw.funding_coverage_id)
    metrics = compute_performance_metrics(outcomes, initial_equity=initial_equity)
    if metrics.max_drawdown is not None and metrics.max_drawdown > 1:
        raise ResearchValidationError(
            "NEGATIVE_SIMULATED_EQUITY",
            f"{request.window.window_id}/{request.scenario.scenario_id}",
        )
    return _ScenarioResult(
        window_id=request.window.window_id,
        scenario_id=request.scenario.scenario_id,
        trade_ids=tuple(sorted(seen)),
        outcomes=tuple(outcomes),
        raw_digest=_digest(normalized),
        source_ids=tuple(sorted(sources)),
        metrics=metrics,
    )


def _cost_stress_outcomes(
    outcomes: Sequence[TradeOutcome], multiplier: float
) -> tuple[TradeOutcome, ...]:
    stressed: list[TradeOutcome] = []
    for trade in outcomes:
        funding = trade.funding_cost * multiplier if trade.funding_cost > 0 else trade.funding_cost
        stressed.append(
            TradeOutcome(
                symbol=trade.symbol,
                closed_at=trade.closed_at,
                gross_pnl=trade.gross_pnl,
                fees=trade.fees * multiplier,
                slippage_cost=trade.slippage_cost * multiplier,
                funding_cost=funding,
            )
        )
    return tuple(stressed)


def _is_profitable(metrics: PerformanceMetrics) -> bool:
    return metrics.net_return is not None and metrics.net_return > 0


def _valid_statistics(
    validator: StatisticalValidator | None,
    request: StatisticalValidationRequest,
    *,
    expected_fold_count: int,
) -> tuple[float | None, float | None, dict[str, Any]]:
    if validator is None:
        return None, None, {"status": "NOT_PROVIDED"}
    try:
        result = validator.calculate(request)
        if isinstance(result.dsr_significance_probability, bool) or isinstance(
            result.pbo_probability, bool
        ):
            raise ValueError("boolean statistical probabilities are invalid")
        probabilities = (
            float(result.dsr_significance_probability),
            float(result.pbo_probability),
        )
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if result.dsr_method != "BAILEY_LOPEZ_DE_PRADO_DSR":
            raise ValueError("unsupported DSR method")
        if result.pbo_method != "CSCV_PBO":
            raise ValueError("unsupported PBO method")
        if not isinstance(result.source_digest, str) or not result.source_digest.strip():
            raise ValueError("statistical source digest is missing")
        if (
            request.trade_count < 30
            or isinstance(result.observation_count, bool)
            or isinstance(result.independent_trial_count, bool)
            or isinstance(result.fold_count, bool)
            or not isinstance(result.observation_count, int)
            or not isinstance(result.independent_trial_count, int)
            or not isinstance(result.fold_count, int)
            or result.observation_count != request.trade_count
            or result.observation_count != len(request.trade_net_returns)
            or result.independent_trial_count < 2
        ):
            raise ValueError("statistical sample is too small")
        if result.fold_count != expected_fold_count or result.fold_count < 4:
            raise ValueError("statistical fold evidence is incomplete")
    except Exception as error:
        return (
            None,
            None,
            {
                "status": "INVALID_FAIL_CLOSED",
                "error_type": type(error).__name__,
                "detail": str(error),
            },
        )
    return (
        probabilities[0],
        probabilities[1],
        {
            "status": "VALIDATED",
            "dsr_method": result.dsr_method,
            "pbo_method": result.pbo_method,
            "source_digest": result.source_digest,
            "observation_count": result.observation_count,
            "independent_trial_count": result.independent_trial_count,
            "fold_count": result.fold_count,
        },
    )


@dataclass(frozen=True, slots=True)
class ResearchValidationReport:
    spec_id: str
    trace_id: str
    started_at: datetime
    ended_at: datetime
    evidence: BacktestEvidence
    input_summary: Mapping[str, Any]
    raw_metrics: Mapping[str, Any]
    report_digest: str

    @property
    def audited_input_digest(self) -> str:
        return str(self.input_summary["audited_input_digest"])

    def append_backtest_run(self, audit: AuditRepository) -> str:
        """Persist one completed job for discovery by `LearningPromotionScheduler`.

        The scheduler already discovers persisted evidence during its daily tick.  A research
        service should call this method first and then call `LearningPromotionScheduler.tick`;
        no historical result is generated or inferred by the scheduler itself.
        """

        if self.evidence.completed is not True:
            raise ResearchValidationError("BACKTEST_JOB_INCOMPLETE", self.spec_id)
        summary_without_digest = dict(self.input_summary)
        summary_without_digest.pop("audited_input_digest", None)
        if _digest(summary_without_digest) != self.audited_input_digest:
            raise ResearchValidationError("BACKTEST_REPORT_TAMPERED", self.spec_id)
        raw_input_summary = self.raw_metrics.get("input_summary")
        if not isinstance(raw_input_summary, Mapping) or _canonical(
            raw_input_summary
        ) != _canonical(self.input_summary):
            raise ResearchValidationError("BACKTEST_REPORT_TAMPERED", self.spec_id)
        expected_report_digest = _digest(
            {
                "spec_id": self.spec_id,
                "trace_id": self.trace_id,
                "started_at": _iso(self.started_at),
                "ended_at": _iso(self.ended_at),
                "evidence": self.evidence.as_dict(),
                "input_digest": self.audited_input_digest,
                "raw_metrics": self.raw_metrics,
            }
        )
        if expected_report_digest != self.report_digest:
            raise ResearchValidationError("BACKTEST_REPORT_TAMPERED", self.spec_id)
        metrics = self.evidence.metrics
        backtest_id = f"backtest_{self.audited_input_digest[:32]}"
        trace = audit.get_trace(self.trace_id)
        if not any(str(item["spec_id"]) == self.spec_id for item in trace["strategy_specs"]):
            raise ResearchValidationError(
                "CHALLENGER_SPEC_NOT_IN_TRACE", f"{self.trace_id}/{self.spec_id}"
            )
        existing = next(
            (
                item
                for item in trace["backtest_runs"]
                if str(item["backtest_run_id"]) == backtest_id
            ),
            None,
        )
        if existing is not None:
            existing_digest = (
                existing.get("raw_metrics", {}).get("input_summary", {}).get("audited_input_digest")
            )
            if str(existing["spec_id"]) != self.spec_id or existing_digest != (
                self.audited_input_digest
            ):
                raise ResearchValidationError("BACKTEST_ID_CONFLICT", backtest_id)
            return backtest_id
        return audit.append_backtest_run(
            backtest_run_id=backtest_id,
            trace_id=self.trace_id,
            spec_id=self.spec_id,
            started_at=self.started_at,
            ended_at=self.ended_at,
            completed=True,
            net_profit=metrics.net_profit,
            net_return=metrics.net_return,
            max_drawdown=metrics.max_drawdown,
            total_cost=metrics.total_cost,
            stressed_net_return_2x=metrics.stressed_net_return_2x,
            dsr_significance_probability=self.evidence.dsr_significance_probability,
            pbo_probability=self.evidence.pbo_probability,
            symbol_concentration=metrics.symbol_concentration,
            month_concentration=metrics.month_concentration,
            trade_count=metrics.trade_count,
            holdout_months=self.evidence.holdout_months,
            validation={
                "walk_forward_passed": self.evidence.walk_forward_passed,
                "holdout_passed": self.evidence.holdout_passed,
                "parameter_perturbation_passed": (self.evidence.parameter_perturbation_passed),
                "latency_stress_passed": self.evidence.latency_stress_passed,
                "social_placebo_passed": self.evidence.social_placebo_passed,
                "cost_stress_2x_passed": self.raw_metrics["cost_stress_2x_passed"],
                "cost_stress_3x_passed": self.raw_metrics["cost_stress_3x_passed"],
            },
            raw_metrics=self.raw_metrics,
            created_at=self.ended_at,
        )


class ResearchBacktestValidator:
    """Run a deterministic expanding-window matrix and produce append-only evidence."""

    def __init__(
        self,
        *,
        spec_id: str,
        trace_id: str,
        strategy_parameters: Mapping[str, Any],
        config: ExpandingWalkForwardConfig,
        initial_equity: float,
        evaluator: PointInTimeScenarioEvaluator,
        statistical_validator: StatisticalValidator | None = None,
    ) -> None:
        if (
            not isinstance(spec_id, str)
            or not spec_id.strip()
            or not isinstance(trace_id, str)
            or not trace_id.strip()
        ):
            raise ResearchValidationError("MISSING_AUDIT_IDENTITY", "spec_id/trace_id")
        self.spec_id = spec_id
        self.trace_id = trace_id
        self.strategy_parameters = dict(strategy_parameters)
        try:
            _canonical(self.strategy_parameters)
        except (TypeError, ValueError) as error:
            raise ResearchValidationError(
                "NON_CANONICAL_STRATEGY_PARAMETERS", str(error)
            ) from error
        self.config = config
        self.initial_equity = _require_finite("initial_equity", initial_equity, positive=True)
        self.evaluator = evaluator
        self.statistical_validator = statistical_validator

    def run(self) -> ResearchValidationReport:
        walk_windows, holdout = build_expanding_windows(self.config)
        strategy_digest = _digest(self.strategy_parameters)
        results: list[_ScenarioResult] = []

        # All pre-holdout evaluation is completed first.  Once frozen, this method performs
        # only holdout requests and never gives the evaluator another optimization window.
        for window in walk_windows:
            for scenario in STANDARD_SCENARIOS:
                results.append(
                    _evaluate_request(
                        self.evaluator,
                        ScenarioRequest(strategy_digest, window, scenario),
                        initial_equity=self.initial_equity,
                    )
                )
        pre_holdout_digest = _digest([item.summary() for item in results])
        seal_id = self.evaluator.freeze_for_holdout(
            strategy_digest=strategy_digest,
            pre_holdout_results_digest=pre_holdout_digest,
        )
        if not isinstance(seal_id, str) or not seal_id.strip():
            raise ResearchValidationError("HOLDOUT_NOT_SEALED", self.spec_id)
        for scenario in STANDARD_SCENARIOS:
            results.append(
                _evaluate_request(
                    self.evaluator,
                    ScenarioRequest(strategy_digest, holdout, scenario, seal_id),
                    initial_equity=self.initial_equity,
                )
            )

        result_by_key = {(item.window_id, item.scenario_id): item for item in results}
        all_windows = (*walk_windows, holdout)

        for scenario in STANDARD_SCENARIOS:
            seen_trade_ids: set[str] = set()
            for window in all_windows:
                item = result_by_key[(window.window_id, scenario.scenario_id)]
                duplicate_ids = seen_trade_ids.intersection(item.trade_ids)
                if duplicate_ids:
                    raise ResearchValidationError(
                        "CROSS_WINDOW_DUPLICATE_TRADE_ID", sorted(duplicate_ids)[0]
                    )
                seen_trade_ids.update(item.trade_ids)

        def scenario_results(scenario_id: str) -> tuple[_ScenarioResult, ...]:
            return tuple(result_by_key[(window.window_id, scenario_id)] for window in all_windows)

        def combined_outcomes(scenario_id: str) -> tuple[TradeOutcome, ...]:
            return tuple(
                outcome for item in scenario_results(scenario_id) for outcome in item.outcomes
            )

        combined_metrics: dict[str, PerformanceMetrics] = {}
        for scenario in STANDARD_SCENARIOS:
            combined_metrics[scenario.scenario_id] = compute_performance_metrics(
                combined_outcomes(scenario.scenario_id),
                initial_equity=self.initial_equity,
            )
        baseline_outcomes = combined_outcomes("baseline")
        baseline_metrics = combined_metrics["baseline"]
        baseline_2x = compute_performance_metrics(
            _cost_stress_outcomes(baseline_outcomes, 2.0),
            initial_equity=self.initial_equity,
        )
        baseline_3x = compute_performance_metrics(
            _cost_stress_outcomes(baseline_outcomes, 3.0),
            initial_equity=self.initial_equity,
        )
        if baseline_metrics.max_drawdown is not None and baseline_metrics.max_drawdown > 1:
            raise ResearchValidationError("NEGATIVE_SIMULATED_EQUITY", "combined baseline")

        walk_forward_passed = all(
            _is_profitable(result_by_key[(window.window_id, "baseline")].metrics)
            for window in walk_windows
        )
        holdout_passed = _is_profitable(result_by_key[(holdout.window_id, "baseline")].metrics)
        perturbation_passed = all(
            _is_profitable(result_by_key[(window.window_id, scenario_id)].metrics)
            for window in all_windows
            for scenario_id in ("parameter-075", "parameter-125")
        )
        latency_passed = all(
            _is_profitable(result_by_key[(window.window_id, scenario_id)].metrics)
            for window in all_windows
            for scenario_id in ("latency-1m", "latency-5m", "latency-15m")
        )
        social_placebo_passed = True
        for window in all_windows:
            baseline_return = result_by_key[(window.window_id, "baseline")].metrics.net_return
            placebo_return = result_by_key[(window.window_id, "social-placebo")].metrics.net_return
            if (
                baseline_return is None
                or placebo_return is None
                or baseline_return <= 0
                or placebo_return > baseline_return + abs(baseline_return) * 0.05 + 1e-12
            ):
                social_placebo_passed = False

        cost_stress_2x_passed = baseline_2x.net_return is not None and baseline_2x.net_return > 0
        cost_stress_3x_passed = baseline_3x.net_return is not None and baseline_3x.net_return > 0

        result_summaries = [item.summary() for item in results]
        preliminary_summary = {
            "schema_version": 1,
            "spec_id": self.spec_id,
            "trace_id": self.trace_id,
            "strategy_parameters": self.strategy_parameters,
            "strategy_digest": strategy_digest,
            "config": {
                "research_started_at": _iso(self.config.research_started_at),
                "research_ended_at": _iso(self.config.research_ended_at),
                "initial_training_months": self.config.initial_training_months,
                "test_window_months": self.config.test_window_months,
                "holdout_months": self.config.holdout_months,
                "initial_equity": self.initial_equity,
            },
            "walk_forward_windows": [window.as_dict() for window in walk_windows],
            "sealed_holdout": holdout.as_dict(),
            "standard_scenarios": [scenario.as_dict() for scenario in STANDARD_SCENARIOS],
            "pre_holdout_results_digest": pre_holdout_digest,
            "holdout_seal_id": seal_id,
            "result_summaries": result_summaries,
            "cost_accounting": {
                "fees": "explicit entry and exit venue/simulator evidence",
                "slippage": "reference-price PnL minus actual-fill PnL",
                "funding": "explicit signed cost with complete coverage evidence",
                "stress": (
                    "fees, adverse slippage, and funding payments multiplied; credits unchanged"
                ),
            },
        }
        audited_input_digest = _digest(preliminary_summary)
        input_summary = {
            **preliminary_summary,
            "audited_input_digest": audited_input_digest,
        }
        statistical_outcomes = tuple(
            outcome
            for window in walk_windows
            for outcome in result_by_key[(window.window_id, "baseline")].outcomes
        )
        fold_returns = tuple(
            float(result_by_key[(window.window_id, "baseline")].metrics.net_return or 0.0)
            for window in walk_windows
        )
        dsr, pbo, statistics_summary = _valid_statistics(
            self.statistical_validator,
            StatisticalValidationRequest(
                strategy_digest=strategy_digest,
                # Statistical selection is bound only to the pre-holdout manifest.  The
                # sealed final year is never supplied to a PBO/DSR implementation as a
                # strategy-selection sample.
                audited_input_digest=pre_holdout_digest,
                fold_net_returns=fold_returns,
                trade_net_returns=tuple(
                    outcome.net_pnl / self.initial_equity for outcome in statistical_outcomes
                ),
                trade_count=len(statistical_outcomes),
            ),
            expected_fold_count=len(walk_windows),
        )

        raw_metrics: dict[str, Any] = {
            "input_summary": input_summary,
            "aggregate_by_scenario": {
                key: value.as_dict() for key, value in combined_metrics.items()
            },
            "baseline_cost_stress_2x": baseline_2x.as_dict(),
            "baseline_cost_stress_3x": baseline_3x.as_dict(),
            "cost_stress_2x_passed": cost_stress_2x_passed,
            "cost_stress_3x_passed": cost_stress_3x_passed,
            "statistics": statistics_summary,
        }
        evidence = BacktestEvidence(
            metrics=baseline_metrics,
            completed=True,
            dsr_significance_probability=dsr,
            pbo_probability=pbo,
            holdout_months=12,
            walk_forward_passed=walk_forward_passed,
            holdout_passed=holdout_passed,
            parameter_perturbation_passed=perturbation_passed,
            latency_stress_passed=latency_passed,
            social_placebo_passed=social_placebo_passed,
        )
        report_digest = _digest(
            {
                "spec_id": self.spec_id,
                "trace_id": self.trace_id,
                "started_at": _iso(self.config.research_started_at),
                "ended_at": _iso(self.config.research_ended_at),
                "evidence": evidence.as_dict(),
                "input_digest": audited_input_digest,
                "raw_metrics": raw_metrics,
            }
        )
        return ResearchValidationReport(
            spec_id=self.spec_id,
            trace_id=self.trace_id,
            started_at=self.config.research_started_at,
            ended_at=self.config.research_ended_at,
            evidence=evidence,
            input_summary=input_summary,
            raw_metrics=raw_metrics,
            report_digest=report_digest,
        )


@dataclass(frozen=True, slots=True)
class AuditedShadowTrade:
    trade_id: str
    outcome: TradeOutcome
    fee_evidence_id: str
    slippage_evidence_id: str
    funding_evidence_id: str
    accounting_complete: bool


@dataclass(frozen=True, slots=True)
class ShadowAppendResult:
    appended: bool
    reason_codes: tuple[str, ...]
    champion_shadow_result_id: str | None = None
    challenger_shadow_result_id: str | None = None


class PairedShadowAccumulator:
    """Accumulate real paired shadow outcomes and persist only after full maturity.

    Call `record_daily_coverage` from the shadow journal every UTC day and `record_trade` from
    immutable fill/funding accounting.  After both strategies have at least 30 closed trades
    and coverage spans 90 days, `append_if_mature` writes comparable rows that the existing
    `LearningPromotionScheduler` can discover on its next daily tick.
    """

    def __init__(
        self,
        *,
        trace_id: str,
        champion_spec_id: str,
        challenger_spec_id: str,
        started_at: datetime,
        initial_equity: float,
        audit: AuditRepository | None = None,
    ) -> None:
        if (
            not isinstance(trace_id, str)
            or not trace_id.strip()
            or not isinstance(champion_spec_id, str)
            or not champion_spec_id.strip()
            or not isinstance(challenger_spec_id, str)
            or not challenger_spec_id.strip()
        ):
            raise ResearchValidationError("MISSING_AUDIT_IDENTITY", "shadow IDs")
        if champion_spec_id == challenger_spec_id:
            raise ResearchValidationError("UNPAIRED_SHADOW_SPECS", champion_spec_id)
        self.trace_id = trace_id
        self.champion_spec_id = champion_spec_id
        self.challenger_spec_id = challenger_spec_id
        self.started_at = _utc(started_at, "started_at")
        self.initial_equity = _require_finite("initial_equity", initial_equity, positive=True)
        self.pair_id = "shadow_pair_" + _digest(
            {
                "trace_id": trace_id,
                "champion_spec_id": champion_spec_id,
                "challenger_spec_id": challenger_spec_id,
                "started_at": _iso(self.started_at),
            }
        )[:32]
        self.audit = audit
        self._audit_strategy_versions: dict[str, str] = {}
        self._verified_audit_trace_ids: set[str] = set()
        self._trades: dict[str, dict[str, AuditedShadowTrade]] = {
            champion_spec_id: {},
            challenger_spec_id: {},
        }
        self._coverage: dict[date, str] = {}
        if audit is not None:
            trace = audit.get_trace(self.trace_id)
            specs = {
                str(row["spec_id"]): str(row["strategy_version"])
                for row in trace["strategy_specs"]
            }
            for spec_id in (self.champion_spec_id, self.challenger_spec_id):
                strategy_version = specs.get(spec_id)
                if not strategy_version:
                    raise ResearchValidationError(
                        "SHADOW_SPEC_NOT_IN_AUDIT_TRACE", f"{self.trace_id}/{spec_id}"
                    )
                self._audit_strategy_versions[spec_id] = strategy_version
            self.hydrate_from_audit(audit)

    def hydrate_from_audit(self, audit: AuditRepository) -> int:
        """Rebuild the 90-day accumulator from immutable journal events."""

        restored = 0
        for row in audit.shadow_journal_events(self.pair_id):
            if (
                str(row["trace_id"]) != self.trace_id
                or str(row["champion_spec_id"]) != self.champion_spec_id
                or str(row["challenger_spec_id"]) != self.challenger_spec_id
                or _utc(
                    datetime.fromisoformat(str(row["pair_started_at"]).replace("Z", "+00:00")),
                    "pair_started_at",
                )
                != self.started_at
            ):
                raise ResearchValidationError("SHADOW_JOURNAL_PAIR_MISMATCH", self.pair_id)
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                raise ResearchValidationError("INVALID_SHADOW_JOURNAL_PAYLOAD", self.pair_id)
            if row["event_type"] == "COVERAGE":
                observed_at = datetime.fromisoformat(
                    str(row["observed_at"]).replace("Z", "+00:00")
                )
                self._record_daily_coverage(
                    observed_at=observed_at,
                    evidence_id=str(payload.get("evidence_id") or ""),
                    persist=False,
                )
            elif row["event_type"] == "TRADE":
                raw_outcome = payload.get("outcome")
                if not isinstance(raw_outcome, Mapping):
                    raise ResearchValidationError(
                        "INVALID_SHADOW_JOURNAL_PAYLOAD", str(row["event_key"])
                    )
                trade = AuditedShadowTrade(
                    trade_id=str(payload.get("trade_id") or ""),
                    outcome=TradeOutcome(
                        symbol=str(raw_outcome.get("symbol") or ""),
                        closed_at=datetime.fromisoformat(
                            str(raw_outcome.get("closed_at") or "").replace("Z", "+00:00")
                        ),
                        gross_pnl=float(raw_outcome["gross_pnl"]),
                        fees=float(raw_outcome["fees"]),
                        slippage_cost=float(raw_outcome["slippage_cost"]),
                        funding_cost=float(raw_outcome["funding_cost"]),
                        episode_id=(
                            str(raw_outcome["episode_id"])
                            if raw_outcome.get("episode_id") is not None
                            else None
                        ),
                        trace_ids=tuple(str(item) for item in raw_outcome.get("trace_ids", ())),
                        strategy_versions=tuple(
                            str(item) for item in raw_outcome.get("strategy_versions", ())
                        ),
                        source_record_ids=tuple(
                            str(item) for item in raw_outcome.get("source_record_ids", ())
                        ),
                    ),
                    fee_evidence_id=str(payload.get("fee_evidence_id") or ""),
                    slippage_evidence_id=str(payload.get("slippage_evidence_id") or ""),
                    funding_evidence_id=str(payload.get("funding_evidence_id") or ""),
                    accounting_complete=payload.get("accounting_complete") is True,
                )
                self._record_trade(spec_id=str(row["spec_id"]), trade=trade, persist=False)
            else:
                raise ResearchValidationError(
                    "INVALID_SHADOW_JOURNAL_EVENT", str(row["event_type"])
                )
            restored += 1
        return restored

    def record_daily_coverage(self, *, observed_at: datetime, evidence_id: str) -> None:
        self._record_daily_coverage(
            observed_at=observed_at,
            evidence_id=evidence_id,
            persist=True,
        )

    def _record_daily_coverage(
        self,
        *,
        observed_at: datetime,
        evidence_id: str,
        persist: bool,
    ) -> None:
        observed = _utc(observed_at, "observed_at")
        if (
            observed < self.started_at
            or not isinstance(evidence_id, str)
            or not evidence_id.strip()
        ):
            raise ResearchValidationError("INVALID_SHADOW_COVERAGE", str(evidence_id))
        existing = self._coverage.get(observed.date())
        if existing is not None and existing != evidence_id:
            raise ResearchValidationError(
                "CONFLICTING_SHADOW_COVERAGE", observed.date().isoformat()
            )
        if persist and self.audit is not None:
            self.audit.append_shadow_journal_event(
                trace_id=self.trace_id,
                pair_id=self.pair_id,
                champion_spec_id=self.champion_spec_id,
                challenger_spec_id=self.challenger_spec_id,
                pair_started_at=self.started_at,
                event_type="COVERAGE",
                event_key=observed.date().isoformat(),
                payload={
                    "coverage_date": observed.date().isoformat(),
                    "evidence_id": evidence_id,
                },
                observed_at=observed,
            )
        self._coverage[observed.date()] = evidence_id

    def record_trade(self, *, spec_id: str, trade: AuditedShadowTrade) -> None:
        self._record_trade(spec_id=spec_id, trade=trade, persist=True)

    def _validate_paired_shadow_cost_evidence(
        self,
        *,
        trade: AuditedShadowTrade,
        trace_id: str,
        strategy_version: str,
        closed_at: datetime,
    ) -> None:
        if self.audit is None:  # pragma: no cover - guarded by the caller
            return
        expected_records = (
            (trade.fee_evidence_id, "FEE", trade.outcome.fees),
            (trade.slippage_evidence_id, "SLIPPAGE", trade.outcome.slippage_cost),
            (trade.funding_evidence_id, "FUNDING", trade.outcome.funding_cost),
        )
        rows = self.audit.external_evidence_records(
            tuple(record_id for record_id, _, _ in expected_records)
        )
        if len(rows) != 3:
            raise ResearchValidationError(
                "UNVERIFIED_SHADOW_COST_EVIDENCE", trade.trade_id
            )

        for record_id, cost_type, expected_amount in expected_records:
            row = rows.get(record_id)
            payload = row.get("payload") if isinstance(row, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or row.get("deleted_at") is not None
                or not isinstance(payload, Mapping)
                or payload.get("schema") != PAIRED_SHADOW_COST_SCHEMA
                or payload.get("cost_type") != cost_type
            ):
                raise ResearchValidationError(
                    "UNVERIFIED_SHADOW_COST_EVIDENCE", record_id
                )

            amount = payload.get("amount")
            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(float(amount))
                or not math.isfinite(float(expected_amount))
                or float(amount) != float(expected_amount)
            ):
                raise ResearchValidationError(
                    "SHADOW_COST_EVIDENCE_BINDING_MISMATCH", f"{record_id}:amount"
                )

            raw_closed_at = payload.get("closed_at")
            try:
                evidence_closed_at = _utc(
                    datetime.fromisoformat(str(raw_closed_at).replace("Z", "+00:00")),
                    "closed_at",
                )
                record_occurred_at = _utc(
                    datetime.fromisoformat(str(row.get("occurred_at")).replace("Z", "+00:00")),
                    "occurred_at",
                )
            except (TypeError, ValueError) as error:
                raise ResearchValidationError(
                    "SHADOW_COST_EVIDENCE_BINDING_MISMATCH", f"{record_id}:closed_at"
                ) from error

            expected_fields = {
                "trade_id": trade.trade_id,
                "episode_id": trade.outcome.episode_id,
                "trace_id": trace_id,
                "symbol": trade.outcome.symbol,
                "strategy_version": strategy_version,
            }
            mismatched_fields = tuple(
                field
                for field, expected in expected_fields.items()
                if payload.get(field) != expected
            )
            if row.get("trace_id") != trace_id:
                mismatched_fields = (*mismatched_fields, "audit_trace_id")
            if evidence_closed_at != closed_at or record_occurred_at != closed_at:
                mismatched_fields = (*mismatched_fields, "closed_at")
            if mismatched_fields:
                raise ResearchValidationError(
                    "SHADOW_COST_EVIDENCE_BINDING_MISMATCH",
                    f"{record_id}:{','.join(mismatched_fields)}",
                )

    def _record_trade(
        self,
        *,
        spec_id: str,
        trade: AuditedShadowTrade,
        persist: bool,
    ) -> None:
        if spec_id not in self._trades:
            raise ResearchValidationError("UNKNOWN_SHADOW_SPEC", spec_id)
        if (
            not isinstance(trade.trade_id, str)
            or not trade.trade_id.strip()
            or trade.accounting_complete is not True
        ):
            raise ResearchValidationError("INCOMPLETE_SHADOW_ACCOUNTING", str(trade.trade_id))
        evidence_ids = (
            trade.fee_evidence_id,
            trade.slippage_evidence_id,
            trade.funding_evidence_id,
        )
        if not all(isinstance(item, str) and item.strip() for item in evidence_ids):
            raise ResearchValidationError("MISSING_SHADOW_COST_EVIDENCE", trade.trade_id)
        if len(set(evidence_ids)) != 3:
            raise ResearchValidationError(
                "DUPLICATE_SHADOW_COST_EVIDENCE", trade.trade_id
            )
        if not isinstance(trade.outcome.symbol, str) or not trade.outcome.symbol.strip():
            raise ResearchValidationError("MISSING_SHADOW_SYMBOL", trade.trade_id)
        closed_at = _utc(trade.outcome.closed_at, "closed_at")
        if closed_at < self.started_at:
            raise ResearchValidationError("SHADOW_TRADE_BEFORE_START", trade.trade_id)
        if self.audit is not None:
            outcome = trade.outcome
            if not (
                isinstance(outcome.episode_id, str)
                and outcome.episode_id.strip()
                and outcome.trace_ids
                and outcome.source_record_ids
                and outcome.strategy_versions
                and all(str(item).strip() for item in outcome.trace_ids)
                and all(str(item).strip() for item in outcome.source_record_ids)
                and all(str(item).strip() for item in outcome.strategy_versions)
            ):
                raise ResearchValidationError("INCOMPLETE_SHADOW_LINEAGE", trade.trade_id)
            source_ids = set(outcome.source_record_ids)
            if not set(evidence_ids).issubset(source_ids):
                raise ResearchValidationError(
                    "UNVERIFIED_SHADOW_COST_EVIDENCE", trade.trade_id
                )
            expected_version = self._audit_strategy_versions.get(spec_id)
            if expected_version is None or outcome.strategy_versions != (expected_version,):
                raise ResearchValidationError(
                    "SHADOW_STRATEGY_LINEAGE_MISMATCH", trade.trade_id
                )
            if len(outcome.trace_ids) != 1:
                raise ResearchValidationError(
                    "INVALID_SHADOW_TRACE_BINDING", trade.trade_id
                )
            trade_trace_id = outcome.trace_ids[0]
            if trade_trace_id not in self._verified_audit_trace_ids:
                if not self.audit.audit_fact_trace_ids_exist((trade_trace_id,)):
                    raise ResearchValidationError(
                        "UNVERIFIED_SHADOW_TRACE", trade.trade_id
                    )
                self._verified_audit_trace_ids.add(trade_trace_id)
            if not self.audit.performance_source_ids_exist(outcome.source_record_ids):
                raise ResearchValidationError(
                    "UNVERIFIED_SHADOW_SOURCE_RECORD", trade.trade_id
                )
            self._validate_paired_shadow_cost_evidence(
                trade=trade,
                trace_id=trade_trace_id,
                strategy_version=expected_version,
                closed_at=closed_at,
            )
        # Reuse the production metric validator for finite values and non-negative fee/slippage.
        compute_performance_metrics([trade.outcome], initial_equity=self.initial_equity)
        existing = self._trades[spec_id].get(trade.trade_id)
        if existing is not None:
            if existing != trade:
                raise ResearchValidationError("DUPLICATE_SHADOW_TRADE", trade.trade_id)
            return
        if persist and self.audit is not None:
            outcome_payload = asdict(trade.outcome)
            outcome_payload["closed_at"] = _iso(trade.outcome.closed_at)
            self.audit.append_shadow_journal_event(
                trace_id=self.trace_id,
                pair_id=self.pair_id,
                champion_spec_id=self.champion_spec_id,
                challenger_spec_id=self.challenger_spec_id,
                pair_started_at=self.started_at,
                event_type="TRADE",
                event_key=f"{spec_id}:{trade.trade_id}",
                spec_id=spec_id,
                payload={
                    "trade_id": trade.trade_id,
                    "outcome": outcome_payload,
                    "fee_evidence_id": trade.fee_evidence_id,
                    "slippage_evidence_id": trade.slippage_evidence_id,
                    "funding_evidence_id": trade.funding_evidence_id,
                    "accounting_complete": trade.accounting_complete,
                },
                observed_at=closed_at,
            )
        self._trades[spec_id][trade.trade_id] = trade

    def append_if_mature(
        self,
        audit: AuditRepository,
        *,
        ended_at: datetime,
    ) -> ShadowAppendResult:
        if self.audit is not None and audit is not self.audit:
            raise ResearchValidationError("SHADOW_AUDIT_MISMATCH", self.pair_id)
        ended = _utc(ended_at, "ended_at")
        elapsed_days = int((ended - self.started_at).total_seconds() // 86_400)
        reasons: list[str] = []
        if elapsed_days < 90:
            reasons.append("INSUFFICIENT_SHADOW_DAYS")

        required_dates: list[date] = []
        cursor = self.started_at.date()
        while cursor <= ended.date():
            required_dates.append(cursor)
            cursor += timedelta(days=1)
        missing_dates = [item.isoformat() for item in required_dates if item not in self._coverage]
        if missing_dates:
            reasons.append("INCOMPLETE_DAILY_SHADOW_COVERAGE")

        for label, spec_id in (
            ("CHAMPION", self.champion_spec_id),
            ("CHALLENGER", self.challenger_spec_id),
        ):
            trades = self._trades[spec_id]
            if len(trades) < 30:
                reasons.append(f"INSUFFICIENT_{label}_SHADOW_TRADES")
            if any(_utc(item.outcome.closed_at, "closed_at") > ended for item in trades.values()):
                reasons.append(f"FUTURE_{label}_SHADOW_TRADE")
        if reasons:
            return ShadowAppendResult(False, tuple(reasons))

        if self.audit is None:
            # An in-memory accumulator is useful for simulation, but it must not become a
            # persistence bypass. Re-run every otherwise-mature trade through an audit-bound
            # verifier before any promotion evidence is written.
            verifier = PairedShadowAccumulator(
                trace_id=self.trace_id,
                champion_spec_id=self.champion_spec_id,
                challenger_spec_id=self.challenger_spec_id,
                started_at=self.started_at,
                initial_equity=self.initial_equity,
                audit=audit,
            )
            for spec_id in (self.champion_spec_id, self.challenger_spec_id):
                for trade in self._trades[spec_id].values():
                    verifier._record_trade(spec_id=spec_id, trade=trade, persist=False)

        metrics: dict[str, PerformanceMetrics] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for spec_id in (self.champion_spec_id, self.challenger_spec_id):
            records = sorted(
                self._trades[spec_id].values(),
                key=lambda item: (_utc(item.outcome.closed_at, "closed_at"), item.trade_id),
            )
            outcomes = tuple(item.outcome for item in records)
            metrics[spec_id] = compute_performance_metrics(
                outcomes, initial_equity=self.initial_equity
            )
            if metrics[spec_id].max_drawdown is not None and metrics[spec_id].max_drawdown > 1:
                raise ResearchValidationError("NEGATIVE_SIMULATED_EQUITY", f"shadow/{spec_id}")
            cost_3x = compute_performance_metrics(
                _cost_stress_outcomes(outcomes, 3.0), initial_equity=self.initial_equity
            )
            audit_inputs = [
                {
                    "trade_id": item.trade_id,
                    "closed_at": _iso(item.outcome.closed_at),
                    "symbol": item.outcome.symbol.upper(),
                    "gross_pnl": item.outcome.gross_pnl,
                    "fees": item.outcome.fees,
                    "slippage_cost": item.outcome.slippage_cost,
                    "funding_cost": item.outcome.funding_cost,
                    "fee_evidence_id": item.fee_evidence_id,
                    "slippage_evidence_id": item.slippage_evidence_id,
                    "funding_evidence_id": item.funding_evidence_id,
                }
                for item in records
            ]
            summaries[spec_id] = {
                "schema_version": 1,
                "initial_equity": self.initial_equity,
                "trade_input_digest": _digest(audit_inputs),
                "trade_count": len(records),
                "coverage_evidence_digest": _digest(
                    [self._coverage[item] for item in required_dates]
                ),
                "coverage_day_count": len(required_dates),
                "metrics": metrics[spec_id].as_dict(),
                "cost_stress_3x": cost_3x.as_dict(),
            }

        pair_digest = _digest(
            {
                "trace_id": self.trace_id,
                "champion": summaries[self.champion_spec_id],
                "challenger": summaries[self.challenger_spec_id],
                "started_at": _iso(self.started_at),
                "ended_at": _iso(ended),
            }
        )
        champion_result_id = f"shadow_champion_{pair_digest[:24]}"
        challenger_result_id = f"shadow_challenger_{pair_digest[:24]}"
        existing = {
            str(item["shadow_result_id"]): item
            for item in audit.get_trace(self.trace_id)["shadow_results"]
        }
        if champion_result_id in existing and challenger_result_id in existing:
            return ShadowAppendResult(
                False,
                ("ALREADY_APPENDED",),
                champion_result_id,
                challenger_result_id,
            )

        for spec_id, result_id in (
            (self.champion_spec_id, champion_result_id),
            (self.challenger_spec_id, challenger_result_id),
        ):
            if result_id in existing:
                continue
            item_metrics = metrics[spec_id]
            audit.append_shadow_result(
                shadow_result_id=result_id,
                trace_id=self.trace_id,
                spec_id=spec_id,
                started_at=self.started_at,
                ended_at=ended,
                completed=True,
                elapsed_days=elapsed_days,
                closed_trades=len(self._trades[spec_id]),
                net_return=item_metrics.net_return,
                max_drawdown=item_metrics.max_drawdown,
                stressed_net_return_2x=item_metrics.stressed_net_return_2x,
                symbol_concentration=item_metrics.symbol_concentration,
                month_concentration=item_metrics.month_concentration,
                raw_metrics={
                    "paired_shadow_digest": pair_digest,
                    "input_summary": summaries[spec_id],
                },
                created_at=ended,
            )
        return ShadowAppendResult(
            True,
            (),
            champion_result_id,
            challenger_result_id,
        )


__all__ = [
    "AuditedShadowTrade",
    "ExecutedBacktestTrade",
    "ExpandingWalkForwardConfig",
    "FundingPayment",
    "PairedShadowAccumulator",
    "PointInTimeScenarioEvaluator",
    "ResearchBacktestValidator",
    "ResearchValidationError",
    "ResearchValidationReport",
    "STANDARD_SCENARIOS",
    "ScenarioRequest",
    "ShadowAppendResult",
    "StatisticalValidation",
    "StatisticalValidationRequest",
    "StatisticalValidator",
    "ValidationScenario",
    "build_expanding_windows",
]
