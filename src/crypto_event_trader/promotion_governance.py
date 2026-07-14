from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from .audit import AuditRepository
from .contracts import PromotionRecord, PromotionStatus
from .learning import (
    BacktestEvidence,
    PerformanceMetrics,
    PromotionEvaluation,
    PromotionPolicy,
    ShadowEvidence,
    evaluate_promotion,
)
from .strategy_registry import StrategyRegistry


@dataclass(frozen=True, slots=True)
class PersistedPromotionEvidence:
    """Primary keys for one already persisted champion/challenger comparison."""

    trace_id: str
    champion_spec_id: str
    challenger_spec_id: str
    backtest_run_id: str
    champion_shadow_result_id: str
    challenger_shadow_result_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("trace_id", self.trace_id),
            ("champion_spec_id", self.champion_spec_id),
            ("challenger_spec_id", self.challenger_spec_id),
            ("backtest_run_id", self.backtest_run_id),
            ("champion_shadow_result_id", self.champion_shadow_result_id),
            ("challenger_shadow_result_id", self.challenger_shadow_result_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class GovernedPromotionResult:
    audit_record_id: str
    evaluation: PromotionEvaluation
    registry_record: PromotionRecord

    @property
    def promoted(self) -> bool:
        return self.registry_record.status is PromotionStatus.PROMOTED


class RollbackSignal(StrEnum):
    RECONCILIATION_ERROR = "reconciliation_error"
    RISK_BOUNDARY_VIOLATION = "risk_boundary_violation"
    PERFORMANCE_DRIFT = "performance_drift"


class GovernedPromotionCoordinator:
    """The only production path from research evidence to a champion mutation.

    This coordinator has no trading-stage or capital-allocation dependency.  A strategy can
    be promoted automatically, while Demo/canary/live allocation remains a separate, manual
    control-plane decision.
    """

    def __init__(self, audit: AuditRepository, registry: StrategyRegistry) -> None:
        self.audit = audit
        self.registry = registry

    def evaluate_and_apply(
        self,
        evidence: PersistedPromotionEvidence,
        *,
        policy: PromotionPolicy | None = None,
        caller_context: Mapping[str, Any] | None = None,
        evaluated_at: datetime | None = None,
    ) -> GovernedPromotionResult:
        """Persist the deterministic gate before applying its exact database result."""

        trace = self.audit.get_trace(evidence.trace_id)
        gate = self._evaluate_trace(trace, evidence, policy=policy, evaluated_at=evaluated_at)
        promotion_record_id = f"promotion_{uuid4().hex}"
        persisted_id = self.audit.append_promotion_record(
            trace_id=evidence.trace_id,
            champion_spec_id=evidence.champion_spec_id,
            challenger_spec_id=evidence.challenger_spec_id,
            backtest_run_id=evidence.backtest_run_id,
            champion_shadow_result_id=evidence.champion_shadow_result_id,
            challenger_shadow_result_id=evidence.challenger_shadow_result_id,
            eligible=gate.eligible,
            reason_codes=gate.reason_codes,
            evaluation={
                "coordinator": "governed-promotion-v1",
                "caller_context": dict(caller_context or {}),
            },
            promotion_policy=policy,
            promotion_record_id=promotion_record_id,
            created_at=gate.evaluated_at,
        )
        if persisted_id != promotion_record_id:
            raise RuntimeError("audit repository returned a different promotion record ID")

        # Re-read the append-only row.  The registry never trusts the pre-insert in-memory gate.
        persisted_trace = self.audit.get_trace(evidence.trace_id)
        record = self._registry_record_from_persisted(
            persisted_trace,
            promotion_record_id=persisted_id,
            expected_evidence=evidence,
        )
        applied = self.registry._apply_audited_record(record)
        return GovernedPromotionResult(
            audit_record_id=persisted_id,
            evaluation=gate,
            registry_record=applied,
        )

    def apply_persisted_record(
        self,
        evidence: PersistedPromotionEvidence,
        promotion_record_id: str,
    ) -> PromotionRecord:
        """Idempotently recover after the audit commit but before registry persistence."""

        trace = self.audit.get_trace(evidence.trace_id)
        record = self._registry_record_from_persisted(
            trace,
            promotion_record_id=promotion_record_id,
            expected_evidence=evidence,
        )
        return self.registry._apply_audited_record(record)

    def handle_rollback_signal(
        self,
        signal: RollbackSignal,
        *,
        observed_at: datetime | None = None,
        detail: str | None = None,
    ) -> PromotionRecord | None:
        """Immediately restore the prior champion for an explicit safety signal.

        Repeated signals after the same rollback are idempotent.  ``None`` means the registry
        has never promoted a challenger, so there is no earlier champion to restore.
        """

        if not isinstance(signal, RollbackSignal):
            raise TypeError("signal must be an explicit RollbackSignal")
        current = self.registry.champion.version
        prior_promotion = next(
            (
                item
                for item in reversed(self.registry.promotion_records)
                if item.status is PromotionStatus.PROMOTED
                and item.resulting_champion_version == current
            ),
            None,
        )
        if prior_promotion is None:
            return None
        reason = signal.value if not detail else f"{signal.value}:{detail[:160]}"
        effective_at = observed_at or datetime.now(UTC)
        champion_row = self.audit.strategy_spec_by_version(current)
        if champion_row is None or not champion_row.get("trace_id"):
            raise ValueError("current champion has no audit trace for rollback")
        event_id = self.audit.append_strategy_registry_rollback(
            trace_id=str(champion_row["trace_id"]),
            source_promotion_record_id=prior_promotion.record_id,
            previous_champion_version=current,
            resulting_champion_version=prior_promotion.previous_champion_version,
            reason=reason,
            created_at=effective_at,
        )
        return self.registry.rollback(
            reason=reason,
            target_version=prior_promotion.previous_champion_version,
            evaluated_at=effective_at,
            record_id=event_id,
        )

    @staticmethod
    def _evaluate_trace(
        trace: Mapping[str, Any],
        evidence: PersistedPromotionEvidence,
        *,
        policy: PromotionPolicy | None,
        evaluated_at: datetime | None,
    ) -> PromotionEvaluation:
        backtest = _one(trace.get("backtest_runs", ()), "backtest_run_id", evidence.backtest_run_id)
        champion_shadow = _one(
            trace.get("shadow_results", ()),
            "shadow_result_id",
            evidence.champion_shadow_result_id,
        )
        challenger_shadow = _one(
            trace.get("shadow_results", ()),
            "shadow_result_id",
            evidence.challenger_shadow_result_id,
        )
        _one(trace.get("strategy_specs", ()), "spec_id", evidence.champion_spec_id)
        _one(trace.get("strategy_specs", ()), "spec_id", evidence.challenger_spec_id)
        return evaluate_promotion(
            champion_shadow=_shadow_evidence(champion_shadow),
            challenger_backtest=_backtest_evidence(backtest),
            challenger_shadow=_shadow_evidence(challenger_shadow),
            policy=policy,
            evaluated_at=evaluated_at,
        )

    def _registry_record_from_persisted(
        self,
        trace: Mapping[str, Any],
        *,
        promotion_record_id: str,
        expected_evidence: PersistedPromotionEvidence,
    ) -> PromotionRecord:
        persisted = _one(
            trace.get("promotion_records", ()),
            "promotion_record_id",
            promotion_record_id,
        )
        expected_ids = {
            "champion_spec_id": expected_evidence.champion_spec_id,
            "challenger_spec_id": expected_evidence.challenger_spec_id,
            "backtest_run_id": expected_evidence.backtest_run_id,
            "champion_shadow_result_id": expected_evidence.champion_shadow_result_id,
            "challenger_shadow_result_id": expected_evidence.challenger_shadow_result_id,
        }
        if any(persisted.get(name) != value for name, value in expected_ids.items()):
            raise ValueError("persisted promotion record does not match requested evidence IDs")

        evaluation = persisted.get("evaluation")
        if not isinstance(evaluation, Mapping) or not isinstance(evaluation.get("gate"), Mapping):
            raise ValueError("persisted promotion record has no deterministic gate")
        gate = evaluation["gate"]
        eligible = persisted.get("eligible") is True
        if gate.get("eligible") is not eligible:
            raise ValueError("persisted promotion row and deterministic gate disagree")
        persisted_reasons = tuple(str(item) for item in persisted.get("reason_codes", ()))
        gate_reasons = tuple(str(item) for item in gate.get("reason_codes", ()))
        if persisted_reasons != gate_reasons:
            raise ValueError("persisted promotion reason codes and gate disagree")

        champion_spec = _one(
            trace.get("strategy_specs", ()),
            "spec_id",
            expected_evidence.champion_spec_id,
        )
        challenger_spec = _one(
            trace.get("strategy_specs", ()),
            "spec_id",
            expected_evidence.challenger_spec_id,
        )
        champion_version = str(champion_spec["strategy_version"])
        challenger_version = str(challenger_spec["strategy_version"])
        already_applied = next(
            (
                item
                for item in self.registry.promotion_records
                if item.record_id == promotion_record_id
            ),
            None,
        )
        if self.registry.champion.version != champion_version:
            if already_applied is None:
                raise ValueError("persisted champion version does not match registry champion")
        elif (
            already_applied is None
            and challenger_version not in {item.version for item in self.registry.challengers}
        ):
            raise ValueError("persisted challenger is not registered")

        observed = gate.get("observed_relative_improvement")
        improvement = float(observed) if observed is not None else 0.0
        status = PromotionStatus.PROMOTED if eligible else PromotionStatus.NOT_ELIGIBLE
        reasons = gate_reasons if gate_reasons else ("strict_persisted_gate_passed",)
        expected_record = PromotionRecord(
            record_id=promotion_record_id,
            challenger_version=challenger_version,
            previous_champion_version=champion_version,
            resulting_champion_version=(challenger_version if eligible else champion_version),
            status=status,
            reasons=reasons,
            net_return_improvement=improvement,
            evaluated_at=_as_datetime(persisted["created_at"]),
        )
        if already_applied is not None:
            if already_applied != expected_record:
                raise ValueError("registry promotion record disagrees with append-only audit")
            return already_applied
        return expected_record


def _one(rows: Sequence[Mapping[str, Any]], key: str, expected: str) -> Mapping[str, Any]:
    matches = [row for row in rows if row.get(key) == expected]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one persisted {key}={expected!r}")
    return matches[0]


def _metrics(row: Mapping[str, Any]) -> PerformanceMetrics:
    return PerformanceMetrics(
        net_profit=row.get("net_profit"),
        net_return=row.get("net_return"),
        max_drawdown=row.get("max_drawdown"),
        total_cost=row.get("total_cost"),
        stressed_net_return_2x=row.get("stressed_net_return_2x"),
        symbol_concentration=row.get("symbol_concentration"),
        month_concentration=row.get("month_concentration"),
        trade_count=row.get("trade_count") or row.get("closed_trades"),
        period_days=row.get("elapsed_days"),
    )


def _backtest_evidence(row: Mapping[str, Any]) -> BacktestEvidence:
    validation = row.get("validation")
    if not isinstance(validation, Mapping):
        validation = {}
    return BacktestEvidence(
        metrics=_metrics(row),
        completed=row.get("completed"),
        dsr_significance_probability=row.get("dsr_significance_probability"),
        pbo_probability=row.get("pbo_probability"),
        holdout_months=row.get("holdout_months"),
        walk_forward_passed=validation.get("walk_forward_passed"),
        holdout_passed=validation.get("holdout_passed"),
        parameter_perturbation_passed=validation.get("parameter_perturbation_passed"),
        latency_stress_passed=validation.get("latency_stress_passed"),
        social_placebo_passed=validation.get("social_placebo_passed"),
    )


def _shadow_evidence(row: Mapping[str, Any]) -> ShadowEvidence:
    return ShadowEvidence(
        metrics=_metrics(row),
        completed=row.get("completed"),
        elapsed_days=row.get("elapsed_days"),
        closed_trades=row.get("closed_trades"),
    )


def _as_datetime(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
