from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from .audit import AuditRepository
from .contracts import (
    CandleInterval,
    MarketBar,
    PromotionRecord,
    PromotionStatus,
    StrategySpec,
    TradeDirection,
)
from .openai_research import (
    OpenAIStrategyResearcher,
    ResearchRecommendation,
    StrategyResearchResult,
)
from .promotion_governance import (
    GovernedPromotionCoordinator,
    GovernedPromotionResult,
    PersistedPromotionEvidence,
    RollbackSignal,
)
from .strategy import TrendBreakoutStrategy
from .strategy_registry import StrategyRegistry


class CounterfactualMarketData(Protocol):
    def closed_bars_between(
        self,
        symbol: str,
        interval: CandleInterval,
        *,
        start: datetime,
        end: datetime,
        limit: int = 100,
    ) -> Sequence[MarketBar]: ...


class StrategyResearchProvider(Protocol):
    def research(
        self,
        champion: StrategySpec,
        research_context: Mapping[str, Any],
        *,
        available_evidence_ids: Sequence[str] = (),
        now: datetime | None = None,
    ) -> StrategyResearchResult: ...


@dataclass(frozen=True, slots=True)
class CounterfactualSettlement:
    appended: int
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LearningTickResult:
    counterfactuals_appended: int = 0
    counterfactual_skips: tuple[str, ...] = ()
    research_status: str = "NOT_DUE"
    promotion_statuses: tuple[str, ...] = ()
    performance_status: str = "NOT_CONFIGURED"
    performance_reason_codes: tuple[str, ...] = ()
    authoritative_trade_count: int = 0
    champion_version: str = ""


@dataclass(frozen=True, slots=True)
class AuthoritativePerformanceResult:
    status: str
    reason_codes: tuple[str, ...]
    strategy_version: str
    trade_count: int
    excluded_trade_count: int
    net_profit: float | None = None
    stressed_net_profit_2x: float | None = None


class AuthoritativePerformanceMonitor:
    """Evaluate only fully attributed venue episodes for the active strategy.

    This monitor deliberately does not estimate missing fees, funding, equity, or strategy
    ownership.  Mixed-version position episodes are excluded, and a small sample can never
    trigger either a favorable conclusion or an automatic rollback.
    """

    def __init__(self, audit: AuditRepository, *, venue: str, min_trades: int = 30) -> None:
        if not venue.strip():
            raise ValueError("performance venue must be non-empty")
        if min_trades < 1:
            raise ValueError("performance min_trades must be positive")
        self.audit = audit
        self.venue = venue.strip()
        self.min_trades = min_trades

    def evaluate(self, strategy_version: str) -> AuthoritativePerformanceResult:
        outcomes = self.audit.build_trade_outcomes(venue=self.venue)
        eligible = tuple(
            item
            for item in outcomes
            if item.episode_id
            and item.trace_ids
            and item.source_record_ids
            and item.strategy_versions == (strategy_version,)
        )
        excluded = len(outcomes) - len(eligible)
        if len(eligible) < self.min_trades:
            return AuthoritativePerformanceResult(
                status="NOT_READY",
                reason_codes=("INSUFFICIENT_AUTHORITATIVE_STRATEGY_OUTCOMES",),
                strategy_version=strategy_version,
                trade_count=len(eligible),
                excluded_trade_count=excluded,
            )
        net_profit = sum(item.net_pnl for item in eligible)
        stressed = sum(item.gross_pnl - item.stressed_total_cost_2x for item in eligible)
        if not math.isfinite(net_profit) or not math.isfinite(stressed):
            raise ValueError("authoritative performance totals must be finite")
        reasons: list[str] = []
        # This is intentionally a conservative, scale-free drift condition: both actual and
        # doubled-cost results must be below zero over a mature, exactly attributed sample.
        if net_profit < 0 and stressed < 0:
            reasons.append("MATURE_AUTHORITATIVE_SAMPLE_NET_LOSS")
        return AuthoritativePerformanceResult(
            status="DRIFT" if reasons else "HEALTHY",
            reason_codes=tuple(reasons),
            strategy_version=strategy_version,
            trade_count=len(eligible),
            excluded_trade_count=excluded,
            net_profit=net_profit,
            stressed_net_profit_2x=stressed,
        )


def build_champion_strategy(registry: StrategyRegistry) -> TrendBreakoutStrategy:
    """Construct the executable strategy only from the durable bounded champion spec."""

    return TrendBreakoutStrategy(spec=registry.champion)


class CounterfactualOutcomeScheduler:
    """Settle matured candidates from historical candles, never from a current quote."""

    def __init__(
        self,
        audit: AuditRepository,
        market_data: CounterfactualMarketData,
    ) -> None:
        self.audit = audit
        self.market_data = market_data

    def settle_due(self, *, as_of: datetime, limit: int = 1_000) -> CounterfactualSettlement:
        reference = _aware_utc(as_of)
        appended = 0
        skipped: list[str] = []
        for work in self.audit.pending_counterfactual_work(as_of=reference, limit=limit):
            candidate_id = str(work["candidate_id"])
            entry_price = _entry_price(work.get("feature_snapshot"))
            if entry_price is None:
                skipped.append(f"{candidate_id}:ENTRY_PRICE_MISSING")
                continue
            created_at = _parse_datetime(work["created_at"])
            direction = TradeDirection(str(work["direction"]))
            for horizon in work["due_horizons"]:
                target = created_at + timedelta(hours=int(horizon))
                # A one-hour candle can close up to one hour after an arbitrary 15-minute
                # candidate timestamp.  Bound the request to two hours around the target so
                # a later current price can never leak into an older outcome.
                window_end = min(reference, target + timedelta(hours=2))
                try:
                    bars = self.market_data.closed_bars_between(
                        str(work["symbol"]),
                        CandleInterval.ONE_HOUR,
                        start=target - timedelta(hours=1),
                        end=window_end,
                        limit=4,
                    )
                except Exception as error:
                    skipped.append(
                        f"{candidate_id}:{horizon}H:MARKET_DATA:{type(error).__name__}"
                    )
                    continue
                eligible = sorted(
                    (
                        bar
                        for bar in bars
                        if bar.is_closed and target <= bar.close_time <= window_end
                    ),
                    key=lambda bar: bar.close_time,
                )
                if not eligible:
                    skipped.append(f"{candidate_id}:{horizon}H:CLOSED_BAR_MISSING")
                    continue
                outcome_bar = eligible[0]
                signed_return = direction.sign * (outcome_bar.close / entry_price - 1)
                if not math.isfinite(signed_return):
                    skipped.append(f"{candidate_id}:{horizon}H:NON_FINITE_RETURN")
                    continue
                action = str(work.get("action") or "REJECT").upper()
                confidence = work.get("confidence")
                favorable = 1.0 if signed_return > 0 else 0.0
                calibration_error = (
                    abs(float(confidence) - favorable) if confidence is not None else None
                )
                regret = (
                    max(0.0, signed_return)
                    if action in {"REJECT", "HOLD", "REDUCE", "CLOSE"}
                    else max(0.0, -signed_return)
                )
                try:
                    self.audit.append_counterfactual_outcome(
                        trace_id=str(work["trace_id"]),
                        candidate_id=candidate_id,
                        decision_id=(
                            str(work["decision_id"]) if work.get("decision_id") else None
                        ),
                        horizon_hours=int(horizon),
                        realized_return=signed_return,
                        decision_regret=regret,
                        confidence_calibration_error=calibration_error,
                        observed_at=outcome_bar.close_time,
                        created_at=reference,
                    )
                except Exception as error:
                    # A concurrent idempotent settlement may win the unique
                    # (candidate_id, horizon_hours) key.  Leave any other failure visible in
                    # the bounded status result and retry it on the next daily run.
                    skipped.append(
                        f"{candidate_id}:{horizon}H:AUDIT:{type(error).__name__}"
                    )
                    continue
                appended += 1
        return CounterfactualSettlement(appended=appended, skipped=tuple(skipped))


class LearningPromotionScheduler:
    """Daily outcome settlement plus weekly bounded research and governed promotion.

    The scheduler has no reference to trading stage, unlock controls, or capital allocation.
    A model proposal is only registered as a challenger.  Promotion is attempted solely when
    complete, persisted backtest and paired shadow evidence can be discovered in the audit
    trace; missing evidence leaves the champion unchanged.
    """

    def __init__(
        self,
        *,
        audit: AuditRepository,
        registry: StrategyRegistry,
        market_data: CounterfactualMarketData,
        researcher: StrategyResearchProvider | None = None,
        state_path: str | Path | None = None,
        learning_trace_id: str = "strategy_learning",
        performance_venue: str | None = None,
        performance_min_trades: int = 30,
    ) -> None:
        self.audit = audit
        self.registry = registry
        self.researcher = researcher
        self.counterfactuals = CounterfactualOutcomeScheduler(audit, market_data)
        self.coordinator = GovernedPromotionCoordinator(audit, registry)
        self.performance = (
            AuthoritativePerformanceMonitor(
                audit,
                venue=performance_venue,
                min_trades=performance_min_trades,
            )
            if performance_venue is not None
            else None
        )
        self.state_path = Path(state_path) if state_path is not None else None
        self.learning_trace_id = learning_trace_id
        self._state = self._load_state()
        self._ensure_champion_audited()
        self._recover_registered_challengers()
        self._replay_audited_registry_state()

    def tick(
        self,
        *,
        now: datetime,
        force_daily: bool = False,
        force_weekly: bool = False,
    ) -> LearningTickResult:
        reference = _aware_utc(now)
        daily_key = reference.date().isoformat()
        iso = reference.isocalendar()
        weekly_key = f"{iso.year}-W{iso.week:02d}"
        settlement = CounterfactualSettlement(0)
        promotion_statuses: tuple[str, ...] = ()
        research_status = "NOT_DUE"
        performance = AuthoritativePerformanceResult(
            status="NOT_CONFIGURED",
            reason_codes=(),
            strategy_version=self.registry.champion.version,
            trade_count=0,
            excluded_trade_count=0,
        )

        if force_daily or self._state.get("last_daily_settlement") != daily_key:
            settlement = self.counterfactuals.settle_due(as_of=reference)
            if self.performance is not None:
                try:
                    performance = self.performance.evaluate(self.registry.champion.version)
                except Exception as error:
                    performance = AuthoritativePerformanceResult(
                        status="FAIL_CLOSED",
                        reason_codes=(f"AUTHORITATIVE_ACCOUNTING:{type(error).__name__}",),
                        strategy_version=self.registry.champion.version,
                        trade_count=0,
                        excluded_trade_count=0,
                    )
                    self.handle_rollback_signal(
                        RollbackSignal.RECONCILIATION_ERROR,
                        observed_at=reference,
                        detail=performance.reason_codes[0],
                    )
                if performance.status == "DRIFT":
                    self.handle_rollback_signal(
                        RollbackSignal.PERFORMANCE_DRIFT,
                        observed_at=reference,
                        detail=",".join(performance.reason_codes),
                    )
            if performance.status not in {"DRIFT", "FAIL_CLOSED"}:
                promotion_statuses = self._attempt_promotions_fail_closed(reference)
            self._state["last_daily_settlement"] = daily_key
            self._persist_state()

        if force_weekly or self._state.get("last_weekly_research") != weekly_key:
            try:
                research_status = self._run_weekly_research(reference)
            except Exception as error:
                # Research is auxiliary.  Its failure must neither stop position management
                # nor manufacture a challenger/promotion.
                research_status = f"FAIL_CLOSED:{type(error).__name__}"
            self._state["last_weekly_research"] = weekly_key
            self._persist_state()
            promotion_statuses = (
                *promotion_statuses,
                *self._attempt_promotions_fail_closed(reference),
            )

        return LearningTickResult(
            counterfactuals_appended=settlement.appended,
            counterfactual_skips=settlement.skipped,
            research_status=research_status,
            promotion_statuses=promotion_statuses,
            performance_status=performance.status,
            performance_reason_codes=performance.reason_codes,
            authoritative_trade_count=performance.trade_count,
            champion_version=self.registry.champion.version,
        )

    def handle_rollback_signal(
        self,
        signal: RollbackSignal,
        *,
        observed_at: datetime | None = None,
        detail: str | None = None,
    ) -> bool:
        return (
            self.coordinator.handle_rollback_signal(
                signal,
                observed_at=observed_at,
                detail=detail,
            )
            is not None
        )

    def _run_weekly_research(self, now: datetime) -> str:
        if self.researcher is None:
            return "DISABLED"
        outcomes = self.audit.recent_counterfactual_outcomes(limit=500)
        horizons_by_candidate: dict[str, set[int]] = {}
        for item in outcomes:
            horizons_by_candidate.setdefault(str(item["candidate_id"]), set()).add(
                int(item["horizon_hours"])
            )
        complete_candidates = {
            candidate_id
            for candidate_id, horizons in horizons_by_candidate.items()
            if horizons == {1, 4, 24}
        }
        if not complete_candidates:
            return "INSUFFICIENT_COUNTERFACTUAL_EVIDENCE"
        evidence = [
            {
                "evidence_id": str(item["outcome_id"]),
                "candidate_id": str(item["candidate_id"]),
                "strategy_version": str(item["strategy_version"]),
                "symbol": str(item["symbol"]),
                "direction": str(item["direction"]),
                "action": item.get("action"),
                "confidence": item.get("confidence"),
                "horizon_hours": int(item["horizon_hours"]),
                "realized_return": float(item["realized_return"]),
                "decision_regret": item.get("decision_regret"),
                "confidence_calibration_error": item.get(
                    "confidence_calibration_error"
                ),
                "observed_at": str(item["observed_at"]),
            }
            for item in outcomes
            if str(item["candidate_id"]) in complete_candidates
        ]
        evidence_ids = tuple(str(item["evidence_id"]) for item in evidence)
        result = self.researcher.research(
            self.registry.champion,
            {
                "evaluation_type": "point_in_time_counterfactuals",
                "cost_and_shadow_gates_required": True,
                "outcomes": evidence,
            },
            available_evidence_ids=evidence_ids,
            now=now,
        )
        if result.parent_version != self.registry.champion.version:
            raise ValueError("research result parent does not match the current champion")
        self.audit.append_strategy_research_run(
            **result.audit_run_kwargs(trace_id=self.learning_trace_id)
        )
        if self.audit.strategy_spec_by_version(result.spec.version) is not None:
            raise ValueError("research strategy version already exists in the audit ledger")
        if result.recommendation is ResearchRecommendation.PROPOSE:
            if result.spec.version in {
                self.registry.champion.version,
                *(item.version for item in self.registry.challengers),
            }:
                raise ValueError("research strategy version already exists in the registry")
            kwargs = result.audit_repository_kwargs()
            kwargs["trace_id"] = self.learning_trace_id
            self.audit.append_strategy_spec(**kwargs)
            self.registry.register_challenger(result.spec)
            return f"CHALLENGER_REGISTERED:{result.spec.version}"
        kwargs = result.audit_repository_kwargs()
        kwargs["trace_id"] = self.learning_trace_id
        self.audit.append_strategy_spec(**kwargs)
        return "NO_CHANGE_RECORDED"

    def _attempt_promotions(self, now: datetime) -> tuple[str, ...]:
        statuses: list[str] = []
        champion_row = self.audit.strategy_spec_by_version(self.registry.champion.version)
        if champion_row is None or not champion_row.get("trace_id"):
            return ("FAIL_CLOSED:CHAMPION_AUDIT_MISSING",)
        trace_id = str(champion_row["trace_id"])
        trace = self.audit.get_trace(trace_id)
        for challenger in self.registry.challengers:
            challenger_row = next(
                (
                    item
                    for item in trace["strategy_specs"]
                    if item["strategy_version"] == challenger.version
                    and item["parent_version"] == self.registry.champion.version
                    and item["status"] == "CHALLENGER"
                ),
                None,
            )
            if challenger_row is None:
                statuses.append(f"{challenger.version}:FAIL_CLOSED:AUDIT_SPEC_MISSING")
                continue
            evidence = _latest_promotion_evidence(
                trace,
                champion_spec_id=str(champion_row["spec_id"]),
                challenger_spec_id=str(challenger_row["spec_id"]),
            )
            if evidence is None:
                statuses.append(f"{challenger.version}:WAITING_FOR_COMPLETE_EVIDENCE")
                continue
            existing = next(
                (
                    row
                    for row in trace["promotion_records"]
                    if row["champion_spec_id"] == evidence.champion_spec_id
                    and row["challenger_spec_id"] == evidence.challenger_spec_id
                    and row["backtest_run_id"] == evidence.backtest_run_id
                    and row["champion_shadow_result_id"]
                    == evidence.champion_shadow_result_id
                    and row["challenger_shadow_result_id"]
                    == evidence.challenger_shadow_result_id
                ),
                None,
            )
            if existing is not None:
                recovered = self.coordinator.apply_persisted_record(
                    evidence,
                    str(existing["promotion_record_id"]),
                )
                recovery_status = (
                    "PROMOTED_RECOVERED"
                    if recovered.status is PromotionStatus.PROMOTED
                    else "ALREADY_EVALUATED"
                )
                statuses.append(
                    f"{challenger.version}:{recovery_status}"
                )
                if recovered.status is PromotionStatus.PROMOTED:
                    break
                continue
            result: GovernedPromotionResult = self.coordinator.evaluate_and_apply(
                evidence,
                caller_context={
                    "scheduler": "learning-runtime-v1",
                    "capital_stage_changed": False,
                },
                evaluated_at=now,
            )
            statuses.append(
                f"{challenger.version}:{'PROMOTED' if result.promoted else 'NOT_ELIGIBLE'}"
            )
            if result.promoted:
                break
        return tuple(statuses)

    def _attempt_promotions_fail_closed(self, now: datetime) -> tuple[str, ...]:
        try:
            return self._attempt_promotions(now)
        except Exception as error:
            return (f"FAIL_CLOSED:PROMOTION:{type(error).__name__}",)

    def _ensure_champion_audited(self) -> None:
        row = self.audit.strategy_spec_by_version(self.registry.champion.version)
        if row is not None:
            if row.get("trace_id"):
                self.learning_trace_id = str(row["trace_id"])
            return
        spec = self.registry.champion
        self.audit.append_strategy_spec(
            trace_id=self.learning_trace_id,
            strategy_version=spec.version,
            status="CHAMPION",
            parameters=_spec_parameters(spec),
            prompt_version=spec.prompt_version,
        )

    def _recover_registered_challengers(self) -> None:
        trace = self.audit.get_trace(self.learning_trace_id)
        known = {
            self.registry.champion.version,
            *(item.version for item in self.registry.challengers),
        }
        for row in trace["strategy_specs"]:
            if (
                row["status"] != "CHALLENGER"
                or row["strategy_version"] in known
                or row["parent_version"] != self.registry.champion.version
            ):
                continue
            spec = _spec_from_audit(row)
            if spec is None:
                continue
            self.registry.register_challenger(spec)
            known.add(spec.version)

    def _replay_audited_registry_state(self) -> None:
        """Reapply committed promotions and rollbacks after file loss or a crash window."""

        trace = self.audit.get_trace(self.learning_trace_id)
        specs = {
            str(row["spec_id"]): row
            for row in trace["strategy_specs"]
        }
        actions = [
            (str(row["created_at"]), 0, "PROMOTION", row)
            for row in trace["promotion_records"]
        ]
        actions.extend(
            (str(row["created_at"]), 1, "ROLLBACK", row)
            for row in trace.get("strategy_registry_events", ())
        )
        for _created_at, _order, kind, row in sorted(
            actions, key=lambda item: (item[0], item[1])
        ):
            if kind == "PROMOTION":
                champion_spec_id = str(row["champion_spec_id"])
                challenger_spec_id = str(row["challenger_spec_id"])
                challenger_row = specs.get(challenger_spec_id)
                if challenger_row is None:
                    raise ValueError("audited promotion challenger spec is missing")
                challenger_version = str(challenger_row["strategy_version"])
                known_versions = {
                    self.registry.champion.version,
                    *(item.version for item in self.registry.challengers),
                    *(item.challenger_version for item in self.registry.promotion_records),
                    *(item.resulting_champion_version for item in self.registry.promotion_records),
                }
                if challenger_version not in known_versions:
                    spec = _spec_from_audit(challenger_row)
                    if spec is None:
                        raise ValueError("audited promotion challenger spec is incomplete")
                    self.registry.register_challenger(spec)
                evidence = PersistedPromotionEvidence(
                    trace_id=self.learning_trace_id,
                    champion_spec_id=champion_spec_id,
                    challenger_spec_id=challenger_spec_id,
                    backtest_run_id=str(row["backtest_run_id"]),
                    champion_shadow_result_id=str(row["champion_shadow_result_id"]),
                    challenger_shadow_result_id=str(row["challenger_shadow_result_id"]),
                )
                self.coordinator.apply_persisted_record(
                    evidence,
                    str(row["promotion_record_id"]),
                )
                continue
            record = PromotionRecord(
                record_id=str(row["registry_event_id"]),
                challenger_version=str(row["previous_champion_version"]),
                previous_champion_version=str(row["previous_champion_version"]),
                resulting_champion_version=str(row["resulting_champion_version"]),
                status=PromotionStatus.ROLLED_BACK,
                reasons=(str(row["reason"]),),
                net_return_improvement=0,
                evaluated_at=_parse_datetime(row["created_at"]),
            )
            self.registry._apply_audited_rollback(record)

    def _load_state(self) -> dict[str, str]:
        if self.state_path is None or not self.state_path.exists():
            return {}
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in payload.items()
        ):
            raise ValueError("invalid learning scheduler state")
        return payload

    def _persist_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.state_path.parent, delete=False
        ) as handle:
            json.dump(self._state, handle, sort_keys=True, separators=(",", ":"))
            temporary = Path(handle.name)
        os.replace(temporary, self.state_path)


def _latest_promotion_evidence(
    trace: Mapping[str, Any],
    *,
    champion_spec_id: str,
    challenger_spec_id: str,
) -> PersistedPromotionEvidence | None:
    backtests = sorted(
        (
            row
            for row in trace["backtest_runs"]
            if row["spec_id"] == challenger_spec_id
        ),
        key=lambda row: str(row["created_at"]),
        reverse=True,
    )
    champion_shadows = [
        row for row in trace["shadow_results"] if row["spec_id"] == champion_spec_id
    ]
    challenger_shadows = sorted(
        (
            row
            for row in trace["shadow_results"]
            if row["spec_id"] == challenger_spec_id
        ),
        key=lambda row: str(row["created_at"]),
        reverse=True,
    )
    if not backtests or not champion_shadows or not challenger_shadows:
        return None
    for challenger_shadow in challenger_shadows:
        champion_shadow = next(
            (
                row
                for row in champion_shadows
                if row["started_at"] == challenger_shadow["started_at"]
                and row["ended_at"] == challenger_shadow["ended_at"]
            ),
            None,
        )
        if champion_shadow is not None:
            return PersistedPromotionEvidence(
                trace_id=str(trace["trace_id"]),
                champion_spec_id=champion_spec_id,
                challenger_spec_id=challenger_spec_id,
                backtest_run_id=str(backtests[0]["backtest_run_id"]),
                champion_shadow_result_id=str(champion_shadow["shadow_result_id"]),
                challenger_shadow_result_id=str(challenger_shadow["shadow_result_id"]),
            )
    return None


def _spec_parameters(spec: StrategySpec) -> dict[str, Any]:
    return {
        "momentum_windows_1h": list(spec.momentum_windows_1h),
        "donchian_windows_4h": list(spec.donchian_windows_4h),
        "minimum_directional_votes": spec.minimum_directional_votes,
        "ewma_span_hours": spec.ewma_span_hours,
        "target_annualized_volatility": spec.target_annualized_volatility,
        "normal_risk_scale": spec.normal_risk_scale,
        "caution_risk_scale": spec.caution_risk_scale,
        "blocked_risk_scale": spec.blocked_risk_scale,
    }


def _spec_from_audit(row: Mapping[str, Any]) -> StrategySpec | None:
    parameters = row.get("parameters")
    required = set(_spec_parameters(StrategySpec(version="schema-probe")))
    if not isinstance(parameters, Mapping) or not required.issubset(parameters):
        return None
    return StrategySpec(
        version=str(row["strategy_version"]),
        prompt_version=str(row["prompt_version"]),
        **{key: parameters[key] for key in required},
    )


def _entry_price(snapshot: Any) -> float | None:
    if not isinstance(snapshot, Mapping):
        return None
    for key in ("last_price", "mark_price"):
        try:
            value = float(snapshot[key])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0 and math.isfinite(value):
            return value
    return None


def _parse_datetime(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _aware_utc(value)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("learning scheduler timestamps must be timezone-aware")
    return value.astimezone(UTC)


__all__ = [
    "AuthoritativePerformanceMonitor",
    "AuthoritativePerformanceResult",
    "CounterfactualOutcomeScheduler",
    "CounterfactualSettlement",
    "LearningPromotionScheduler",
    "LearningTickResult",
    "OpenAIStrategyResearcher",
    "RollbackSignal",
    "build_champion_strategy",
]
