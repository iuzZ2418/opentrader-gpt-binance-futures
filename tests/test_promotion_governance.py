from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.contracts import PromotionStatus, StrategySpec
from crypto_event_trader.promotion_governance import (
    GovernedPromotionCoordinator,
    PersistedPromotionEvidence,
    RollbackSignal,
)
from crypto_event_trader.strategy_registry import StrategyRegistry


def _repository(tmp_path: Path) -> AuditRepository:
    repository = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    repository.initialize()
    return repository


def _persist_evidence(
    repository: AuditRepository,
    *,
    trace_id: str = "promotion-trace",
    challenger_return: float = 0.12,
) -> PersistedPromotionEvidence:
    champion_id = repository.append_strategy_spec(
        trace_id=trace_id,
        strategy_version="trend-breakout-v1",
        status="CHAMPION",
        parameters={"vote_threshold": 3},
        prompt_version="research-v1",
    )
    challenger_id = repository.append_strategy_spec(
        trace_id=trace_id,
        strategy_version="trend-breakout-v2",
        status="CHALLENGER",
        parent_version="trend-breakout-v1",
        parameters={"vote_threshold": 4},
        prompt_version="research-v2",
        source_response_id="gpt-research-proposal-only",
    )
    backtest_id = repository.append_backtest_run(
        trace_id=trace_id,
        spec_id=challenger_id,
        started_at="2025-01-01T00:00:00Z",
        ended_at="2026-06-30T00:00:00Z",
        completed=True,
        net_profit=2500,
        net_return=0.25,
        max_drawdown=0.10,
        total_cost=200,
        stressed_net_return_2x=0.20,
        dsr_significance_probability=0.96,
        pbo_probability=0.08,
        symbol_concentration=0.30,
        month_concentration=0.30,
        trade_count=100,
        holdout_months=12,
        validation={
            "walk_forward_passed": True,
            "holdout_passed": True,
            "parameter_perturbation_passed": True,
            "latency_stress_passed": True,
            "social_placebo_passed": True,
        },
    )
    champion_shadow_id = repository.append_shadow_result(
        trace_id=trace_id,
        spec_id=champion_id,
        started_at="2026-04-01T00:00:00Z",
        ended_at="2026-07-01T00:00:00Z",
        completed=True,
        elapsed_days=91,
        closed_trades=60,
        net_return=0.10,
        max_drawdown=0.12,
        stressed_net_return_2x=0.06,
        symbol_concentration=0.30,
        month_concentration=0.30,
    )
    challenger_shadow_id = repository.append_shadow_result(
        trace_id=trace_id,
        spec_id=challenger_id,
        started_at="2026-04-01T00:00:00Z",
        ended_at="2026-07-01T00:00:00Z",
        completed=True,
        elapsed_days=91,
        closed_trades=30,
        net_return=challenger_return,
        max_drawdown=0.10,
        stressed_net_return_2x=0.08,
        symbol_concentration=0.30,
        month_concentration=0.30,
    )
    return PersistedPromotionEvidence(
        trace_id=trace_id,
        champion_spec_id=champion_id,
        challenger_spec_id=challenger_id,
        backtest_run_id=backtest_id,
        champion_shadow_result_id=champion_shadow_id,
        challenger_shadow_result_id=challenger_shadow_id,
    )


def _registry(tmp_path: Path) -> StrategyRegistry:
    registry = StrategyRegistry(tmp_path / "registry.json")
    registry.register_challenger(
        StrategySpec(version="trend-breakout-v2", minimum_directional_votes=4)
    )
    return registry


def test_persists_strict_gate_before_promoting_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    evidence = _persist_evidence(repository)
    registry = _registry(tmp_path)
    coordinator = GovernedPromotionCoordinator(repository, registry)
    champion_seen_during_append: list[str] = []
    original_append = repository.append_promotion_record

    def recording_append(**kwargs: Any) -> str:
        champion_seen_during_append.append(registry.champion.version)
        return original_append(**kwargs)

    monkeypatch.setattr(repository, "append_promotion_record", recording_append)
    result = coordinator.evaluate_and_apply(
        evidence,
        caller_context={"research_response_id": "gpt-proposal-not-authority"},
        evaluated_at=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert champion_seen_during_append == ["trend-breakout-v1"]
    assert result.promoted is True
    assert registry.champion.version == "trend-breakout-v2"
    persisted = repository.get_trace(evidence.trace_id)["promotion_records"][0]
    assert persisted["promotion_record_id"] == result.audit_record_id
    assert persisted["eligible"] is True
    assert persisted["evaluation"]["gate"]["eligible"] is True
    assert result.registry_record.record_id == result.audit_record_id


def test_ineligible_persisted_gate_never_changes_champion(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    evidence = _persist_evidence(repository, challenger_return=0.105)
    registry = _registry(tmp_path)

    result = GovernedPromotionCoordinator(repository, registry).evaluate_and_apply(evidence)

    assert result.promoted is False
    assert result.registry_record.status is PromotionStatus.NOT_ELIGIBLE
    assert "RELATIVE_RETURN_IMPROVEMENT_TOO_LOW" in result.registry_record.reasons
    assert registry.champion.version == "trend-breakout-v1"
    persisted = repository.get_trace(evidence.trace_id)["promotion_records"][0]
    assert persisted["eligible"] is False
    assert persisted["evaluation"]["gate"]["eligible"] is False


def test_can_recover_after_audit_commit_without_duplicating_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    evidence = _persist_evidence(repository)
    registry = _registry(tmp_path)
    coordinator = GovernedPromotionCoordinator(repository, registry)
    original_apply = registry._apply_audited_record

    def fail_after_audit(_record: object) -> object:
        raise RuntimeError("simulated registry persistence failure")

    monkeypatch.setattr(registry, "_apply_audited_record", fail_after_audit)
    with pytest.raises(RuntimeError, match="simulated registry"):
        coordinator.evaluate_and_apply(evidence)
    assert registry.champion.version == "trend-breakout-v1"
    promotion = repository.get_trace(evidence.trace_id)["promotion_records"][0]

    monkeypatch.setattr(registry, "_apply_audited_record", original_apply)
    applied = coordinator.apply_persisted_record(
        evidence, promotion_record_id=promotion["promotion_record_id"]
    )
    repeated = coordinator.apply_persisted_record(
        evidence, promotion_record_id=promotion["promotion_record_id"]
    )

    assert applied.status is PromotionStatus.PROMOTED
    assert repeated == applied
    assert registry.champion.version == "trend-breakout-v2"
    assert len(registry.promotion_records) == 1


@pytest.mark.parametrize("signal", list(RollbackSignal))
def test_explicit_safety_signals_auto_rollback(
    tmp_path: Path, signal: RollbackSignal
) -> None:
    repository = _repository(tmp_path)
    evidence = _persist_evidence(repository)
    registry = _registry(tmp_path)
    coordinator = GovernedPromotionCoordinator(repository, registry)
    coordinator.evaluate_and_apply(evidence)

    rollback = coordinator.handle_rollback_signal(signal, detail="detector-42")

    assert rollback is not None
    assert rollback.status is PromotionStatus.ROLLED_BACK
    assert rollback.reasons == (f"{signal.value}:detector-42",)
    assert registry.champion.version == "trend-breakout-v1"
    assert coordinator.handle_rollback_signal(signal) is None


def test_rollback_requires_typed_signal_and_promotion_has_no_capital_controls(
    tmp_path: Path,
) -> None:
    coordinator = GovernedPromotionCoordinator(_repository(tmp_path), _registry(tmp_path))

    with pytest.raises(TypeError, match="explicit RollbackSignal"):
        coordinator.handle_rollback_signal("risk_boundary_violation")  # type: ignore[arg-type]
    parameters = inspect.signature(coordinator.evaluate_and_apply).parameters
    assert "capital_stage" not in parameters
    assert "capital_allocation" not in parameters
    assert "trading_stage" not in parameters
