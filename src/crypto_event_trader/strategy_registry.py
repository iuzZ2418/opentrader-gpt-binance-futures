from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from pydantic import Field, model_validator

from .contracts import (
    PromotionRecord,
    PromotionStatus,
    StrategySpec,
    StrictContract,
    utc_now,
)
from .strategy import default_champion_spec


class ChallengerMetrics(StrictContract):
    champion_net_return: float
    challenger_net_return: float
    champion_max_drawdown: float = Field(ge=0, le=1)
    challenger_max_drawdown: float = Field(ge=0, le=1)
    double_cost_net_return: float
    dsr_probability: float = Field(ge=0, le=1)
    pbo_probability: float = Field(ge=0, le=1)
    max_symbol_contribution: float = Field(ge=0, le=1)
    max_month_contribution: float = Field(ge=0, le=1)
    shadow_days: int = Field(ge=0)
    shadow_closed_trades: int = Field(ge=0)
    walk_forward_passed: bool
    sealed_holdout_passed: bool
    parameter_perturbation_passed: bool
    latency_stress_passed: bool
    social_placebo_passed: bool

    @model_validator(mode="after")
    def returns_are_finite(self) -> ChallengerMetrics:
        values = (
            self.champion_net_return,
            self.challenger_net_return,
            self.double_cost_net_return,
        )
        if any(value != value or value in {float("inf"), float("-inf")} for value in values):
            raise ValueError("returns must be finite")
        return self

    @property
    def relative_net_return_improvement(self) -> float:
        denominator = max(abs(self.champion_net_return), 1e-12)
        return (self.challenger_net_return - self.champion_net_return) / denominator


class StrategyRegistry:
    """Small durable champion/challenger registry with bounded automatic promotion."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        initial_champion: StrategySpec | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._champion = initial_champion or default_champion_spec()
        self._challengers: dict[str, StrategySpec] = {}
        self._versions: dict[str, StrategySpec] = {self._champion.version: self._champion}
        self._records: list[PromotionRecord] = []
        if self.path is not None and self.path.exists():
            self._load()
        elif self.path is not None:
            self._persist()

    @property
    def champion(self) -> StrategySpec:
        return self._champion

    @property
    def challengers(self) -> tuple[StrategySpec, ...]:
        return tuple(self._challengers[key] for key in sorted(self._challengers))

    @property
    def promotion_records(self) -> tuple[PromotionRecord, ...]:
        return tuple(self._records)

    def register_challenger(self, spec: StrategySpec) -> None:
        if spec.version in self._versions:
            raise ValueError(f"strategy version already exists: {spec.version}")
        self._challengers[spec.version] = spec
        self._versions[spec.version] = spec
        self._persist()

    def evaluate_and_promote(
        self,
        challenger_version: str,
        metrics: ChallengerMetrics,
        *,
        evaluated_at: datetime | None = None,
    ) -> PromotionRecord:
        """Reject the legacy metrics-only promotion path.

        Promotion is a state-changing operation, so an in-memory metrics object is not
        acceptable authority.  Callers must use ``GovernedPromotionCoordinator``; it reads
        append-only research evidence, asks ``AuditRepository`` to recompute and persist the
        strict gate, and only then applies the persisted record here.
        """

        del challenger_version, metrics, evaluated_at
        raise PermissionError(
            "direct promotion is disabled; use GovernedPromotionCoordinator with persisted "
            "audit evidence"
        )

    def _apply_audited_record(self, record: PromotionRecord) -> PromotionRecord:
        """Apply an already persisted audit result.

        This package-private boundary deliberately accepts a ``PromotionRecord`` whose
        ``record_id`` is the append-only audit record ID.  The promotion coordinator is the
        sole production caller and verifies the database row before invoking this method.
        """

        existing = next(
            (item for item in self._records if item.record_id == record.record_id),
            None,
        )
        if existing is not None:
            if existing != record:
                raise ValueError("audit promotion record ID was already applied differently")
            return existing
        if record.status is PromotionStatus.ROLLED_BACK:
            raise ValueError("an audit promotion evaluation cannot directly apply a rollback")
        challenger = self._challengers.get(record.challenger_version)
        if challenger is None:
            raise KeyError(f"unknown challenger: {record.challenger_version}")
        if record.previous_champion_version != self._champion.version:
            raise ValueError("audit record previous champion does not match registry champion")

        if record.status is PromotionStatus.PROMOTED:
            if record.resulting_champion_version != challenger.version:
                raise ValueError("promoted audit record must result in the challenger")
            self._champion = challenger
            del self._challengers[challenger.version]
        elif record.resulting_champion_version != self._champion.version:
            raise ValueError("ineligible audit record cannot change the champion")

        self._records.append(record)
        self._persist()
        return record

    def rollback(
        self,
        *,
        reason: str,
        target_version: str | None = None,
        evaluated_at: datetime | None = None,
        record_id: str | None = None,
    ) -> PromotionRecord:
        if target_version is None:
            promoted = next(
                (
                    record
                    for record in reversed(self._records)
                    if record.status is PromotionStatus.PROMOTED
                    and record.resulting_champion_version == self._champion.version
                ),
                None,
            )
            if promoted is None:
                raise ValueError("no prior champion is available for rollback")
            target_version = promoted.previous_champion_version
        previous = self._champion
        record = PromotionRecord(
            record_id=record_id or uuid4().hex,
            challenger_version=previous.version,
            previous_champion_version=previous.version,
            resulting_champion_version=target_version,
            status=PromotionStatus.ROLLED_BACK,
            reasons=(reason,),
            net_return_improvement=0,
            evaluated_at=evaluated_at or utc_now(),
        )
        return self._apply_audited_rollback(record)

    def _apply_audited_rollback(self, record: PromotionRecord) -> PromotionRecord:
        """Apply an immutable rollback event, including crash-recovery replays."""

        existing = next(
            (item for item in self._records if item.record_id == record.record_id),
            None,
        )
        if existing is not None:
            if existing != record:
                raise ValueError("audit rollback record ID was already applied differently")
            return existing
        if record.status is not PromotionStatus.ROLLED_BACK:
            raise ValueError("registry rollback application requires a rollback record")
        if (
            record.previous_champion_version != self._champion.version
            or record.challenger_version != self._champion.version
        ):
            raise ValueError("audit rollback previous champion does not match registry")
        target = self._versions.get(record.resulting_champion_version)
        if target is None:
            raise KeyError(f"unknown rollback target: {record.resulting_champion_version}")
        if target.version == self._champion.version:
            raise ValueError("rollback target is already champion")
        previous = self._champion
        self._challengers[previous.version] = previous
        self._challengers.pop(target.version, None)
        self._champion = target
        self._records.append(record)
        self._persist()
        return record

    @staticmethod
    def _failed_checks(metrics: ChallengerMetrics) -> list[str]:
        checks = {
            "walk_forward_failed": metrics.walk_forward_passed,
            "sealed_holdout_failed": metrics.sealed_holdout_passed,
            "parameter_perturbation_failed": metrics.parameter_perturbation_passed,
            "latency_stress_failed": metrics.latency_stress_passed,
            "social_placebo_failed": metrics.social_placebo_passed,
            "maximum_drawdown_exceeds_20_percent": metrics.challenger_max_drawdown <= 0.20,
            "risk_worse_than_champion": (
                metrics.challenger_max_drawdown <= metrics.champion_max_drawdown
            ),
            "double_cost_return_not_positive": metrics.double_cost_net_return > 0,
            "dsr_probability_below_0.95": metrics.dsr_probability >= 0.95,
            "pbo_probability_above_0.10": metrics.pbo_probability <= 0.10,
            "symbol_concentration_above_0.35": metrics.max_symbol_contribution <= 0.35,
            "month_concentration_above_0.35": metrics.max_month_contribution <= 0.35,
            "shadow_period_below_90_days": metrics.shadow_days >= 90,
            "shadow_trades_below_30": metrics.shadow_closed_trades >= 30,
            "net_return_improvement_below_10_percent": (
                metrics.relative_net_return_improvement >= 0.10
            ),
        }
        return [reason for reason, passed in checks.items() if not passed]

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "champion": self._champion.model_dump(mode="json"),
            "challengers": [item.model_dump(mode="json") for item in self.challengers],
            "versions": [
                self._versions[key].model_dump(mode="json") for key in sorted(self._versions)
            ],
            "promotion_records": [item.model_dump(mode="json") for item in self._records],
        }
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            temporary = Path(handle.name)
        os.replace(temporary, self.path)

    def _load(self) -> None:
        if self.path is None:
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._champion = StrategySpec.model_validate(payload["champion"])
        self._challengers = {
            item.version: item
            for item in (
                StrategySpec.model_validate(value) for value in payload.get("challengers", [])
            )
        }
        versions = [
            StrategySpec.model_validate(value) for value in payload.get("versions", [])
        ]
        self._versions = {item.version: item for item in versions}
        self._versions[self._champion.version] = self._champion
        self._versions.update(self._challengers)
        self._records = [
            PromotionRecord.model_validate(value)
            for value in payload.get("promotion_records", [])
        ]
