from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class StrictContract(BaseModel):
    """Base class for immutable, forward-compatible boundary contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class TradeDirection(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is TradeDirection.LONG else -1


class TradeAction(StrEnum):
    OPEN = "OPEN"
    ADD = "ADD"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    REJECT = "REJECT"


class RiskRegime(StrEnum):
    NORMAL = "normal"
    CAUTION = "caution"
    BLOCKED = "blocked"


class CandleInterval(StrEnum):
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"


class MarketBar(StrictContract):
    symbol: str = Field(min_length=1, max_length=30)
    interval: CandleInterval
    open_time: datetime
    close_time: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(default=0, ge=0)
    is_closed: bool = True

    @model_validator(mode="after")
    def validate_bar(self) -> MarketBar:
        open_time = _require_aware(self.open_time, "open_time")
        close_time = _require_aware(self.close_time, "close_time")
        if close_time <= open_time:
            raise ValueError("close_time must be after open_time")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC values are inconsistent")
        if self.low > self.high:
            raise ValueError("low cannot exceed high")
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "open_time", open_time)
        object.__setattr__(self, "close_time", close_time)
        return self


class TradeCandidate(StrictContract):
    candidate_id: str = Field(default_factory=lambda: uuid4().hex, min_length=8, max_length=64)
    strategy_version: str = Field(min_length=1, max_length=80)
    symbol: str = Field(min_length=1, max_length=30)
    direction: TradeDirection
    max_quantity: float = Field(gt=0)
    max_risk_fraction: float = Field(gt=0, le=0.01)
    feature_snapshot: dict[str, JsonValue]
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(
        default_factory=lambda data: data["created_at"] + timedelta(seconds=120)
    )

    @model_validator(mode="after")
    def validate_candidate(self) -> TradeCandidate:
        created_at = _require_aware(self.created_at, "created_at")
        expires_at = _require_aware(self.expires_at, "expires_at")
        lifetime = (expires_at - created_at).total_seconds()
        if lifetime <= 0 or lifetime > 120:
            raise ValueError("candidate lifetime must be within 120 seconds")
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "expires_at", expires_at)
        return self

    def is_valid(self, at: datetime | None = None) -> bool:
        reference = _require_aware(at or utc_now(), "at")
        return self.created_at <= reference < self.expires_at


class TradeDecision(StrictContract):
    decision_id: str = Field(default_factory=lambda: uuid4().hex, min_length=8, max_length=64)
    candidate_id: str | None = None
    symbol: str = Field(min_length=1, max_length=30)
    action: TradeAction
    direction: TradeDirection | None
    position_multiplier: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: tuple[str, ...] = ()
    position_thesis: str = Field(default="", max_length=4_000)
    invalidation_conditions: tuple[str, ...] = ()
    next_review_at: datetime
    reason: str = Field(min_length=1, max_length=2_000)
    provider_model: str | None = None
    response_id: str | None = None
    prompt_version: str = "trade-approval-v1"
    latency_ms: int | None = Field(default=None, ge=0)
    source_urls: tuple[str, ...] = ()
    decided_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_shape(self) -> TradeDecision:
        review_at = _require_aware(self.next_review_at, "next_review_at")
        decided_at = _require_aware(self.decided_at, "decided_at")
        if self.action in {TradeAction.OPEN, TradeAction.ADD}:
            if self.direction is None:
                raise ValueError("OPEN and ADD require a direction")
            if self.position_multiplier <= 0:
                raise ValueError("OPEN and ADD require a positive position_multiplier")
        elif self.action is TradeAction.HOLD and self.position_multiplier != 1:
            raise ValueError("HOLD requires position_multiplier=1")
        elif self.action is TradeAction.REDUCE and self.position_multiplier <= 0:
            raise ValueError("REDUCE requires a positive reduction fraction")
        elif self.action in {TradeAction.CLOSE, TradeAction.REJECT}:
            if self.position_multiplier != 0:
                raise ValueError("CLOSE and REJECT require position_multiplier=0")
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "next_review_at", review_at)
        object.__setattr__(self, "decided_at", decided_at)
        return self


class PositionThesis(StrictContract):
    thesis_id: str = Field(default_factory=lambda: uuid4().hex, min_length=8, max_length=64)
    previous_version_id: str | None = None
    version: int = Field(default=1, ge=1)
    symbol: str = Field(min_length=1, max_length=30)
    direction: TradeDirection
    entry_reason: str = Field(min_length=1, max_length=4_000)
    expected_horizon_minutes: int = Field(gt=0, le=43_200)
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    invalidation_conditions: tuple[str, ...] = ()
    add_count: int = Field(default=0, ge=0, le=1)
    pnl_r: float = 0
    decision_history: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_thesis(self) -> PositionThesis:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "created_at", _require_aware(self.created_at, "created_at"))
        return self

    def append_version(
        self,
        *,
        entry_reason: str,
        supporting_evidence_ids: Sequence[str],
        contradicting_evidence_ids: Sequence[str],
        invalidation_conditions: Sequence[str] | None = None,
        pnl_r: float,
        decision_id: str,
        add_count: int | None = None,
        created_at: datetime | None = None,
    ) -> PositionThesis:
        """Return an append-only successor; the prior thesis is never mutated."""

        return PositionThesis(
            previous_version_id=self.thesis_id,
            version=self.version + 1,
            symbol=self.symbol,
            direction=self.direction,
            entry_reason=entry_reason,
            expected_horizon_minutes=self.expected_horizon_minutes,
            supporting_evidence_ids=tuple(supporting_evidence_ids),
            contradicting_evidence_ids=tuple(contradicting_evidence_ids),
            invalidation_conditions=(
                self.invalidation_conditions
                if invalidation_conditions is None
                else tuple(invalidation_conditions)
            ),
            add_count=self.add_count if add_count is None else add_count,
            pnl_r=pnl_r,
            decision_history=(*self.decision_history, decision_id),
            created_at=created_at or utc_now(),
        )


class StrategySpec(StrictContract):
    version: str = Field(min_length=1, max_length=80)
    momentum_windows_1h: tuple[int, int, int] = (24, 72, 168)
    donchian_windows_4h: tuple[int, int] = (42, 126)
    minimum_directional_votes: int = Field(default=3, ge=3, le=5)
    ewma_span_hours: int = Field(default=720, ge=168, le=1_440)
    target_annualized_volatility: float = Field(default=0.40, gt=0, le=2)
    normal_risk_scale: float = Field(default=1.0, ge=0, le=1)
    caution_risk_scale: float = Field(default=0.5, ge=0, le=1)
    blocked_risk_scale: float = Field(default=0.0, ge=0, le=1)
    prompt_version: str = Field(default="trade-approval-v1", min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_allowed_parameters(self) -> StrategySpec:
        if tuple(sorted(set(self.momentum_windows_1h))) != self.momentum_windows_1h:
            raise ValueError("momentum windows must be unique and ascending")
        if tuple(sorted(set(self.donchian_windows_4h))) != self.donchian_windows_4h:
            raise ValueError("Donchian windows must be unique and ascending")
        if self.normal_risk_scale != 1 or self.caution_risk_scale != 0.5:
            raise ValueError("risk scales are hard-limited to normal=1 and caution=0.5")
        if self.blocked_risk_scale != 0:
            raise ValueError("blocked risk scale must be zero")
        return self

    def risk_scale(self, regime: RiskRegime) -> float:
        return {
            RiskRegime.NORMAL: self.normal_risk_scale,
            RiskRegime.CAUTION: self.caution_risk_scale,
            RiskRegime.BLOCKED: self.blocked_risk_scale,
        }[regime]


class PromotionStatus(StrEnum):
    PROMOTED = "promoted"
    NOT_ELIGIBLE = "not_eligible"
    ROLLED_BACK = "rolled_back"


class PromotionRecord(StrictContract):
    record_id: str = Field(default_factory=lambda: uuid4().hex, min_length=8, max_length=64)
    challenger_version: str
    previous_champion_version: str
    resulting_champion_version: str
    status: PromotionStatus
    reasons: tuple[str, ...]
    net_return_improvement: float
    evaluated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_record(self) -> PromotionRecord:
        object.__setattr__(
            self, "evaluated_at", _require_aware(self.evaluated_at, "evaluated_at")
        )
        return self


@runtime_checkable
class MarketDataProvider(Protocol):
    def closed_bars(
        self, symbol: str, interval: CandleInterval, limit: int
    ) -> Sequence[MarketBar]: ...


@runtime_checkable
class AccountProvider(Protocol):
    def open_positions(self) -> Sequence[PositionThesis]: ...


@runtime_checkable
class OrderGateway(Protocol):
    def submit(self, candidate: TradeCandidate, decision: TradeDecision) -> Mapping[str, Any]: ...


@runtime_checkable
class DecisionProvider(Protocol):
    def decide(
        self,
        candidate: TradeCandidate | None,
        *,
        position: PositionThesis | None = None,
        evidence: Sequence[Mapping[str, JsonValue]] = (),
        signal_strengthening: bool = False,
        now: datetime | None = None,
    ) -> TradeDecision: ...
