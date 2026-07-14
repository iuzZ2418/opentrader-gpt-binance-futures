from __future__ import annotations

import hashlib
import hmac
import json
import stat
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from .audit import AuditRepository
from .learning import TradeOutcome
from .research_validation import (
    STANDARD_SCENARIOS,
    AuditedShadowTrade,
    ExecutedBacktestTrade,
    ExpandingWalkForwardConfig,
    FundingPayment,
    PairedShadowAccumulator,
    ResearchBacktestValidator,
    ResearchValidationReport,
    ScenarioRequest,
    StatisticalValidation,
    StatisticalValidationRequest,
    build_expanding_windows,
)

MAX_MANIFEST_BYTES = 64 * 1024 * 1024
SHA256_LENGTH = 64
MAX_CLOCK_SKEW_SECONDS = 300


class ResearchManifestError(ValueError):
    """A stable fail-closed rejection code for an offline research manifest."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _text(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require_sha256(value: str, field_name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return normalized


class ManifestStrategyIdentity(_StrictModel):
    trace_id: str = Field(min_length=1, max_length=128)
    spec_id: str = Field(min_length=1, max_length=128)
    strategy_version: str = Field(min_length=1, max_length=128)
    parent_version: str | None = Field(default=None, max_length=128)
    status: Literal["CHAMPION", "CHALLENGER"]
    prompt_version: str = Field(min_length=1, max_length=128)
    strategy_parameters: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_identity(self) -> ManifestStrategyIdentity:
        for name in ("trace_id", "spec_id", "strategy_version", "prompt_version"):
            _text(getattr(self, name), name)
        if self.parent_version is not None:
            _text(self.parent_version, "parent_version")
        if self.status == "CHALLENGER" and not self.parent_version:
            raise ValueError("challenger identity requires parent_version")
        if self.status == "CHAMPION" and self.parent_version is not None:
            raise ValueError("champion identity cannot carry parent_version")
        _canonical(self.strategy_parameters)
        return self


class ManifestResearchConfig(_StrictModel):
    research_started_at: str = Field(min_length=1, max_length=64)
    research_ended_at: str = Field(min_length=1, max_length=64)
    initial_training_months: int = Field(ge=1, le=240)
    test_window_months: int = Field(ge=1, le=120)
    holdout_months: Literal[12]
    initial_equity: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_timestamps(self) -> ManifestResearchConfig:
        started = _timestamp(self.research_started_at, "research_started_at")
        ended = _timestamp(self.research_ended_at, "research_ended_at")
        if ended <= started:
            raise ValueError("research_ended_at must follow research_started_at")
        return self

    def to_domain(self) -> ExpandingWalkForwardConfig:
        return ExpandingWalkForwardConfig(
            research_started_at=_timestamp(self.research_started_at, "research_started_at"),
            research_ended_at=_timestamp(self.research_ended_at, "research_ended_at"),
            initial_training_months=self.initial_training_months,
            test_window_months=self.test_window_months,
            holdout_months=self.holdout_months,
        )


class ManifestFundingPayment(_StrictModel):
    event_id: str = Field(min_length=1, max_length=256)
    effective_at: str = Field(min_length=1, max_length=64)
    cost: float

    @model_validator(mode="after")
    def validate_payment(self) -> ManifestFundingPayment:
        _text(self.event_id, "event_id")
        _timestamp(self.effective_at, "funding.effective_at")
        return self

    def to_domain(self) -> FundingPayment:
        return FundingPayment(
            event_id=self.event_id,
            effective_at=_timestamp(self.effective_at, "funding.effective_at"),
            cost=self.cost,
        )


class ManifestExecutedTrade(_StrictModel):
    trade_id: str = Field(min_length=1, max_length=256)
    symbol: str = Field(min_length=1, max_length=32)
    direction: Literal[-1, 1]
    quantity: float = Field(gt=0)
    signal_at: str = Field(min_length=1, max_length=64)
    information_cutoff_at: str = Field(min_length=1, max_length=64)
    opened_at: str = Field(min_length=1, max_length=64)
    closed_at: str = Field(min_length=1, max_length=64)
    entry_reference_price: float = Field(gt=0)
    entry_fill_price: float = Field(gt=0)
    exit_reference_price: float = Field(gt=0)
    exit_fill_price: float = Field(gt=0)
    entry_reference_available_at: str = Field(min_length=1, max_length=64)
    exit_reference_available_at: str = Field(min_length=1, max_length=64)
    entry_fee: float = Field(ge=0)
    exit_fee: float = Field(ge=0)
    funding_events: list[ManifestFundingPayment]
    entry_fee_evidence_ids: list[str] = Field(min_length=1, max_length=100)
    exit_fee_evidence_ids: list[str] = Field(min_length=1, max_length=100)
    funding_coverage_id: str = Field(min_length=1, max_length=256)
    source_ids: list[str] = Field(min_length=1, max_length=1_000)
    market_data_digest: str
    fees_complete: Literal[True]
    funding_complete: Literal[True]

    @model_validator(mode="after")
    def validate_trade(self) -> ManifestExecutedTrade:
        for name in ("trade_id", "symbol", "funding_coverage_id"):
            _text(getattr(self, name), name)
        for name in (
            "signal_at",
            "information_cutoff_at",
            "opened_at",
            "closed_at",
            "entry_reference_available_at",
            "exit_reference_available_at",
        ):
            _timestamp(getattr(self, name), name)
        for name in ("entry_fee_evidence_ids", "exit_fee_evidence_ids", "source_ids"):
            values = getattr(self, name)
            if any(not item.strip() for item in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique non-empty IDs")
        _require_sha256(self.market_data_digest, "market_data_digest")
        return self

    def to_domain(self) -> ExecutedBacktestTrade:
        return ExecutedBacktestTrade(
            trade_id=self.trade_id,
            symbol=self.symbol.upper(),
            direction=self.direction,
            quantity=self.quantity,
            signal_at=_timestamp(self.signal_at, "signal_at"),
            information_cutoff_at=_timestamp(self.information_cutoff_at, "information_cutoff_at"),
            opened_at=_timestamp(self.opened_at, "opened_at"),
            closed_at=_timestamp(self.closed_at, "closed_at"),
            entry_reference_price=self.entry_reference_price,
            entry_fill_price=self.entry_fill_price,
            exit_reference_price=self.exit_reference_price,
            exit_fill_price=self.exit_fill_price,
            entry_reference_available_at=_timestamp(
                self.entry_reference_available_at, "entry_reference_available_at"
            ),
            exit_reference_available_at=_timestamp(
                self.exit_reference_available_at, "exit_reference_available_at"
            ),
            entry_fee=self.entry_fee,
            exit_fee=self.exit_fee,
            funding_events=tuple(item.to_domain() for item in self.funding_events),
            entry_fee_evidence_ids=tuple(self.entry_fee_evidence_ids),
            exit_fee_evidence_ids=tuple(self.exit_fee_evidence_ids),
            funding_coverage_id=self.funding_coverage_id,
            source_ids=tuple(self.source_ids),
            market_data_digest=self.market_data_digest.lower(),
            fees_complete=True,
            funding_complete=True,
        )


class ManifestScenarioResult(_StrictModel):
    window_id: str = Field(min_length=1, max_length=128)
    scenario_id: str = Field(min_length=1, max_length=128)
    holdout_seal_id: str | None = Field(default=None, max_length=256)
    trades: list[ManifestExecutedTrade] = Field(max_length=1_000_000)

    @model_validator(mode="after")
    def validate_result(self) -> ManifestScenarioResult:
        _text(self.window_id, "window_id")
        _text(self.scenario_id, "scenario_id")
        if self.holdout_seal_id is not None:
            _text(self.holdout_seal_id, "holdout_seal_id")
        return self


class ManifestHoldoutSeal(_StrictModel):
    seal_id: str = Field(min_length=1, max_length=256)
    strategy_digest: str
    pre_holdout_results_digest: str

    @model_validator(mode="after")
    def validate_seal(self) -> ManifestHoldoutSeal:
        _text(self.seal_id, "seal_id")
        _require_sha256(self.strategy_digest, "holdout.strategy_digest")
        _require_sha256(
            self.pre_holdout_results_digest,
            "holdout.pre_holdout_results_digest",
        )
        return self


class ManifestStatisticalValidation(_StrictModel):
    request_digest: str
    dsr_significance_probability: float = Field(ge=0, le=1)
    pbo_probability: float = Field(ge=0, le=1)
    dsr_method: Literal["BAILEY_LOPEZ_DE_PRADO_DSR"]
    pbo_method: Literal["CSCV_PBO"]
    source_digest: str
    observation_count: int = Field(ge=30)
    independent_trial_count: int = Field(ge=2)
    fold_count: int = Field(ge=4)

    @model_validator(mode="after")
    def validate_digests(self) -> ManifestStatisticalValidation:
        _require_sha256(self.request_digest, "statistics.request_digest")
        _require_sha256(self.source_digest, "statistics.source_digest")
        return self


class BacktestResearchManifest(_StrictModel):
    schema_version: Literal[1]
    manifest_type: Literal["BACKTEST"]
    manifest_id: str = Field(min_length=1, max_length=128)
    generator_id: str = Field(min_length=1, max_length=256)
    generated_at: str = Field(min_length=1, max_length=64)
    strategy: ManifestStrategyIdentity
    research: ManifestResearchConfig
    holdout_seal: ManifestHoldoutSeal
    scenario_results: list[ManifestScenarioResult] = Field(min_length=1)
    statistical_validation: ManifestStatisticalValidation

    @model_validator(mode="after")
    def validate_manifest(self) -> BacktestResearchManifest:
        _text(self.manifest_id, "manifest_id")
        _text(self.generator_id, "generator_id")
        _timestamp(self.generated_at, "generated_at")
        if self.strategy.status != "CHALLENGER":
            raise ValueError("backtest manifest must identify a challenger")
        return self


class ManifestShadowCoverage(_StrictModel):
    observed_at: str = Field(min_length=1, max_length=64)
    evidence_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_coverage(self) -> ManifestShadowCoverage:
        _timestamp(self.observed_at, "coverage.observed_at")
        _text(self.evidence_id, "coverage.evidence_id")
        return self


class ManifestShadowOutcome(_StrictModel):
    symbol: str = Field(min_length=1, max_length=32)
    closed_at: str = Field(min_length=1, max_length=64)
    gross_pnl: float
    fees: float = Field(ge=0)
    slippage_cost: float = Field(ge=0)
    funding_cost: float
    episode_id: str = Field(min_length=1, max_length=256)
    trace_ids: list[str] = Field(min_length=1, max_length=1_000)
    strategy_versions: list[str] = Field(min_length=1, max_length=100)
    source_record_ids: list[str] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_outcome(self) -> ManifestShadowOutcome:
        _text(self.symbol, "shadow.symbol")
        _timestamp(self.closed_at, "shadow.closed_at")
        _text(self.episode_id, "shadow.episode_id")
        for name in ("trace_ids", "strategy_versions", "source_record_ids"):
            values = getattr(self, name)
            if any(not item.strip() for item in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique non-empty IDs")
        return self

    def to_domain(self) -> TradeOutcome:
        return TradeOutcome(
            symbol=self.symbol.upper(),
            closed_at=_timestamp(self.closed_at, "shadow.closed_at"),
            gross_pnl=self.gross_pnl,
            fees=self.fees,
            slippage_cost=self.slippage_cost,
            funding_cost=self.funding_cost,
            episode_id=self.episode_id,
            trace_ids=tuple(self.trace_ids),
            strategy_versions=tuple(self.strategy_versions),
            source_record_ids=tuple(self.source_record_ids),
        )


class ManifestShadowTrade(_StrictModel):
    spec_id: str = Field(min_length=1, max_length=128)
    trade_id: str = Field(min_length=1, max_length=256)
    outcome: ManifestShadowOutcome
    fee_evidence_id: str = Field(min_length=1, max_length=256)
    slippage_evidence_id: str = Field(min_length=1, max_length=256)
    funding_evidence_id: str = Field(min_length=1, max_length=256)
    accounting_complete: Literal[True]

    @model_validator(mode="after")
    def validate_trade(self) -> ManifestShadowTrade:
        for name in (
            "spec_id",
            "trade_id",
            "fee_evidence_id",
            "slippage_evidence_id",
            "funding_evidence_id",
        ):
            _text(getattr(self, name), name)
        if not {
            self.fee_evidence_id,
            self.slippage_evidence_id,
            self.funding_evidence_id,
        }.issubset(self.outcome.source_record_ids):
            raise ValueError("shadow cost IDs must be included in source_record_ids")
        return self

    def to_domain(self) -> AuditedShadowTrade:
        return AuditedShadowTrade(
            trade_id=self.trade_id,
            outcome=self.outcome.to_domain(),
            fee_evidence_id=self.fee_evidence_id,
            slippage_evidence_id=self.slippage_evidence_id,
            funding_evidence_id=self.funding_evidence_id,
            accounting_complete=True,
        )


class PairedShadowResearchManifest(_StrictModel):
    schema_version: Literal[1]
    manifest_type: Literal["PAIRED_SHADOW"]
    manifest_id: str = Field(min_length=1, max_length=128)
    generator_id: str = Field(min_length=1, max_length=256)
    generated_at: str = Field(min_length=1, max_length=64)
    champion: ManifestStrategyIdentity
    challenger: ManifestStrategyIdentity
    started_at: str = Field(min_length=1, max_length=64)
    ended_at: str = Field(min_length=1, max_length=64)
    initial_equity: float = Field(gt=0)
    finalize: bool
    daily_coverage: list[ManifestShadowCoverage]
    trades: list[ManifestShadowTrade]

    @model_validator(mode="after")
    def validate_manifest(self) -> PairedShadowResearchManifest:
        _text(self.manifest_id, "manifest_id")
        _text(self.generator_id, "generator_id")
        _timestamp(self.generated_at, "generated_at")
        started = _timestamp(self.started_at, "started_at")
        ended = _timestamp(self.ended_at, "ended_at")
        if ended < started:
            raise ValueError("ended_at cannot precede started_at")
        if self.champion.status != "CHAMPION" or self.challenger.status != "CHALLENGER":
            raise ValueError("paired shadow requires champion and challenger identities")
        if self.champion.trace_id != self.challenger.trace_id:
            raise ValueError("paired shadow specs must share one trace")
        if self.challenger.parent_version != self.champion.strategy_version:
            raise ValueError("challenger parent must be the paired champion")
        if self.champion.spec_id == self.challenger.spec_id:
            raise ValueError("paired shadow specs must be distinct")
        coverage_dates: set[str] = set()
        coverage_evidence_ids: set[str] = set()
        for item in self.daily_coverage:
            observed = _timestamp(item.observed_at, "coverage.observed_at")
            if observed < started or observed > ended:
                raise ValueError("coverage timestamp is outside the declared interval")
            date_key = observed.date().isoformat()
            if date_key in coverage_dates:
                raise ValueError("daily coverage dates must be unique")
            if item.evidence_id in coverage_evidence_ids:
                raise ValueError("daily coverage evidence IDs must be unique")
            coverage_dates.add(date_key)
            coverage_evidence_ids.add(item.evidence_id)
        trade_keys: set[tuple[str, str]] = set()
        identities = {
            self.champion.spec_id: self.champion.strategy_version,
            self.challenger.spec_id: self.challenger.strategy_version,
        }
        for item in self.trades:
            version = identities.get(item.spec_id)
            if version is None:
                raise ValueError("shadow trade is outside the paired specs")
            if item.outcome.strategy_versions != [version]:
                raise ValueError("shadow outcome does not bind its strategy version")
            closed = _timestamp(item.outcome.closed_at, "shadow.closed_at")
            if closed < started or closed > ended:
                raise ValueError("shadow trade close is outside the declared interval")
            key = (item.spec_id, item.trade_id)
            if key in trade_keys:
                raise ValueError("shadow trade IDs must be unique per spec")
            trade_keys.add(key)
        return self


ResearchManifest = BacktestResearchManifest | PairedShadowResearchManifest


@dataclass(frozen=True, slots=True)
class LoadedResearchManifest:
    manifest: ResearchManifest
    sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResearchManifestError("DUPLICATE_JSON_KEY", key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ResearchManifestError("NON_FINITE_JSON_VALUE", value)


def load_verified_research_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    now: datetime | None = None,
) -> LoadedResearchManifest:
    try:
        expected = _require_sha256(expected_sha256, "expected_sha256")
    except ValueError as error:
        raise ResearchManifestError("INVALID_EXPECTED_SHA256") from error
    manifest_path = Path(path)
    try:
        metadata = manifest_path.stat()
    except OSError as error:
        raise ResearchManifestError("MANIFEST_UNREADABLE") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ResearchManifestError("MANIFEST_NOT_REGULAR_FILE")
    if metadata.st_size <= 0 or metadata.st_size > MAX_MANIFEST_BYTES:
        raise ResearchManifestError("MANIFEST_SIZE_INVALID")
    try:
        with manifest_path.open("rb") as handle:
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
    except OSError as error:
        raise ResearchManifestError("MANIFEST_UNREADABLE") from error
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ResearchManifestError("MANIFEST_SIZE_INVALID")
    observed = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise ResearchManifestError("MANIFEST_SHA256_MISMATCH")
    try:
        decoded = raw.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(payload, dict):
            raise ResearchManifestError("MANIFEST_ROOT_INVALID")
        manifest_type = payload.get("manifest_type")
        if manifest_type == "BACKTEST":
            manifest: ResearchManifest = BacktestResearchManifest.model_validate(
                payload, strict=True
            )
        elif manifest_type == "PAIRED_SHADOW":
            manifest = PairedShadowResearchManifest.model_validate(payload, strict=True)
        else:
            raise ResearchManifestError("MANIFEST_TYPE_UNSUPPORTED")
    except ResearchManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ResearchManifestError("MANIFEST_SCHEMA_INVALID") from error
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None or reference.utcoffset() is None:
        raise ResearchManifestError("VALIDATION_CLOCK_NAIVE")
    reference = reference.astimezone(UTC)
    generated = _timestamp(manifest.generated_at, "generated_at")
    if generated.timestamp() > reference.timestamp() + MAX_CLOCK_SKEW_SECONDS:
        raise ResearchManifestError("MANIFEST_GENERATED_IN_FUTURE")
    period_end = (
        _timestamp(manifest.research.research_ended_at, "research_ended_at")
        if isinstance(manifest, BacktestResearchManifest)
        else _timestamp(manifest.ended_at, "ended_at")
    )
    if period_end > generated:
        raise ResearchManifestError("MANIFEST_PERIOD_ENDS_AFTER_GENERATION")
    return LoadedResearchManifest(manifest=manifest, sha256=observed)


def _validate_strategy_identity(
    audit: AuditRepository,
    identity: ManifestStrategyIdentity,
) -> dict[str, Any]:
    trace = audit.get_trace(identity.trace_id)
    matching = [row for row in trace["strategy_specs"] if str(row["spec_id"]) == identity.spec_id]
    if len(matching) != 1:
        raise ResearchManifestError("STRATEGY_SPEC_NOT_IN_TRACE")
    row = matching[0]
    expected = {
        "trace_id": identity.trace_id,
        "strategy_version": identity.strategy_version,
        "parent_version": identity.parent_version,
        "status": identity.status,
        "prompt_version": identity.prompt_version,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ResearchManifestError("STRATEGY_IDENTITY_CONFLICT")
    if _canonical(row.get("parameters")) != _canonical(identity.strategy_parameters):
        raise ResearchManifestError("STRATEGY_PARAMETERS_CONFLICT")
    if identity.status == "CHALLENGER" and not any(
        candidate.get("strategy_version") == identity.parent_version
        and candidate.get("status") == "CHAMPION"
        for candidate in trace["strategy_specs"]
    ):
        raise ResearchManifestError("CHALLENGER_PARENT_NOT_IN_TRACE")
    return row


class ManifestScenarioEvaluator:
    """A fixed lookup evaluator; it never imports or executes simulator code."""

    def __init__(self, manifest: BacktestResearchManifest) -> None:
        self.manifest = manifest
        self.config = manifest.research.to_domain()
        walks, holdout = build_expanding_windows(self.config)
        self._windows = {item.window_id: item for item in (*walks, holdout)}
        expected = {
            (window.window_id, scenario.scenario_id)
            for window in (*walks, holdout)
            for scenario in STANDARD_SCENARIOS
        }
        results: dict[tuple[str, str], ManifestScenarioResult] = {}
        for item in manifest.scenario_results:
            key = (item.window_id, item.scenario_id)
            if key in results:
                raise ResearchManifestError("DUPLICATE_SCENARIO_RESULT")
            results[key] = item
        actual = set(results)
        if actual != expected:
            code = "MISSING_SCENARIO_RESULT" if expected - actual else "UNEXPECTED_SCENARIO_RESULT"
            raise ResearchManifestError(code)
        for key, item in results.items():
            window = self._windows[key[0]]
            if window.kind == "SEALED_HOLDOUT":
                if item.holdout_seal_id != manifest.holdout_seal.seal_id:
                    raise ResearchManifestError("HOLDOUT_RESULT_SEAL_MISMATCH")
            elif item.holdout_seal_id is not None:
                raise ResearchManifestError("PRE_HOLDOUT_RESULT_HAS_SEAL")
        strategy_digest = _digest(manifest.strategy.strategy_parameters)
        if strategy_digest != manifest.holdout_seal.strategy_digest.lower():
            raise ResearchManifestError("HOLDOUT_STRATEGY_DIGEST_MISMATCH")
        self._results = results
        self._expected = expected
        self._walk_keys = {
            key for key in expected if self._windows[key[0]].kind != "SEALED_HOLDOUT"
        }
        self._consumed: set[tuple[str, str]] = set()
        self._sealed = False

    def evaluate(self, request: ScenarioRequest) -> tuple[ExecutedBacktestTrade, ...]:
        key = (request.window.window_id, request.scenario.scenario_id)
        item = self._results.get(key)
        if item is None:
            raise ResearchManifestError("SCENARIO_RESULT_NOT_FOUND")
        is_holdout = request.window.kind == "SEALED_HOLDOUT"
        if is_holdout and not self._sealed:
            raise ResearchManifestError("HOLDOUT_EVALUATED_BEFORE_SEAL")
        if not is_holdout and self._sealed:
            raise ResearchManifestError("PRE_HOLDOUT_EVALUATED_AFTER_SEAL")
        if request.holdout_seal_id != item.holdout_seal_id:
            raise ResearchManifestError("REQUEST_SEAL_MISMATCH")
        self._consumed.add(key)
        return tuple(trade.to_domain() for trade in item.trades)

    def freeze_for_holdout(
        self,
        *,
        strategy_digest: str,
        pre_holdout_results_digest: str,
    ) -> str:
        if self._sealed:
            raise ResearchManifestError("HOLDOUT_ALREADY_SEALED")
        if not self._walk_keys.issubset(self._consumed):
            raise ResearchManifestError("PRE_HOLDOUT_MATRIX_INCOMPLETE")
        seal = self.manifest.holdout_seal
        if strategy_digest != seal.strategy_digest.lower():
            raise ResearchManifestError("HOLDOUT_STRATEGY_DIGEST_MISMATCH")
        if pre_holdout_results_digest != seal.pre_holdout_results_digest.lower():
            raise ResearchManifestError("HOLDOUT_PRE_RESULTS_DIGEST_MISMATCH")
        self._sealed = True
        return seal.seal_id

    def assert_complete(self) -> None:
        if self._consumed != self._expected:
            raise ResearchManifestError("SCENARIO_MATRIX_NOT_CONSUMED")


def statistical_request_sha256(request: StatisticalValidationRequest) -> str:
    return _digest(
        {
            "strategy_digest": request.strategy_digest,
            "audited_input_digest": request.audited_input_digest,
            "fold_net_returns": list(request.fold_net_returns),
            "trade_net_returns": list(request.trade_net_returns),
            "trade_count": request.trade_count,
        }
    )


class ManifestStatisticalValidator:
    """Return only the precomputed, request-bound independent statistics in the manifest."""

    def __init__(self, evidence: ManifestStatisticalValidation) -> None:
        self.evidence = evidence
        self.used = False

    def calculate(self, request: StatisticalValidationRequest) -> StatisticalValidation:
        if self.used:
            raise ResearchManifestError("STATISTICS_REUSED")
        self.used = True
        if statistical_request_sha256(request) != self.evidence.request_digest.lower():
            raise ResearchManifestError("STATISTICS_REQUEST_DIGEST_MISMATCH")
        if self.evidence.observation_count != request.trade_count:
            raise ResearchManifestError("STATISTICS_OBSERVATION_COUNT_MISMATCH")
        return StatisticalValidation(
            dsr_significance_probability=self.evidence.dsr_significance_probability,
            pbo_probability=self.evidence.pbo_probability,
            dsr_method=self.evidence.dsr_method,
            pbo_method=self.evidence.pbo_method,
            source_digest=self.evidence.source_digest.lower(),
            observation_count=self.evidence.observation_count,
            independent_trial_count=self.evidence.independent_trial_count,
            fold_count=self.evidence.fold_count,
        )

    def assert_used(self) -> None:
        if not self.used:
            raise ResearchManifestError("STATISTICS_NOT_CONSUMED")


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _attach_manifest_provenance(
    report: ResearchValidationReport,
    loaded: LoadedResearchManifest,
) -> ResearchValidationReport:
    manifest = loaded.manifest
    input_without_digest = dict(report.input_summary)
    input_without_digest.pop("audited_input_digest", None)
    input_without_digest["manifest"] = {
        "manifest_id": manifest.manifest_id,
        "manifest_type": manifest.manifest_type,
        "generator_id": manifest.generator_id,
        "generated_at": manifest.generated_at,
        "sha256": loaded.sha256,
    }
    audited_input_digest = _digest(input_without_digest)
    input_summary = {
        **input_without_digest,
        "audited_input_digest": audited_input_digest,
    }
    raw_metrics = dict(report.raw_metrics)
    raw_metrics["input_summary"] = input_summary
    report_digest = _digest(
        {
            "spec_id": report.spec_id,
            "trace_id": report.trace_id,
            "started_at": _iso(report.started_at),
            "ended_at": _iso(report.ended_at),
            "evidence": report.evidence.as_dict(),
            "input_digest": audited_input_digest,
            "raw_metrics": raw_metrics,
        }
    )
    return replace(
        report,
        input_summary=input_summary,
        raw_metrics=raw_metrics,
        report_digest=report_digest,
    )


def _run_backtest_manifest(
    loaded: LoadedResearchManifest,
    audit: AuditRepository,
) -> dict[str, Any]:
    manifest = loaded.manifest
    if not isinstance(manifest, BacktestResearchManifest):
        raise ResearchManifestError("MANIFEST_TYPE_MISMATCH")
    _validate_strategy_identity(audit, manifest.strategy)
    evaluator = ManifestScenarioEvaluator(manifest)
    statistical_validator = ManifestStatisticalValidator(manifest.statistical_validation)
    validator = ResearchBacktestValidator(
        spec_id=manifest.strategy.spec_id,
        trace_id=manifest.strategy.trace_id,
        strategy_parameters=manifest.strategy.strategy_parameters,
        config=manifest.research.to_domain(),
        initial_equity=manifest.research.initial_equity,
        evaluator=evaluator,
        statistical_validator=statistical_validator,
    )
    report = validator.run()
    evaluator.assert_complete()
    statistical_validator.assert_used()
    if report.raw_metrics.get("statistics", {}).get("status") != "VALIDATED":
        raise ResearchManifestError("STATISTICS_FAIL_CLOSED")
    report = _attach_manifest_provenance(report, loaded)
    backtest_run_id = report.append_backtest_run(audit)
    return {
        "status": "APPENDED",
        "manifest_type": manifest.manifest_type,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": loaded.sha256,
        "trace_id": manifest.strategy.trace_id,
        "spec_id": manifest.strategy.spec_id,
        "backtest_run_id": backtest_run_id,
        "report_digest": report.report_digest,
        "validation": {
            "walk_forward_passed": report.evidence.walk_forward_passed,
            "holdout_passed": report.evidence.holdout_passed,
            "parameter_perturbation_passed": report.evidence.parameter_perturbation_passed,
            "latency_stress_passed": report.evidence.latency_stress_passed,
            "social_placebo_passed": report.evidence.social_placebo_passed,
        },
    }


def _run_shadow_manifest(
    loaded: LoadedResearchManifest,
    audit: AuditRepository,
) -> dict[str, Any]:
    manifest = loaded.manifest
    if not isinstance(manifest, PairedShadowResearchManifest):
        raise ResearchManifestError("MANIFEST_TYPE_MISMATCH")
    _validate_strategy_identity(audit, manifest.champion)
    _validate_strategy_identity(audit, manifest.challenger)
    started = _timestamp(manifest.started_at, "started_at")
    ended = _timestamp(manifest.ended_at, "ended_at")
    coverage_source_ids = tuple(item.evidence_id for item in manifest.daily_coverage)
    if coverage_source_ids and not audit.performance_source_ids_exist(coverage_source_ids):
        raise ResearchManifestError("UNVERIFIED_SHADOW_COVERAGE_EVIDENCE")

    # Validate the complete submitted batch against both its own shape and the existing journal
    # before writing any new event.  This prevents a late conflict from partially importing a
    # manifest.
    preflight = PairedShadowAccumulator(
        trace_id=manifest.champion.trace_id,
        champion_spec_id=manifest.champion.spec_id,
        challenger_spec_id=manifest.challenger.spec_id,
        started_at=started,
        initial_equity=manifest.initial_equity,
        audit=audit,
    )
    for item in manifest.daily_coverage:
        preflight._record_daily_coverage(  # noqa: SLF001
            observed_at=_timestamp(item.observed_at, "coverage.observed_at"),
            evidence_id=item.evidence_id,
            persist=False,
        )
    for item in manifest.trades:
        preflight._record_trade(  # noqa: SLF001
            spec_id=item.spec_id,
            trade=item.to_domain(),
            persist=False,
        )

    writer = PairedShadowAccumulator(
        trace_id=manifest.champion.trace_id,
        champion_spec_id=manifest.champion.spec_id,
        challenger_spec_id=manifest.challenger.spec_id,
        started_at=started,
        initial_equity=manifest.initial_equity,
        audit=audit,
    )
    for item in manifest.daily_coverage:
        writer.record_daily_coverage(
            observed_at=_timestamp(item.observed_at, "coverage.observed_at"),
            evidence_id=item.evidence_id,
        )
    for item in manifest.trades:
        writer.record_trade(spec_id=item.spec_id, trade=item.to_domain())

    result = writer.append_if_mature(audit, ended_at=ended) if manifest.finalize else None
    if result is None:
        status = "JOURNALED"
        reasons: tuple[str, ...] = ()
    elif result.appended:
        status = "APPENDED"
        reasons = ()
    elif result.reason_codes == ("ALREADY_APPENDED",):
        status = "ALREADY_APPENDED"
        reasons = result.reason_codes
    else:
        status = "NOT_MATURE"
        reasons = result.reason_codes
    return {
        "status": status,
        "manifest_type": manifest.manifest_type,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": loaded.sha256,
        "trace_id": manifest.champion.trace_id,
        "champion_spec_id": manifest.champion.spec_id,
        "challenger_spec_id": manifest.challenger.spec_id,
        "pair_id": writer.pair_id,
        "reason_codes": list(reasons),
        "champion_shadow_result_id": (
            result.champion_shadow_result_id if result is not None else None
        ),
        "challenger_shadow_result_id": (
            result.challenger_shadow_result_id if result is not None else None
        ),
    }


def validate_and_append_research_manifest(
    loaded: LoadedResearchManifest,
    audit: AuditRepository,
) -> dict[str, Any]:
    if isinstance(loaded.manifest, BacktestResearchManifest):
        return _run_backtest_manifest(loaded, audit)
    return _run_shadow_manifest(loaded, audit)


__all__ = [
    "BacktestResearchManifest",
    "LoadedResearchManifest",
    "ManifestScenarioEvaluator",
    "ManifestStatisticalValidator",
    "PairedShadowResearchManifest",
    "ResearchManifestError",
    "load_verified_research_manifest",
    "statistical_request_sha256",
    "validate_and_append_research_manifest",
]
