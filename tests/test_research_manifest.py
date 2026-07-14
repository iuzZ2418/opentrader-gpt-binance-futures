from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.research_manifest import (
    ResearchManifestError,
    load_verified_research_manifest,
    statistical_request_sha256,
    validate_and_append_research_manifest,
)
from crypto_event_trader.research_validation import (
    ExecutedBacktestTrade,
    ExpandingWalkForwardConfig,
    FundingPayment,
    ResearchBacktestValidator,
    ResearchValidationError,
    ScenarioRequest,
    StatisticalValidation,
    StatisticalValidationRequest,
)
from crypto_event_trader.research_validator_cli import run_cli

TRACE_ID = "manifest-research-trace"
CHAMPION_SPEC_ID = "manifest-champion-spec"
CHALLENGER_SPEC_ID = "manifest-challenger-spec"
CHAMPION_VERSION = "manifest-champion-v1"
CHALLENGER_VERSION = "manifest-challenger-v2"
CHAMPION_PARAMETERS = {"minimum_directional_votes": 3}
CHALLENGER_PARAMETERS = {"minimum_directional_votes": 4}


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return _sha_bytes(raw)


def _write_manifest(path: Path, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return _sha_bytes(raw)


def _identity(
    *,
    champion: bool,
) -> dict[str, Any]:
    if champion:
        return {
            "trace_id": TRACE_ID,
            "spec_id": CHAMPION_SPEC_ID,
            "strategy_version": CHAMPION_VERSION,
            "parent_version": None,
            "status": "CHAMPION",
            "prompt_version": "manifest-prompt-v1",
            "strategy_parameters": CHAMPION_PARAMETERS,
        }
    return {
        "trace_id": TRACE_ID,
        "spec_id": CHALLENGER_SPEC_ID,
        "strategy_version": CHALLENGER_VERSION,
        "parent_version": CHAMPION_VERSION,
        "status": "CHALLENGER",
        "prompt_version": "manifest-prompt-v2",
        "strategy_parameters": CHALLENGER_PARAMETERS,
    }


def _audit(tmp_path: Path, name: str = "manifest.db") -> AuditRepository:
    audit = AuditRepository(tmp_path / name)
    audit.initialize()
    audit.append_strategy_spec(
        trace_id=TRACE_ID,
        spec_id=CHAMPION_SPEC_ID,
        strategy_version=CHAMPION_VERSION,
        status="CHAMPION",
        parameters=CHAMPION_PARAMETERS,
        prompt_version="manifest-prompt-v1",
    )
    audit.append_strategy_spec(
        trace_id=TRACE_ID,
        spec_id=CHALLENGER_SPEC_ID,
        strategy_version=CHALLENGER_VERSION,
        parent_version=CHAMPION_VERSION,
        status="CHALLENGER",
        parameters=CHALLENGER_PARAMETERS,
        prompt_version="manifest-prompt-v2",
    )
    return audit


class _CaptureEvaluator:
    def __init__(self) -> None:
        self.rows: list[tuple[ScenarioRequest, tuple[ExecutedBacktestTrade, ...]]] = []
        self.pre_holdout_digest = ""

    def evaluate(self, request: ScenarioRequest) -> tuple[ExecutedBacktestTrade, ...]:
        rows = tuple(self._trade(request, index) for index in range(8))
        self.rows.append((request, rows))
        return rows

    def freeze_for_holdout(
        self,
        *,
        strategy_digest: str,
        pre_holdout_results_digest: str,
    ) -> str:
        assert strategy_digest == _canonical_digest(CHALLENGER_PARAMETERS)
        self.pre_holdout_digest = pre_holdout_results_digest
        return "manifest-holdout-seal-v1"

    @staticmethod
    def _trade(request: ScenarioRequest, index: int) -> ExecutedBacktestTrade:
        signal = request.window.test_started_at + timedelta(days=1 + index * 2)
        opened = signal + timedelta(minutes=request.scenario.execution_delay_minutes)
        closed = opened + timedelta(days=1)
        trade_id = f"{request.window.window_id}-{request.scenario.scenario_id}-{index}"
        return ExecutedBacktestTrade(
            trade_id=trade_id,
            symbol="BTCUSDT" if index % 2 else "ETHUSDT",
            direction=1,
            quantity=1.0,
            signal_at=signal,
            information_cutoff_at=signal,
            opened_at=opened,
            closed_at=closed,
            entry_reference_price=100.0,
            entry_fill_price=100.1,
            exit_reference_price=110.0,
            exit_fill_price=109.9,
            entry_reference_available_at=opened,
            exit_reference_available_at=closed,
            entry_fee=0.5,
            exit_fee=0.5,
            funding_events=(
                FundingPayment(
                    event_id=f"{trade_id}-funding",
                    effective_at=opened + timedelta(hours=4),
                    cost=0.2,
                ),
            ),
            entry_fee_evidence_ids=(f"{trade_id}-entry-fee",),
            exit_fee_evidence_ids=(f"{trade_id}-exit-fee",),
            funding_coverage_id=f"{trade_id}-funding-coverage",
            source_ids=(f"{trade_id}-point-in-time-bars",),
            market_data_digest=hashlib.sha256(f"market:{trade_id}".encode()).hexdigest(),
            fees_complete=True,
            funding_complete=True,
        )


class _CaptureStatistics:
    def __init__(self) -> None:
        self.request: StatisticalValidationRequest | None = None

    def calculate(self, request: StatisticalValidationRequest) -> StatisticalValidation:
        self.request = request
        return StatisticalValidation(
            dsr_significance_probability=0.97,
            pbo_probability=0.05,
            dsr_method="BAILEY_LOPEZ_DE_PRADO_DSR",
            pbo_method="CSCV_PBO",
            source_digest=hashlib.sha256(b"independent-statistics").hexdigest(),
            observation_count=request.trade_count,
            independent_trial_count=8,
            fold_count=len(request.fold_net_returns),
        )


def _trade_payload(trade: ExecutedBacktestTrade) -> dict[str, Any]:
    return {
        "trade_id": trade.trade_id,
        "symbol": trade.symbol,
        "direction": trade.direction,
        "quantity": trade.quantity,
        "signal_at": trade.signal_at.isoformat(),
        "information_cutoff_at": trade.information_cutoff_at.isoformat(),
        "opened_at": trade.opened_at.isoformat(),
        "closed_at": trade.closed_at.isoformat(),
        "entry_reference_price": trade.entry_reference_price,
        "entry_fill_price": trade.entry_fill_price,
        "exit_reference_price": trade.exit_reference_price,
        "exit_fill_price": trade.exit_fill_price,
        "entry_reference_available_at": trade.entry_reference_available_at.isoformat(),
        "exit_reference_available_at": trade.exit_reference_available_at.isoformat(),
        "entry_fee": trade.entry_fee,
        "exit_fee": trade.exit_fee,
        "funding_events": [
            {
                "event_id": item.event_id,
                "effective_at": item.effective_at.isoformat(),
                "cost": item.cost,
            }
            for item in trade.funding_events
        ],
        "entry_fee_evidence_ids": list(trade.entry_fee_evidence_ids),
        "exit_fee_evidence_ids": list(trade.exit_fee_evidence_ids),
        "funding_coverage_id": trade.funding_coverage_id,
        "source_ids": list(trade.source_ids),
        "market_data_digest": trade.market_data_digest,
        "fees_complete": True,
        "funding_complete": True,
    }


@pytest.fixture(scope="module")
def backtest_manifest() -> dict[str, Any]:
    config = ExpandingWalkForwardConfig(
        research_started_at=datetime(2020, 1, 1, tzinfo=UTC),
        research_ended_at=datetime(2024, 1, 1, tzinfo=UTC),
        initial_training_months=12,
        test_window_months=6,
    )
    evaluator = _CaptureEvaluator()
    statistics = _CaptureStatistics()
    ResearchBacktestValidator(
        spec_id=CHALLENGER_SPEC_ID,
        trace_id=TRACE_ID,
        strategy_parameters=CHALLENGER_PARAMETERS,
        config=config,
        initial_equity=10_000.0,
        evaluator=evaluator,
        statistical_validator=statistics,
    ).run()
    assert statistics.request is not None
    return {
        "schema_version": 1,
        "manifest_type": "BACKTEST",
        "manifest_id": "audited-backtest-manifest-v1",
        "generator_id": "external-point-in-time-simulator/build-123",
        "generated_at": "2024-01-02T00:00:00Z",
        "strategy": _identity(champion=False),
        "research": {
            "research_started_at": "2020-01-01T00:00:00Z",
            "research_ended_at": "2024-01-01T00:00:00Z",
            "initial_training_months": 12,
            "test_window_months": 6,
            "holdout_months": 12,
            "initial_equity": 10_000.0,
        },
        "holdout_seal": {
            "seal_id": "manifest-holdout-seal-v1",
            "strategy_digest": _canonical_digest(CHALLENGER_PARAMETERS),
            "pre_holdout_results_digest": evaluator.pre_holdout_digest,
        },
        "scenario_results": [
            {
                "window_id": request.window.window_id,
                "scenario_id": request.scenario.scenario_id,
                "holdout_seal_id": request.holdout_seal_id,
                "trades": [_trade_payload(trade) for trade in trades],
            }
            for request, trades in evaluator.rows
        ],
        "statistical_validation": {
            "request_digest": statistical_request_sha256(statistics.request),
            "dsr_significance_probability": 0.97,
            "pbo_probability": 0.05,
            "dsr_method": "BAILEY_LOPEZ_DE_PRADO_DSR",
            "pbo_method": "CSCV_PBO",
            "source_digest": hashlib.sha256(b"independent-statistics").hexdigest(),
            "observation_count": statistics.request.trade_count,
            "independent_trial_count": 8,
            "fold_count": len(statistics.request.fold_net_returns),
        },
    }


def _clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def test_backtest_manifest_success_is_audited_and_idempotent(
    tmp_path: Path,
    backtest_manifest: dict[str, Any],
) -> None:
    audit = _audit(tmp_path)
    path = tmp_path / "manifest.json"
    digest = _write_manifest(path, backtest_manifest)
    loaded = load_verified_research_manifest(path, expected_sha256=digest)

    result = validate_and_append_research_manifest(loaded, audit)

    assert result["status"] == "APPENDED"
    rows = audit.get_trace(TRACE_ID)["backtest_runs"]
    assert len(rows) == 1
    assert rows[0]["raw_metrics"]["input_summary"]["manifest"]["sha256"] == digest
    assert validate_and_append_research_manifest(loaded, audit)["backtest_run_id"] == result[
        "backtest_run_id"
    ]
    assert len(audit.get_trace(TRACE_ID)["backtest_runs"]) == 1


def test_manifest_tamper_missing_scenario_and_identity_conflict_never_write(
    tmp_path: Path,
    backtest_manifest: dict[str, Any],
) -> None:
    audit = _audit(tmp_path)
    path = tmp_path / "tampered.json"
    digest = _write_manifest(path, backtest_manifest)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ResearchManifestError, match="MANIFEST_SHA256_MISMATCH"):
        load_verified_research_manifest(path, expected_sha256=digest)

    missing = _clone(backtest_manifest)
    missing["scenario_results"].pop()
    missing_digest = _write_manifest(path, missing)
    loaded = load_verified_research_manifest(path, expected_sha256=missing_digest)
    with pytest.raises(ResearchManifestError, match="MISSING_SCENARIO_RESULT"):
        validate_and_append_research_manifest(loaded, audit)

    conflict = _clone(backtest_manifest)
    conflict["strategy"]["prompt_version"] = "wrong-audit-identity"
    conflict_digest = _write_manifest(path, conflict)
    loaded = load_verified_research_manifest(path, expected_sha256=conflict_digest)
    with pytest.raises(ResearchManifestError, match="STRATEGY_IDENTITY_CONFLICT"):
        validate_and_append_research_manifest(loaded, audit)

    assert audit.get_trace(TRACE_ID)["backtest_runs"] == []


def test_future_manifest_is_rejected_before_audit_use(
    tmp_path: Path,
    backtest_manifest: dict[str, Any],
) -> None:
    payload = _clone(backtest_manifest)
    payload["research"]["research_ended_at"] = "2099-01-01T00:00:00Z"
    payload["generated_at"] = "2099-01-02T00:00:00Z"
    path = tmp_path / "future.json"
    digest = _write_manifest(path, payload)

    with pytest.raises(ResearchManifestError, match="MANIFEST_GENERATED_IN_FUTURE"):
        load_verified_research_manifest(
            path,
            expected_sha256=digest,
            now=datetime(2026, 7, 14, tzinfo=UTC),
        )


def test_cli_rejects_wrong_digest_before_database_access(
    tmp_path: Path,
    backtest_manifest: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "cli-manifest.json"
    _write_manifest(path, backtest_manifest)
    monkeypatch.delenv("AUDIT_DATABASE_URL", raising=False)

    assert run_cli(["--manifest", str(path), "--sha256", "0" * 64]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "reason_code": "MANIFEST_SHA256_MISMATCH",
        "status": "REJECTED",
    }


def test_cli_fails_closed_without_leaking_database_adapter_errors(
    tmp_path: Path,
    backtest_manifest: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "cli-valid-manifest.json"
    digest = _write_manifest(path, backtest_manifest)
    monkeypatch.setenv(
        "AUDIT_DATABASE_URL",
        "postgresql://validator:do-not-leak@postgres:5432/trader",
    )

    class UnexpectedAdapterFailure(Exception):
        pass

    def fail_to_connect(_: str) -> AuditRepository:
        raise UnexpectedAdapterFailure("do-not-leak")

    monkeypatch.setattr(
        "crypto_event_trader.research_validator_cli.AuditRepository",
        fail_to_connect,
    )

    assert run_cli(["--manifest", str(path), "--sha256", digest]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {
        "reason_code": "VALIDATION_INTERNAL_ERROR",
        "status": "REJECTED",
    }
    assert "do-not-leak" not in captured.err


def _append_source(audit: AuditRepository, source_id: str, observed_at: datetime) -> str:
    return audit.append_external_evidence(
        source="audited_shadow_runner",
        source_id=source_id,
        evidence_id=f"shadow:{source_id}",
        evidence_record_id=source_id,
        occurred_at=observed_at,
        first_observed_at=observed_at,
        payload={"source_id": source_id, "observed_at": observed_at.isoformat()},
        created_at=observed_at,
    )


def _append_shadow_cost_source(
    audit: AuditRepository,
    *,
    source_id: str,
    cost_type: str,
    amount: float,
    trade_id: str,
    episode_id: str,
    symbol: str,
    strategy_version: str,
    closed_at: datetime,
) -> str:
    return audit.append_external_evidence(
        trace_id=TRACE_ID,
        source="audited_shadow_runner",
        source_id=source_id,
        evidence_id=f"shadow-cost:{source_id}",
        evidence_record_id=source_id,
        occurred_at=closed_at,
        first_observed_at=closed_at,
        payload={
            "schema": "paired-shadow-cost-v1",
            "cost_type": cost_type,
            "trade_id": trade_id,
            "episode_id": episode_id,
            "trace_id": TRACE_ID,
            "symbol": symbol,
            "strategy_version": strategy_version,
            "closed_at": closed_at.isoformat(),
            "amount": amount,
        },
        created_at=closed_at,
    )


def _shadow_manifest(
    audit: AuditRepository,
    *,
    omit_coverage_day: int | None = None,
    register_sources: bool = True,
    typed_cost_sources: bool = True,
) -> dict[str, Any]:
    started = datetime(2026, 1, 1, tzinfo=UTC)
    ended = started + timedelta(days=90)
    coverage: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for day in range(91):
        if day == omit_coverage_day:
            continue
        evidence_id = f"shadow-coverage-{day}"
        observed = started + timedelta(days=day)
        if register_sources:
            _append_source(audit, evidence_id, observed)
        coverage.append({"observed_at": observed.isoformat(), "evidence_id": evidence_id})
    for identity in (_identity(champion=True), _identity(champion=False)):
        for number in range(30):
            prefix = "champion" if identity["status"] == "CHAMPION" else "challenger"
            closed = started + timedelta(days=number + 1)
            trade_id = f"{prefix}-{number}"
            episode_id = f"episode-{prefix}-{number}"
            symbol = "BTCUSDT" if number % 2 else "ETHUSDT"
            cost_ids = [
                f"{prefix}-{number}-fee",
                f"{prefix}-{number}-slippage",
                f"{prefix}-{number}-funding",
            ]
            if register_sources:
                if typed_cost_sources:
                    for source_id, cost_type, amount in zip(
                        cost_ids,
                        ("FEE", "SLIPPAGE", "FUNDING"),
                        (1.0, 0.5, 0.25),
                        strict=True,
                    ):
                        _append_shadow_cost_source(
                            audit,
                            source_id=source_id,
                            cost_type=cost_type,
                            amount=amount,
                            trade_id=trade_id,
                            episode_id=episode_id,
                            symbol=symbol,
                            strategy_version=identity["strategy_version"],
                            closed_at=closed,
                        )
                else:
                    for source_id in cost_ids:
                        _append_source(audit, source_id, closed)
            trades.append(
                {
                    "spec_id": identity["spec_id"],
                    "trade_id": trade_id,
                    "outcome": {
                        "symbol": symbol,
                        "closed_at": closed.isoformat(),
                        "gross_pnl": 10.0,
                        "fees": 1.0,
                        "slippage_cost": 0.5,
                        "funding_cost": 0.25,
                        "episode_id": episode_id,
                        "trace_ids": [TRACE_ID],
                        "strategy_versions": [identity["strategy_version"]],
                        "source_record_ids": cost_ids,
                    },
                    "fee_evidence_id": cost_ids[0],
                    "slippage_evidence_id": cost_ids[1],
                    "funding_evidence_id": cost_ids[2],
                    "accounting_complete": True,
                }
            )
    return {
        "schema_version": 1,
        "manifest_type": "PAIRED_SHADOW",
        "manifest_id": "audited-shadow-manifest-v1",
        "generator_id": "external-shadow-runner/build-456",
        "generated_at": "2026-04-02T00:00:00Z",
        "champion": _identity(champion=True),
        "challenger": _identity(champion=False),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "initial_equity": 10_000.0,
        "finalize": True,
        "daily_coverage": coverage,
        "trades": trades,
    }


def test_future_paired_shadow_manifest_is_rejected_before_audit_use(
    tmp_path: Path,
) -> None:
    audit = _audit(tmp_path, "future-shadow.db")
    payload = _shadow_manifest(audit, register_sources=False)
    payload["generated_at"] = "2099-04-02T00:00:00Z"
    path = tmp_path / "future-shadow.json"
    digest = _write_manifest(path, payload)

    with pytest.raises(ResearchManifestError, match="MANIFEST_GENERATED_IN_FUTURE"):
        load_verified_research_manifest(
            path,
            expected_sha256=digest,
            now=datetime(2026, 7, 14, tzinfo=UTC),
        )


def test_paired_shadow_manifest_persists_real_journal_and_finalizes(tmp_path: Path) -> None:
    audit = _audit(tmp_path, "shadow-success.db")
    payload = _shadow_manifest(audit)
    path = tmp_path / "shadow.json"
    digest = _write_manifest(path, payload)
    loaded = load_verified_research_manifest(path, expected_sha256=digest)

    result = validate_and_append_research_manifest(loaded, audit)

    assert result["status"] == "APPENDED"
    assert len(audit.shadow_journal_events(result["pair_id"])) == 151
    assert len(audit.get_trace(TRACE_ID)["shadow_results"]) == 2


def test_paired_shadow_manifest_rejects_legacy_untyped_cost_evidence(
    tmp_path: Path,
) -> None:
    audit = _audit(tmp_path, "shadow-legacy-costs.db")
    payload = _shadow_manifest(audit, typed_cost_sources=False)
    path = tmp_path / "shadow-legacy-costs.json"
    digest = _write_manifest(path, payload)

    with pytest.raises(ResearchValidationError, match="UNVERIFIED_SHADOW_COST_EVIDENCE"):
        validate_and_append_research_manifest(
            load_verified_research_manifest(path, expected_sha256=digest),
            audit,
        )

    assert audit.get_trace(TRACE_ID)["shadow_journal_events"] == []
    assert audit.get_trace(TRACE_ID)["shadow_results"] == []


def test_shadow_missing_or_unverified_coverage_never_writes_results(tmp_path: Path) -> None:
    incomplete_audit = _audit(tmp_path, "shadow-incomplete.db")
    incomplete = _shadow_manifest(incomplete_audit, omit_coverage_day=45)
    path = tmp_path / "shadow-incomplete.json"
    digest = _write_manifest(path, incomplete)
    result = validate_and_append_research_manifest(
        load_verified_research_manifest(path, expected_sha256=digest),
        incomplete_audit,
    )
    assert result["status"] == "NOT_MATURE"
    assert result["reason_codes"] == ["INCOMPLETE_DAILY_SHADOW_COVERAGE"]
    assert incomplete_audit.get_trace(TRACE_ID)["shadow_results"] == []

    fake_audit = _audit(tmp_path, "shadow-fake.db")
    fake = _shadow_manifest(fake_audit, register_sources=False)
    fake_digest = _write_manifest(path, fake)
    with pytest.raises(ResearchManifestError, match="UNVERIFIED_SHADOW_COVERAGE_EVIDENCE"):
        validate_and_append_research_manifest(
            load_verified_research_manifest(path, expected_sha256=fake_digest),
            fake_audit,
        )
    assert fake_audit.get_trace(TRACE_ID)["shadow_results"] == []
    assert fake_audit.shadow_journal_events(
        "shadow_pair_" + _canonical_digest(
            {
                "trace_id": TRACE_ID,
                "champion_spec_id": CHAMPION_SPEC_ID,
                "challenger_spec_id": CHALLENGER_SPEC_ID,
                "started_at": "2026-01-01T00:00:00Z",
            }
        )[:32]
    ) == []
