from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote
from uuid import uuid4

from .migrations import apply_postgres_migrations, apply_sqlite_migrations

if TYPE_CHECKING:
    from .learning import PromotionPolicy, TradeOutcome

JSON_COLUMNS = {
    "payload_json",
    "feature_snapshot_json",
    "evidence_ids_json",
    "invalidation_conditions_json",
    "raw_response_json",
    "supporting_evidence_json",
    "opposing_evidence_json",
    "reason_codes_json",
    "limits_snapshot_json",
    "positions_json",
    "parameters_json",
    "validation_json",
    "raw_metrics_json",
    "evaluation_json",
    "source_reliability_json",
    "sources_json",
    "rationale_json",
    "expected_failure_modes_json",
}

BOOLEAN_COLUMNS = {"reduce_only", "completed", "eligible"}

PAPER_FUNDING_COVERAGE_SCHEMA = "paper-funding-coverage-v1"
PAPER_FUNDING_COVERAGE_SOURCE = "binance_public_funding_history"

TRACE_TABLES = (
    "external_evidence",
    "trade_candidates",
    "candidate_evidence_links",
    "llm_decisions",
    "position_theses",
    "risk_decisions",
    "venue_orders",
    "venue_order_events",
    "venue_fills",
    "venue_accounting_events",
    "venue_accounting_attributions",
    "venue_fee_conversions",
    "account_snapshots",
    "strategy_specs",
    "strategy_research_runs",
    "backtest_runs",
    "shadow_results",
    "promotion_records",
    "strategy_registry_events",
    "shadow_journal_events",
    "counterfactual_outcomes",
)


class IncompleteVenueAccountingError(RuntimeError):
    """The immutable venue ledger cannot support an exact risk/performance number."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True, slots=True)
class FundingAttribution:
    status: str
    reason: str
    trace_id: str | None = None
    venue_order_id: str | None = None


ALLOWED_STRATEGY_PARAMETERS = frozenset(
    {
        "momentum_windows_1h",
        "donchian_windows_4h",
        "minimum_directional_votes",
        "ewma_span_hours",
        "target_annualized_volatility",
        "normal_risk_scale",
        "caution_risk_scale",
        "blocked_risk_scale",
        "vote_threshold",
        "target_volatility",
        "risk_scale",
        "funding_scale_thresholds",
        "basis_scale_thresholds",
        "oi_scale_thresholds",
        "adl_scale_thresholds",
        "order_book_scale_thresholds",
    }
)


def _utc_iso(value: datetime | str | None = None) -> str:
    if value is None:
        value = datetime.now(UTC)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid ISO-8601 timestamp: {value}") from exc
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def paper_funding_coverage_evidence_id(venue: str, episode_id: str) -> str:
    normalized_venue = venue.strip()
    normalized_episode = episode_id.strip()
    if not normalized_venue or not normalized_episode:
        raise ValueError("paper funding coverage venue and episode_id must be non-empty")
    return f"paper-funding-coverage:{normalized_venue}:{normalized_episode}"


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _optional_finite(name: str, value: float | None) -> float | None:
    return _require_finite(name, value) if value is not None else None


def _optional_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _optional_probability(name: str, value: float | None) -> float | None:
    numeric = _optional_finite(name, value)
    if numeric is not None and not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def _row_value(row: Any, name: str, index: int) -> Any:
    return row[name] if isinstance(row, (sqlite3.Row, dict)) else row[index]


def _as_utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _order_semantic_errors(
    order: Mapping[str, Any],
    candidate: Mapping[str, Any],
    decision: Mapping[str, Any],
    risk: Mapping[str, Any],
    theses: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    errors: list[str] = []

    def add(code: str) -> None:
        if code not in errors:
            errors.append(code)

    if order["symbol"] != candidate["symbol"]:
        add("ORDER_SYMBOL_MISMATCH")
    if decision["candidate_id"] != candidate["candidate_id"]:
        add("DECISION_CANDIDATE_MISMATCH")
    if decision.get("direction") is not None and decision["direction"] != candidate["direction"]:
        add("DECISION_DIRECTION_MISMATCH")
    if (
        risk["candidate_id"] != candidate["candidate_id"]
        or risk["decision_id"] != decision["decision_id"]
    ):
        add("RISK_LINEAGE_MISMATCH")
    if float(order["quantity"]) > float(candidate["max_quantity"]) + 1e-12:
        add("ORDER_EXCEEDS_CANDIDATE_QUANTITY")
    if float(order["quantity"]) > float(risk["approved_quantity"]) + 1e-12:
        add("ORDER_EXCEEDS_RISK_QUANTITY")
    if risk["outcome"] == "REJECT" or float(risk["approved_quantity"]) <= 0:
        add("RISK_DID_NOT_APPROVE_ORDER")

    action = decision["action"]
    direction = candidate["direction"]
    expected_entry_side = "BUY" if direction == "LONG" else "SELL"
    expected_exit_side = "SELL" if direction == "LONG" else "BUY"
    if action in {"OPEN", "ADD"}:
        if bool(order["reduce_only"]):
            add("ENTRY_MARKED_REDUCE_ONLY")
        if order["side"] != expected_entry_side:
            add("ENTRY_SIDE_DIRECTION_MISMATCH")
        if risk["outcome"] not in {"ALLOW", "RESIZE"}:
            add("ENTRY_RISK_OUTCOME_INVALID")
        minimum_confidence = 0.70 if action == "OPEN" else 0.80
        if float(decision["confidence"]) < minimum_confidence:
            add("DECISION_CONFIDENCE_TOO_LOW")
        if float(decision["position_multiplier"]) <= 0:
            add("POSITION_MULTIPLIER_NOT_POSITIVE")
        requested = float(candidate["max_quantity"]) * float(decision["position_multiplier"])
        if float(order["quantity"]) > requested + 1e-12:
            add("ORDER_EXCEEDS_DECISION_SIZE")
    elif action in {"REDUCE", "CLOSE"}:
        if not bool(order["reduce_only"]):
            add("EXIT_NOT_REDUCE_ONLY")
        if order["side"] != expected_exit_side:
            add("EXIT_SIDE_DIRECTION_MISMATCH")
        if risk["outcome"] != "EXIT":
            add("EXIT_RISK_OUTCOME_INVALID")
    else:
        add("DECISION_ACTION_NOT_EXECUTABLE")

    if not candidate.get("strategy_version"):
        add("MISSING_STRATEGY_VERSION")
    if not candidate.get("feature_snapshot"):
        add("MISSING_FEATURE_SNAPSHOT")
    if not candidate.get("evidence_ids"):
        add("MISSING_SOURCE_EVIDENCE")
    if not decision.get("evidence_ids"):
        add("MISSING_DECISION_EVIDENCE")
    elif not set(decision["evidence_ids"]).issubset(set(candidate.get("evidence_ids") or ())):
        add("UNKNOWN_DECISION_EVIDENCE")
    if not str(decision.get("thesis", "")).strip():
        add("MISSING_DECISION_THESIS")
    if not decision.get("invalidation_conditions"):
        add("MISSING_INVALIDATION_CONDITIONS")
    if not theses:
        add("MISSING_POSITION_THESIS")
    elif not any(thesis.get("supporting_evidence") for thesis in theses):
        add("MISSING_THESIS_EVIDENCE")
    try:
        if _as_utc_datetime(order["observed_at"]) > _as_utc_datetime(candidate["valid_until"]):
            add("CANDIDATE_EXPIRED")
    except (TypeError, ValueError):
        add("INVALID_CANDIDATE_TIMESTAMP")
    return tuple(errors)


def _raw_evidence_errors(
    candidate: Mapping[str, Any], evidence_rows: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    errors: list[str] = []

    def add(code: str) -> None:
        if code not in errors:
            errors.append(code)

    declared = set(candidate.get("evidence_ids") or ())
    linked = {str(row["evidence_id"]) for row in evidence_rows}
    if not evidence_rows:
        add("MISSING_RAW_EVIDENCE")
    if declared != linked:
        add("RAW_EVIDENCE_LINK_MISMATCH")
    candidate_created = _as_utc_datetime(candidate["created_at"])
    for row in evidence_rows:
        if row.get("deleted_at") is not None:
            add("DELETED_EVIDENCE_LINKED")
        try:
            canonical = _json(row["payload"])
        except (TypeError, ValueError):
            add("INVALID_EVIDENCE_PAYLOAD")
        else:
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if digest != row.get("content_hash"):
                add("EVIDENCE_HASH_MISMATCH")
        try:
            if _as_utc_datetime(row["occurred_at"]) > candidate_created:
                add("FUTURE_EVIDENCE_LINKED")
            if _as_utc_datetime(row["first_observed_at"]) > candidate_created:
                add("EVIDENCE_OBSERVED_AFTER_CANDIDATE")
        except (TypeError, ValueError):
            add("INVALID_EVIDENCE_TIMESTAMP")
    return tuple(errors)


def _validate_strategy_parameters(parameters: Mapping[str, Any]) -> None:
    unknown = set(parameters) - ALLOWED_STRATEGY_PARAMETERS
    if unknown:
        raise ValueError(
            "StrategySpec contains non-approved parameters: " + ", ".join(sorted(unknown))
        )
    # Canonical serialization also rejects unsupported objects and NaN/Infinity recursively.
    _json(dict(parameters))
    for key, maximum in (
        ("momentum_windows_1h", 24 * 365),
        ("donchian_windows_4h", 6 * 365),
    ):
        if key not in parameters:
            continue
        raw_windows = parameters[key]
        if not isinstance(raw_windows, Sequence) or isinstance(raw_windows, (str, bytes)):
            raise ValueError(f"{key} must be a non-empty sequence of integers")
        windows = list(raw_windows)
        if (
            not windows
            or len(windows) > 10
            or any(isinstance(value, bool) or not isinstance(value, int) for value in windows)
            or any(value <= 0 or value > maximum for value in windows)
            or windows != sorted(set(windows))
        ):
            raise ValueError(
                f"{key} must contain 1-10 unique, increasing, positive integer windows"
            )
    if "vote_threshold" in parameters:
        threshold = parameters["vote_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, int) or not 1 <= threshold <= 5:
            raise ValueError("vote_threshold must be an integer in [1, 5]")
    if "minimum_directional_votes" in parameters:
        threshold = parameters["minimum_directional_votes"]
        if isinstance(threshold, bool) or not isinstance(threshold, int) or not 3 <= threshold <= 5:
            raise ValueError("minimum_directional_votes must be an integer in [3, 5]")
    if "ewma_span_hours" in parameters:
        span = parameters["ewma_span_hours"]
        if isinstance(span, bool) or not isinstance(span, int) or not 168 <= span <= 1_440:
            raise ValueError("ewma_span_hours must be an integer in [168, 1440]")
    for key, upper_bound in (("target_volatility", 2.0), ("risk_scale", 1.0)):
        if key not in parameters:
            continue
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        numeric = float(value)
        lower_ok = numeric > 0 if key == "target_volatility" else numeric >= 0
        if not math.isfinite(numeric) or not lower_ok or numeric > upper_bound:
            interval = "(0, 2]" if key == "target_volatility" else "[0, 1]"
            raise ValueError(f"{key} must be finite and in {interval}")
    for key, lower_bound, upper_bound in (
        ("target_annualized_volatility", 0.0, 2.0),
        ("normal_risk_scale", -1.0, 1.0),
        ("caution_risk_scale", -1.0, 1.0),
        ("blocked_risk_scale", -1.0, 1.0),
    ):
        if key not in parameters:
            continue
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= lower_bound or numeric > upper_bound:
            raise ValueError(f"{key} is outside its approved finite range")
    exact_scales = {
        "normal_risk_scale": 1.0,
        "caution_risk_scale": 0.5,
        "blocked_risk_scale": 0.0,
    }
    for key, expected in exact_scales.items():
        if key in parameters and float(parameters[key]) != expected:
            raise ValueError(f"{key} must remain exactly {expected}")
    for key in (
        "funding_scale_thresholds",
        "basis_scale_thresholds",
        "oi_scale_thresholds",
        "adl_scale_thresholds",
        "order_book_scale_thresholds",
    ):
        if key not in parameters:
            continue
        root = parameters[key]
        if not isinstance(root, Mapping) or not root:
            raise ValueError(f"{key} must be a non-empty numeric mapping")

        def validate_tree(value: Any, depth: int = 0, field_name: str = key) -> None:
            if depth > 3:
                raise ValueError(f"{field_name} nesting is too deep")
            if isinstance(value, Mapping):
                if not value or any(not isinstance(name, str) or not name for name in value):
                    raise ValueError(f"{field_name} mapping keys must be non-empty strings")
                for child in value.values():
                    validate_tree(child, depth + 1, field_name)
                return
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{field_name} leaves must be finite numbers")

        validate_tree(root)


def _database_target(database_url: str | Path) -> tuple[str, str]:
    if isinstance(database_url, Path):
        return "sqlite", str(database_url)
    raw = str(database_url)
    if raw.startswith(("postgresql://", "postgres://")):
        return "postgres", raw
    if raw.startswith("sqlite:///"):
        path = unquote(raw.removeprefix("sqlite:///"))
        if path == ":memory:":
            return "sqlite", ":memory:"
        target = Path(path)
        if not target.is_absolute():
            target = Path.cwd() / target
        return "sqlite", str(target)
    if "://" in raw:
        raise ValueError("Only sqlite:/// and postgresql:// database URLs are supported")
    return "sqlite", raw


class AuditRepository:
    """Append-oriented audit ledger isolated from the legacy trading tables.

    SQLite is built in and is intended for tests. A PostgreSQL URL uses psycopg 3 when that
    optional production dependency is installed. The language model only supplies records;
    this repository never gives it database or exchange credentials.
    """

    def __init__(self, database_url: str | Path) -> None:
        self.dialect, self.target = _database_target(database_url)
        self._memory_connection: sqlite3.Connection | None = None

    @contextmanager
    def connect(self) -> Iterator[Any]:
        if self.dialect == "sqlite":
            connection = self._sqlite_connection()
            close = self.target != ":memory:"
        else:
            connection = self._postgres_connection()
            close = True
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            if close:
                connection.close()

    def _sqlite_connection(self) -> sqlite3.Connection:
        if self.target == ":memory:" and self._memory_connection is not None:
            return self._memory_connection
        if self.target != ":memory:":
            Path(self.target).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.target, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if self.target == ":memory:":
            self._memory_connection = connection
        return connection

    def _postgres_connection(self) -> Any:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - optional production adapter
            raise RuntimeError(
                "PostgreSQL audit storage requires the optional 'psycopg[binary]' package"
            ) from exc
        return psycopg.connect(self.target, row_factory=dict_row)

    def initialize(self) -> None:
        with self.connect() as connection:
            if self.dialect == "sqlite":
                apply_sqlite_migrations(connection)
            else:  # pragma: no cover - requires an external PostgreSQL service
                apply_postgres_migrations(connection)

    def close(self) -> None:
        if self._memory_connection is not None:
            self._memory_connection.close()
            self._memory_connection = None

    def _sql(self, query: str) -> str:
        return query if self.dialect == "sqlite" else query.replace("?", "%s")

    def _insert(self, table: str, values: Mapping[str, Any]) -> str:
        if table not in TRACE_TABLES:
            raise ValueError(f"Unknown audit table: {table}")
        columns = tuple(values)
        placeholders = ", ".join("?" for _ in columns)
        query = self._sql(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"  # noqa: S608
        )
        with self.connect() as connection:
            connection.execute(query, tuple(values[column] for column in columns))
        identifier = next((str(value) for key, value in values.items() if key.endswith("_id")), "")
        return identifier

    @staticmethod
    def _assert_trace(
        connection: Any,
        query: str,
        identifier: str,
        trace_id: str,
        *,
        dialect: str,
    ) -> None:
        if dialect != "sqlite":
            query = query.replace("?", "%s")
        row = connection.execute(query, (identifier,)).fetchone()
        if row is None:
            raise ValueError(f"Referenced audit record does not exist: {identifier}")
        actual = row["trace_id"] if isinstance(row, (sqlite3.Row, dict)) else row[0]
        if str(actual) != trace_id:
            raise ValueError(f"Referenced record {identifier} belongs to a different trace")

    def append_external_evidence(
        self,
        *,
        source: str,
        source_id: str,
        occurred_at: datetime | str,
        payload: Mapping[str, Any],
        first_observed_at: datetime | str | None = None,
        source_url: str | None = None,
        deleted_at: datetime | str | None = None,
        evidence_id: str | None = None,
        evidence_record_id: str | None = None,
        content_hash: str | None = None,
        trace_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        """Append an exact source version; edits and deletions become successor rows."""

        source = source.strip()
        source_id = source_id.strip()
        if not source or not source_id:
            raise ValueError("source and source_id must be non-empty")
        stable_id = evidence_id or f"{source}:{source_id}"
        canonical_payload = _json(dict(payload))
        calculated_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
        if content_hash is not None and content_hash.lower() != calculated_hash:
            raise ValueError("content_hash does not match the canonical evidence payload")
        evidence_record_id = evidence_record_id or _new_id("evidence")
        created_timestamp = _utc_iso(created_at)
        with self.connect() as connection:
            if self.dialect == "sqlite":
                connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                self._sql(
                    """
                    SELECT * FROM external_evidence WHERE evidence_id=?
                    ORDER BY version DESC LIMIT 1
                    """
                    + (" FOR UPDATE" if self.dialect == "postgres" else "")
                ),
                (stable_id,),
            ).fetchone()
            if row is None:
                version = 1
                prior_record_id = None
                first_observed_timestamp = _utc_iso(first_observed_at or created_timestamp)
            else:
                prior = self._decode_row(row)
                if prior["source"] != source or prior["source_id"] != source_id:
                    raise ValueError("source and source_id cannot change across evidence versions")
                version = int(prior["version"]) + 1
                prior_record_id = str(prior["evidence_record_id"])
                first_observed_timestamp = str(prior["first_observed_at"])
            values = {
                "evidence_record_id": evidence_record_id,
                "trace_id": trace_id,
                "evidence_id": stable_id,
                "version": version,
                "prior_evidence_record_id": prior_record_id,
                "source": source,
                "source_id": source_id,
                "source_url": source_url,
                "first_observed_at": first_observed_timestamp,
                "occurred_at": _utc_iso(occurred_at),
                "payload_json": canonical_payload,
                "content_hash": calculated_hash,
                "deleted_at": _utc_iso(deleted_at) if deleted_at else None,
                "created_at": created_timestamp,
            }
            columns = tuple(values)
            query = self._sql(
                f"INSERT INTO external_evidence ({', '.join(columns)}) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in columns)})"
            )
            connection.execute(query, tuple(values[column] for column in columns))
        return evidence_record_id

    def ensure_external_evidence(
        self,
        *,
        source: str,
        source_id: str,
        occurred_at: datetime | str,
        payload: Mapping[str, Any],
        evidence_id: str,
        first_observed_at: datetime | str | None = None,
        source_url: str | None = None,
        created_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Return the current immutable version, appending only when content changed.

        Closed-candle windows are collected more frequently than they change.  This helper
        keeps the exact normalized strategy inputs in the append-only evidence ledger without
        writing a duplicate 700-bar payload every 15 minutes.  A changed payload becomes the
        next version of the same stable evidence ID.
        """

        source = source.strip()
        source_id = source_id.strip()
        evidence_id = evidence_id.strip()
        if not source or not source_id or not evidence_id:
            raise ValueError("source, source_id, and evidence_id must be non-empty")
        canonical_payload = _json(dict(payload))
        content_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
        latest = self.latest_external_evidence(evidence_id)
        if latest is not None:
            if latest["source"] != source or latest["source_id"] != source_id:
                raise ValueError("source and source_id cannot change across evidence versions")
            if latest.get("deleted_at") is None and latest.get("content_hash") == content_hash:
                return latest
        record_id = self.append_external_evidence(
            source=source,
            source_id=source_id,
            source_url=source_url,
            evidence_id=evidence_id,
            occurred_at=occurred_at,
            first_observed_at=first_observed_at,
            payload=payload,
            content_hash=content_hash,
            created_at=created_at,
        )
        stored = self.latest_external_evidence(evidence_id)
        if stored is None or stored["evidence_record_id"] != record_id:
            raise RuntimeError("external evidence was not durably stored as the latest version")
        return stored

    def link_candidate_evidence(
        self,
        *,
        trace_id: str,
        candidate_id: str,
        evidence_record_ids: Sequence[str],
        role: str = "PRIMARY",
        created_at: datetime | str | None = None,
    ) -> tuple[str, ...]:
        if not evidence_record_ids:
            return ()
        role = role.strip().upper()
        if not role:
            raise ValueError("evidence role must be non-empty")
        links: list[str] = []
        with self.connect() as connection:
            self._assert_trace(
                connection,
                "SELECT trace_id FROM trade_candidates WHERE candidate_id=?",
                candidate_id,
                trace_id,
                dialect=self.dialect,
            )
            candidate_row = connection.execute(
                self._sql("SELECT * FROM trade_candidates WHERE candidate_id=?"),
                (candidate_id,),
            ).fetchone()
            assert candidate_row is not None
            expected_ids = set(self._decode_row(candidate_row)["evidence_ids"])
            for evidence_record_id in dict.fromkeys(evidence_record_ids):
                evidence_row = connection.execute(
                    self._sql("SELECT * FROM external_evidence WHERE evidence_record_id=?"),
                    (evidence_record_id,),
                ).fetchone()
                if evidence_row is None:
                    raise ValueError(f"Unknown evidence record: {evidence_record_id}")
                evidence = self._decode_row(evidence_row)
                if evidence["evidence_id"] not in expected_ids:
                    raise ValueError(
                        f"Evidence {evidence['evidence_id']} was not declared by the candidate"
                    )
                if evidence["deleted_at"] is not None:
                    raise ValueError("A candidate cannot link a deleted evidence version")
                latest_row = connection.execute(
                    self._sql(
                        """
                        SELECT evidence_record_id FROM external_evidence
                        WHERE evidence_id=? ORDER BY version DESC LIMIT 1
                        """
                    ),
                    (evidence["evidence_id"],),
                ).fetchone()
                assert latest_row is not None
                if str(_row_value(latest_row, "evidence_record_id", 0)) != (evidence_record_id):
                    raise ValueError("Candidate evidence must link the latest observed version")
                existing = connection.execute(
                    self._sql(
                        """
                        SELECT evidence_link_id FROM candidate_evidence_links
                        WHERE candidate_id=? AND evidence_record_id=?
                        """
                    ),
                    (candidate_id, evidence_record_id),
                ).fetchone()
                if existing is not None:
                    links.append(str(_row_value(existing, "evidence_link_id", 0)))
                    continue
                link_id = _new_id("evidence_link")
                connection.execute(
                    self._sql(
                        """
                        INSERT INTO candidate_evidence_links
                            (evidence_link_id, trace_id, candidate_id,
                             evidence_record_id, role, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """
                    ),
                    (
                        link_id,
                        trace_id,
                        candidate_id,
                        evidence_record_id,
                        role,
                        _utc_iso(created_at),
                    ),
                )
                links.append(link_id)
        return tuple(links)

    def append_trade_candidate(
        self,
        *,
        trace_id: str,
        strategy_version: str,
        symbol: str,
        direction: str,
        max_quantity: float,
        max_risk_fraction: float,
        feature_snapshot: Mapping[str, Any],
        evidence_ids: Sequence[str],
        evidence_record_ids: Sequence[str] = (),
        valid_until: datetime | str,
        candidate_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        candidate_id = candidate_id or _new_id("cand")
        direction = direction.upper()
        if direction not in {"LONG", "SHORT"}:
            raise ValueError("direction must be LONG or SHORT")
        stored_id = self._insert(
            "trade_candidates",
            {
                "candidate_id": candidate_id,
                "trace_id": trace_id,
                "strategy_version": strategy_version,
                "symbol": symbol.upper(),
                "direction": direction,
                "max_quantity": _require_finite("max_quantity", max_quantity),
                "max_risk_fraction": _require_finite("max_risk_fraction", max_risk_fraction),
                "feature_snapshot_json": _json(dict(feature_snapshot)),
                "evidence_ids_json": _json(list(evidence_ids)),
                "valid_until": _utc_iso(valid_until),
                "created_at": _utc_iso(created_at),
            },
        )
        if evidence_record_ids:
            self.link_candidate_evidence(
                trace_id=trace_id,
                candidate_id=candidate_id,
                evidence_record_ids=evidence_record_ids,
                created_at=created_at,
            )
        return stored_id

    def append_llm_decision(
        self,
        *,
        trace_id: str,
        candidate_id: str,
        action: str,
        direction: str | None = None,
        position_multiplier: float,
        confidence: float,
        evidence_ids: Sequence[str],
        thesis: str,
        invalidation_conditions: Sequence[str],
        model: str,
        prompt_version: str,
        next_review_at: datetime | str | None = None,
        response_id: str | None = None,
        latency_ms: int | None = None,
        raw_response: Mapping[str, Any] | None = None,
        decision_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        decision_id = decision_id or _new_id("decision")
        action = action.upper()
        if action not in {"OPEN", "ADD", "HOLD", "REDUCE", "CLOSE", "REJECT"}:
            raise ValueError(f"Unsupported decision action: {action}")
        if direction is not None:
            direction = direction.upper()
            if direction not in {"LONG", "SHORT"}:
                raise ValueError("direction must be LONG, SHORT, or None")
        with self.connect() as connection:
            self._assert_trace(
                connection,
                "SELECT trace_id FROM trade_candidates WHERE candidate_id=?",
                candidate_id,
                trace_id,
                dialect=self.dialect,
            )
            values = {
                "decision_id": decision_id,
                "trace_id": trace_id,
                "candidate_id": candidate_id,
                "action": action,
                "direction": direction,
                "position_multiplier": _require_finite("position_multiplier", position_multiplier),
                "confidence": _require_finite("confidence", confidence),
                "evidence_ids_json": _json(list(evidence_ids)),
                "thesis": thesis,
                "invalidation_conditions_json": _json(list(invalidation_conditions)),
                "next_review_at": _utc_iso(next_review_at) if next_review_at else None,
                "model": model,
                "prompt_version": prompt_version,
                "response_id": response_id,
                "latency_ms": latency_ms,
                "raw_response_json": _json(dict(raw_response or {})),
                "created_at": _utc_iso(created_at),
            }
            columns = tuple(values)
            placeholders = ", ".join("?" for _ in columns)
            query = self._sql(
                f"INSERT INTO llm_decisions ({', '.join(columns)}) VALUES ({placeholders})"  # noqa: S608
            )
            connection.execute(query, tuple(values[column] for column in columns))
        return decision_id

    def append_position_thesis(
        self,
        *,
        trace_id: str,
        position_id: str,
        decision_id: str,
        entry_reason: str,
        expected_horizon: str,
        supporting_evidence: Sequence[str],
        opposing_evidence: Sequence[str],
        add_count: int,
        pnl_r: float,
        invalidation_conditions: Sequence[str],
        thesis_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        thesis_id = thesis_id or _new_id("thesis")
        if isinstance(add_count, bool) or not isinstance(add_count, int) or not 0 <= add_count <= 1:
            raise ValueError("add_count must be the integer 0 or 1")
        if not entry_reason.strip() or not expected_horizon.strip():
            raise ValueError("entry_reason and expected_horizon must be non-empty")
        if not supporting_evidence:
            raise ValueError("position thesis must include supporting evidence")
        if not invalidation_conditions:
            raise ValueError("position thesis must include invalidation conditions")
        with self.connect() as connection:
            if self.dialect == "sqlite":
                connection.execute("BEGIN IMMEDIATE")
            else:  # pragma: no cover - requires an external PostgreSQL service
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (position_id,),
                )
            self._assert_trace(
                connection,
                "SELECT trace_id FROM llm_decisions WHERE decision_id=?",
                decision_id,
                trace_id,
                dialect=self.dialect,
            )
            row = connection.execute(
                self._sql(
                    """
                    SELECT thesis_id, version, add_count FROM position_theses
                    WHERE position_id=? ORDER BY version DESC LIMIT 1
                    """
                ),
                (position_id,),
            ).fetchone()
            if row is None:
                version, prior_thesis_id = 1, None
            else:
                version = (
                    int(row["version"] if isinstance(row, (sqlite3.Row, dict)) else row[1]) + 1
                )
                prior_thesis_id = str(
                    row["thesis_id"] if isinstance(row, (sqlite3.Row, dict)) else row[0]
                )
                prior_add_count = int(_row_value(row, "add_count", 2))
                if add_count < prior_add_count:
                    raise ValueError("add_count cannot decrease across thesis versions")
            values = {
                "thesis_id": thesis_id,
                "trace_id": trace_id,
                "position_id": position_id,
                "version": version,
                "prior_thesis_id": prior_thesis_id,
                "decision_id": decision_id,
                "entry_reason": entry_reason,
                "expected_horizon": expected_horizon,
                "supporting_evidence_json": _json(list(supporting_evidence)),
                "opposing_evidence_json": _json(list(opposing_evidence)),
                "add_count": add_count,
                "pnl_r": _require_finite("pnl_r", pnl_r),
                "invalidation_conditions_json": _json(list(invalidation_conditions)),
                "created_at": _utc_iso(created_at),
            }
            columns = tuple(values)
            query = self._sql(
                f"INSERT INTO position_theses ({', '.join(columns)}) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in columns)})"
            )
            connection.execute(query, tuple(values[column] for column in columns))
        return thesis_id

    def append_risk_decision(
        self,
        *,
        trace_id: str,
        candidate_id: str,
        decision_id: str,
        outcome: str,
        approved_quantity: float,
        reason_codes: Sequence[str],
        limits_snapshot: Mapping[str, Any],
        risk_decision_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        risk_decision_id = risk_decision_id or _new_id("risk")
        outcome = outcome.upper()
        if outcome not in {"ALLOW", "RESIZE", "REJECT", "EXIT"}:
            raise ValueError(f"Unsupported risk outcome: {outcome}")
        with self.connect() as connection:
            self._assert_trace(
                connection,
                "SELECT trace_id FROM trade_candidates WHERE candidate_id=?",
                candidate_id,
                trace_id,
                dialect=self.dialect,
            )
            self._assert_trace(
                connection,
                "SELECT trace_id FROM llm_decisions WHERE decision_id=?",
                decision_id,
                trace_id,
                dialect=self.dialect,
            )
            decision_row = connection.execute(
                self._sql("SELECT candidate_id FROM llm_decisions WHERE decision_id=?"),
                (decision_id,),
            ).fetchone()
            assert decision_row is not None
            if str(_row_value(decision_row, "candidate_id", 0)) != candidate_id:
                raise ValueError("Risk decision candidate does not match the LLM decision")
            values = {
                "risk_decision_id": risk_decision_id,
                "trace_id": trace_id,
                "candidate_id": candidate_id,
                "decision_id": decision_id,
                "outcome": outcome,
                "approved_quantity": _require_finite("approved_quantity", approved_quantity),
                "reason_codes_json": _json(list(reason_codes)),
                "limits_snapshot_json": _json(dict(limits_snapshot)),
                "created_at": _utc_iso(created_at),
            }
            columns = tuple(values)
            query = self._sql(
                f"INSERT INTO risk_decisions ({', '.join(columns)}) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in columns)})"
            )
            connection.execute(query, tuple(values[column] for column in columns))
        return risk_decision_id

    def append_venue_order(
        self,
        *,
        trace_id: str,
        candidate_id: str,
        decision_id: str,
        risk_decision_id: str,
        venue: str,
        client_order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        status: str,
        external_order_id: str | None = None,
        price: float | None = None,
        reduce_only: bool = False,
        raw_response: Mapping[str, Any] | None = None,
        observed_at: datetime | str | None = None,
        venue_order_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        venue_order_id = venue_order_id or _new_id("order")
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        observed_timestamp = _utc_iso(observed_at)
        created_timestamp = _utc_iso(created_at)
        with self.connect() as connection:
            for table, id_column, identifier in (
                ("trade_candidates", "candidate_id", candidate_id),
                ("llm_decisions", "decision_id", decision_id),
                ("risk_decisions", "risk_decision_id", risk_decision_id),
            ):
                self._assert_trace(
                    connection,
                    f"SELECT trace_id FROM {table} WHERE {id_column}=?",  # noqa: S608
                    identifier,
                    trace_id,
                    dialect=self.dialect,
                )
            candidate_row = connection.execute(
                self._sql("SELECT * FROM trade_candidates WHERE candidate_id=?"),
                (candidate_id,),
            ).fetchone()
            decision_row = connection.execute(
                self._sql("SELECT * FROM llm_decisions WHERE decision_id=?"),
                (decision_id,),
            ).fetchone()
            risk_row = connection.execute(
                self._sql(
                    """
                    SELECT * FROM risk_decisions WHERE risk_decision_id=?
                    """
                ),
                (risk_decision_id,),
            ).fetchone()
            thesis_rows = connection.execute(
                self._sql("SELECT * FROM position_theses WHERE decision_id=? ORDER BY version"),
                (decision_id,),
            ).fetchall()
            evidence_rows = connection.execute(
                self._sql(
                    """
                    SELECT e.* FROM candidate_evidence_links l
                    JOIN external_evidence e
                      ON e.evidence_record_id=l.evidence_record_id
                    WHERE l.candidate_id=?
                    ORDER BY l.created_at
                    """
                ),
                (candidate_id,),
            ).fetchall()
            assert candidate_row is not None and decision_row is not None and risk_row is not None
            values = {
                "venue_order_id": venue_order_id,
                "trace_id": trace_id,
                "candidate_id": candidate_id,
                "decision_id": decision_id,
                "risk_decision_id": risk_decision_id,
                "venue": venue,
                "client_order_id": client_order_id,
                "external_order_id": external_order_id,
                "symbol": symbol.upper(),
                "side": side,
                "order_type": order_type,
                "quantity": _require_finite("quantity", quantity),
                "price": _require_finite("price", price) if price is not None else None,
                "reduce_only": int(reduce_only),
                "status": status,
                "raw_response_json": _json(dict(raw_response or {})),
                "observed_at": observed_timestamp,
                "created_at": created_timestamp,
            }
            semantic_errors = _order_semantic_errors(
                values,
                self._decode_row(candidate_row),
                self._decode_row(decision_row),
                self._decode_row(risk_row),
                [self._decode_row(row) for row in thesis_rows],
            )
            semantic_errors += _raw_evidence_errors(
                self._decode_row(candidate_row),
                [self._decode_row(row) for row in evidence_rows],
            )
            if semantic_errors:
                raise ValueError("Order audit invariants failed: " + ", ".join(semantic_errors))
            columns = tuple(values)
            query = self._sql(
                f"INSERT INTO venue_orders ({', '.join(columns)}) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in columns)})"
            )
            connection.execute(query, tuple(values[column] for column in columns))
            event_values = {
                "order_event_id": _new_id("order_event"),
                "trace_id": trace_id,
                "venue_order_id": venue_order_id,
                "event_sequence": 1,
                "event_type": "ORDER_RECORDED",
                "status": status,
                "source_event_id": None,
                "external_order_id": external_order_id,
                "executed_quantity": 0.0,
                "average_price": None,
                "raw_response_json": _json(dict(raw_response or {})),
                "observed_at": observed_timestamp,
                "created_at": created_timestamp,
            }
            event_columns = tuple(event_values)
            event_query = self._sql(
                f"INSERT INTO venue_order_events ({', '.join(event_columns)}) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in event_columns)})"
            )
            connection.execute(
                event_query,
                tuple(event_values[column] for column in event_columns),
            )
        return venue_order_id

    def append_venue_order_event(
        self,
        *,
        trace_id: str,
        venue_order_id: str,
        event_type: str,
        status: str,
        observed_at: datetime | str | None = None,
        executed_quantity: float = 0.0,
        average_price: float | None = None,
        source_event_id: str | None = None,
        external_order_id: str | None = None,
        raw_response: Mapping[str, Any] | None = None,
        order_event_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        """Append one locally sequenced venue observation without rewriting order history."""

        event_type = event_type.strip().upper()
        status = status.strip().upper()
        if not event_type or not status:
            raise ValueError("event_type and status must be non-empty")
        order_event_id = order_event_id or _new_id("order_event")
        with self.connect() as connection:
            if self.dialect == "sqlite":
                # Serialize the MAX(sequence)+1 allocation across WS and REST reconcilers.
                connection.execute("BEGIN IMMEDIATE")
            else:  # pragma: no cover - requires an external PostgreSQL service
                connection.execute(
                    self._sql(
                        "SELECT venue_order_id FROM venue_orders WHERE venue_order_id=? FOR UPDATE"
                    ),
                    (venue_order_id,),
                )
            self._assert_trace(
                connection,
                "SELECT trace_id FROM venue_orders WHERE venue_order_id=?",
                venue_order_id,
                trace_id,
                dialect=self.dialect,
            )
            if source_event_id:
                existing = connection.execute(
                    self._sql(
                        """
                        SELECT order_event_id FROM venue_order_events
                        WHERE venue_order_id=? AND source_event_id=?
                        """
                    ),
                    (venue_order_id, source_event_id),
                ).fetchone()
                if existing is not None:
                    return str(_row_value(existing, "order_event_id", 0))
            row = connection.execute(
                self._sql(
                    """
                    SELECT MAX(event_sequence) AS max_sequence FROM venue_order_events
                    WHERE venue_order_id=?
                    """
                ),
                (venue_order_id,),
            ).fetchone()
            assert row is not None
            maximum = _row_value(row, "max_sequence", 0)
            event_sequence = int(maximum or 0) + 1
            values = {
                "order_event_id": order_event_id,
                "trace_id": trace_id,
                "venue_order_id": venue_order_id,
                "event_sequence": event_sequence,
                "event_type": event_type,
                "status": status,
                "source_event_id": source_event_id,
                "external_order_id": external_order_id,
                "executed_quantity": _require_finite("executed_quantity", executed_quantity),
                "average_price": (
                    _require_finite("average_price", average_price)
                    if average_price is not None
                    else None
                ),
                "raw_response_json": _json(dict(raw_response or {})),
                "observed_at": _utc_iso(observed_at),
                "created_at": _utc_iso(created_at),
            }
            columns = tuple(values)
            query = self._sql(
                f"INSERT INTO venue_order_events ({', '.join(columns)}) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in columns)})"
            )
            connection.execute(query, tuple(values[column] for column in columns))
        return order_event_id

    def append_venue_fill(
        self,
        *,
        trace_id: str,
        venue_order_id: str,
        price: float,
        quantity: float,
        fee: float,
        fee_asset: str,
        filled_at: datetime | str,
        external_fill_id: str | None = None,
        realized_pnl: float | None = None,
        raw_response: Mapping[str, Any] | None = None,
        venue_fill_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        venue_fill_id = venue_fill_id or _new_id("fill")
        with self.connect() as connection:
            if self.dialect == "sqlite":
                connection.execute("BEGIN IMMEDIATE")
            else:  # pragma: no cover - requires an external PostgreSQL service
                connection.execute(
                    self._sql(
                        "SELECT venue_order_id FROM venue_orders "
                        "WHERE venue_order_id=? FOR UPDATE"
                    ),
                    (venue_order_id,),
                )
            self._assert_trace(
                connection,
                "SELECT trace_id FROM venue_orders WHERE venue_order_id=?",
                venue_order_id,
                trace_id,
                dialect=self.dialect,
            )
            if external_fill_id:
                existing = connection.execute(
                    self._sql(
                        """
                        SELECT venue_fill_id FROM venue_fills
                        WHERE venue_order_id=? AND external_fill_id=?
                        """
                    ),
                    (venue_order_id, external_fill_id),
                ).fetchone()
                if existing is not None:
                    return str(_row_value(existing, "venue_fill_id", 0))
            values = {
                "venue_fill_id": venue_fill_id,
                "trace_id": trace_id,
                "venue_order_id": venue_order_id,
                "external_fill_id": external_fill_id,
                "price": _require_finite("price", price),
                "quantity": _require_finite("quantity", quantity),
                "fee": _require_finite("fee", fee),
                "fee_asset": fee_asset,
                "realized_pnl": (
                    _require_finite("realized_pnl", realized_pnl)
                    if realized_pnl is not None
                    else None
                ),
                "raw_response_json": _json(dict(raw_response or {})),
                "filled_at": _utc_iso(filled_at),
                "created_at": _utc_iso(created_at),
            }
            columns = tuple(values)
            query = self._sql(
                f"INSERT INTO venue_fills ({', '.join(columns)}) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in columns)})"
            )
            connection.execute(query, tuple(values[column] for column in columns))
        return venue_fill_id

    def append_venue_accounting_event(
        self,
        *,
        venue: str,
        external_income_id: str,
        income_type: str,
        asset: str,
        amount: float,
        transaction_time: datetime | str,
        symbol: str | None = None,
        trace_id: str | None = None,
        venue_order_id: str | None = None,
        trade_id: str | int | None = None,
        raw_response: Mapping[str, Any] | None = None,
        accounting_event_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        """Append a signed venue cash-flow observation exactly once.

        Positive ``amount`` means cash received and negative means cash paid. Binance
        ``tranId``/trade-derived identifiers are scoped by venue and are used as the
        immutable idempotency key.
        """

        venue = venue.strip()
        external_income_id = external_income_id.strip()
        income_type = income_type.strip().upper()
        asset = asset.strip().upper()
        normalized_symbol = symbol.strip().upper() if symbol else None
        normalized_amount = _require_finite("amount", amount)
        normalized_time = _utc_iso(transaction_time)
        normalized_trade_id = str(trade_id) if trade_id is not None else None
        if not venue or not external_income_id or not income_type or not asset:
            raise ValueError(
                "venue, external_income_id, income_type, and asset must be non-empty"
            )
        accounting_event_id = accounting_event_id or _new_id("accounting")
        with self.connect() as connection:
            if self.dialect == "sqlite":
                connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                self._sql(
                    """
                    SELECT * FROM venue_accounting_events
                    WHERE venue=? AND external_income_id=?
                    """
                ),
                (venue, external_income_id),
            ).fetchone()
            if existing is not None:
                item = self._decode_row(existing)
                immutable = {
                    "symbol": normalized_symbol,
                    "income_type": income_type,
                    "asset": asset,
                    "amount": normalized_amount,
                    "transaction_time": normalized_time,
                    "trade_id": normalized_trade_id,
                }
                for field, expected in immutable.items():
                    observed = item.get(field)
                    if field == "amount":
                        matches = float(observed) == float(expected)
                    else:
                        matches = observed == expected
                    if not matches:
                        raise ValueError(
                            "Conflicting accounting replay for immutable venue income "
                            f"{venue}:{external_income_id}: {field}"
                        )
                return str(_row_value(existing, "accounting_event_id", 0))
            if venue_order_id is not None:
                row = connection.execute(
                    self._sql(
                        "SELECT trace_id FROM venue_orders WHERE venue_order_id=?"
                    ),
                    (venue_order_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        f"Referenced audit record does not exist: {venue_order_id}"
                    )
                order_trace = str(_row_value(row, "trace_id", 0))
                if trace_id is None:
                    trace_id = order_trace
                elif trace_id != order_trace:
                    raise ValueError(
                        f"Referenced record {venue_order_id} belongs to a different trace"
                    )
            values = {
                "accounting_event_id": accounting_event_id,
                "trace_id": trace_id,
                "venue_order_id": venue_order_id,
                "venue": venue,
                "external_income_id": external_income_id,
                "symbol": normalized_symbol,
                "income_type": income_type,
                "asset": asset,
                "amount": normalized_amount,
                "transaction_time": normalized_time,
                "trade_id": normalized_trade_id,
                "raw_response_json": _json(dict(raw_response or {})),
                "created_at": _utc_iso(created_at),
            }
            columns = tuple(values)
            query = self._sql(
                f"INSERT INTO venue_accounting_events ({', '.join(columns)}) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in columns)})"
                + (
                    " ON CONFLICT (venue, external_income_id) DO NOTHING "
                    "RETURNING accounting_event_id"
                    if self.dialect == "postgres"
                    else ""
                )
            )
            inserted = connection.execute(
                query, tuple(values[column] for column in columns)
            )
            if self.dialect == "postgres":  # pragma: no cover - external service
                row = inserted.fetchone()
                if row is None:
                    row = connection.execute(
                        self._sql(
                            """
                            SELECT * FROM venue_accounting_events
                            WHERE venue=? AND external_income_id=?
                            """
                        ),
                        (venue, external_income_id),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("Accounting idempotency conflict could not be resolved")
                    item = self._decode_row(row)
                    immutable = {
                        "symbol": normalized_symbol,
                        "income_type": income_type,
                        "asset": asset,
                        "amount": normalized_amount,
                        "transaction_time": normalized_time,
                        "trade_id": normalized_trade_id,
                    }
                    if any(
                        (
                            float(item[field]) != float(expected)
                            if field == "amount"
                            else item[field] != expected
                        )
                        for field, expected in immutable.items()
                    ):
                        raise ValueError(
                            "Conflicting accounting replay for immutable venue income "
                            f"{venue}:{external_income_id}"
                        )
                    return str(_row_value(row, "accounting_event_id", 0))
        return accounting_event_id

    def append_venue_accounting_attribution(
        self,
        *,
        accounting_event_id: str,
        status: str,
        reason: str,
        resolved_at: datetime | str,
        trace_id: str | None = None,
        venue_order_id: str | None = None,
        attribution_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        """Append an idempotent attribution observation; later corrections stay append-only."""

        status = status.strip().upper()
        reason = reason.strip()
        if status not in {"ATTRIBUTED", "UNATTRIBUTED"}:
            raise ValueError("status must be ATTRIBUTED or UNATTRIBUTED")
        if not reason:
            raise ValueError("attribution reason must be non-empty")
        if status == "ATTRIBUTED" and (not trace_id or not venue_order_id):
            raise ValueError("ATTRIBUTED accounting requires a trace and venue order")
        if status == "UNATTRIBUTED" and (trace_id is not None or venue_order_id is not None):
            raise ValueError("UNATTRIBUTED accounting cannot claim a trace or venue order")
        fingerprint = _json(
            {
                "accounting_event_id": accounting_event_id,
                "status": status,
                "reason": reason,
                "trace_id": trace_id,
                "venue_order_id": venue_order_id,
            }
        )
        attribution_id = attribution_id or (
            "attribution_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
        )
        with self.connect() as connection:
            if self.dialect == "sqlite":
                connection.execute("BEGIN IMMEDIATE")
            event = connection.execute(
                self._sql(
                    "SELECT symbol FROM venue_accounting_events WHERE accounting_event_id=?"
                    + (" FOR UPDATE" if self.dialect == "postgres" else "")
                ),
                (accounting_event_id,),
            ).fetchone()
            if event is None:
                raise ValueError(f"Unknown venue accounting event: {accounting_event_id}")
            if status == "ATTRIBUTED":
                owner = connection.execute(
                    self._sql(
                        "SELECT trace_id, symbol FROM venue_orders WHERE venue_order_id=?"
                    ),
                    (venue_order_id,),
                ).fetchone()
                if owner is None:
                    raise ValueError(f"Unknown venue order: {venue_order_id}")
                owner_trace = str(_row_value(owner, "trace_id", 0))
                owner_symbol = str(_row_value(owner, "symbol", 1)).upper()
                event_symbol = str(_row_value(event, "symbol", 0) or "").upper()
                if owner_trace != trace_id or not event_symbol or owner_symbol != event_symbol:
                    raise ValueError("Accounting attribution does not match its audited owner")
            existing = connection.execute(
                self._sql(
                    "SELECT attribution_id FROM venue_accounting_attributions "
                    "WHERE attribution_id=?"
                ),
                (attribution_id,),
            ).fetchone()
            if existing is not None:
                return str(_row_value(existing, "attribution_id", 0))
            values = {
                "attribution_id": attribution_id,
                "accounting_event_id": accounting_event_id,
                "trace_id": trace_id,
                "venue_order_id": venue_order_id,
                "status": status,
                "reason": reason,
                "resolved_at": _utc_iso(resolved_at),
                "created_at": _utc_iso(created_at),
            }
            columns = tuple(values)
            connection.execute(
                self._sql(
                    f"INSERT INTO venue_accounting_attributions "  # noqa: S608
                    f"({', '.join(columns)}) VALUES "
                    f"({', '.join('?' for _ in columns)})"
                ),
                tuple(values[column] for column in columns),
            )
        return attribution_id

    def append_venue_fee_conversion(
        self,
        *,
        trace_id: str,
        venue_fill_id: str,
        quote_asset: str,
        rate: float,
        effective_at: datetime | str,
        source: str,
        source_record_id: str,
        conversion_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        """Record an exact fill-time fee conversion without guessing a cross-asset value."""

        quote_asset = quote_asset.strip().upper()
        source = source.strip()
        source_record_id = source_record_id.strip()
        normalized_rate = _require_finite("rate", rate)
        if not quote_asset or not source or not source_record_id or normalized_rate <= 0:
            raise ValueError("fee conversion fields must be non-empty and rate must be positive")
        normalized_effective_at = _utc_iso(effective_at)
        conversion_id = conversion_id or _new_id("fee_conversion")
        with self.connect() as connection:
            if self.dialect == "sqlite":
                connection.execute("BEGIN IMMEDIATE")
            fill = connection.execute(
                self._sql(
                    "SELECT trace_id, fee_asset, filled_at FROM venue_fills "
                    "WHERE venue_fill_id=?"
                    + (" FOR UPDATE" if self.dialect == "postgres" else "")
                ),
                (venue_fill_id,),
            ).fetchone()
            if fill is None:
                raise ValueError(f"Unknown venue fill: {venue_fill_id}")
            fill_trace = str(_row_value(fill, "trace_id", 0))
            from_asset = str(_row_value(fill, "fee_asset", 1)).upper()
            filled_at = _utc_iso(str(_row_value(fill, "filled_at", 2)))
            if fill_trace != trace_id:
                raise ValueError("Fee conversion belongs to a different trace")
            if from_asset == quote_asset:
                raise ValueError("Same-asset fees do not require a conversion record")
            if normalized_effective_at != filled_at:
                raise ValueError("Fee conversion must be effective at the exact fill timestamp")
            existing = connection.execute(
                self._sql(
                    "SELECT * FROM venue_fee_conversions "
                    "WHERE venue_fill_id=? AND quote_asset=?"
                ),
                (venue_fill_id, quote_asset),
            ).fetchone()
            if existing is not None:
                item = self._decode_row(existing)
                if (
                    item["from_asset"] != from_asset
                    or float(item["rate"]) != normalized_rate
                    or item["effective_at"] != normalized_effective_at
                    or item["source"] != source
                    or item["source_record_id"] != source_record_id
                ):
                    raise ValueError("Conflicting point-in-time fee conversion replay")
                return str(item["conversion_id"])
            values = {
                "conversion_id": conversion_id,
                "trace_id": trace_id,
                "venue_fill_id": venue_fill_id,
                "from_asset": from_asset,
                "quote_asset": quote_asset,
                "rate": normalized_rate,
                "effective_at": normalized_effective_at,
                "source": source,
                "source_record_id": source_record_id,
                "created_at": _utc_iso(created_at),
            }
            columns = tuple(values)
            connection.execute(
                self._sql(
                    f"INSERT INTO venue_fee_conversions "  # noqa: S608
                    f"({', '.join(columns)}) VALUES "
                    f"({', '.join('?' for _ in columns)})"
                ),
                tuple(values[column] for column in columns),
            )
        return conversion_id

    def append_account_snapshot(
        self,
        *,
        equity: float,
        cash: float,
        gross_exposure: float,
        net_exposure: float,
        daily_pnl: float,
        drawdown: float,
        positions: Sequence[Mapping[str, Any]],
        source: str,
        observed_at: datetime | str,
        trace_id: str | None = None,
        snapshot_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        snapshot_id = snapshot_id or _new_id("account")
        return self._insert(
            "account_snapshots",
            {
                "snapshot_id": snapshot_id,
                "trace_id": trace_id,
                "equity": _require_finite("equity", equity),
                "cash": _require_finite("cash", cash),
                "gross_exposure": _require_finite("gross_exposure", gross_exposure),
                "net_exposure": _require_finite("net_exposure", net_exposure),
                "daily_pnl": _require_finite("daily_pnl", daily_pnl),
                "drawdown": _optional_probability("drawdown", drawdown),
                "positions_json": _json([dict(position) for position in positions]),
                "source": source,
                "observed_at": _utc_iso(observed_at),
                "created_at": _utc_iso(created_at),
            },
        )

    def append_strategy_spec(
        self,
        *,
        strategy_version: str,
        status: str,
        parameters: Mapping[str, Any],
        prompt_version: str,
        parent_version: str | None = None,
        source_response_id: str | None = None,
        trace_id: str | None = None,
        spec_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        _validate_strategy_parameters(parameters)
        if not strategy_version.strip() or not prompt_version.strip():
            raise ValueError("strategy_version and prompt_version must be non-empty")
        status = status.upper()
        if status not in {"CHAMPION", "CHALLENGER", "RETIRED", "REJECTED"}:
            raise ValueError(f"Unsupported strategy status: {status}")
        spec_id = spec_id or _new_id("spec")
        return self._insert(
            "strategy_specs",
            {
                "spec_id": spec_id,
                "trace_id": trace_id,
                "strategy_version": strategy_version,
                "parent_version": parent_version,
                "status": status,
                "parameters_json": _json(dict(parameters)),
                "prompt_version": prompt_version,
                "source_response_id": source_response_id,
                "created_at": _utc_iso(created_at),
            },
        )

    def append_strategy_research_run(
        self,
        *,
        trace_id: str,
        strategy_version: str,
        parent_version: str,
        recommendation: str,
        model: str,
        prompt_version: str,
        response_id: str,
        latency_ms: int,
        evidence_ids: Sequence[str],
        sources: Sequence[Mapping[str, Any]],
        hypothesis: str,
        rationale: Sequence[str],
        expected_failure_modes: Sequence[str],
        research_run_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        recommendation = recommendation.strip().upper()
        if recommendation not in {"PROPOSE", "NO_CHANGE"}:
            raise ValueError("Unsupported strategy research recommendation")
        required = {
            "trace_id": trace_id,
            "strategy_version": strategy_version,
            "parent_version": parent_version,
            "model": model,
            "prompt_version": prompt_version,
            "response_id": response_id,
            "hypothesis": hypothesis,
        }
        if any(not value.strip() for value in required.values()):
            raise ValueError("strategy research audit fields must be non-empty")
        if isinstance(latency_ms, bool) or not isinstance(latency_ms, int) or latency_ms < 0:
            raise ValueError("strategy research latency_ms must be a non-negative integer")
        normalized_evidence = tuple(str(value).strip() for value in evidence_ids)
        if not normalized_evidence or any(not value for value in normalized_evidence):
            raise ValueError("strategy research requires non-empty evidence IDs")
        if len(set(normalized_evidence)) != len(normalized_evidence):
            raise ValueError("strategy research evidence IDs must be unique")
        research_run_id = research_run_id or _new_id("research")
        return self._insert(
            "strategy_research_runs",
            {
                "research_run_id": research_run_id,
                "trace_id": trace_id,
                "strategy_version": strategy_version,
                "parent_version": parent_version,
                "recommendation": recommendation,
                "model": model,
                "prompt_version": prompt_version,
                "response_id": response_id,
                "latency_ms": latency_ms,
                "evidence_ids_json": _json(normalized_evidence),
                "sources_json": _json([dict(item) for item in sources]),
                "hypothesis": hypothesis,
                "rationale_json": _json(tuple(rationale)),
                "expected_failure_modes_json": _json(
                    tuple(expected_failure_modes)
                ),
                "created_at": _utc_iso(created_at),
            },
        )

    def append_backtest_run(
        self,
        *,
        spec_id: str,
        started_at: datetime | str,
        ended_at: datetime | str,
        completed: bool,
        validation: Mapping[str, Any],
        raw_metrics: Mapping[str, Any] | None = None,
        net_profit: float | None = None,
        net_return: float | None = None,
        max_drawdown: float | None = None,
        total_cost: float | None = None,
        stressed_net_return_2x: float | None = None,
        dsr_significance_probability: float | None = None,
        pbo_probability: float | None = None,
        symbol_concentration: float | None = None,
        month_concentration: float | None = None,
        trade_count: int | None = None,
        holdout_months: int | None = None,
        trace_id: str | None = None,
        backtest_run_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        backtest_run_id = backtest_run_id or _new_id("backtest")
        if not isinstance(completed, bool):
            raise ValueError("completed must be bool")
        started_timestamp = _utc_iso(started_at)
        ended_timestamp = _utc_iso(ended_at)
        if _as_utc_datetime(ended_timestamp) <= _as_utc_datetime(started_timestamp):
            raise ValueError("backtest ended_at must be after started_at")
        return self._insert(
            "backtest_runs",
            {
                "backtest_run_id": backtest_run_id,
                "trace_id": trace_id,
                "spec_id": spec_id,
                "started_at": started_timestamp,
                "ended_at": ended_timestamp,
                "completed": int(completed),
                "net_profit": _optional_finite("net_profit", net_profit),
                "net_return": _optional_finite("net_return", net_return),
                "max_drawdown": _optional_probability("max_drawdown", max_drawdown),
                "total_cost": _optional_finite("total_cost", total_cost),
                "stressed_net_return_2x": _optional_finite(
                    "stressed_net_return_2x", stressed_net_return_2x
                ),
                "dsr_significance_probability": _optional_probability(
                    "dsr_significance_probability", dsr_significance_probability
                ),
                "pbo_probability": _optional_probability("pbo_probability", pbo_probability),
                "symbol_concentration": _optional_probability(
                    "symbol_concentration", symbol_concentration
                ),
                "month_concentration": _optional_probability(
                    "month_concentration", month_concentration
                ),
                "trade_count": _optional_count("trade_count", trade_count),
                "holdout_months": _optional_count("holdout_months", holdout_months),
                "validation_json": _json(dict(validation)),
                "raw_metrics_json": _json(dict(raw_metrics or {})),
                "created_at": _utc_iso(created_at),
            },
        )

    def append_shadow_result(
        self,
        *,
        spec_id: str,
        started_at: datetime | str,
        ended_at: datetime | str,
        completed: bool,
        raw_metrics: Mapping[str, Any] | None = None,
        elapsed_days: int | None = None,
        closed_trades: int | None = None,
        net_return: float | None = None,
        max_drawdown: float | None = None,
        stressed_net_return_2x: float | None = None,
        symbol_concentration: float | None = None,
        month_concentration: float | None = None,
        trace_id: str | None = None,
        shadow_result_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        shadow_result_id = shadow_result_id or _new_id("shadow")
        if not isinstance(completed, bool):
            raise ValueError("completed must be bool")
        started_timestamp = _utc_iso(started_at)
        ended_timestamp = _utc_iso(ended_at)
        if _as_utc_datetime(ended_timestamp) <= _as_utc_datetime(started_timestamp):
            raise ValueError("shadow ended_at must be after started_at")
        validated_days = _optional_count("elapsed_days", elapsed_days)
        derived_days = int(
            (
                _as_utc_datetime(ended_timestamp) - _as_utc_datetime(started_timestamp)
            ).total_seconds()
            // 86400
        )
        if validated_days is not None and validated_days != derived_days:
            raise ValueError("elapsed_days does not match the shadow timestamps")
        return self._insert(
            "shadow_results",
            {
                "shadow_result_id": shadow_result_id,
                "trace_id": trace_id,
                "spec_id": spec_id,
                "started_at": started_timestamp,
                "ended_at": ended_timestamp,
                "completed": int(completed),
                "elapsed_days": validated_days,
                "closed_trades": _optional_count("closed_trades", closed_trades),
                "net_return": _optional_finite("net_return", net_return),
                "max_drawdown": _optional_probability("max_drawdown", max_drawdown),
                "stressed_net_return_2x": _optional_finite(
                    "stressed_net_return_2x", stressed_net_return_2x
                ),
                "symbol_concentration": _optional_probability(
                    "symbol_concentration", symbol_concentration
                ),
                "month_concentration": _optional_probability(
                    "month_concentration", month_concentration
                ),
                "raw_metrics_json": _json(dict(raw_metrics or {})),
                "created_at": _utc_iso(created_at),
            },
        )

    def append_promotion_record(
        self,
        *,
        trace_id: str,
        champion_spec_id: str,
        challenger_spec_id: str,
        backtest_run_id: str,
        champion_shadow_result_id: str,
        challenger_shadow_result_id: str,
        eligible: bool,
        reason_codes: Sequence[str],
        evaluation: Mapping[str, Any],
        promotion_policy: PromotionPolicy | None = None,
        promotion_record_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        from .learning import (
            BacktestEvidence,
            PerformanceMetrics,
            ShadowEvidence,
            evaluate_promotion,
        )

        promotion_record_id = promotion_record_id or _new_id("promotion")
        if champion_spec_id == challenger_spec_id:
            raise ValueError("Champion and challenger specs must be distinct")
        created_timestamp = _utc_iso(created_at)
        with self.connect() as connection:
            backtest = connection.execute(
                self._sql("SELECT * FROM backtest_runs WHERE backtest_run_id=?"),
                (backtest_run_id,),
            ).fetchone()
            champion_shadow = connection.execute(
                self._sql("SELECT * FROM shadow_results WHERE shadow_result_id=?"),
                (champion_shadow_result_id,),
            ).fetchone()
            challenger_shadow = connection.execute(
                self._sql("SELECT * FROM shadow_results WHERE shadow_result_id=?"),
                (challenger_shadow_result_id,),
            ).fetchone()
            champion_spec = connection.execute(
                self._sql("SELECT * FROM strategy_specs WHERE spec_id=?"),
                (champion_spec_id,),
            ).fetchone()
            challenger_spec = connection.execute(
                self._sql("SELECT * FROM strategy_specs WHERE spec_id=?"),
                (challenger_spec_id,),
            ).fetchone()
            evidence_rows = (
                backtest,
                champion_shadow,
                challenger_shadow,
                champion_spec,
                challenger_spec,
            )
            if any(row is None for row in evidence_rows):
                raise ValueError("Promotion evidence record does not exist")
            assert all(row is not None for row in evidence_rows)
            if str(_row_value(backtest, "spec_id", 0)) != challenger_spec_id:
                raise ValueError("Backtest does not belong to the challenger strategy")
            if str(_row_value(champion_shadow, "spec_id", 0)) != champion_spec_id:
                raise ValueError("Champion shadow result does not belong to the champion")
            if str(_row_value(challenger_shadow, "spec_id", 0)) != challenger_spec_id:
                raise ValueError("Challenger shadow result does not belong to the challenger")
            if _row_value(champion_shadow, "started_at", 3) != _row_value(
                challenger_shadow, "started_at", 3
            ) or _row_value(champion_shadow, "ended_at", 4) != _row_value(
                challenger_shadow, "ended_at", 4
            ):
                raise ValueError("Champion and challenger shadow intervals must match")
            champion_status = str(_row_value(champion_spec, "status", 4))
            if champion_status != "CHAMPION":
                promoted_row = connection.execute(
                    self._sql(
                        """
                        SELECT promotion_record_id FROM promotion_records
                        WHERE challenger_spec_id=? AND eligible=1
                        ORDER BY created_at DESC LIMIT 1
                        """
                    ),
                    (champion_spec_id,),
                ).fetchone()
                if champion_status != "CHALLENGER" or promoted_row is None:
                    raise ValueError(
                        "champion_spec_id is neither the initial CHAMPION nor an audited "
                        "promoted challenger"
                    )
            if str(_row_value(challenger_spec, "status", 4)) != "CHALLENGER":
                raise ValueError("challenger_spec_id does not reference a CHALLENGER spec")
            champion_version = str(_row_value(champion_spec, "strategy_version", 2))
            challenger_parent = _row_value(challenger_spec, "parent_version", 3)
            if challenger_parent != champion_version:
                raise ValueError("Challenger parent_version must reference the champion")
            if any(str(_row_value(row, "trace_id", 1)) != trace_id for row in evidence_rows):
                raise ValueError("Promotion evidence must belong to the promotion trace")

        backtest_row = self._decode_row(backtest)
        champion_shadow_row = self._decode_row(champion_shadow)
        challenger_shadow_row = self._decode_row(challenger_shadow)

        def metrics(row: Mapping[str, Any]) -> PerformanceMetrics:
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

        validation = backtest_row["validation"]
        backtest_evidence = BacktestEvidence(
            metrics=metrics(backtest_row),
            completed=backtest_row["completed"],
            dsr_significance_probability=backtest_row["dsr_significance_probability"],
            pbo_probability=backtest_row["pbo_probability"],
            holdout_months=backtest_row["holdout_months"],
            walk_forward_passed=validation.get("walk_forward_passed"),
            holdout_passed=validation.get("holdout_passed"),
            parameter_perturbation_passed=validation.get("parameter_perturbation_passed"),
            latency_stress_passed=validation.get("latency_stress_passed"),
            social_placebo_passed=validation.get("social_placebo_passed"),
        )
        champion_evidence = ShadowEvidence(
            metrics=metrics(champion_shadow_row),
            completed=champion_shadow_row["completed"],
            elapsed_days=champion_shadow_row["elapsed_days"],
            closed_trades=champion_shadow_row["closed_trades"],
        )
        challenger_evidence = ShadowEvidence(
            metrics=metrics(challenger_shadow_row),
            completed=challenger_shadow_row["completed"],
            elapsed_days=challenger_shadow_row["elapsed_days"],
            closed_trades=challenger_shadow_row["closed_trades"],
        )
        gate = evaluate_promotion(
            champion_shadow=champion_evidence,
            challenger_backtest=backtest_evidence,
            challenger_shadow=challenger_evidence,
            policy=promotion_policy,
            evaluated_at=datetime.fromisoformat(created_timestamp.replace("Z", "+00:00")),
        )
        if bool(eligible) != gate.eligible:
            raise ValueError("Promotion eligibility does not match the deterministic gate")
        if tuple(reason_codes) != gate.reason_codes:
            raise ValueError("Promotion reason codes do not match the deterministic gate")
        return self._insert(
            "promotion_records",
            {
                "promotion_record_id": promotion_record_id,
                "trace_id": trace_id,
                "champion_spec_id": champion_spec_id,
                "challenger_spec_id": challenger_spec_id,
                "backtest_run_id": backtest_run_id,
                "champion_shadow_result_id": champion_shadow_result_id,
                "challenger_shadow_result_id": challenger_shadow_result_id,
                "eligible": int(gate.eligible),
                "reason_codes_json": _json(list(gate.reason_codes)),
                "evaluation_json": _json(
                    {"gate": gate.as_dict(), "caller_context": dict(evaluation)}
                ),
                "created_at": created_timestamp,
            },
        )

    def append_strategy_registry_rollback(
        self,
        *,
        trace_id: str,
        source_promotion_record_id: str,
        previous_champion_version: str,
        resulting_champion_version: str,
        reason: str,
        registry_event_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        """Audit a rollback before mutating the file-backed registry.

        The source promotion uniquely identifies the champion transition being reversed.  A
        retry after an audit-commit/process-crash window returns the same immutable event.
        """

        values = (
            trace_id.strip(),
            source_promotion_record_id.strip(),
            previous_champion_version.strip(),
            resulting_champion_version.strip(),
            reason.strip(),
        )
        if any(not value for value in values):
            raise ValueError("strategy registry rollback fields must be non-empty")
        if previous_champion_version == resulting_champion_version:
            raise ValueError("rollback must change the champion version")
        with self.connect() as connection:
            source = connection.execute(
                self._sql(
                    """
                    SELECT p.trace_id, p.eligible,
                           challenger.strategy_version AS challenger_version,
                           champion.strategy_version AS champion_version
                    FROM promotion_records p
                    JOIN strategy_specs challenger
                      ON challenger.spec_id=p.challenger_spec_id
                    JOIN strategy_specs champion
                      ON champion.spec_id=p.champion_spec_id
                    WHERE p.promotion_record_id=?
                    """
                ),
                (source_promotion_record_id,),
            ).fetchone()
            if source is None:
                raise ValueError("rollback source promotion does not exist")
            decoded_source = dict(source)
            if (
                str(decoded_source["trace_id"]) != trace_id
                or not bool(decoded_source["eligible"])
                or str(decoded_source["challenger_version"])
                != previous_champion_version
                or str(decoded_source["champion_version"])
                != resulting_champion_version
            ):
                raise ValueError("rollback does not reverse the audited eligible promotion")
            existing = connection.execute(
                self._sql(
                    "SELECT * FROM strategy_registry_events "
                    "WHERE source_promotion_record_id=?"
                ),
                (source_promotion_record_id,),
            ).fetchone()
        if existing is not None:
            decoded = self._decode_row(existing)
            expected = {
                "trace_id": trace_id,
                "event_type": "ROLLBACK",
                "previous_champion_version": previous_champion_version,
                "resulting_champion_version": resulting_champion_version,
                "reason": reason,
            }
            if any(decoded.get(key) != value for key, value in expected.items()):
                raise ValueError("audited rollback source was already used differently")
            return str(decoded["registry_event_id"])
        return self._insert(
            "strategy_registry_events",
            {
                "registry_event_id": registry_event_id or _new_id("registry"),
                "trace_id": trace_id,
                "event_type": "ROLLBACK",
                "source_promotion_record_id": source_promotion_record_id,
                "previous_champion_version": previous_champion_version,
                "resulting_champion_version": resulting_champion_version,
                "reason": reason,
                "created_at": _utc_iso(created_at),
            },
        )

    def append_shadow_journal_event(
        self,
        *,
        trace_id: str,
        pair_id: str,
        champion_spec_id: str,
        challenger_spec_id: str,
        pair_started_at: datetime | str,
        event_type: str,
        event_key: str,
        payload: Mapping[str, Any],
        observed_at: datetime | str,
        spec_id: str | None = None,
        shadow_journal_event_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        """Append one durable paired-shadow coverage or trade fact idempotently."""

        event_type = event_type.strip().upper()
        required = (
            trace_id.strip(),
            pair_id.strip(),
            champion_spec_id.strip(),
            challenger_spec_id.strip(),
            event_key.strip(),
        )
        if any(not value for value in required):
            raise ValueError("shadow journal identity fields must be non-empty")
        if champion_spec_id == challenger_spec_id:
            raise ValueError("shadow journal requires distinct strategy specs")
        if event_type not in {"COVERAGE", "TRADE"}:
            raise ValueError("unsupported shadow journal event type")
        if (event_type == "COVERAGE") != (spec_id is None):
            raise ValueError("only TRADE shadow events may carry spec_id")
        payload_json = _json(dict(payload))
        with self.connect() as connection:
            specs = connection.execute(
                self._sql(
                    "SELECT spec_id, trace_id FROM strategy_specs WHERE spec_id IN (?,?)"
                ),
                (champion_spec_id, challenger_spec_id),
            ).fetchall()
            if len(specs) != 2 or any(
                str(_row_value(row, "trace_id", 1)) != trace_id for row in specs
            ):
                raise ValueError("shadow journal specs must exist in the same audit trace")
            if spec_id is not None and spec_id not in {
                champion_spec_id,
                challenger_spec_id,
            }:
                raise ValueError("shadow trade spec is outside the paired strategies")
            existing = connection.execute(
                self._sql(
                    "SELECT * FROM shadow_journal_events "
                    "WHERE pair_id=? AND event_type=? AND event_key=?"
                ),
                (pair_id, event_type, event_key),
            ).fetchone()
        if existing is not None:
            decoded = self._decode_row(existing)
            if (
                decoded.get("trace_id") != trace_id
                or decoded.get("champion_spec_id") != champion_spec_id
                or decoded.get("challenger_spec_id") != challenger_spec_id
                or decoded.get("spec_id") != spec_id
                or decoded.get("pair_started_at") != _utc_iso(pair_started_at)
                or decoded.get("observed_at") != _utc_iso(observed_at)
                or _json(decoded.get("payload")) != payload_json
            ):
                raise ValueError("shadow journal identity was already used differently")
            return str(decoded["shadow_journal_event_id"])
        return self._insert(
            "shadow_journal_events",
            {
                "shadow_journal_event_id": shadow_journal_event_id
                or _new_id("shadow_event"),
                "trace_id": trace_id,
                "pair_id": pair_id,
                "champion_spec_id": champion_spec_id,
                "challenger_spec_id": challenger_spec_id,
                "pair_started_at": _utc_iso(pair_started_at),
                "event_type": event_type,
                "event_key": event_key,
                "spec_id": spec_id,
                "payload_json": payload_json,
                "observed_at": _utc_iso(observed_at),
                "created_at": _utc_iso(created_at),
            },
        )

    def shadow_journal_events(self, pair_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    "SELECT * FROM shadow_journal_events WHERE pair_id=? "
                    "ORDER BY observed_at, created_at, shadow_journal_event_id"
                ),
                (pair_id,),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def append_counterfactual_outcome(
        self,
        *,
        trace_id: str,
        candidate_id: str,
        horizon_hours: int,
        realized_return: float,
        observed_at: datetime | str,
        decision_id: str | None = None,
        decision_regret: float | None = None,
        confidence_calibration_error: float | None = None,
        source_reliability: Mapping[str, float] | None = None,
        outcome_id: str | None = None,
        created_at: datetime | str | None = None,
    ) -> str:
        if horizon_hours not in {1, 4, 24}:
            raise ValueError("horizon_hours must be 1, 4, or 24")
        outcome_id = outcome_id or _new_id("counterfactual")
        with self.connect() as connection:
            self._assert_trace(
                connection,
                "SELECT trace_id FROM trade_candidates WHERE candidate_id=?",
                candidate_id,
                trace_id,
                dialect=self.dialect,
            )
            if decision_id is not None:
                self._assert_trace(
                    connection,
                    "SELECT trace_id FROM llm_decisions WHERE decision_id=?",
                    decision_id,
                    trace_id,
                    dialect=self.dialect,
                )
                row = connection.execute(
                    self._sql("SELECT candidate_id FROM llm_decisions WHERE decision_id=?"),
                    (decision_id,),
                ).fetchone()
                assert row is not None
                if str(_row_value(row, "candidate_id", 0)) != candidate_id:
                    raise ValueError("Counterfactual candidate does not match the LLM decision")
        return self._insert(
            "counterfactual_outcomes",
            {
                "outcome_id": outcome_id,
                "trace_id": trace_id,
                "candidate_id": candidate_id,
                "decision_id": decision_id,
                "horizon_hours": horizon_hours,
                "realized_return": _require_finite("realized_return", realized_return),
                "decision_regret": _optional_finite("decision_regret", decision_regret),
                "confidence_calibration_error": _optional_probability(
                    "confidence_calibration_error", confidence_calibration_error
                ),
                "source_reliability_json": _json(dict(source_reliability or {})),
                "observed_at": _utc_iso(observed_at),
                "created_at": _utc_iso(created_at),
            },
        )

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {"trace_id": trace_id}
        with self.connect() as connection:
            for table in TRACE_TABLES:
                rows = connection.execute(
                    self._sql(
                        f"SELECT * FROM {table} WHERE trace_id=? ORDER BY created_at, rowid"  # noqa: S608
                        if self.dialect == "sqlite"
                        else f"SELECT * FROM {table} WHERE trace_id=? ORDER BY created_at"  # noqa: S608
                    ),
                    (trace_id,),
                ).fetchall()
                result[table] = [self._decode_row(row) for row in rows]
            # A funding event first observed as UNATTRIBUTED remains global forever. If a later
            # append-only attribution safely resolves it, expose the immutable event through the
            # owning trace without rewriting the original row.
            linked_accounting_rows = connection.execute(
                self._sql(
                    """
                    SELECT DISTINCT e.* FROM venue_accounting_events e
                    JOIN venue_accounting_attributions a
                      ON a.accounting_event_id=e.accounting_event_id
                    WHERE a.trace_id=? AND e.trace_id IS NULL
                    ORDER BY e.created_at, e.accounting_event_id
                    """
                ),
                (trace_id,),
            ).fetchall()
            existing_accounting_ids = {
                str(item["accounting_event_id"])
                for item in result["venue_accounting_events"]
            }
            result["venue_accounting_events"].extend(
                self._decode_row(row)
                for row in linked_accounting_rows
                if str(_row_value(row, "accounting_event_id", 0))
                not in existing_accounting_ids
            )
            candidate_ids = [candidate["candidate_id"] for candidate in result["trade_candidates"]]
            if candidate_ids:
                placeholders = ", ".join("?" for _ in candidate_ids)
                rows = connection.execute(
                    self._sql(
                        f"""
                        SELECT e.*, l.evidence_link_id, l.candidate_id AS linked_candidate_id,
                               l.role,
                               l.trace_id AS link_trace_id,
                               l.created_at AS linked_at
                        FROM candidate_evidence_links l
                        JOIN external_evidence e
                          ON e.evidence_record_id=l.evidence_record_id
                        WHERE l.candidate_id IN ({placeholders})
                        ORDER BY l.created_at, e.evidence_id
                        """  # noqa: S608
                    ),
                    tuple(candidate_ids),
                ).fetchall()
                result["linked_external_evidence"] = [self._decode_row(row) for row in rows]
            else:
                result["linked_external_evidence"] = []
        return result

    def latest_external_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql(
                    """
                    SELECT * FROM external_evidence WHERE evidence_id=?
                    ORDER BY version DESC LIMIT 1
                    """
                ),
                (evidence_id,),
            ).fetchone()
        return self._decode_row(row) if row is not None else None

    def latest_external_evidence_batch(self, *, limit: int = 1_000) -> list[dict[str, Any]]:
        """Return latest immutable source versions for outbox recovery."""

        if not 1 <= limit <= 10_000:
            raise ValueError("external evidence batch limit must be between 1 and 10000")
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    """
                    SELECT evidence.*
                    FROM external_evidence evidence
                    WHERE NOT EXISTS (
                        SELECT 1 FROM external_evidence newer
                        WHERE newer.evidence_id=evidence.evidence_id
                          AND newer.version>evidence.version
                    )
                    ORDER BY evidence.created_at DESC
                    LIMIT ?
                    """
                ),
                (limit,),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def evidence_for_candidate(self, candidate_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    """
                    SELECT e.*, l.evidence_link_id, l.candidate_id AS linked_candidate_id,
                           l.role,
                           l.trace_id AS link_trace_id,
                           l.created_at AS linked_at
                    FROM candidate_evidence_links l
                    JOIN external_evidence e
                      ON e.evidence_record_id=l.evidence_record_id
                    WHERE l.candidate_id=?
                    ORDER BY l.created_at, e.evidence_id
                    """
                ),
                (candidate_id,),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def trace_for_order(self, venue_order_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                self._sql("SELECT trace_id FROM venue_orders WHERE venue_order_id=?"),
                (venue_order_id,),
            ).fetchone()
        if row is None:
            raise LookupError(f"Unknown venue order: {venue_order_id}")
        trace_id = str(row["trace_id"] if isinstance(row, (sqlite3.Row, dict)) else row[0])
        return self.get_trace(trace_id)

    def latest_order_event(self, venue_order_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                self._sql(
                    """
                    SELECT * FROM venue_order_events WHERE venue_order_id=?
                    ORDER BY event_sequence DESC LIMIT 1
                    """
                ),
                (venue_order_id,),
            ).fetchone()
        if row is None:
            raise LookupError(f"Unknown venue order or missing event history: {venue_order_id}")
        return self._decode_row(row)

    def resolve_venue_order_client_id(
        self, client_order_id: str, *, venue: str | None = None
    ) -> dict[str, Any] | None:
        """Resolve either the audited parent ID or a recorded gateway child ID.

        Child IDs are intentionally recovered from the immutable gateway event payload instead
        of trusting an in-memory execution map, so private-stream recovery still works after a
        process restart.
        """

        client_order_id = client_order_id.strip()
        if not client_order_id:
            return None
        venue_clause = " AND venue=?" if venue is not None else ""
        parent_params: tuple[Any, ...] = (
            (client_order_id, venue) if venue is not None else (client_order_id,)
        )
        with self.connect() as connection:
            parent = connection.execute(
                self._sql(
                    "SELECT * FROM venue_orders WHERE client_order_id=?"
                    f"{venue_clause} LIMIT 1"  # noqa: S608
                ),
                parent_params,
            ).fetchone()
            if parent is not None:
                result = self._decode_row(parent)
                result["client_id_role"] = "PARENT"
                result["matched_client_order_id"] = client_order_id
                return result

            child_venue_clause = " WHERE o.venue=?" if venue is not None else ""
            child_params: tuple[Any, ...] = (venue,) if venue is not None else ()
            rows = connection.execute(
                self._sql(
                    """
                    SELECT o.*, e.raw_response_json AS child_event_raw_response_json,
                           e.order_event_id AS child_source_order_event_id,
                           e.created_at AS child_event_created_at
                    FROM venue_order_events e
                    JOIN venue_orders o ON o.venue_order_id=e.venue_order_id
                    """
                    + child_venue_clause
                    + " ORDER BY e.created_at DESC"
                ),
                child_params,
            ).fetchall()
        for row in rows:
            item = dict(row)
            try:
                raw = json.loads(item.pop("child_event_raw_response_json") or "{}")
            except (TypeError, ValueError):
                continue
            gateway_event = (
                raw.get("gateway_order_event") if isinstance(raw, dict) else None
            )
            if not isinstance(gateway_event, dict):
                continue
            child_id = str(gateway_event.get("child_client_order_id") or "")
            if child_id != client_order_id:
                continue
            for column in tuple(item):
                if column in JSON_COLUMNS:
                    decoded_name = column.removesuffix("_json")
                    item[decoded_name] = json.loads(item.pop(column) or "{}")
                elif column in BOOLEAN_COLUMNS and item[column] is not None:
                    item[column] = bool(item[column])
            item["client_id_role"] = str(gateway_event.get("role") or "CHILD").upper()
            item["matched_client_order_id"] = client_order_id
            item["gateway_order_event"] = gateway_event
            return item
        return None

    def resolve_protective_algo_id(
        self, external_algo_id: str | int, *, venue: str
    ) -> dict[str, Any] | None:
        """Resolve a venue algo ID only through an audited protective-child event."""

        normalized_id = str(external_algo_id).strip()
        normalized_venue = venue.strip()
        if not normalized_id or not normalized_venue:
            return None
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    """
                    SELECT o.*, e.raw_response_json AS child_event_raw_response_json,
                           e.order_event_id AS child_source_order_event_id,
                           e.external_order_id AS matched_external_algo_id,
                           e.created_at AS child_event_created_at
                    FROM venue_order_events e
                    JOIN venue_orders o ON o.venue_order_id=e.venue_order_id
                    WHERE o.venue=? AND e.external_order_id=?
                    ORDER BY e.created_at DESC
                    """
                ),
                (normalized_venue, normalized_id),
            ).fetchall()
        for row in rows:
            item = dict(row)
            try:
                raw = json.loads(item.pop("child_event_raw_response_json") or "{}")
            except (TypeError, ValueError):
                continue
            gateway_event = (
                raw.get("gateway_order_event") if isinstance(raw, dict) else None
            )
            if not isinstance(gateway_event, dict):
                continue
            if str(gateway_event.get("role") or "").upper() != "PROTECTIVE_STOP":
                continue
            child_id = str(gateway_event.get("child_client_order_id") or "")
            if not child_id:
                continue
            for column in tuple(item):
                if column in JSON_COLUMNS:
                    decoded_name = column.removesuffix("_json")
                    item[decoded_name] = json.loads(item.pop(column) or "{}")
                elif column in BOOLEAN_COLUMNS and item[column] is not None:
                    item[column] = bool(item[column])
            item["client_id_role"] = "PROTECTIVE_STOP"
            item["matched_client_order_id"] = child_id
            item["gateway_order_event"] = gateway_event
            return item
        return None

    def latest_venue_accounting_event(
        self, *, venue: str, income_type: str | None = None
    ) -> dict[str, Any] | None:
        where = "venue=?"
        params: tuple[Any, ...] = (venue,)
        if income_type is not None:
            where += " AND income_type=?"
            params += (income_type.strip().upper(),)
        with self.connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT * FROM venue_accounting_events "
                    f"WHERE {where} "  # noqa: S608
                    "ORDER BY transaction_time DESC, created_at DESC LIMIT 1"
                ),
                params,
            ).fetchone()
        return self._decode_row(row) if row is not None else None

    def list_protective_order_events(
        self,
        *,
        venue: str,
        limit: int = 10_000,
    ) -> list[dict[str, Any]]:
        """Return append-only protective child events with their parent-order lineage."""

        normalized = venue.strip()
        if not normalized:
            raise ValueError("venue must be non-empty")
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    """
                    SELECT e.*, o.symbol, o.side AS parent_side,
                           o.client_order_id AS parent_client_order_id,
                           o.reduce_only AS parent_reduce_only
                    FROM venue_order_events e
                    JOIN venue_orders o ON o.venue_order_id=e.venue_order_id
                    WHERE o.venue=? AND e.event_type LIKE ?
                    ORDER BY e.created_at DESC, e.observed_at DESC, e.event_sequence DESC
                    LIMIT ?
                    """
                ),
                (normalized, "PROTECTIVE_%", limit),
            ).fetchall()
        # Bound recovery cost without ever dropping the newest lifecycle facts.  Reversing
        # restores append order for the state-machine replay performed by paper_runtime.
        return [self._decode_row(row) for row in reversed(rows)]

    def list_venue_accounting_events(
        self,
        *,
        venue: str | None = None,
        income_type: str | None = None,
        limit: int = 1_000,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        clauses: list[str] = []
        params: list[Any] = []
        if venue is not None:
            clauses.append("venue=?")
            params.append(venue)
        if income_type is not None:
            clauses.append("income_type=?")
            params.append(income_type.strip().upper())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    "SELECT * FROM venue_accounting_events"
                    f"{where} "  # noqa: S608
                    "ORDER BY transaction_time DESC, created_at DESC LIMIT ?"
                ),
                (*params, limit),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def venue_accounting_event_by_external_id(
        self,
        *,
        venue: str,
        external_income_id: str,
    ) -> dict[str, Any] | None:
        normalized_venue = venue.strip()
        normalized_external_id = external_income_id.strip()
        if not normalized_venue or not normalized_external_id:
            raise ValueError("venue and external_income_id must be non-empty")
        with self.connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT * FROM venue_accounting_events "
                    "WHERE venue=? AND external_income_id=?"
                ),
                (normalized_venue, normalized_external_id),
            ).fetchone()
        return self._decode_row(row) if row is not None else None

    def resolve_funding_attribution(
        self,
        *,
        venue: str,
        symbol: str,
        transaction_time: datetime | str,
    ) -> FundingAttribution:
        """Resolve funding against the audited one-way position at the venue timestamp.

        The position is reconstructed solely from authoritative fills. Ambiguous ordering,
        incomplete lineage, missing realized accounting, or an invalid reduce/open transition
        is returned as ``UNATTRIBUTED`` instead of inventing an owner.
        """

        venue = venue.strip()
        symbol = symbol.strip().upper()
        at = _utc_iso(transaction_time)
        if not venue or not symbol:
            return FundingAttribution("UNATTRIBUTED", "MISSING_VENUE_OR_SYMBOL")
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    """
                    SELECT f.venue_fill_id, f.trace_id, f.venue_order_id, f.price,
                           f.quantity, f.fee, f.fee_asset, f.realized_pnl,
                           f.filled_at, f.created_at AS fill_created_at,
                           o.side, o.reduce_only
                    FROM venue_fills f
                    JOIN venue_orders o ON o.venue_order_id=f.venue_order_id
                    WHERE o.venue=? AND o.symbol=? AND f.filled_at<=?
                    ORDER BY f.filled_at, f.created_at, f.venue_fill_id
                    """
                ),
                (venue, symbol, at),
            ).fetchall()
        fills = [self._decode_row(row) for row in rows]
        if not fills:
            return FundingAttribution("UNATTRIBUTED", "NO_AUDITED_FILL_AT_TRANSACTION_TIME")

        order_ids = {str(item["venue_order_id"]) for item in fills}
        for venue_order_id in sorted(order_ids):
            complete, missing = self.validate_order_trace(venue_order_id)
            if not complete:
                return FundingAttribution(
                    "UNATTRIBUTED",
                    "INVALID_AUDIT_TRACE:" + ",".join(missing),
                )

        net_quantity = 0.0
        total_quantity = 0.0
        for fill in fills:
            values = (fill["price"], fill["quantity"], fill["fee"])
            if not all(math.isfinite(float(value)) for value in values):
                return FundingAttribution("UNATTRIBUTED", "NON_FINITE_AUDITED_FILL")
            if fill["realized_pnl"] is None or not math.isfinite(
                float(fill["realized_pnl"])
            ):
                return FundingAttribution("UNATTRIBUTED", "MISSING_REALIZED_PNL")
            quantity = float(fill["quantity"])
            total_quantity += quantity
            side = str(fill["side"]).upper()
            if quantity <= 0 or side not in {"BUY", "SELL"}:
                return FundingAttribution("UNATTRIBUTED", "INVALID_AUDITED_FILL_SHAPE")
            signed = quantity if side == "BUY" else -quantity
            next_quantity = net_quantity + signed
            tolerance = max(1e-12, total_quantity * 1e-10)
            if bool(fill["reduce_only"]):
                if (
                    abs(net_quantity) <= tolerance
                    or net_quantity * signed >= 0
                    or abs(next_quantity) > abs(net_quantity) + tolerance
                    or net_quantity * next_quantity < -tolerance
                ):
                    return FundingAttribution(
                        "UNATTRIBUTED", "INVALID_REDUCE_ONLY_POSITION_TRANSITION"
                    )
            elif abs(net_quantity) > tolerance and net_quantity * signed < 0:
                return FundingAttribution(
                    "UNATTRIBUTED", "NON_REDUCE_ORDER_CHANGED_POSITION_DIRECTION"
                )
            net_quantity = 0.0 if abs(next_quantity) <= tolerance else next_quantity

        if net_quantity == 0:
            return FundingAttribution("UNATTRIBUTED", "NO_OPEN_POSITION_AT_TRANSACTION_TIME")
        latest_time = str(fills[-1]["filled_at"])
        latest = [item for item in fills if str(item["filled_at"]) == latest_time]
        latest_owners = {
            (str(item["trace_id"]), str(item["venue_order_id"])) for item in latest
        }
        if len(latest_owners) != 1:
            return FundingAttribution("UNATTRIBUTED", "AMBIGUOUS_LATEST_AUDITED_OWNER")
        trace_id, venue_order_id = latest_owners.pop()
        return FundingAttribution(
            "ATTRIBUTED",
            "LATEST_VALID_AUDITED_POSITION_OWNER",
            trace_id=trace_id,
            venue_order_id=venue_order_id,
        )

    def latest_venue_accounting_attribution(
        self, accounting_event_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT * FROM venue_accounting_attributions "
                    "WHERE accounting_event_id=? "
                    "ORDER BY resolved_at DESC, created_at DESC, attribution_id DESC LIMIT 1"
                ),
                (accounting_event_id,),
            ).fetchone()
        return self._decode_row(row) if row is not None else None

    def authoritative_accounting_totals(
        self,
        *,
        venue: str,
        quote_asset: str = "USDT",
        as_of: datetime | str | None = None,
    ) -> dict[str, float]:
        """Return exact realized/funding totals or fail closed on any ledger gap."""

        venue = venue.strip()
        quote_asset = quote_asset.strip().upper()
        if not venue or not quote_asset:
            raise ValueError("venue and quote_asset must be non-empty")
        fill_time_clause = " AND f.filled_at<=?" if as_of is not None else ""
        income_time_clause = " AND transaction_time<=?" if as_of is not None else ""
        time_params: tuple[Any, ...] = ((_utc_iso(as_of),) if as_of is not None else ())
        with self.connect() as connection:
            fill_rows = connection.execute(
                self._sql(
                    "SELECT f.* FROM venue_fills f "
                    "JOIN venue_orders o ON o.venue_order_id=f.venue_order_id "
                    "WHERE o.venue=?" + fill_time_clause
                ),
                (venue, *time_params),
            ).fetchall()
            funding_rows = connection.execute(
                self._sql(
                    "SELECT * FROM venue_accounting_events "
                    "WHERE venue=? AND income_type='FUNDING_FEE'" + income_time_clause
                ),
                (venue, *time_params),
            ).fetchall()

        realized_pnl = 0.0
        for row in fill_rows:
            fill = self._decode_row(row)
            value = fill.get("realized_pnl")
            if value is None or not math.isfinite(float(value)):
                raise IncompleteVenueAccountingError(
                    "MISSING_REALIZED_PNL",
                    f"venue fill {fill['venue_fill_id']} has no exact realized PnL",
                )
            realized_pnl += float(value)

        funding_pnl = 0.0
        for row in funding_rows:
            event = self._decode_row(row)
            attribution = self.latest_venue_accounting_attribution(
                str(event["accounting_event_id"])
            )
            if attribution is None or attribution["status"] != "ATTRIBUTED":
                reason = attribution["reason"] if attribution is not None else "MISSING"
                raise IncompleteVenueAccountingError(
                    "UNATTRIBUTED_FUNDING",
                    f"funding event {event['accounting_event_id']} attribution={reason}",
                )
            if str(event["asset"]).upper() != quote_asset:
                raise IncompleteVenueAccountingError(
                    "UNCONVERTED_FUNDING_ASSET",
                    f"funding event {event['accounting_event_id']} is in {event['asset']}",
                )
            amount = float(event["amount"])
            if not math.isfinite(amount):
                raise IncompleteVenueAccountingError(
                    "NON_FINITE_FUNDING", str(event["accounting_event_id"])
                )
            funding_pnl += amount
        if not math.isfinite(realized_pnl) or not math.isfinite(funding_pnl):
            raise IncompleteVenueAccountingError(
                "NON_FINITE_ACCOUNT_TOTAL", "aggregated accounting is not finite"
            )
        return {"realized_pnl": realized_pnl, "funding_pnl": funding_pnl}

    def audited_performance_records(self, *, venue: str) -> dict[str, list[dict[str, Any]]]:
        """Export the immutable venue records consumed by the strict outcome builder."""

        with self.connect() as connection:
            fill_rows = connection.execute(
                self._sql(
                    """
                    SELECT f.*, o.venue, o.symbol, o.side, o.reduce_only,
                           c.strategy_version
                    FROM venue_fills f
                    JOIN venue_orders o ON o.venue_order_id=f.venue_order_id
                    JOIN trade_candidates c ON c.candidate_id=o.candidate_id
                    WHERE o.venue=?
                    ORDER BY f.filled_at, f.created_at, f.venue_fill_id
                    """
                ),
                (venue,),
            ).fetchall()
            funding_rows = connection.execute(
                self._sql(
                    "SELECT * FROM venue_accounting_events "
                    "WHERE venue=? AND income_type='FUNDING_FEE' "
                    "ORDER BY transaction_time, created_at, accounting_event_id"
                ),
                (venue,),
            ).fetchall()
            conversion_rows = connection.execute(
                self._sql(
                    """
                    SELECT c.* FROM venue_fee_conversions c
                    JOIN venue_fills f ON f.venue_fill_id=c.venue_fill_id
                    JOIN venue_orders o ON o.venue_order_id=f.venue_order_id
                    WHERE o.venue=?
                    """
                ),
                (venue,),
            ).fetchall()
        decoded_fills = [self._decode_row(row) for row in fill_rows]
        for venue_order_id in sorted(
            {str(item["venue_order_id"]) for item in decoded_fills}
        ):
            complete, missing = self.validate_order_trace(venue_order_id)
            if not complete:
                raise IncompleteVenueAccountingError(
                    "INVALID_AUDIT_TRACE",
                    f"{venue_order_id}: {','.join(missing)}",
                )
        funding = []
        for row in funding_rows:
            item = self._decode_row(row)
            item["attribution"] = self.latest_venue_accounting_attribution(
                str(item["accounting_event_id"])
            )
            funding.append(item)
        return {
            "fills": decoded_fills,
            "funding": funding,
            "fee_conversions": [self._decode_row(row) for row in conversion_rows],
        }

    def build_trade_outcomes(
        self, *, venue: str, quote_asset: str = "USDT"
    ) -> tuple[TradeOutcome, ...]:
        """Build exact closed-position outcomes suitable for production promotion metrics."""

        from .learning import build_trade_outcomes_from_audit_records

        records = self.audited_performance_records(venue=venue)
        outcomes = build_trade_outcomes_from_audit_records(
            fills=records["fills"],
            funding_events=records["funding"],
            fee_conversions=records["fee_conversions"],
            quote_asset=quote_asset,
        )
        if venue == "internal-paper":
            return self.require_paper_funding_coverage(venue=venue, outcomes=outcomes)
        return outcomes

    def require_paper_funding_coverage(
        self,
        *,
        venue: str,
        outcomes: Sequence[TradeOutcome],
    ) -> tuple[TradeOutcome, ...]:
        """Bind each paper outcome to proof that public funding was queried through close."""

        verified: list[TradeOutcome] = []
        for outcome in outcomes:
            episode_id = str(outcome.episode_id or "").strip()
            if not episode_id:
                raise IncompleteVenueAccountingError(
                    "INCOMPLETE_FUNDING_COVERAGE",
                    f"{outcome.symbol}: closed paper outcome has no episode identity",
                )
            evidence_id = paper_funding_coverage_evidence_id(venue, episode_id)
            coverage = self.latest_external_evidence(evidence_id)
            if coverage is None or coverage.get("deleted_at") is not None:
                raise IncompleteVenueAccountingError(
                    "INCOMPLETE_FUNDING_COVERAGE",
                    f"{episode_id}: durable coverage evidence is missing",
                )
            payload = coverage.get("payload")
            if not isinstance(payload, Mapping):
                raise IncompleteVenueAccountingError(
                    "INVALID_FUNDING_COVERAGE",
                    f"{episode_id}: coverage payload is not an object",
                )
            expected_identity = {
                "schema": PAPER_FUNDING_COVERAGE_SCHEMA,
                "venue": venue,
                "episode_id": episode_id,
                "symbol": outcome.symbol,
            }
            if any(payload.get(key) != value for key, value in expected_identity.items()):
                raise IncompleteVenueAccountingError(
                    "INVALID_FUNDING_COVERAGE",
                    f"{episode_id}: coverage identity does not match the outcome",
                )
            raw_episode_close = payload.get("episode_closed_at")
            if raw_episode_close is None:
                raise IncompleteVenueAccountingError(
                    "INCOMPLETE_FUNDING_COVERAGE",
                    f"{episode_id}: coverage has not been sealed to a close",
                )
            try:
                covered_through = _as_utc_datetime(str(payload["covered_through"]))
                episode_closed_at = _as_utc_datetime(str(raw_episode_close))
            except (KeyError, TypeError, ValueError) as error:
                raise IncompleteVenueAccountingError(
                    "INVALID_FUNDING_COVERAGE",
                    f"{episode_id}: coverage timestamps are incomplete",
                ) from error
            closed_at = _as_utc_datetime(outcome.closed_at)
            if episode_closed_at != closed_at:
                raise IncompleteVenueAccountingError(
                    "INCOMPLETE_FUNDING_COVERAGE",
                    f"{episode_id}: coverage is not sealed to the audited close",
                )
            if covered_through < closed_at:
                raise IncompleteVenueAccountingError(
                    "INCOMPLETE_FUNDING_COVERAGE",
                    f"{episode_id}: coverage ends before the audited close",
                )
            record_id = str(coverage["evidence_record_id"])
            verified.append(
                replace(
                    outcome,
                    source_record_ids=tuple(
                        dict.fromkeys((*outcome.source_record_ids, record_id))
                    ),
                )
            )
        return tuple(verified)

    def performance_source_ids_exist(self, source_record_ids: Sequence[str]) -> bool:
        """Verify shadow-accounting lineage against immutable audit record IDs."""

        normalized = tuple(dict.fromkeys(str(item).strip() for item in source_record_ids))
        if (
            not normalized
            or any(not item for item in normalized)
            or len(normalized) > 1_000
        ):
            return False
        found: set[str] = set()
        placeholders = ",".join("?" for _ in normalized)
        sources = (
            ("venue_fills", "venue_fill_id"),
            ("venue_accounting_events", "accounting_event_id"),
            ("venue_fee_conversions", "conversion_id"),
            ("external_evidence", "evidence_record_id"),
        )
        with self.connect() as connection:
            for table, column in sources:
                rows = connection.execute(
                    self._sql(
                        f"SELECT {column} FROM {table} "  # noqa: S608
                        f"WHERE {column} IN ({placeholders})"  # noqa: S608
                    ),
                    normalized,
                ).fetchall()
                found.update(str(_row_value(row, column, 0)) for row in rows)
        return found == set(normalized)

    def external_evidence_records(
        self,
        evidence_record_ids: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Load exact immutable external-evidence versions by record ID.

        Callers that need a typed evidence contract must inspect these exact rows rather than
        accepting an ID that merely exists in any accounting table.
        """

        normalized = tuple(dict.fromkeys(str(item).strip() for item in evidence_record_ids))
        if (
            not normalized
            or any(not item for item in normalized)
            or len(normalized) > 1_000
        ):
            return {}
        placeholders = ",".join("?" for _ in normalized)
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    "SELECT * FROM external_evidence "
                    f"WHERE evidence_record_id IN ({placeholders})"  # noqa: S608
                ),
                normalized,
            ).fetchall()
        return {
            str(_row_value(row, "evidence_record_id", 0)): self._decode_row(row)
            for row in rows
        }

    def audit_fact_trace_ids_exist(self, trace_ids: Sequence[str]) -> bool:
        """Require traces to exist independently of external-evidence assertions.

        A typed cost payload cannot make its own arbitrary trace ID real.  At least one other
        immutable audit fact (for example a strategy spec, candidate, order, fill, or shadow
        journal event) must already carry every requested trace ID.
        """

        normalized = tuple(dict.fromkeys(str(item).strip() for item in trace_ids))
        if not normalized or any(not item for item in normalized) or len(normalized) > 1_000:
            return False
        placeholders = ",".join("?" for _ in normalized)
        expected = set(normalized)
        found: set[str] = set()
        fact_tables = tuple(table for table in TRACE_TABLES if table != "external_evidence")
        with self.connect() as connection:
            for table in fact_tables:
                rows = connection.execute(
                    self._sql(
                        f"SELECT DISTINCT trace_id FROM {table} "  # noqa: S608
                        f"WHERE trace_id IN ({placeholders})"  # noqa: S608
                    ),
                    normalized,
                ).fetchall()
                found.update(str(_row_value(row, "trace_id", 0)) for row in rows)
                if found == expected:
                    return True
        return found == expected

    def latest_account_snapshot(self, source: str | None = None) -> dict[str, Any] | None:
        where = " WHERE source=?" if source is not None else ""
        params: tuple[Any, ...] = (source,) if source is not None else ()
        with self.connect() as connection:
            row = connection.execute(
                self._sql(
                    f"SELECT * FROM account_snapshots{where} "  # noqa: S608
                    "ORDER BY observed_at DESC, created_at DESC LIMIT 1"
                ),
                params,
            ).fetchone()
        return self._decode_row(row) if row is not None else None

    def account_risk_state(
        self,
        source: str | None = None,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Restore durable daily-loss and high-water context after a process restart."""

        as_of = now or datetime.now(UTC)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        else:
            as_of = as_of.astimezone(UTC)
        day_start = as_of.replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = day_start + timedelta(days=1)
        source_clause = " AND source=?" if source is not None else ""
        source_params: tuple[Any, ...] = (source,) if source is not None else ()
        with self.connect() as connection:
            latest = connection.execute(
                self._sql(
                    "SELECT * FROM account_snapshots WHERE 1=1"
                    f"{source_clause} "  # noqa: S608
                    "ORDER BY observed_at DESC, created_at DESC LIMIT 1"
                ),
                source_params,
            ).fetchone()
            high_water = connection.execute(
                self._sql(
                    "SELECT MAX(equity) AS high_water_equity FROM account_snapshots "
                    f"WHERE 1=1{source_clause}"  # noqa: S608
                ),
                source_params,
            ).fetchone()
            day_first = connection.execute(
                self._sql(
                    """
                    SELECT * FROM account_snapshots
                    WHERE observed_at>=? AND observed_at<?
                    """
                    + source_clause
                    + " ORDER BY observed_at, created_at LIMIT 1"
                ),
                (_utc_iso(day_start), _utc_iso(next_day), *source_params),
            ).fetchone()
        high_water_value = (
            _row_value(high_water, "high_water_equity", 0) if high_water is not None else None
        )
        decoded_day_first = self._decode_row(day_first) if day_first is not None else None
        return {
            "as_of": _utc_iso(as_of),
            "source": source,
            "latest": self._decode_row(latest) if latest is not None else None,
            "historical_high_water_equity": (
                float(high_water_value) if high_water_value is not None else None
            ),
            "utc_day_start_equity": (
                float(decoded_day_first["equity"]) if decoded_day_first is not None else None
            ),
            "utc_day_start_snapshot": decoded_day_first,
        }

    def position_thesis_history(self, position_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    """
                    SELECT * FROM position_theses WHERE position_id=?
                    ORDER BY version
                    """
                ),
                (position_id,),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def latest_position_thesis(self, position_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                self._sql(
                    """
                    SELECT * FROM position_theses WHERE position_id=?
                    ORDER BY version DESC LIMIT 1
                    """
                ),
                (position_id,),
            ).fetchone()
        return self._decode_row(row) if row is not None else None

    def strategy_spec_by_version(self, strategy_version: str) -> dict[str, Any] | None:
        """Return the immutable research row for one registry version, if it exists."""

        with self.connect() as connection:
            row = connection.execute(
                self._sql(
                    """
                    SELECT * FROM strategy_specs
                    WHERE strategy_version=?
                    LIMIT 1
                    """
                ),
                (strategy_version,),
            ).fetchone()
        return self._decode_row(row) if row is not None else None

    def pending_counterfactual_work(
        self,
        *,
        as_of: datetime | str,
        limit: int = 1_000,
    ) -> list[dict[str, Any]]:
        """List candidates with at least one matured 1h/4h/24h outcome still missing.

        The query only establishes that a horizon is old enough.  The learning runtime must
        still obtain a candle that closed no earlier than the target and no later than
        ``as_of`` before it may append the outcome.
        """

        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        cutoff = _as_utc_datetime(_utc_iso(as_of))
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    """
                    SELECT c.*, d.decision_id, d.action, d.confidence
                    FROM trade_candidates c
                    LEFT JOIN llm_decisions d ON d.candidate_id=c.candidate_id
                    WHERE c.created_at <= ?
                    ORDER BY c.created_at, c.candidate_id
                    LIMIT ?
                    """
                ),
                (_utc_iso(cutoff - timedelta(hours=1)), limit),
            ).fetchall()
            decoded = [self._decode_row(row) for row in rows]
            if not decoded:
                return []
            candidate_ids = tuple(str(row["candidate_id"]) for row in decoded)
            placeholders = ", ".join("?" for _ in candidate_ids)
            outcome_rows = connection.execute(
                self._sql(
                    f"""
                    SELECT candidate_id, horizon_hours
                    FROM counterfactual_outcomes
                    WHERE candidate_id IN ({placeholders})
                    """  # noqa: S608
                ),
                candidate_ids,
            ).fetchall()
        existing: dict[str, set[int]] = {}
        for row in outcome_rows:
            existing.setdefault(str(_row_value(row, "candidate_id", 0)), set()).add(
                int(_row_value(row, "horizon_hours", 1))
            )
        result: list[dict[str, Any]] = []
        for row in decoded:
            created_at = _as_utc_datetime(row["created_at"])
            due = tuple(
                horizon
                for horizon in (1, 4, 24)
                if created_at + timedelta(hours=horizon) <= cutoff
                and horizon not in existing.get(str(row["candidate_id"]), set())
            )
            if due:
                row["due_horizons"] = due
                result.append(row)
        return result

    def recent_counterfactual_outcomes(self, *, limit: int = 500) -> list[dict[str, Any]]:
        """Return bounded, model-safe learning facts without raw social-media content."""

        if not 1 <= limit <= 5_000:
            raise ValueError("limit must be between 1 and 5000")
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    """
                    SELECT o.*, c.strategy_version, c.symbol, c.direction,
                           d.action, d.confidence
                    FROM counterfactual_outcomes o
                    JOIN trade_candidates c ON c.candidate_id=o.candidate_id
                    LEFT JOIN llm_decisions d ON d.decision_id=o.decision_id
                    ORDER BY o.observed_at DESC, o.outcome_id DESC
                    LIMIT ?
                    """
                ),
                (limit,),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def list_traces(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        union_query = " UNION ALL ".join(
            f"SELECT trace_id, created_at FROM {table} WHERE trace_id IS NOT NULL"  # noqa: S608
            for table in TRACE_TABLES
        )
        query = self._sql(
            f"""
            SELECT trace_id, MAX(created_at) AS last_activity_at, COUNT(*) AS record_count
            FROM ({union_query}) AS audit_records
            GROUP BY trace_id
            ORDER BY last_activity_at DESC, trace_id
            LIMIT ?
            """  # noqa: S608
        )
        with self.connect() as connection:
            rows = connection.execute(query, (limit,)).fetchall()
        return [dict(row) for row in rows]

    def list_venue_order_trace_ids(self, *, venue: str) -> tuple[str, ...]:
        """Return every trace owning an order at ``venue`` without a safety cutoff.

        Startup reconciliation must recover protection for a position held longer than the
        generic activity feed.  A presentation-oriented ``list_traces`` limit therefore cannot
        be used as an accounting or ownership boundary.
        """

        normalized = venue.strip()
        if not normalized:
            raise ValueError("venue must be non-empty")
        with self.connect() as connection:
            rows = connection.execute(
                self._sql(
                    """
                    SELECT trace_id, MIN(created_at) AS first_order_at
                    FROM venue_orders
                    WHERE venue=?
                    GROUP BY trace_id
                    ORDER BY first_order_at, trace_id
                    """
                ),
                (normalized,),
            ).fetchall()
        return tuple(str(_row_value(row, "trace_id", 0)) for row in rows)

    def validate_order_trace(
        self, venue_order_id: str, *, require_outcome: bool = False
    ) -> tuple[bool, tuple[str, ...]]:
        trace = self.trace_for_order(venue_order_id)
        orders = [
            order for order in trace["venue_orders"] if order["venue_order_id"] == venue_order_id
        ]
        if not orders:
            return False, ("MISSING_ORDER",)
        order = orders[0]
        missing: list[str] = []

        def add(code: str) -> None:
            if code not in missing:
                missing.append(code)

        candidates = {item["candidate_id"]: item for item in trace["trade_candidates"]}
        decisions = {item["decision_id"]: item for item in trace["llm_decisions"]}
        risks = {item["risk_decision_id"]: item for item in trace["risk_decisions"]}
        candidate = candidates.get(order["candidate_id"])
        decision = decisions.get(order["decision_id"])
        risk = risks.get(order["risk_decision_id"])
        if candidate is None:
            add("MISSING_CANDIDATE")
        elif not candidate["evidence_ids"]:
            add("MISSING_SOURCE_EVIDENCE")
        if decision is None:
            add("MISSING_LLM_DECISION")
        elif decision["candidate_id"] != order["candidate_id"]:
            add("DECISION_CANDIDATE_MISMATCH")
        if risk is None:
            add("MISSING_RISK_DECISION")
        elif (
            risk["candidate_id"] != order["candidate_id"]
            or risk["decision_id"] != order["decision_id"]
        ):
            add("RISK_LINEAGE_MISMATCH")
        theses = [
            thesis
            for thesis in trace["position_theses"]
            if thesis["decision_id"] == order["decision_id"]
        ]
        if candidate is not None and decision is not None and risk is not None:
            for code in _order_semantic_errors(order, candidate, decision, risk, theses):
                add(code)
            raw_evidence = [
                evidence
                for evidence in trace["linked_external_evidence"]
                if evidence["linked_candidate_id"] == candidate["candidate_id"]
            ]
            for code in _raw_evidence_errors(candidate, raw_evidence):
                add(code)
        elif not theses:
            add("MISSING_POSITION_THESIS")
        order_events = [
            event
            for event in trace["venue_order_events"]
            if event["venue_order_id"] == venue_order_id
        ]
        if not order_events:
            add("MISSING_ORDER_EVENT_HISTORY")
        elif sorted(event["event_sequence"] for event in order_events) != list(
            range(1, len(order_events) + 1)
        ):
            add("ORDER_EVENT_SEQUENCE_GAP")
        if require_outcome:
            has_realized_fill = any(
                bool(order["reduce_only"])
                and fill["venue_order_id"] == venue_order_id
                and fill["realized_pnl"] is not None
                for fill in trace["venue_fills"]
            )
            counterfactual_horizons = {
                outcome["horizon_hours"]
                for outcome in trace["counterfactual_outcomes"]
                if outcome["candidate_id"] == order["candidate_id"]
                and outcome["decision_id"] == order["decision_id"]
            }
            if not has_realized_fill and counterfactual_horizons != {1, 4, 24}:
                add("MISSING_FINAL_OR_1H_4H_24H_OUTCOMES")
        return not missing, tuple(missing)

    @staticmethod
    def _decode_row(row: Any) -> dict[str, Any]:
        item = dict(row)
        for column in tuple(item):
            if column in JSON_COLUMNS:
                decoded_name = column.removesuffix("_json")
                item[decoded_name] = json.loads(item.pop(column) or "{}")
            elif column in BOOLEAN_COLUMNS and item[column] is not None:
                item[column] = bool(item[column])
        return item
