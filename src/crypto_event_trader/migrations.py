from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sqlite_script: str
    postgres_statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        payload = "\n".join((self.name, self.sqlite_script, *self.postgres_statements))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MigrationConnection(Protocol):
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any: ...


AUDIT_TABLES_SQLITE = """
CREATE TABLE IF NOT EXISTS trade_candidates (
    candidate_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('LONG','SHORT')),
    max_quantity REAL NOT NULL CHECK(max_quantity > 0),
    max_risk_fraction REAL NOT NULL CHECK(max_risk_fraction > 0),
    feature_snapshot_json TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_decisions (
    decision_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL REFERENCES trade_candidates(candidate_id) ON DELETE RESTRICT,
    action TEXT NOT NULL CHECK(action IN ('OPEN','ADD','HOLD','REDUCE','CLOSE','REJECT')),
    position_multiplier REAL NOT NULL CHECK(position_multiplier BETWEEN 0 AND 1),
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    evidence_ids_json TEXT NOT NULL,
    thesis TEXT NOT NULL,
    invalidation_conditions_json TEXT NOT NULL,
    next_review_at TEXT,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    response_id TEXT,
    latency_ms INTEGER,
    raw_response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS position_theses (
    thesis_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    position_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK(version > 0),
    prior_thesis_id TEXT REFERENCES position_theses(thesis_id) ON DELETE RESTRICT,
    decision_id TEXT NOT NULL REFERENCES llm_decisions(decision_id) ON DELETE RESTRICT,
    entry_reason TEXT NOT NULL,
    expected_horizon TEXT NOT NULL,
    supporting_evidence_json TEXT NOT NULL,
    opposing_evidence_json TEXT NOT NULL,
    add_count INTEGER NOT NULL CHECK(add_count >= 0),
    pnl_r REAL NOT NULL,
    invalidation_conditions_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(position_id, version)
);

CREATE TABLE IF NOT EXISTS risk_decisions (
    risk_decision_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL REFERENCES trade_candidates(candidate_id) ON DELETE RESTRICT,
    decision_id TEXT NOT NULL REFERENCES llm_decisions(decision_id) ON DELETE RESTRICT,
    outcome TEXT NOT NULL CHECK(outcome IN ('ALLOW','RESIZE','REJECT','EXIT')),
    approved_quantity REAL NOT NULL CHECK(approved_quantity >= 0),
    reason_codes_json TEXT NOT NULL,
    limits_snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS venue_orders (
    venue_order_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL REFERENCES trade_candidates(candidate_id) ON DELETE RESTRICT,
    decision_id TEXT NOT NULL REFERENCES llm_decisions(decision_id) ON DELETE RESTRICT,
    risk_decision_id TEXT NOT NULL REFERENCES risk_decisions(risk_decision_id) ON DELETE RESTRICT,
    venue TEXT NOT NULL,
    client_order_id TEXT NOT NULL UNIQUE,
    external_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
    order_type TEXT NOT NULL,
    quantity REAL NOT NULL CHECK(quantity > 0),
    price REAL,
    reduce_only INTEGER NOT NULL CHECK(reduce_only IN (0,1)),
    status TEXT NOT NULL,
    raw_response_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS venue_fills (
    venue_fill_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    venue_order_id TEXT NOT NULL REFERENCES venue_orders(venue_order_id) ON DELETE RESTRICT,
    external_fill_id TEXT,
    price REAL NOT NULL CHECK(price > 0),
    quantity REAL NOT NULL CHECK(quantity > 0),
    fee REAL NOT NULL CHECK(fee >= 0),
    fee_asset TEXT NOT NULL,
    realized_pnl REAL,
    raw_response_json TEXT NOT NULL,
    filled_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(venue_order_id, external_fill_id)
);

CREATE TABLE IF NOT EXISTS account_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    trace_id TEXT,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    gross_exposure REAL NOT NULL CHECK(gross_exposure >= 0),
    net_exposure REAL NOT NULL,
    daily_pnl REAL NOT NULL,
    drawdown REAL NOT NULL CHECK(drawdown >= 0),
    positions_json TEXT NOT NULL,
    source TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_specs (
    spec_id TEXT PRIMARY KEY,
    trace_id TEXT,
    strategy_version TEXT NOT NULL UNIQUE,
    parent_version TEXT,
    status TEXT NOT NULL CHECK(status IN ('CHAMPION','CHALLENGER','RETIRED','REJECTED')),
    parameters_json TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    source_response_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    backtest_run_id TEXT PRIMARY KEY,
    trace_id TEXT,
    spec_id TEXT NOT NULL REFERENCES strategy_specs(spec_id) ON DELETE RESTRICT,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    completed INTEGER NOT NULL CHECK(completed IN (0,1)),
    net_profit REAL,
    net_return REAL,
    max_drawdown REAL,
    total_cost REAL,
    stressed_net_return_2x REAL,
    dsr_significance_probability REAL,
    pbo_probability REAL,
    symbol_concentration REAL,
    month_concentration REAL,
    trade_count INTEGER,
    holdout_months INTEGER,
    validation_json TEXT NOT NULL,
    raw_metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_results (
    shadow_result_id TEXT PRIMARY KEY,
    trace_id TEXT,
    spec_id TEXT NOT NULL REFERENCES strategy_specs(spec_id) ON DELETE RESTRICT,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    completed INTEGER NOT NULL CHECK(completed IN (0,1)),
    elapsed_days INTEGER,
    closed_trades INTEGER,
    net_return REAL,
    max_drawdown REAL,
    stressed_net_return_2x REAL,
    symbol_concentration REAL,
    month_concentration REAL,
    raw_metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotion_records (
    promotion_record_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    champion_spec_id TEXT NOT NULL REFERENCES strategy_specs(spec_id) ON DELETE RESTRICT,
    challenger_spec_id TEXT NOT NULL REFERENCES strategy_specs(spec_id) ON DELETE RESTRICT,
    backtest_run_id TEXT NOT NULL REFERENCES backtest_runs(backtest_run_id) ON DELETE RESTRICT,
    champion_shadow_result_id TEXT NOT NULL
        REFERENCES shadow_results(shadow_result_id) ON DELETE RESTRICT,
    challenger_shadow_result_id TEXT NOT NULL
        REFERENCES shadow_results(shadow_result_id) ON DELETE RESTRICT,
    eligible INTEGER NOT NULL CHECK(eligible IN (0,1)),
    reason_codes_json TEXT NOT NULL,
    evaluation_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS counterfactual_outcomes (
    outcome_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL REFERENCES trade_candidates(candidate_id) ON DELETE RESTRICT,
    decision_id TEXT REFERENCES llm_decisions(decision_id) ON DELETE RESTRICT,
    horizon_hours INTEGER NOT NULL CHECK(horizon_hours IN (1,4,24)),
    realized_return REAL NOT NULL,
    decision_regret REAL,
    confidence_calibration_error REAL,
    source_reliability_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(candidate_id, horizon_hours)
);

CREATE INDEX IF NOT EXISTS idx_trade_candidates_trace ON trade_candidates(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_decisions_trace ON llm_decisions(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_position_theses_trace ON position_theses(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_position_theses_position ON position_theses(position_id, version);
CREATE INDEX IF NOT EXISTS idx_risk_decisions_trace ON risk_decisions(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_venue_orders_trace ON venue_orders(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_venue_fills_trace ON venue_fills(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_counterfactual_trace
    ON counterfactual_outcomes(trace_id, created_at);
"""


def _postgres_portable_schema(schema_text: str) -> tuple[str, ...]:
    # The portable schema deliberately keeps JSON and timestamps as text. This makes exports
    # byte-for-byte comparable between SQLite tests and PostgreSQL production deployments.
    schema = (
        schema_text.replace(" REAL", " DOUBLE PRECISION")
        .replace("CREATE INDEX IF NOT EXISTS", "CREATE INDEX IF NOT EXISTS")
        .replace("reduce_only INTEGER", "reduce_only INTEGER")
    )
    return tuple(statement.strip() for statement in schema.split(";") if statement.strip())


def _postgres_audit_schema() -> tuple[str, ...]:
    return _postgres_portable_schema(AUDIT_TABLES_SQLITE)


APPEND_ONLY_TABLES = (
    "llm_decisions",
    "position_theses",
    "risk_decisions",
    "promotion_records",
)


def _sqlite_append_only_triggers() -> str:
    statements: list[str] = []
    for table in APPEND_ONLY_TABLES:
        statements.extend(
            (
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_update_{table}
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_delete_{table}
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
            )
        )
    return ";\n".join(statement.strip() for statement in statements) + ";"


POSTGRES_APPEND_ONLY_FUNCTION = """
CREATE OR REPLACE FUNCTION deny_crypto_audit_mutation()
RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql
"""


def _postgres_append_only_triggers() -> tuple[str, ...]:
    statements = [POSTGRES_APPEND_ONLY_FUNCTION]
    for table in APPEND_ONLY_TABLES:
        statements.extend(
            (
                f"DROP TRIGGER IF EXISTS deny_update_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_update_{table}
                BEFORE UPDATE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
                f"DROP TRIGGER IF EXISTS deny_delete_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_delete_{table}
                BEFORE DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
            )
        )
    return tuple(statements)


ORDER_EVENT_HISTORY_SQLITE = """
CREATE TABLE IF NOT EXISTS venue_order_events (
    order_event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    venue_order_id TEXT NOT NULL REFERENCES venue_orders(venue_order_id) ON DELETE RESTRICT,
    event_sequence INTEGER NOT NULL CHECK(event_sequence > 0),
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    source_event_id TEXT,
    external_order_id TEXT,
    executed_quantity REAL NOT NULL CHECK(executed_quantity >= 0),
    average_price REAL,
    raw_response_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(venue_order_id, event_sequence),
    UNIQUE(venue_order_id, source_event_id)
);

CREATE INDEX IF NOT EXISTS idx_venue_order_events_trace
    ON venue_order_events(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_venue_order_events_order
    ON venue_order_events(venue_order_id, event_sequence);
"""


def _sqlite_order_history_triggers() -> str:
    statements: list[str] = []
    for table in ("venue_orders", "venue_order_events"):
        statements.extend(
            (
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_update_{table}
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_delete_{table}
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
            )
        )
    return ";\n".join(statement.strip() for statement in statements) + ";"


def _postgres_order_history() -> tuple[str, ...]:
    statements = list(_postgres_portable_schema(ORDER_EVENT_HISTORY_SQLITE))
    for table in ("venue_orders", "venue_order_events"):
        statements.extend(
            (
                f"DROP TRIGGER IF EXISTS deny_update_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_update_{table}
                BEFORE UPDATE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
                f"DROP TRIGGER IF EXISTS deny_delete_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_delete_{table}
                BEFORE DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
            )
        )
    return tuple(statements)


REMAINING_APPEND_ONLY_TABLES = (
    "trade_candidates",
    "venue_fills",
    "account_snapshots",
    "strategy_specs",
    "backtest_runs",
    "shadow_results",
    "counterfactual_outcomes",
)


def _sqlite_remaining_append_only_triggers() -> str:
    statements: list[str] = []
    for table in REMAINING_APPEND_ONLY_TABLES:
        statements.extend(
            (
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_update_{table}
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_delete_{table}
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
            )
        )
    return ";\n".join(statement.strip() for statement in statements) + ";"


def _postgres_remaining_append_only_triggers() -> tuple[str, ...]:
    statements: list[str] = []
    for table in REMAINING_APPEND_ONLY_TABLES:
        statements.extend(
            (
                f"DROP TRIGGER IF EXISTS deny_update_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_update_{table}
                BEFORE UPDATE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
                f"DROP TRIGGER IF EXISTS deny_delete_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_delete_{table}
                BEFORE DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
            )
        )
    return tuple(statements)


EXTERNAL_EVIDENCE_SQLITE = """
CREATE TABLE IF NOT EXISTS external_evidence (
    evidence_record_id TEXT PRIMARY KEY,
    trace_id TEXT,
    evidence_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK(version > 0),
    prior_evidence_record_id TEXT
        REFERENCES external_evidence(evidence_record_id) ON DELETE RESTRICT,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_url TEXT,
    first_observed_at TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    deleted_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(evidence_id, version)
);

CREATE TABLE IF NOT EXISTS candidate_evidence_links (
    evidence_link_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL
        REFERENCES trade_candidates(candidate_id) ON DELETE RESTRICT,
    evidence_record_id TEXT NOT NULL
        REFERENCES external_evidence(evidence_record_id) ON DELETE RESTRICT,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(candidate_id, evidence_record_id)
);

CREATE INDEX IF NOT EXISTS idx_external_evidence_stable_id
    ON external_evidence(evidence_id, version);
CREATE INDEX IF NOT EXISTS idx_external_evidence_trace
    ON external_evidence(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_candidate_evidence_trace
    ON candidate_evidence_links(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_candidate_evidence_candidate
    ON candidate_evidence_links(candidate_id, created_at);
"""


VENUE_ACCOUNTING_EVENTS_SQLITE = """
CREATE TABLE IF NOT EXISTS venue_accounting_events (
    accounting_event_id TEXT PRIMARY KEY,
    trace_id TEXT,
    venue_order_id TEXT
        REFERENCES venue_orders(venue_order_id) ON DELETE RESTRICT,
    venue TEXT NOT NULL,
    external_income_id TEXT NOT NULL,
    symbol TEXT,
    income_type TEXT NOT NULL,
    asset TEXT NOT NULL,
    amount REAL NOT NULL,
    transaction_time TEXT NOT NULL,
    trade_id TEXT,
    raw_response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(venue, external_income_id)
);

CREATE INDEX IF NOT EXISTS idx_venue_accounting_trace
    ON venue_accounting_events(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_venue_accounting_lookup
    ON venue_accounting_events(venue, income_type, transaction_time);
"""


VENUE_PERFORMANCE_ACCOUNTING_SQLITE = """
CREATE TABLE IF NOT EXISTS venue_accounting_attributions (
    attribution_id TEXT PRIMARY KEY,
    accounting_event_id TEXT NOT NULL
        REFERENCES venue_accounting_events(accounting_event_id) ON DELETE RESTRICT,
    trace_id TEXT,
    venue_order_id TEXT
        REFERENCES venue_orders(venue_order_id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK(status IN ('ATTRIBUTED','UNATTRIBUTED')),
    reason TEXT NOT NULL,
    resolved_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS venue_fee_conversions (
    conversion_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    venue_fill_id TEXT NOT NULL
        REFERENCES venue_fills(venue_fill_id) ON DELETE RESTRICT,
    from_asset TEXT NOT NULL,
    quote_asset TEXT NOT NULL,
    rate REAL NOT NULL CHECK(rate > 0),
    effective_at TEXT NOT NULL,
    source TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(venue_fill_id, quote_asset)
);

CREATE INDEX IF NOT EXISTS idx_venue_accounting_attribution_trace
    ON venue_accounting_attributions(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_venue_accounting_attribution_event
    ON venue_accounting_attributions(accounting_event_id, resolved_at, created_at);
CREATE INDEX IF NOT EXISTS idx_venue_fee_conversion_trace
    ON venue_fee_conversions(trace_id, created_at);
"""


def _sqlite_venue_accounting_triggers() -> str:
    return """
    CREATE TRIGGER IF NOT EXISTS deny_update_venue_accounting_events
    BEFORE UPDATE ON venue_accounting_events
    BEGIN
        SELECT RAISE(ABORT, 'venue_accounting_events is append-only');
    END;

    CREATE TRIGGER IF NOT EXISTS deny_delete_venue_accounting_events
    BEFORE DELETE ON venue_accounting_events
    BEGIN
        SELECT RAISE(ABORT, 'venue_accounting_events is append-only');
    END;
    """


def _postgres_venue_accounting() -> tuple[str, ...]:
    statements = list(_postgres_portable_schema(VENUE_ACCOUNTING_EVENTS_SQLITE))
    statements.extend(
        (
            "DROP TRIGGER IF EXISTS deny_update_venue_accounting_events "
            "ON venue_accounting_events",
            """
            CREATE TRIGGER deny_update_venue_accounting_events
            BEFORE UPDATE ON venue_accounting_events
            FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
            """,
            "DROP TRIGGER IF EXISTS deny_delete_venue_accounting_events "
            "ON venue_accounting_events",
            """
            CREATE TRIGGER deny_delete_venue_accounting_events
            BEFORE DELETE ON venue_accounting_events
            FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
            """,
        )
    )
    return tuple(statements)


def _sqlite_performance_accounting_triggers() -> str:
    statements: list[str] = []
    for table in ("venue_accounting_attributions", "venue_fee_conversions"):
        statements.extend(
            (
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_update_{table}
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_delete_{table}
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
            )
        )
    return ";\n".join(statement.strip() for statement in statements) + ";"


def _postgres_performance_accounting() -> tuple[str, ...]:
    statements = list(_postgres_portable_schema(VENUE_PERFORMANCE_ACCOUNTING_SQLITE))
    for table in ("venue_accounting_attributions", "venue_fee_conversions"):
        statements.extend(
            (
                f"DROP TRIGGER IF EXISTS deny_update_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_update_{table}
                BEFORE UPDATE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
                f"DROP TRIGGER IF EXISTS deny_delete_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_delete_{table}
                BEFORE DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
            )
        )
    return tuple(statements)


def _sqlite_external_evidence_triggers() -> str:
    statements: list[str] = []
    for table in ("external_evidence", "candidate_evidence_links"):
        statements.extend(
            (
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_update_{table}
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_delete_{table}
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
            )
        )
    return ";\n".join(statement.strip() for statement in statements) + ";"


def _postgres_external_evidence() -> tuple[str, ...]:
    statements = list(_postgres_portable_schema(EXTERNAL_EVIDENCE_SQLITE))
    for table in ("external_evidence", "candidate_evidence_links"):
        statements.extend(
            (
                f"DROP TRIGGER IF EXISTS deny_update_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_update_{table}
                BEFORE UPDATE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
                f"DROP TRIGGER IF EXISTS deny_delete_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_delete_{table}
                BEFORE DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
            )
        )
    return tuple(statements)


STRATEGY_RESEARCH_RUNS_SQLITE = """
CREATE TABLE IF NOT EXISTS strategy_research_runs (
    research_run_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    parent_version TEXT NOT NULL,
    recommendation TEXT NOT NULL CHECK(recommendation IN ('PROPOSE','NO_CHANGE')),
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    response_id TEXT NOT NULL,
    latency_ms INTEGER NOT NULL CHECK(latency_ms >= 0),
    evidence_ids_json TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    rationale_json TEXT NOT NULL,
    expected_failure_modes_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_research_response
    ON strategy_research_runs(model, response_id);
CREATE INDEX IF NOT EXISTS idx_strategy_research_trace
    ON strategy_research_runs(trace_id, created_at);
"""


def _sqlite_strategy_research_triggers() -> str:
    return """
    CREATE TRIGGER IF NOT EXISTS deny_update_strategy_research_runs
    BEFORE UPDATE ON strategy_research_runs
    BEGIN
        SELECT RAISE(ABORT, 'strategy_research_runs is append-only');
    END;
    CREATE TRIGGER IF NOT EXISTS deny_delete_strategy_research_runs
    BEFORE DELETE ON strategy_research_runs
    BEGIN
        SELECT RAISE(ABORT, 'strategy_research_runs is append-only');
    END;
    """


def _postgres_strategy_research_runs() -> tuple[str, ...]:
    return (
        *_postgres_portable_schema(STRATEGY_RESEARCH_RUNS_SQLITE),
        "DROP TRIGGER IF EXISTS deny_update_strategy_research_runs "
        "ON strategy_research_runs",
        """
        CREATE TRIGGER deny_update_strategy_research_runs
        BEFORE UPDATE ON strategy_research_runs
        FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
        """,
        "DROP TRIGGER IF EXISTS deny_delete_strategy_research_runs "
        "ON strategy_research_runs",
        """
        CREATE TRIGGER deny_delete_strategy_research_runs
        BEFORE DELETE ON strategy_research_runs
        FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
        """,
    )


STRATEGY_REGISTRY_AND_SHADOW_JOURNAL_SQLITE = """
CREATE TABLE IF NOT EXISTS strategy_registry_events (
    registry_event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('ROLLBACK')),
    source_promotion_record_id TEXT NOT NULL UNIQUE
        REFERENCES promotion_records(promotion_record_id) ON DELETE RESTRICT,
    previous_champion_version TEXT NOT NULL,
    resulting_champion_version TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_strategy_registry_events_trace
    ON strategy_registry_events(trace_id, created_at);

CREATE TABLE IF NOT EXISTS shadow_journal_events (
    shadow_journal_event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    pair_id TEXT NOT NULL,
    champion_spec_id TEXT NOT NULL REFERENCES strategy_specs(spec_id) ON DELETE RESTRICT,
    challenger_spec_id TEXT NOT NULL REFERENCES strategy_specs(spec_id) ON DELETE RESTRICT,
    pair_started_at TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('COVERAGE','TRADE')),
    event_key TEXT NOT NULL,
    spec_id TEXT REFERENCES strategy_specs(spec_id) ON DELETE RESTRICT,
    payload_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK(champion_spec_id <> challenger_spec_id),
    CHECK(
        (event_type='COVERAGE' AND spec_id IS NULL)
        OR (event_type='TRADE' AND spec_id IS NOT NULL)
    ),
    UNIQUE(pair_id, event_type, event_key)
);

CREATE INDEX IF NOT EXISTS idx_shadow_journal_pair
    ON shadow_journal_events(pair_id, observed_at, created_at);
CREATE INDEX IF NOT EXISTS idx_shadow_journal_trace
    ON shadow_journal_events(trace_id, created_at);
"""


def _sqlite_registry_shadow_journal_triggers() -> str:
    statements: list[str] = []
    for table in ("strategy_registry_events", "shadow_journal_events"):
        statements.extend(
            (
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_update_{table}
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
                f"""
                CREATE TRIGGER IF NOT EXISTS deny_delete_{table}
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{table} is append-only');
                END
                """,
            )
        )
    return ";".join(statements) + ";"


def _postgres_registry_shadow_journal() -> tuple[str, ...]:
    statements = list(_postgres_portable_schema(STRATEGY_REGISTRY_AND_SHADOW_JOURNAL_SQLITE))
    for table in ("strategy_registry_events", "shadow_journal_events"):
        statements.extend(
            (
                f"DROP TRIGGER IF EXISTS deny_update_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_update_{table}
                BEFORE UPDATE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
                f"DROP TRIGGER IF EXISTS deny_delete_{table} ON {table}",
                f"""
                CREATE TRIGGER deny_delete_{table}
                BEFORE DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION deny_crypto_audit_mutation()
                """,
            )
        )
    return tuple(statements)


MIGRATIONS = (
    Migration(
        version=1,
        name="create_crypto_audit_ledger",
        sqlite_script=AUDIT_TABLES_SQLITE,
        postgres_statements=_postgres_audit_schema(),
    ),
    Migration(
        version=2,
        name="enforce_append_only_decisions_and_theses",
        sqlite_script=_sqlite_append_only_triggers(),
        postgres_statements=_postgres_append_only_triggers(),
    ),
    Migration(
        version=3,
        name="add_append_only_venue_order_event_history",
        sqlite_script=ORDER_EVENT_HISTORY_SQLITE + _sqlite_order_history_triggers(),
        postgres_statements=_postgres_order_history(),
    ),
    Migration(
        version=4,
        name="make_all_crypto_audit_evidence_append_only",
        sqlite_script=_sqlite_remaining_append_only_triggers(),
        postgres_statements=_postgres_remaining_append_only_triggers(),
    ),
    Migration(
        version=5,
        name="add_point_in_time_external_evidence",
        sqlite_script=EXTERNAL_EVIDENCE_SQLITE + _sqlite_external_evidence_triggers(),
        postgres_statements=_postgres_external_evidence(),
    ),
    Migration(
        version=6,
        name="record_first_class_llm_attempted_direction",
        sqlite_script="""
        ALTER TABLE llm_decisions ADD COLUMN direction TEXT
            CHECK(direction IN ('LONG','SHORT'));
        """,
        postgres_statements=(
            """
            ALTER TABLE llm_decisions ADD COLUMN direction TEXT
                CHECK(direction IN ('LONG','SHORT'))
            """,
        ),
    ),
    Migration(
        version=7,
        name="add_append_only_venue_accounting_events",
        sqlite_script=(
            VENUE_ACCOUNTING_EVENTS_SQLITE + _sqlite_venue_accounting_triggers()
        ),
        postgres_statements=_postgres_venue_accounting(),
    ),
    Migration(
        version=8,
        name="add_attributed_performance_accounting",
        sqlite_script=(
            VENUE_PERFORMANCE_ACCOUNTING_SQLITE
            + _sqlite_performance_accounting_triggers()
        ),
        postgres_statements=_postgres_performance_accounting(),
    ),
    Migration(
        version=9,
        name="add_append_only_strategy_research_runs",
        sqlite_script=(
            STRATEGY_RESEARCH_RUNS_SQLITE + _sqlite_strategy_research_triggers()
        ),
        postgres_statements=_postgres_strategy_research_runs(),
    ),
    Migration(
        version=10,
        name="add_registry_replay_and_shadow_journal",
        sqlite_script=(
            STRATEGY_REGISTRY_AND_SHADOW_JOURNAL_SQLITE
            + _sqlite_registry_shadow_journal_triggers()
        ),
        postgres_statements=_postgres_registry_shadow_journal(),
    ),
)


SQLITE_MIGRATION_TABLE = """
CREATE TABLE IF NOT EXISTS crypto_schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL
)
"""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_applied_versions(applied: dict[int, str]) -> None:
    known_versions = {migration.version for migration in MIGRATIONS}
    unknown = sorted(set(applied) - known_versions)
    if unknown:
        raise RuntimeError(f"Database has unknown newer audit migrations: {unknown}")
    if applied:
        expected_prefix = set(range(1, max(applied) + 1))
        if set(applied) != expected_prefix:
            raise RuntimeError("Audit migration history contains a version gap")


def apply_sqlite_migrations(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(SQLITE_MIGRATION_TABLE)
    applied = {
        int(row[0]): str(row[1])
        for row in connection.execute(
            "SELECT version, checksum FROM crypto_schema_migrations"
        ).fetchall()
    }
    _validate_applied_versions(applied)
    for migration in MIGRATIONS:
        existing = applied.get(migration.version)
        if existing is not None:
            if existing != migration.checksum:
                raise RuntimeError(
                    f"Audit migration {migration.version} checksum mismatch; refusing to continue"
                )
            continue
        connection.executescript(migration.sqlite_script)
        connection.execute(
            """
            INSERT INTO crypto_schema_migrations(version, name, checksum, applied_at)
            VALUES (?, ?, ?, ?)
            """,
            (migration.version, migration.name, migration.checksum, _now()),
        )


def apply_postgres_migrations(connection: MigrationConnection) -> None:
    # Every service may start concurrently under Compose; one transaction owns schema changes.
    connection.execute("SELECT pg_advisory_xact_lock(734821906)")
    connection.execute(SQLITE_MIGRATION_TABLE)
    rows = connection.execute("SELECT version, checksum FROM crypto_schema_migrations").fetchall()
    applied = {
        int(row["version"] if isinstance(row, dict) else row[0]): str(
            row["checksum"] if isinstance(row, dict) else row[1]
        )
        for row in rows
    }
    _validate_applied_versions(applied)
    for migration in MIGRATIONS:
        existing = applied.get(migration.version)
        if existing is not None:
            if existing != migration.checksum:
                raise RuntimeError(
                    f"Audit migration {migration.version} checksum mismatch; refusing to continue"
                )
            continue
        for statement in migration.postgres_statements:
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO crypto_schema_migrations(version, name, checksum, applied_at)
            VALUES (%s, %s, %s, %s)
            """,
            (migration.version, migration.name, migration.checksum, _now()),
        )
