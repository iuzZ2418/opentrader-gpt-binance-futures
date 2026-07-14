from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .domain import (
    ChangeType,
    Company,
    Document,
    EventStatus,
    EventType,
    EvidenceSegment,
    FundamentalEvent,
    SourceTier,
)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return str(value.value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS companies (
    company_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    ticker TEXT NOT NULL UNIQUE,
    aliases_json TEXT NOT NULL,
    industry TEXT NOT NULL DEFAULT '',
    market TEXT NOT NULL DEFAULT '',
    source_org_id TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS company_catalog (
    company_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    market TEXT NOT NULL,
    source_org_id TEXT NOT NULL,
    industry TEXT NOT NULL DEFAULT '',
    aliases_json TEXT NOT NULL DEFAULT '[]',
    cached_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_tier TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    published_at TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    body_hash TEXT NOT NULL DEFAULT '',
    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_name, source_id),
    UNIQUE(content_hash)
);

CREATE TABLE IF NOT EXISTS fundamental_events (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    direction INTEGER NOT NULL,
    standardized_text TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    evidence_page INTEGER,
    evidence_section TEXT NOT NULL DEFAULT '',
    certainty REAL NOT NULL,
    impact_dimensions_json TEXT NOT NULL,
    numeric_evidence_json TEXT NOT NULL,
    change_type TEXT NOT NULL,
    novelty REAL NOT NULL,
    confidence REAL NOT NULL,
    value_score REAL NOT NULL,
    score_reasons_json TEXT NOT NULL,
    cause_text TEXT NOT NULL DEFAULT '',
    importance_reason TEXT NOT NULL DEFAULT '',
    processing_method TEXT NOT NULL DEFAULT 'rule',
    model_version TEXT NOT NULL DEFAULT '',
    review_status TEXT NOT NULL DEFAULT 'verified',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, company_id, event_type, evidence_text)
);

CREATE TABLE IF NOT EXISTS document_segments (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    segment_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    page INTEGER,
    section TEXT NOT NULL DEFAULT '',
    UNIQUE(document_id, segment_index)
);

CREATE TABLE IF NOT EXISTS company_documents (
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    document_id INTEGER NOT NULL REFERENCES documents(id),
    PRIMARY KEY(company_id, document_id)
);

CREATE TABLE IF NOT EXISTS disclosure_catalog (
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    source_name TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    published_at TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    selected_for_analysis INTEGER NOT NULL DEFAULT 0,
    processing_status TEXT NOT NULL DEFAULT 'discovered',
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY(source_name, source_id)
);

CREATE TABLE IF NOT EXISTS analyst_feedback (
    id INTEGER PRIMARY KEY,
    event_id INTEGER NOT NULL REFERENCES fundamental_events(id),
    label TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    analyst_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS research_theses (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    thesis_direction INTEGER NOT NULL DEFAULT 1 CHECK(thesis_direction IN (-1, 1)),
    impact_dimensions_json TEXT NOT NULL DEFAULT '[]',
    invalidation_criteria TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'tracking',
    manual_state TEXT NOT NULL DEFAULT '',
    evidence_score REAL NOT NULL DEFAULT 0,
    support_count INTEGER NOT NULL DEFAULT 0,
    contradict_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS thesis_evidence (
    thesis_id TEXT NOT NULL REFERENCES research_theses(id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL REFERENCES fundamental_events(id) ON DELETE CASCADE,
    stance TEXT NOT NULL CHECK(stance IN ('supports', 'contradicts', 'neutral')),
    relevance REAL NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(thesis_id, event_id)
);

CREATE TABLE IF NOT EXISTS management_commitments (
    id INTEGER PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    source_event_id INTEGER NOT NULL UNIQUE REFERENCES fundamental_events(id) ON DELETE CASCADE,
    commitment_text TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    impact_dimensions_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'open',
    fulfilled_event_id INTEGER REFERENCES fundamental_events(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_changes (
    id INTEGER PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    previous_document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    current_document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_family TEXT NOT NULL,
    similarity REAL NOT NULL,
    change_kind TEXT NOT NULL,
    added_text TEXT NOT NULL DEFAULT '',
    removed_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, current_document_id)
);

CREATE TABLE IF NOT EXISTS event_relations (
    id INTEGER PRIMARY KEY,
    from_event_id INTEGER NOT NULL REFERENCES fundamental_events(id),
    to_event_id INTEGER NOT NULL REFERENCES fundamental_events(id),
    relation_type TEXT NOT NULL,
    similarity REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(from_event_id, to_event_id, relation_type)
);

CREATE TABLE IF NOT EXISTS segment_annotations (
    id INTEGER PRIMARY KEY,
    segment_id INTEGER NOT NULL REFERENCES document_segments(id),
    label TEXT NOT NULL CHECK(label IN ('event','no_event','uncertain')),
    event_type TEXT,
    direction INTEGER,
    status TEXT,
    annotator TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(segment_id, annotator)
);

CREATE TABLE IF NOT EXISTS query_jobs (
    id TEXT PRIMARY KEY,
    company_id TEXT,
    query_text TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    discovered_documents INTEGER NOT NULL DEFAULT 0,
    processed_documents INTEGER NOT NULL DEFAULT 0,
    skipped_documents INTEGER NOT NULL DEFAULT 0,
    inserted_documents INTEGER NOT NULL DEFAULT 0,
    inserted_events INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS company_snapshots (
    id INTEGER PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    query_job_id TEXT REFERENCES query_jobs(id),
    window_days INTEGER NOT NULL,
    summary_json TEXT NOT NULL,
    data_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(company_id, data_version)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY,
    company_id TEXT,
    document_id INTEGER,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comparisons (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    window_days INTEGER NOT NULL,
    data_version TEXT NOT NULL DEFAULT '',
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS comparison_members (
    comparison_id TEXT NOT NULL REFERENCES comparisons(id),
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    update_status TEXT NOT NULL DEFAULT 'pending',
    data_as_of TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(comparison_id, company_id)
);

CREATE TABLE IF NOT EXISTS batch_queries (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    query_mode TEXT NOT NULL,
    criteria_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    total_companies INTEGER NOT NULL DEFAULT 0,
    completed_companies INTEGER NOT NULL DEFAULT 0,
    failed_companies INTEGER NOT NULL DEFAULT 0,
    comparison_id TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS batch_members (
    batch_id TEXT NOT NULL REFERENCES batch_queries(id) ON DELETE CASCADE,
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    position INTEGER NOT NULL,
    query_job_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    data_as_of TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(batch_id, company_id)
);

CREATE TABLE IF NOT EXISTS price_bars (
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    trade_date TEXT NOT NULL,
    open REAL NOT NULL,
    close REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    volume REAL NOT NULL,
    amount REAL NOT NULL,
    amplitude REAL NOT NULL DEFAULT 0,
    pct_change REAL NOT NULL DEFAULT 0,
    turnover REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY(company_id, trade_date)
);

CREATE TABLE IF NOT EXISTS market_analyses (
    id INTEGER PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(company_id),
    as_of TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(company_id, data_hash)
);

CREATE INDEX IF NOT EXISTS idx_documents_published ON documents(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_company_catalog_name ON company_catalog(name);
CREATE INDEX IF NOT EXISTS idx_events_company ON fundamental_events(company_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_events_score ON fundamental_events(value_score DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_relations_to ON event_relations(to_event_id);
CREATE INDEX IF NOT EXISTS idx_segments_document ON document_segments(document_id, segment_index);
CREATE INDEX IF NOT EXISTS idx_company_documents_document ON company_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_disclosure_catalog_company
    ON disclosure_catalog(company_id, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_annotations_segment ON segment_annotations(segment_id);
CREATE INDEX IF NOT EXISTS idx_query_jobs_created ON query_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_company ON company_snapshots(company_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_comparisons_created ON comparisons(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_queries_created ON batch_queries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_members_batch ON batch_members(batch_id, position);
CREATE INDEX IF NOT EXISTS idx_price_bars_company
    ON price_bars(company_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_market_analyses_company
    ON market_analyses(company_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_theses_company ON research_theses(company_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_thesis_evidence_thesis
    ON thesis_evidence(thesis_id, relevance DESC);
CREATE INDEX IF NOT EXISTS idx_commitments_company
    ON management_commitments(company_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_document_changes_company
    ON document_changes(company_id, current_document_id DESC);
"""


def content_hash(document: Document) -> str:
    normalized = " ".join(f"{document.title}\n{document.text}".split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def body_hash(document: Document) -> str:
    normalized = " ".join(document.text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


class EventRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            company_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(companies)")
            }
            if "market" not in company_columns:
                connection.execute(
                    "ALTER TABLE companies ADD COLUMN market TEXT NOT NULL DEFAULT ''"
                )
            if "source_org_id" not in company_columns:
                connection.execute(
                    "ALTER TABLE companies ADD COLUMN source_org_id TEXT NOT NULL DEFAULT ''"
                )
            catalog_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(company_catalog)")
            }
            if "industry" not in catalog_columns:
                connection.execute(
                    "ALTER TABLE company_catalog ADD COLUMN industry TEXT NOT NULL DEFAULT ''"
                )
            query_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(query_jobs)")
            }
            if "skipped_documents" not in query_columns:
                connection.execute(
                    "ALTER TABLE query_jobs ADD COLUMN skipped_documents INTEGER NOT NULL DEFAULT 0"
                )
            document_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(documents)")
            }
            if "body_hash" not in document_columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN body_hash TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_body_hash ON documents(body_hash)"
            )
            document_rows = connection.execute(
                "SELECT id, text FROM documents WHERE body_hash='' AND TRIM(text)<>''"
            ).fetchall()
            for row in document_rows:
                normalized = " ".join(str(row["text"]).split())
                connection.execute(
                    "UPDATE documents SET body_hash=? WHERE id=?",
                    (hashlib.sha256(normalized.encode("utf-8")).hexdigest(), row["id"]),
                )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(fundamental_events)")
            }
            if "evidence_page" not in columns:
                connection.execute(
                    "ALTER TABLE fundamental_events ADD COLUMN evidence_page INTEGER"
                )
            if "evidence_section" not in columns:
                connection.execute(
                    """ALTER TABLE fundamental_events
                       ADD COLUMN evidence_section TEXT NOT NULL DEFAULT ''"""
                )
            event_additions = {
                "cause_text": "TEXT NOT NULL DEFAULT ''",
                "importance_reason": "TEXT NOT NULL DEFAULT ''",
                "processing_method": "TEXT NOT NULL DEFAULT 'rule'",
                "model_version": "TEXT NOT NULL DEFAULT ''",
                "review_status": "TEXT NOT NULL DEFAULT 'verified'",
            }
            for name, definition in event_additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE fundamental_events ADD COLUMN {name} {definition}"
                    )

    def upsert_company(self, company: Company) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO companies
                    (company_id, name, ticker, aliases_json, industry, market, source_org_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(company_id) DO UPDATE SET
                    name=excluded.name, ticker=excluded.ticker,
                    aliases_json=excluded.aliases_json, industry=excluded.industry,
                    market=excluded.market, source_org_id=excluded.source_org_id
                """,
                (
                    company.company_id,
                    company.name,
                    company.ticker,
                    json.dumps(company.aliases),
                    company.industry,
                    company.market,
                    company.source_org_id,
                ),
            )

    def cache_companies(self, companies: list[Company]) -> None:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as connection:
            connection.executemany(
                """INSERT INTO company_catalog
                   (company_id, ticker, name, market, source_org_id, industry,
                    aliases_json, cached_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(company_id) DO UPDATE SET ticker=excluded.ticker,
                     name=excluded.name, market=excluded.market,
                     source_org_id=excluded.source_org_id,
                     industry=excluded.industry,
                     aliases_json=excluded.aliases_json, cached_at=excluded.cached_at""",
                [
                    (
                        company.company_id,
                        company.ticker,
                        company.name,
                        company.market,
                        company.source_org_id,
                        company.industry,
                        json.dumps(company.aliases, ensure_ascii=False),
                        now,
                    )
                    for company in companies
                ],
            )

    def search_cached_companies(self, query: str, max_age_days: int = 7) -> list[Company]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM company_catalog
                   WHERE cached_at >= datetime('now', ?)
                     AND (ticker=? OR name LIKE ? OR aliases_json LIKE ?)
                   ORDER BY CASE WHEN ticker=? OR name=? THEN 0 ELSE 1 END, ticker
                   LIMIT 10""",
                (f"-{max_age_days} days", query, f"%{query}%", f"%{query}%", query, query),
            ).fetchall()
        return [
            Company(
                company_id=row["company_id"],
                name=row["name"],
                ticker=row["ticker"],
                aliases=tuple(json.loads(row["aliases_json"])),
                industry=row["industry"],
                market=row["market"],
                source_org_id=row["source_org_id"],
            )
            for row in rows
        ]

    def list_companies(self) -> list[Company]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM companies ORDER BY ticker").fetchall()
        return [
            Company(
                row["company_id"],
                row["name"],
                row["ticker"],
                tuple(json.loads(row["aliases_json"])),
                row["industry"],
                row["market"],
                row["source_org_id"],
            )
            for row in rows
        ]

    def search_company_pool(
        self,
        keywords: list[str] | None = None,
        markets: list[str] | None = None,
        *,
        local_only: bool = False,
        limit: int = 30,
    ) -> list[Company]:
        """Search the cached official directory and/or local library for batch candidates."""
        table = "companies" if local_only else "company_catalog"
        clauses: list[str] = []
        parameters: list[Any] = []
        normalized = [item.strip() for item in (keywords or []) if item.strip()]
        if normalized:
            keyword_clauses: list[str] = []
            for keyword in normalized:
                token = f"%{keyword}%"
                condition = (
                    "(ticker LIKE ? OR name LIKE ? OR aliases_json LIKE ? OR industry LIKE ?"
                )
                parameters.extend((token, token, token, token))
                if local_only:
                    condition += (
                        " OR EXISTS (SELECT 1 FROM fundamental_events e "
                        "WHERE e.company_id=companies.company_id AND "
                        "(e.standardized_text LIKE ? OR e.evidence_text LIKE ? "
                        "OR e.cause_text LIKE ? OR e.importance_reason LIKE ?))"
                    )
                    parameters.extend((token, token, token, token))
                keyword_clauses.append(condition + ")")
            clauses.append("(" + " OR ".join(keyword_clauses) + ")")
        if markets:
            placeholders = ",".join("?" for _ in markets)
            clauses.append(f"market IN ({placeholders})")
            parameters.extend(markets)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        order = "ORDER BY ticker"
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {table} {where} {order} LIMIT ?",
                (*parameters, max(1, min(limit, 100))),
            ).fetchall()
        return [
            Company(
                company_id=row["company_id"],
                name=row["name"],
                ticker=row["ticker"],
                aliases=tuple(json.loads(row["aliases_json"])),
                industry=row["industry"],
                market=row["market"],
                source_org_id=row["source_org_id"],
            )
            for row in rows
        ]

    def get_company(self, company_id: str) -> Company | None:
        return next(
            (company for company in self.list_companies() if company.company_id == company_id),
            None,
        )

    def company_by_ticker(self, ticker: str) -> Company | None:
        return next(
            (company for company in self.list_companies() if company.ticker == ticker),
            None,
        )

    def library_companies(self) -> list[dict[str, Any]]:
        query = """
            SELECT c.*,
                   COUNT(DISTINCT d.id) document_count,
                   COUNT(DISTINCT e.id) event_count,
                   MAX(d.published_at) data_as_of,
                   MAX(CASE WHEN q.status='completed' THEN q.completed_at END) last_queried_at
            FROM companies c
            LEFT JOIN company_documents cd ON cd.company_id=c.company_id
            LEFT JOIN documents d ON d.id=cd.document_id
            LEFT JOIN fundamental_events e ON e.company_id=c.company_id
            LEFT JOIN query_jobs q ON q.company_id=c.company_id
            GROUP BY c.company_id
            ORDER BY COALESCE(last_queried_at, '') DESC, c.ticker
        """
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
            result = []
            for row in rows:
                latest = connection.execute(
                    """SELECT e.standardized_text, e.value_score, e.direction,
                              e.change_type, d.published_at
                       FROM fundamental_events e
                       JOIN documents d ON d.id=e.document_id
                       WHERE e.company_id=?
                       ORDER BY d.published_at DESC, e.value_score DESC LIMIT 1""",
                    (row["company_id"],),
                ).fetchone()
                item = dict(row)
                item["aliases"] = json.loads(item.pop("aliases_json"))
                catalog = connection.execute(
                    "SELECT COUNT(*) total FROM disclosure_catalog WHERE company_id=?",
                    (row["company_id"],),
                ).fetchone()
                item["discovered_document_count"] = int(catalog["total"] if catalog else 0)
                snapshot = connection.execute(
                    """SELECT summary_json FROM company_snapshots WHERE company_id=?
                       ORDER BY created_at DESC LIMIT 1""",
                    (row["company_id"],),
                ).fetchone()
                if snapshot:
                    saved_summary = json.loads(snapshot["summary_json"])
                    item["discovered_document_count"] = saved_summary.get(
                        "discovered_document_count",
                        item["discovered_document_count"],
                    )
                item["latest_event"] = dict(latest) if latest else None
                result.append(item)
        return result

    def latest_document_at(self, company_id: str) -> datetime | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT MAX(d.published_at) value
                   FROM documents d JOIN company_documents cd ON cd.document_id=d.id
                   WHERE cd.company_id=?""",
                (company_id,),
            ).fetchone()
        return datetime.fromisoformat(row["value"]) if row and row["value"] else None

    def latest_discovery_at(self, company_id: str) -> datetime | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT MAX(published_at) value FROM disclosure_catalog WHERE company_id=?",
                (company_id,),
            ).fetchone()
        return datetime.fromisoformat(row["value"]) if row and row["value"] else None

    def upsert_discoveries(
        self,
        company_id: str,
        documents: list[Document],
        selected_ids: set[tuple[str, str]],
    ) -> None:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as connection:
            connection.executemany(
                """INSERT INTO disclosure_catalog
                   (company_id, source_name, source_id, title, published_at, url,
                    selected_for_analysis, processing_status, last_seen_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'discovered', ?)
                   ON CONFLICT(source_name, source_id) DO UPDATE SET
                     company_id=excluded.company_id, title=excluded.title,
                     published_at=excluded.published_at, url=excluded.url,
                     selected_for_analysis=MAX(
                       disclosure_catalog.selected_for_analysis,
                       excluded.selected_for_analysis
                     ), last_seen_at=excluded.last_seen_at""",
                [
                    (
                        company_id,
                        document.source_name,
                        document.source_id,
                        document.title,
                        document.published_at.isoformat(),
                        document.url,
                        int((document.source_name, document.source_id) in selected_ids),
                        now,
                    )
                    for document in documents
                ],
            )

    def mark_discovery_processed(
        self,
        source_name: str,
        source_id: str,
        status: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE disclosure_catalog SET processing_status=?
                   WHERE source_name=? AND source_id=?""",
                (status, source_name, source_id),
            )

    def processed_disclosure_keys(self, company_id: str) -> set[tuple[str, str]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT source_name, source_id FROM disclosure_catalog
                   WHERE company_id=? AND processing_status='processed'""",
                (company_id,),
            ).fetchall()
        return {(str(row["source_name"]), str(row["source_id"])) for row in rows}

    def company_disclosures(self, company_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM disclosure_catalog WHERE company_id=?
                   ORDER BY published_at DESC LIMIT ?""",
                (company_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def source_counts(self, company_id: str) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT source_name, COUNT(*) total FROM disclosure_catalog
                   WHERE company_id=? GROUP BY source_name ORDER BY total DESC""",
                (company_id,),
            ).fetchall()
        return {str(row["source_name"]): int(row["total"]) for row in rows}

    def company_document_count(self, company_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM company_documents WHERE company_id=?",
                (company_id,),
            ).fetchone()
        return int(row[0])

    def delete_company(self, company_id: str) -> None:
        with self.connect() as connection:
            event_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM fundamental_events WHERE company_id=?", (company_id,)
                ).fetchall()
            ]
            if event_ids:
                placeholders = ",".join("?" for _ in event_ids)
                connection.execute(
                    f"DELETE FROM analyst_feedback WHERE event_id IN ({placeholders})", event_ids
                )
                connection.execute(
                    f"DELETE FROM event_relations WHERE from_event_id IN ({placeholders}) "
                    f"OR to_event_id IN ({placeholders})",
                    (*event_ids, *event_ids),
                )
            connection.execute(
                "DELETE FROM document_changes WHERE company_id=?", (company_id,)
            )
            connection.execute(
                "DELETE FROM management_commitments WHERE company_id=?", (company_id,)
            )
            connection.execute(
                "DELETE FROM research_theses WHERE company_id=?", (company_id,)
            )
            connection.execute("DELETE FROM comparison_members WHERE company_id=?", (company_id,))
            connection.execute("DELETE FROM batch_members WHERE company_id=?", (company_id,))
            connection.execute("DELETE FROM company_snapshots WHERE company_id=?", (company_id,))
            connection.execute("DELETE FROM market_analyses WHERE company_id=?", (company_id,))
            connection.execute("DELETE FROM price_bars WHERE company_id=?", (company_id,))
            connection.execute("DELETE FROM llm_calls WHERE company_id=?", (company_id,))
            connection.execute("DELETE FROM query_jobs WHERE company_id=?", (company_id,))
            connection.execute("DELETE FROM fundamental_events WHERE company_id=?", (company_id,))
            connection.execute("DELETE FROM company_documents WHERE company_id=?", (company_id,))
            connection.execute("DELETE FROM disclosure_catalog WHERE company_id=?", (company_id,))
            connection.execute("DELETE FROM companies WHERE company_id=?", (company_id,))

    def insert_document(self, document: Document) -> tuple[int, bool]:
        digest = content_hash(document)
        text_digest = body_hash(document)
        with self.connect() as connection:
            if text_digest:
                existing = connection.execute(
                    "SELECT id FROM documents WHERE body_hash=? LIMIT 1",
                    (text_digest,),
                ).fetchone()
                if existing is not None:
                    return int(existing["id"]), False
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO documents
                    (source_id, source_name, source_tier, doc_type, title, text,
                     published_at, url, content_hash, body_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.source_id,
                    document.source_name,
                    document.source_tier.value,
                    document.doc_type,
                    document.title,
                    document.text,
                    document.published_at.isoformat(),
                    document.url,
                    digest,
                    text_digest,
                ),
            )
            row = connection.execute(
                "SELECT id FROM documents WHERE (source_name=? AND source_id=?) OR content_hash=?",
                (document.source_name, document.source_id, digest),
            ).fetchone()
            assert row is not None
            document_id = int(row["id"])
            if cursor.rowcount == 1:
                connection.executemany(
                    """INSERT INTO document_segments
                       (document_id, segment_index, text, page, section)
                       VALUES (?, ?, ?, ?, ?)""",
                    [
                        (document_id, index, segment.text, segment.page, segment.section)
                        for index, segment in enumerate(document.segments)
                    ],
                )
            return document_id, cursor.rowcount == 1

    def link_company_document(self, company_id: str, document_id: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO company_documents (company_id, document_id) VALUES (?, ?)",
                (company_id, document_id),
            )

    def register_document_change(
        self,
        company_id: str,
        current_document_id: int,
    ) -> dict[str, Any] | None:
        from .research import compare_document_text, document_family

        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM document_changes WHERE company_id=? AND current_document_id=?",
                (company_id, current_document_id),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            current = connection.execute(
                "SELECT * FROM documents WHERE id=?", (current_document_id,)
            ).fetchone()
            if current is None:
                raise KeyError(current_document_id)
            family = document_family(str(current["title"]), str(current["doc_type"]))
            candidates = connection.execute(
                """SELECT d.* FROM documents d
                   JOIN company_documents cd ON cd.document_id=d.id
                   WHERE cd.company_id=? AND d.id<>? AND d.published_at<?
                   ORDER BY d.published_at DESC LIMIT 100""",
                (company_id, current_document_id, current["published_at"]),
            ).fetchall()
            previous = next(
                (
                    row
                    for row in candidates
                    if document_family(str(row["title"]), str(row["doc_type"])) == family
                ),
                None,
            )
            if previous is None:
                return None
            change = compare_document_text(str(previous["text"]), str(current["text"]))
            connection.execute(
                """INSERT OR IGNORE INTO document_changes
                   (company_id, previous_document_id, current_document_id, document_family,
                    similarity, change_kind, added_text, removed_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    company_id,
                    previous["id"],
                    current_document_id,
                    family,
                    change["similarity"],
                    change["change_kind"],
                    change["added_text"],
                    change["removed_text"],
                ),
            )
            row = connection.execute(
                "SELECT * FROM document_changes WHERE company_id=? AND current_document_id=?",
                (company_id, current_document_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def backfill_document_changes(self, company_id: str) -> int:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT d.id FROM documents d
                   JOIN company_documents cd ON cd.document_id=d.id
                   WHERE cd.company_id=? ORDER BY d.published_at""",
                (company_id,),
            ).fetchall()
            before = connection.execute(
                "SELECT COUNT(*) FROM document_changes WHERE company_id=?", (company_id,)
            ).fetchone()[0]
        for row in rows:
            self.register_document_change(company_id, int(row["id"]))
        with self.connect() as connection:
            after = connection.execute(
                "SELECT COUNT(*) FROM document_changes WHERE company_id=?", (company_id,)
            ).fetchone()[0]
        return int(after) - int(before)

    def document_event_count(self, document_id: int) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) total FROM fundamental_events WHERE document_id=?",
                (document_id,),
            ).fetchone()
        return int(row["total"] if row else 0)

    def insert_event(self, document_id: int, event: FundamentalEvent) -> tuple[int, bool]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO fundamental_events
                    (document_id, company_id, event_type, status, direction,
                     standardized_text, evidence_text, evidence_page, evidence_section, certainty,
                     impact_dimensions_json, numeric_evidence_json, change_type,
                     novelty, confidence, value_score, score_reasons_json,
                     cause_text, importance_reason, processing_method, model_version, review_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    event.company_id,
                    event.event_type.value,
                    event.status.value,
                    event.direction,
                    event.standardized_text,
                    event.evidence_text,
                    event.evidence_page,
                    event.evidence_section,
                    event.certainty,
                    json.dumps(event.impact_dimensions),
                    json.dumps(event.numeric_evidence),
                    event.change_type.value,
                    event.novelty,
                    event.confidence,
                    event.value_score,
                    json.dumps(event.score_reasons, ensure_ascii=False),
                    event.cause_text,
                    event.importance_reason,
                    event.processing_method,
                    event.model_version,
                    event.review_status,
                ),
            )
            row = connection.execute(
                """SELECT id FROM fundamental_events
                   WHERE document_id=? AND company_id=? AND event_type=? AND evidence_text=?""",
                (document_id, event.company_id, event.event_type.value, event.evidence_text),
            ).fetchone()
            assert row is not None
            return int(row["id"]), cursor.rowcount == 1

    def update_event_scoring(self, event_id: int, event: FundamentalEvent) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE fundamental_events
                   SET value_score=?, score_reasons_json=?, novelty=?, confidence=?
                   WHERE id=?""",
                (
                    event.value_score,
                    json.dumps(event.score_reasons, ensure_ascii=False),
                    event.novelty,
                    event.confidence,
                    event_id,
                ),
            )

    def documents_without_segments(self) -> list[tuple[int, str]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT d.id, d.text
                FROM documents d
                LEFT JOIN document_segments s ON s.document_id=d.id
                WHERE s.id IS NULL AND TRIM(d.text) != ''
                ORDER BY d.id
                """
            ).fetchall()
        return [(int(row["id"]), str(row["text"])) for row in rows]

    def insert_segments(
        self,
        document_id: int,
        segments: tuple[EvidenceSegment, ...],
    ) -> int:
        with self.connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """INSERT OR IGNORE INTO document_segments
                   (document_id, segment_index, text, page, section)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (document_id, index, segment.text, segment.page, segment.section)
                    for index, segment in enumerate(segments)
                ],
            )
            return connection.total_changes - before

    def history(self, company_id: str | None = None) -> list[FundamentalEvent]:
        return [self._event_from_row(row) for row in self._event_rows(company_id)]

    def list_events(self, limit: int = 100, company_id: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE e.company_id=?" if company_id else ""
        parameters: tuple[Any, ...] = (company_id, limit) if company_id else (limit,)
        query = f"""
            SELECT e.*, c.name company_name, c.ticker, d.source_id, d.source_name,
                   d.source_tier, d.published_at, d.url, d.title
            FROM fundamental_events e
            JOIN companies c ON c.company_id=e.company_id
            JOIN documents d ON d.id=e.document_id
            {where}
            ORDER BY e.value_score DESC, d.published_at DESC LIMIT ?
        """
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._event_dict(row) for row in rows]

    def events_in_window(self, company_id: str, since: datetime, limit: int = 1000) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT e.*, c.name company_name, c.ticker, d.source_id, d.source_name,
                          d.source_tier, d.published_at, d.url, d.title
                   FROM fundamental_events e
                   JOIN companies c ON c.company_id=e.company_id
                   JOIN documents d ON d.id=e.document_id
                   WHERE e.company_id=? AND d.published_at>=?
                   ORDER BY e.value_score DESC, d.published_at DESC LIMIT ?""",
                (company_id, since.isoformat(), limit),
            ).fetchall()
        return [self._event_dict(row) for row in rows]

    def create_query_job(self, job_id: str, query_text: str, company_id: str = "") -> dict:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO query_jobs
                   (id, company_id, query_text, status, stage, progress, message,
                    created_at, updated_at)
                   VALUES (?, ?, ?, 'pending', 'resolving', 0, '正在识别公司', ?, ?)""",
                (job_id, company_id or None, query_text, now, now),
            )
        return self.query_job(job_id)

    def update_query_job(self, job_id: str, **updates: Any) -> dict:
        allowed = {
            "company_id",
            "status",
            "stage",
            "progress",
            "message",
            "discovered_documents",
            "processed_documents",
            "skipped_documents",
            "inserted_documents",
            "inserted_events",
            "error",
            "completed_at",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        values["updated_at"] = datetime.now().astimezone().isoformat()
        columns = ", ".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE query_jobs SET {columns} WHERE id=?",
                (*values.values(), job_id),
            )
        return self.query_job(job_id)

    def query_job(self, job_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM query_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def active_query_jobs(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT q.*, c.name company_name, c.ticker
                   FROM query_jobs q LEFT JOIN companies c ON c.company_id=q.company_id
                   WHERE q.status IN ('pending', 'running')
                   ORDER BY q.created_at"""
            ).fetchall()
        return [dict(row) for row in rows]

    def save_snapshot(
        self,
        company_id: str,
        job_id: str,
        window_days: int,
        summary: dict,
        data_version: str,
    ) -> dict:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO company_snapshots
                   (company_id, query_job_id, window_days, summary_json, data_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(company_id, data_version) DO UPDATE SET
                     query_job_id=excluded.query_job_id, summary_json=excluded.summary_json,
                     window_days=excluded.window_days, created_at=excluded.created_at""",
                (
                    company_id,
                    job_id,
                    window_days,
                    json.dumps(summary, ensure_ascii=False, default=_json_default),
                    data_version,
                    now,
                ),
            )
        return self.latest_snapshot(company_id) or {}

    def latest_snapshot(self, company_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM company_snapshots WHERE company_id=?
                   ORDER BY created_at DESC LIMIT 1""",
                (company_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["summary"] = json.loads(result.pop("summary_json"))
        return result

    def upsert_price_bars(self, company_id: str, bars: list[Any]) -> int:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """INSERT INTO price_bars
                   (company_id, trade_date, open, close, high, low, volume, amount,
                    amplitude, pct_change, turnover, source, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(company_id, trade_date) DO UPDATE SET
                     open=excluded.open, close=excluded.close, high=excluded.high,
                     low=excluded.low, volume=excluded.volume, amount=excluded.amount,
                     amplitude=excluded.amplitude, pct_change=excluded.pct_change,
                     turnover=excluded.turnover, source=excluded.source,
                     fetched_at=excluded.fetched_at""",
                [
                    (
                        company_id,
                        bar.trade_date.isoformat(),
                        bar.open,
                        bar.close,
                        bar.high,
                        bar.low,
                        bar.volume,
                        bar.amount,
                        bar.amplitude,
                        bar.pct_change,
                        bar.turnover,
                        bar.source,
                        now,
                    )
                    for bar in bars
                ],
            )
            return connection.total_changes - before

    def price_bars(self, company_id: str, limit: int = 750) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM price_bars WHERE company_id=?
                   ORDER BY trade_date DESC LIMIT ?""",
                (company_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def save_market_analysis(self, company_id: str, analysis: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO market_analyses
                   (company_id, as_of, data_hash, analysis_json, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(company_id, data_hash) DO UPDATE SET
                     analysis_json=excluded.analysis_json, created_at=excluded.created_at""",
                (
                    company_id,
                    str(analysis.get("as_of") or ""),
                    str(analysis.get("data_hash") or ""),
                    json.dumps(analysis, ensure_ascii=False, default=_json_default),
                    now,
                ),
            )
        return self.latest_market_analysis(company_id) or analysis

    def latest_market_analysis(self, company_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM market_analyses WHERE company_id=?
                   ORDER BY created_at DESC LIMIT 1""",
                (company_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["analysis"] = json.loads(result.pop("analysis_json"))
        return result

    def record_llm_call(self, **values: Any) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO llm_calls
                   (company_id, document_id, model, prompt_version, request_hash, status,
                    input_tokens, output_tokens, latency_ms, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    values.get("company_id"),
                    values.get("document_id"),
                    values["model"],
                    values.get("prompt_version", "v1"),
                    values["request_hash"],
                    values["status"],
                    values.get("input_tokens", 0),
                    values.get("output_tokens", 0),
                    values.get("latency_ms", 0),
                    values.get("error", ""),
                ),
            )
        return int(cursor.lastrowid)

    def create_comparison(
        self,
        comparison_id: str,
        company_ids: list[str],
        window_days: int,
    ) -> dict:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO comparisons
                   (id, status, stage, progress, window_days, created_at, updated_at)
                   VALUES (?, 'pending', 'updating', 0, ?, ?, ?)""",
                (comparison_id, window_days, now, now),
            )
            connection.executemany(
                "INSERT INTO comparison_members (comparison_id, company_id) VALUES (?, ?)",
                [(comparison_id, company_id) for company_id in company_ids],
            )
        return self.comparison(comparison_id)

    def update_comparison(self, comparison_id: str, **updates: Any) -> dict:
        allowed = {
            "status",
            "stage",
            "progress",
            "window_days",
            "data_version",
            "result_json",
            "error",
            "completed_at",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if isinstance(values.get("result_json"), dict):
            values["result_json"] = json.dumps(
                values["result_json"],
                ensure_ascii=False,
                default=_json_default,
            )
        values["updated_at"] = datetime.now().astimezone().isoformat()
        columns = ", ".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE comparisons SET {columns} WHERE id=?",
                (*values.values(), comparison_id),
            )
        return self.comparison(comparison_id)

    def update_comparison_member(self, comparison_id: str, company_id: str, **updates: str) -> None:
        allowed = {"update_status", "data_as_of", "error"}
        values = {key: value for key, value in updates.items() if key in allowed}
        if not values:
            return
        columns = ", ".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE comparison_members SET {columns} WHERE comparison_id=? AND company_id=?",
                (*values.values(), comparison_id, company_id),
            )

    def comparison(self, comparison_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM comparisons WHERE id=?", (comparison_id,)
            ).fetchone()
            members = connection.execute(
                """SELECT m.*, c.name, c.ticker, c.market
                   FROM comparison_members m JOIN companies c ON c.company_id=m.company_id
                   WHERE m.comparison_id=? ORDER BY c.ticker""",
                (comparison_id,),
            ).fetchall()
        if row is None:
            raise KeyError(comparison_id)
        result = dict(row)
        result["result"] = json.loads(result.pop("result_json") or "{}")
        result["members"] = [dict(item) for item in members]
        return result

    def recent_comparisons(self, limit: int = 10) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM comparisons ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.comparison(str(row["id"])) for row in rows]

    def cached_comparison(
        self,
        data_version: str,
        company_ids: list[str],
        window_days: int,
        *,
        exclude_id: str = "",
    ) -> dict | None:
        placeholders = ",".join("?" for _ in company_ids)
        query = f"""
            SELECT c.id
            FROM comparisons c
            JOIN comparison_members m ON m.comparison_id=c.id
            WHERE c.status='completed' AND c.data_version=? AND c.window_days=?
              AND c.id<>? AND m.company_id IN ({placeholders})
            GROUP BY c.id
            HAVING COUNT(DISTINCT m.company_id)=?
               AND (SELECT COUNT(*) FROM comparison_members x WHERE x.comparison_id=c.id)=?
            ORDER BY c.completed_at DESC LIMIT 1
        """
        parameters = (
            data_version,
            window_days,
            exclude_id,
            *company_ids,
            len(company_ids),
            len(company_ids),
        )
        with self.connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return self.comparison(str(row["id"])) if row else None

    def active_comparisons(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id FROM comparisons WHERE status IN ('pending', 'running')
                   ORDER BY created_at"""
            ).fetchall()
        return [self.comparison(str(row["id"])) for row in rows]

    def create_batch_query(
        self,
        batch_id: str,
        name: str,
        query_mode: str,
        criteria: dict[str, Any],
        company_ids: list[str],
    ) -> dict[str, Any]:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO batch_queries
                   (id, name, query_mode, criteria_json, status, stage, progress,
                    total_companies, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', 'preparing', 0, ?, ?, ?)""",
                (
                    batch_id,
                    name,
                    query_mode,
                    json.dumps(criteria, ensure_ascii=False, default=_json_default),
                    len(company_ids),
                    now,
                    now,
                ),
            )
            connection.executemany(
                """INSERT INTO batch_members
                   (batch_id, company_id, position) VALUES (?, ?, ?)""",
                [(batch_id, company_id, index) for index, company_id in enumerate(company_ids)],
            )
        return self.batch_query(batch_id)

    def update_batch_query(self, batch_id: str, **updates: Any) -> dict[str, Any]:
        allowed = {
            "status",
            "stage",
            "progress",
            "completed_companies",
            "failed_companies",
            "comparison_id",
            "result_json",
            "error",
            "completed_at",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if isinstance(values.get("result_json"), dict):
            values["result_json"] = json.dumps(
                values["result_json"], ensure_ascii=False, default=_json_default
            )
        values["updated_at"] = datetime.now().astimezone().isoformat()
        columns = ", ".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE batch_queries SET {columns} WHERE id=?",
                (*values.values(), batch_id),
            )
        return self.batch_query(batch_id)

    def update_batch_member(self, batch_id: str, company_id: str, **updates: Any) -> None:
        allowed = {"query_job_id", "status", "data_as_of", "error"}
        values = {key: value for key, value in updates.items() if key in allowed}
        if not values:
            return
        columns = ", ".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE batch_members SET {columns} WHERE batch_id=? AND company_id=?",
                (*values.values(), batch_id, company_id),
            )

    def batch_query(self, batch_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM batch_queries WHERE id=?", (batch_id,)
            ).fetchone()
            members = connection.execute(
                """SELECT m.*, c.name, c.ticker, c.market, c.industry
                   FROM batch_members m JOIN companies c ON c.company_id=m.company_id
                   WHERE m.batch_id=? ORDER BY m.position""",
                (batch_id,),
            ).fetchall()
        if row is None:
            raise KeyError(batch_id)
        result = dict(row)
        result["criteria"] = json.loads(result.pop("criteria_json") or "{}")
        result["result"] = json.loads(result.pop("result_json") or "{}")
        result["members"] = [dict(item) for item in members]
        return result

    def recent_batch_queries(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM batch_queries ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.batch_query(str(row["id"])) for row in rows]

    def active_batch_queries(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id FROM batch_queries WHERE status IN ('pending', 'running')
                   ORDER BY created_at"""
            ).fetchall()
        return [self.batch_query(str(row["id"])) for row in rows]

    def create_thesis(
        self,
        company_id: str,
        title: str,
        *,
        description: str = "",
        thesis_direction: int = 1,
        impact_dimensions: list[str] | tuple[str, ...] = (),
        invalidation_criteria: str = "",
    ) -> dict[str, Any]:
        if self.get_company(company_id) is None:
            raise KeyError(company_id)
        normalized_title = title.strip()
        if len(normalized_title) < 2:
            raise ValueError("研究观点标题至少需要2个字符")
        if thesis_direction not in {-1, 1}:
            raise ValueError("观点方向只能为正向或谨慎")
        dimensions = list(dict.fromkeys(str(value).strip() for value in impact_dimensions if value))
        if not dimensions:
            raise ValueError("至少选择一个需要验证的影响维度")
        now = datetime.now().astimezone().isoformat()
        thesis_id = uuid4().hex
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO research_theses
                   (id, company_id, title, description, thesis_direction,
                    impact_dimensions_json, invalidation_criteria, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    thesis_id,
                    company_id,
                    normalized_title,
                    description.strip(),
                    thesis_direction,
                    json.dumps(dimensions, ensure_ascii=False),
                    invalidation_criteria.strip(),
                    now,
                    now,
                ),
            )
        self.refresh_research_workspace(company_id)
        return self.thesis(thesis_id)

    def thesis(self, thesis_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT t.*, c.name company_name, c.ticker
                   FROM research_theses t JOIN companies c ON c.company_id=t.company_id
                   WHERE t.id=?""",
                (thesis_id,),
            ).fetchone()
        if row is None:
            raise KeyError(thesis_id)
        result = dict(row)
        result["impact_dimensions"] = json.loads(result.pop("impact_dimensions_json") or "[]")
        result["evidence"] = self.thesis_evidence(thesis_id)
        return result

    def list_theses(
        self,
        company_id: str | None = None,
        *,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = []
        parameters: list[Any] = []
        if company_id:
            clauses.append("t.company_id=?")
            parameters.append(company_id)
        if not include_archived:
            clauses.append("t.archived=0")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT t.*, c.name company_name, c.ticker
                    FROM research_theses t JOIN companies c ON c.company_id=t.company_id
                    {where} ORDER BY t.updated_at DESC""",
                parameters,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["impact_dimensions"] = json.loads(
                item.pop("impact_dimensions_json") or "[]"
            )
            result.append(item)
        return result

    def update_thesis(self, thesis_id: str, **updates: Any) -> dict[str, Any]:
        allowed = {
            "title",
            "description",
            "thesis_direction",
            "impact_dimensions",
            "invalidation_criteria",
            "manual_state",
            "archived",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if "thesis_direction" in values and int(values["thesis_direction"]) not in {-1, 1}:
            raise ValueError("观点方向只能为正向或谨慎")
        if "manual_state" in values and values["manual_state"] not in {
            "",
            "confirmed",
            "invalidated",
        }:
            raise ValueError("人工状态只能为确认、失效或自动判断")
        if "impact_dimensions" in values:
            dimensions = [str(value) for value in values.pop("impact_dimensions") if value]
            if not dimensions:
                raise ValueError("至少选择一个需要验证的影响维度")
            values["impact_dimensions_json"] = json.dumps(
                list(dict.fromkeys(dimensions)), ensure_ascii=False
            )
        values["updated_at"] = datetime.now().astimezone().isoformat()
        columns = ", ".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT company_id FROM research_theses WHERE id=?", (thesis_id,)
            ).fetchone()
            if row is None:
                raise KeyError(thesis_id)
            connection.execute(
                f"UPDATE research_theses SET {columns} WHERE id=?",
                (*values.values(), thesis_id),
            )
            company_id = str(row["company_id"])
        self.refresh_research_workspace(company_id)
        return self.thesis(thesis_id)

    def thesis_evidence(self, thesis_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT te.*, e.standardized_text, e.evidence_text, e.evidence_page,
                          e.event_type, e.status event_status, e.direction, e.certainty,
                          e.value_score, e.change_type, e.impact_dimensions_json,
                          d.title document_title, d.url, d.published_at, d.source_name
                   FROM thesis_evidence te
                   JOIN fundamental_events e ON e.id=te.event_id
                   JOIN documents d ON d.id=e.document_id
                   WHERE te.thesis_id=?
                   ORDER BY d.published_at DESC, te.relevance DESC""",
                (thesis_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["impact_dimensions"] = json.loads(
                item.pop("impact_dimensions_json") or "[]"
            )
            result.append(item)
        return result

    def refresh_research_workspace(self, company_id: str) -> dict[str, Any]:
        from .research import evaluate_thesis, is_management_commitment, match_thesis_event

        events = self.list_events(10_000, company_id)
        theses = self.list_theses(company_id)
        now = datetime.now().astimezone().isoformat()
        with self.connect() as connection:
            for thesis in theses:
                connection.execute(
                    "DELETE FROM thesis_evidence WHERE thesis_id=?", (thesis["id"],)
                )
                for event in events:
                    match = match_thesis_event(thesis, event)
                    if match is None:
                        continue
                    connection.execute(
                        """INSERT INTO thesis_evidence
                           (thesis_id, event_id, stance, relevance, rationale)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            thesis["id"],
                            event["id"],
                            match["stance"],
                            match["relevance"],
                            match["rationale"],
                        ),
                    )
                evidence_rows = connection.execute(
                    """SELECT te.*, e.value_score, e.certainty
                       FROM thesis_evidence te JOIN fundamental_events e ON e.id=te.event_id
                       WHERE te.thesis_id=?""",
                    (thesis["id"],),
                ).fetchall()
                evaluation = evaluate_thesis(
                    [dict(row) for row in evidence_rows], str(thesis.get("manual_state") or "")
                )
                connection.execute(
                    """UPDATE research_theses
                       SET state=?, evidence_score=?, support_count=?, contradict_count=?,
                           updated_at=? WHERE id=?""",
                    (
                        evaluation["state"],
                        evaluation["score"],
                        evaluation["support_count"],
                        evaluation["contradict_count"],
                        now,
                        thesis["id"],
                    ),
                )

            for event in events:
                if not is_management_commitment(event):
                    continue
                connection.execute(
                    """INSERT OR IGNORE INTO management_commitments
                       (company_id, source_event_id, commitment_text, evidence_text,
                        impact_dimensions_json)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        company_id,
                        event["id"],
                        event["standardized_text"],
                        event["evidence_text"],
                        json.dumps(event.get("impact_dimensions") or [], ensure_ascii=False),
                    ),
                )

            commitments = connection.execute(
                """SELECT mc.*, source.event_type, source.direction,
                          source.impact_dimensions_json, source_doc.published_at source_date
                   FROM management_commitments mc
                   JOIN fundamental_events source ON source.id=mc.source_event_id
                   JOIN documents source_doc ON source_doc.id=source.document_id
                   WHERE mc.company_id=?""",
                (company_id,),
            ).fetchall()
            for commitment in commitments:
                candidates = connection.execute(
                    """SELECT e.id, e.direction, e.status, e.change_type, d.published_at
                       FROM fundamental_events e JOIN documents d ON d.id=e.document_id
                       WHERE e.company_id=? AND e.event_type=? AND e.id<>?
                         AND d.published_at>?
                       ORDER BY d.published_at DESC""",
                    (
                        company_id,
                        commitment["event_type"],
                        commitment["source_event_id"],
                        commitment["source_date"],
                    ),
                ).fetchall()
                fulfilled = next(
                    (
                        row
                        for row in candidates
                        if int(row["direction"]) == int(commitment["direction"])
                        and str(row["status"]) in {"occurred", "resolved"}
                    ),
                    None,
                )
                at_risk = next(
                    (
                        row
                        for row in candidates
                        if int(row["direction"]) != int(commitment["direction"])
                        or str(row["change_type"]) in {"reversal", "conflict"}
                    ),
                    None,
                )
                status = "fulfilled" if fulfilled else "at_risk" if at_risk else "open"
                linked = fulfilled or at_risk
                connection.execute(
                    """UPDATE management_commitments
                       SET status=?, fulfilled_event_id=?, updated_at=? WHERE id=?""",
                    (status, linked["id"] if linked else None, now, commitment["id"]),
                )
        return self.research_workspace(company_id)

    def commitments(self, company_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT mc.*, source.event_type, d.published_at, d.url, d.source_name
                   FROM management_commitments mc
                   JOIN fundamental_events source ON source.id=mc.source_event_id
                   JOIN documents d ON d.id=source.document_id
                   WHERE mc.company_id=? ORDER BY d.published_at DESC LIMIT ?""",
                (company_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["impact_dimensions"] = json.loads(
                item.pop("impact_dimensions_json") or "[]"
            )
            result.append(item)
        return result

    def document_changes(self, company_id: str, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT dc.*, previous.title previous_title,
                          current.title current_title, current.published_at,
                          current.url, current.source_name
                   FROM document_changes dc
                   JOIN documents previous ON previous.id=dc.previous_document_id
                   JOIN documents current ON current.id=dc.current_document_id
                   WHERE dc.company_id=?
                   ORDER BY current.published_at DESC LIMIT ?""",
                (company_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def research_workspace(self, company_id: str) -> dict[str, Any]:
        if self.get_company(company_id) is None:
            raise KeyError(company_id)
        events = self.list_events(10_000, company_id)
        grounded = sum(bool(item.get("evidence_text")) for item in events)
        numeric = sum(bool(item.get("numeric_evidence")) for item in events)
        return {
            "company_id": company_id,
            "theses": [
                self.thesis(str(item["id"])) for item in self.list_theses(company_id)
            ],
            "commitments": self.commitments(company_id),
            "document_changes": self.document_changes(company_id),
            "evidence_coverage": {
                "event_count": len(events),
                "grounded_count": grounded,
                "grounded_ratio": round(grounded / len(events), 3) if events else 0,
                "numeric_count": numeric,
            },
        }

    def add_feedback(self, event_id: int, label: str, note: str = "", analyst_id: str = "") -> int:
        allowed = {"valuable", "known", "irrelevant", "incorrect", "wrong_direction", "track"}
        if label not in allowed:
            raise ValueError(f"Unsupported feedback label: {label}")
        with self.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM fundamental_events WHERE id=?", (event_id,)
            ).fetchone()
            if not exists:
                raise KeyError(event_id)
            cursor = connection.execute(
                """INSERT INTO analyst_feedback (event_id, label, note, analyst_id)
                   VALUES (?, ?, ?, ?)""",
                (event_id, label, note, analyst_id),
            )
            return int(cursor.lastrowid)

    def add_relation(
        self,
        from_event_id: int,
        to_event_id: int,
        relation_type: str,
        similarity: float,
    ) -> int:
        allowed = {"supports", "updates", "escalates", "reverses", "conflicts", "duplicates"}
        if relation_type not in allowed:
            raise ValueError(f"Unsupported relation type: {relation_type}")
        with self.connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO event_relations
                   (from_event_id, to_event_id, relation_type, similarity)
                   VALUES (?, ?, ?, ?)""",
                (from_event_id, to_event_id, relation_type, similarity),
            )
            row = connection.execute(
                """SELECT id FROM event_relations
                   WHERE from_event_id=? AND to_event_id=? AND relation_type=?""",
                (from_event_id, to_event_id, relation_type),
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def relations(self, event_id: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, previous.standardized_text from_text,
                       current.standardized_text to_text
                FROM event_relations r
                JOIN fundamental_events previous ON previous.id=r.from_event_id
                JOIN fundamental_events current ON current.id=r.to_event_id
                WHERE r.from_event_id=? OR r.to_event_id=?
                ORDER BY r.id
                """,
                (event_id, event_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def event_id_for(self, event: FundamentalEvent) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT e.id
                FROM fundamental_events e
                JOIN documents d ON d.id=e.document_id
                WHERE e.company_id=? AND e.event_type=? AND e.evidence_text=?
                  AND d.source_id=? AND d.source_name=?
                """,
                (
                    event.company_id,
                    event.event_type.value,
                    event.evidence_text,
                    event.source_id,
                    event.source_name,
                ),
            ).fetchone()
        if row is None:
            raise KeyError(event.event_key)
        return int(row["id"])

    def annotation_queue(
        self,
        limit: int = 100,
        *,
        company_id: str | None = None,
        annotator: str = "",
    ) -> list[dict[str, Any]]:
        company_filter = "AND c.company_id=?" if company_id else ""
        parameters: list[Any] = [annotator]
        if company_id:
            parameters.append(company_id)
        parameters.append(limit)
        query = f"""
            SELECT s.id segment_id, s.text, s.page, s.section,
                   d.id document_id, d.title, d.url, d.published_at,
                   d.source_name, c.company_id, c.ticker, c.name company_name,
                   GROUP_CONCAT(DISTINCT e.event_type) predicted_event_types
            FROM document_segments s
            JOIN documents d ON d.id=s.document_id
            JOIN companies c ON (
                d.title LIKE '%' || c.name || '%'
                OR d.text LIKE '%' || c.name || '%'
                OR d.text LIKE '%' || c.ticker || '%'
                OR EXISTS (
                    SELECT 1 FROM json_each(c.aliases_json) alias
                    WHERE d.title LIKE '%' || alias.value || '%'
                       OR d.text LIKE '%' || alias.value || '%'
                )
            )
            LEFT JOIN segment_annotations a
                ON a.segment_id=s.id AND a.annotator=?
            LEFT JOIN fundamental_events e
                ON e.document_id=d.id AND e.company_id=c.company_id
               AND e.evidence_text=s.text
            WHERE a.id IS NULL {company_filter}
            GROUP BY s.id, c.company_id
            ORDER BY d.published_at DESC, s.segment_index
            LIMIT ?
        """
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            predicted = item.pop("predicted_event_types")
            item["predicted_event_types"] = predicted.split(",") if predicted else []
            result.append(item)
        return result

    def add_annotation(
        self,
        segment_id: int,
        *,
        label: str,
        event_type: str | None = None,
        direction: int | None = None,
        status: str | None = None,
        annotator: str = "",
        note: str = "",
    ) -> int:
        if label not in {"event", "no_event", "uncertain"}:
            raise ValueError(f"Unsupported annotation label: {label}")
        if label == "event" and not event_type:
            raise ValueError("event_type is required when label is event")
        if event_type and event_type not in {item.value for item in EventType}:
            raise ValueError(f"Unsupported event_type: {event_type}")
        if status and status not in {item.value for item in EventStatus}:
            raise ValueError(f"Unsupported status: {status}")
        if direction not in {None, -1, 0, 1}:
            raise ValueError("direction must be -1, 0, 1, or null")
        with self.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM document_segments WHERE id=?", (segment_id,)
            ).fetchone()
            if not exists:
                raise KeyError(segment_id)
            connection.execute(
                """INSERT INTO segment_annotations
                   (segment_id, label, event_type, direction, status, annotator, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(segment_id, annotator) DO UPDATE SET
                     label=excluded.label, event_type=excluded.event_type,
                     direction=excluded.direction, status=excluded.status,
                     note=excluded.note, created_at=CURRENT_TIMESTAMP""",
                (segment_id, label, event_type, direction, status, annotator, note),
            )
            row = connection.execute(
                "SELECT id FROM segment_annotations WHERE segment_id=? AND annotator=?",
                (segment_id, annotator),
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def annotations(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT a.*, s.text, s.page, s.section, d.title, d.url,
                       d.published_at, d.source_name
                FROM segment_annotations a
                JOIN document_segments s ON s.id=a.segment_id
                JOIN documents d ON d.id=s.document_id
                ORDER BY a.id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "companies",
                    "documents",
                    "fundamental_events",
                    "analyst_feedback",
                    "event_relations",
                    "document_segments",
                    "segment_annotations",
                )
            }

    def _event_rows(self, company_id: str | None = None) -> list[sqlite3.Row]:
        where = "WHERE e.company_id=?" if company_id else ""
        parameters: tuple[Any, ...] = (company_id,) if company_id else ()
        with self.connect() as connection:
            return connection.execute(
                f"""SELECT e.*, c.name company_name, c.ticker, d.source_id, d.source_name,
                          d.source_tier, d.published_at, d.url, d.title
                   FROM fundamental_events e
                   JOIN companies c ON c.company_id=e.company_id
                   JOIN documents d ON d.id=e.document_id
                   {where}
                   ORDER BY d.published_at""",
                parameters,
            ).fetchall()

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> FundamentalEvent:
        return FundamentalEvent(
            company_id=row["company_id"],
            ticker=row["ticker"],
            company_name=row["company_name"],
            event_type=EventType(row["event_type"]),
            status=EventStatus(row["status"]),
            direction=row["direction"],
            standardized_text=row["standardized_text"],
            evidence_text=row["evidence_text"],
            evidence_page=row["evidence_page"],
            evidence_section=row["evidence_section"],
            source_id=row["source_id"],
            source_name=row["source_name"],
            source_tier=SourceTier(row["source_tier"]),
            published_at=datetime.fromisoformat(row["published_at"]),
            certainty=row["certainty"],
            impact_dimensions=tuple(json.loads(row["impact_dimensions_json"])),
            numeric_evidence=tuple(json.loads(row["numeric_evidence_json"])),
            change_type=ChangeType(row["change_type"]),
            novelty=row["novelty"],
            confidence=row["confidence"],
            value_score=row["value_score"],
            score_reasons=json.loads(row["score_reasons_json"]),
            cause_text=row["cause_text"],
            importance_reason=row["importance_reason"],
            processing_method=row["processing_method"],
            model_version=row["model_version"],
            review_status=row["review_status"],
        )

    @classmethod
    def _event_dict(cls, row: sqlite3.Row) -> dict[str, Any]:
        result = asdict(cls._event_from_row(row))
        result.update(
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "url": row["url"],
                "title": row["title"],
            }
        )
        return result
