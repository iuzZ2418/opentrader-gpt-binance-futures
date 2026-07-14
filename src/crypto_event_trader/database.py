from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain import Asset, ExtractedEvent, RawDocument, Signal

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS source_entities (
    id INTEGER PRIMARY KEY,
    source_key TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    display_name TEXT NOT NULL,
    quality_score REAL NOT NULL DEFAULT 0.5,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS canonical_assets (
    asset_id TEXT PRIMARY KEY,
    coingecko_id TEXT,
    symbol TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    chain_id TEXT,
    contract_address TEXT,
    exchange_symbols_json TEXT NOT NULL DEFAULT '{}',
    active INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source_entity_id INTEGER NOT NULL REFERENCES source_entities(id),
    source_id TEXT NOT NULL,
    doc_type TEXT NOT NULL CHECK(doc_type IN ('post','news','announcement','onchain')),
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT 'en',
    engagement_json TEXT NOT NULL DEFAULT '{}',
    raw_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source_entity_id, source_id)
);

CREATE VIEW IF NOT EXISTS raw_posts AS
SELECT * FROM documents WHERE doc_type = 'post';
CREATE VIEW IF NOT EXISTS raw_news AS
SELECT * FROM documents WHERE doc_type = 'news';
CREATE VIEW IF NOT EXISTS exchange_announcements AS
SELECT * FROM documents WHERE doc_type = 'announcement';

CREATE TABLE IF NOT EXISTS onchain_events (
    id INTEGER PRIMARY KEY,
    chain_id TEXT NOT NULL,
    tx_hash TEXT NOT NULL UNIQUE,
    asset_id TEXT REFERENCES canonical_assets(asset_id),
    address_from TEXT,
    address_to TEXT,
    amount REAL,
    label TEXT,
    event_at TEXT NOT NULL,
    raw_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS extracted_events (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    asset_id TEXT NOT NULL REFERENCES canonical_assets(asset_id),
    event_type TEXT NOT NULL,
    polarity TEXT NOT NULL,
    factuality REAL NOT NULL,
    urgency REAL NOT NULL,
    novelty REAL NOT NULL,
    sentiment REAL NOT NULL,
    bot_score REAL NOT NULL,
    source_quality REAL NOT NULL,
    confidence REAL NOT NULL,
    matched_entities_json TEXT NOT NULL,
    reasoning_tags_json TEXT NOT NULL,
    extraction_json TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    UNIQUE(document_id, asset_id)
);

CREATE TABLE IF NOT EXISTS signal_scores (
    id INTEGER PRIMARY KEY,
    event_id INTEGER NOT NULL UNIQUE REFERENCES extracted_events(id),
    asset_id TEXT NOT NULL REFERENCES canonical_assets(asset_id),
    direction INTEGER NOT NULL CHECK(direction BETWEEN -1 AND 1),
    score REAL NOT NULL,
    score_long REAL NOT NULL,
    score_short REAL NOT NULL,
    reason_json TEXT NOT NULL,
    threshold_bucket TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_prices (
    id INTEGER PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES canonical_assets(asset_id),
    price REAL NOT NULL,
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    volume_24h REAL NOT NULL,
    observed_at TEXT NOT NULL,
    UNIQUE(asset_id, observed_at)
);

CREATE TABLE IF NOT EXISTS account_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    cash REAL NOT NULL,
    initial_cash REAL NOT NULL,
    high_water_mark REAL NOT NULL,
    trading_enabled INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_orders (
    id INTEGER PRIMARY KEY,
    signal_id INTEGER NOT NULL REFERENCES signal_scores(id),
    venue TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('buy','sell')),
    intent_px REAL NOT NULL,
    order_type TEXT NOT NULL,
    quantity REAL NOT NULL,
    status TEXT NOT NULL,
    external_order_id TEXT,
    external_client_order_id TEXT,
    raw_response_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_fills (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES paper_orders(id),
    fill_px REAL NOT NULL,
    quantity REAL NOT NULL,
    fee REAL NOT NULL,
    slippage_bps REAL NOT NULL,
    filled_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    quantity REAL NOT NULL,
    average_entry REAL NOT NULL,
    mark_price REAL NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    gross_exposure REAL NOT NULL,
    drawdown REAL NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_published ON documents(published_at);
CREATE INDEX IF NOT EXISTS idx_events_asset ON extracted_events(asset_id, extracted_at);
CREATE INDEX IF NOT EXISTS idx_signals_score ON signal_scores(score DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_orders_created ON paper_orders(created_at);
"""


def _iso(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _now() -> str:
    return _iso(datetime.now(UTC))


class Repository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self, initial_cash: float) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate_paper_orders(connection)
            connection.execute(
                """
                INSERT OR IGNORE INTO account_state
                    (id, cash, initial_cash, high_water_mark, trading_enabled, updated_at)
                VALUES (1, ?, ?, ?, 1, ?)
                """,
                (initial_cash, initial_cash, initial_cash, _now()),
            )

    @staticmethod
    def _migrate_paper_orders(connection: sqlite3.Connection) -> None:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(paper_orders)")}
        additions = {
            "external_order_id": "TEXT",
            "external_client_order_id": "TEXT",
            "raw_response_json": "TEXT NOT NULL DEFAULT '{}'",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE paper_orders ADD COLUMN {name} {definition}")

    def upsert_asset(self, asset: Asset) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO canonical_assets
                    (asset_id, coingecko_id, symbol, name, aliases_json,
                     exchange_symbols_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    coingecko_id=excluded.coingecko_id,
                    symbol=excluded.symbol,
                    name=excluded.name,
                    aliases_json=excluded.aliases_json,
                    exchange_symbols_json=excluded.exchange_symbols_json,
                    updated_at=excluded.updated_at
                """,
                (
                    asset.asset_id,
                    asset.coingecko_id,
                    asset.symbol,
                    asset.name,
                    json.dumps(asset.aliases),
                    json.dumps(asset.exchange_symbols),
                    _now(),
                ),
            )

    def list_assets(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM canonical_assets WHERE active=1 ORDER BY symbol"
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["aliases"] = json.loads(item.pop("aliases_json"))
            item["exchange_symbols"] = json.loads(item.pop("exchange_symbols_json"))
            result.append(item)
        return result

    def _source_id(self, connection: sqlite3.Connection, document: RawDocument) -> int:
        quality = source_quality(document.source)
        source_type = document.doc_type.value
        connection.execute(
            """
            INSERT INTO source_entities
                (source_key, source_type, display_name, quality_score, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET quality_score=excluded.quality_score
            """,
            (document.source, source_type, document.source, quality, _now()),
        )
        row = connection.execute(
            "SELECT id FROM source_entities WHERE source_key=?", (document.source,)
        ).fetchone()
        assert row is not None
        return int(row["id"])

    def insert_document(self, document: RawDocument) -> tuple[int, bool]:
        with self.connect() as connection:
            source_id = self._source_id(connection, document)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO documents
                    (source_entity_id, source_id, doc_type, title, text, author,
                     published_at, ingested_at, url, engagement_json, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    document.source_id,
                    document.doc_type.value,
                    document.title,
                    document.text,
                    document.author,
                    _iso(document.published_at),
                    _now(),
                    document.url,
                    json.dumps(document.engagement),
                    json.dumps(document.raw),
                ),
            )
            row = connection.execute(
                "SELECT id FROM documents WHERE source_entity_id=? AND source_id=?",
                (source_id, document.source_id),
            ).fetchone()
            assert row is not None
            return int(row["id"]), cursor.rowcount == 1

    def unprocessed_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT d.*, s.source_key, s.quality_score
                FROM documents d
                JOIN source_entities s ON s.id=d.source_entity_id
                LEFT JOIN extracted_events e ON e.document_id=d.id
                WHERE e.id IS NULL
                ORDER BY d.published_at
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def insert_event(self, event: ExtractedEvent) -> int:
        payload = asdict(event)
        payload["event_type"] = event.event_type.value
        payload["polarity"] = event.polarity.value
        payload["extracted_at"] = _iso(event.extracted_at)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO extracted_events
                    (document_id, asset_id, event_type, polarity, factuality, urgency,
                     novelty, sentiment, bot_score, source_quality, confidence,
                     matched_entities_json, reasoning_tags_json, extraction_json, extracted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.document_id,
                    event.asset_id,
                    event.event_type.value,
                    event.polarity.value,
                    event.factuality,
                    event.urgency,
                    event.novelty,
                    event.sentiment,
                    event.bot_score,
                    event.source_quality,
                    event.confidence,
                    json.dumps(event.matched_entities),
                    json.dumps(event.reasoning_tags),
                    json.dumps(payload),
                    _iso(event.extracted_at),
                ),
            )
            row = connection.execute(
                "SELECT id FROM extracted_events WHERE document_id=? AND asset_id=?",
                (event.document_id, event.asset_id),
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def insert_signal(self, signal: Signal) -> int:
        reason = asdict(signal.inputs)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO signal_scores
                    (event_id, asset_id, direction, score, score_long, score_short,
                     reason_json, threshold_bucket, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.event_id,
                    signal.asset_id,
                    signal.direction,
                    signal.score,
                    signal.score if signal.direction > 0 else 0,
                    signal.score if signal.direction < 0 else 0,
                    json.dumps(reason),
                    signal.threshold_bucket,
                    _iso(signal.created_at),
                ),
            )
            row = connection.execute(
                "SELECT id FROM signal_scores WHERE event_id=?", (signal.event_id,)
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def list_events(self, limit: int = 100) -> list[dict[str, Any]]:
        query = """
            SELECT e.*, a.symbol, d.title, d.url, d.published_at, s.source_key
            FROM extracted_events e
            JOIN canonical_assets a ON a.asset_id=e.asset_id
            JOIN documents d ON d.id=e.document_id
            JOIN source_entities s ON s.id=d.source_entity_id
            ORDER BY e.extracted_at DESC LIMIT ?
        """
        with self.connect() as connection:
            rows = connection.execute(query, (limit,)).fetchall()
        return [self._decode_row(row) for row in rows]

    def list_signals(self, limit: int = 100) -> list[dict[str, Any]]:
        query = """
            SELECT ss.*, a.symbol, e.event_type, e.polarity, d.title, d.published_at
            FROM signal_scores ss
            JOIN canonical_assets a ON a.asset_id=ss.asset_id
            JOIN extracted_events e ON e.id=ss.event_id
            JOIN documents d ON d.id=e.document_id
            ORDER BY ss.created_at DESC LIMIT ?
        """
        with self.connect() as connection:
            rows = connection.execute(query, (limit,)).fetchall()
        return [self._decode_row(row) for row in rows]

    def signal_has_order(self, signal_id: int) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM paper_orders WHERE signal_id=?", (signal_id,)
            ).fetchone()
        return row is not None

    def pending_signals(self, minimum_score: float) -> list[dict[str, Any]]:
        query = """
            SELECT ss.*, a.symbol
            FROM signal_scores ss
            JOIN canonical_assets a ON a.asset_id=ss.asset_id
            LEFT JOIN paper_orders o ON o.signal_id=ss.id
            WHERE o.id IS NULL AND ss.direction != 0 AND ss.score >= ?
            ORDER BY ss.created_at
        """
        with self.connect() as connection:
            rows = connection.execute(query, (minimum_score,)).fetchall()
        return [self._decode_row(row) for row in rows]

    def account(self) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM account_state WHERE id=1").fetchone()
        if row is None:
            raise RuntimeError("Database not initialized")
        return dict(row)

    def positions(self, include_flat: bool = False) -> list[dict[str, Any]]:
        where = "" if include_flat else "WHERE ABS(quantity) > 1e-12"
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM positions {where} ORDER BY symbol"  # noqa: S608
            ).fetchall()
        return [dict(row) for row in rows]

    def portfolio(self) -> dict[str, Any]:
        account = self.account()
        positions = self.positions()
        market_value = sum(item["quantity"] * item["mark_price"] for item in positions)
        gross = sum(abs(item["quantity"] * item["mark_price"]) for item in positions)
        equity = account["cash"] + market_value
        return {
            "cash": account["cash"],
            "initial_cash": account["initial_cash"],
            "equity": equity,
            "gross_exposure": gross,
            "drawdown": max(0.0, 1 - equity / max(account["high_water_mark"], 1e-12)),
            "trading_enabled": bool(account["trading_enabled"]),
            "positions": positions,
        }

    def record_fill(
        self,
        *,
        signal_id: int,
        venue: str,
        symbol: str,
        side: str,
        intent_price: float,
        fill_price: float,
        quantity: float,
        fee: float,
        slippage_bps: float,
        external_order_id: str | None = None,
        external_client_order_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> tuple[int, int]:
        timestamp = _now()
        delta = quantity if side == "buy" else -quantity
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO paper_orders
                    (signal_id, venue, symbol, side, intent_px, order_type,
                     quantity, status, external_order_id, external_client_order_id,
                     raw_response_json, created_at)
                VALUES (?, ?, ?, ?, ?, 'market', ?, 'filled', ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    venue,
                    symbol,
                    side,
                    intent_price,
                    quantity,
                    external_order_id,
                    external_client_order_id,
                    json.dumps(raw_response or {}),
                    timestamp,
                ),
            )
            order_id = int(cursor.lastrowid)
            cursor = connection.execute(
                """
                INSERT INTO paper_fills
                    (order_id, fill_px, quantity, fee, slippage_bps, filled_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (order_id, fill_price, quantity, fee, slippage_bps, timestamp),
            )
            fill_id = int(cursor.lastrowid)

            current = connection.execute(
                "SELECT * FROM positions WHERE symbol=?", (symbol,)
            ).fetchone()
            old_qty = float(current["quantity"]) if current else 0.0
            old_avg = float(current["average_entry"]) if current else 0.0
            old_realized = float(current["realized_pnl"]) if current else 0.0
            new_qty = old_qty + delta

            if old_qty == 0 or old_qty * delta > 0:
                notional = abs(old_qty) * old_avg + abs(delta) * fill_price
                new_avg = notional / abs(new_qty) if new_qty else 0.0
                realized = old_realized
            else:
                closed = min(abs(old_qty), abs(delta))
                realized = old_realized + closed * (fill_price - old_avg) * (
                    1 if old_qty > 0 else -1
                )
                if new_qty == 0:
                    new_avg = 0.0
                elif old_qty * new_qty > 0:
                    new_avg = old_avg
                else:
                    new_avg = fill_price

            connection.execute(
                """
                INSERT INTO positions
                    (symbol, quantity, average_entry, mark_price, realized_pnl, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    quantity=excluded.quantity,
                    average_entry=excluded.average_entry,
                    mark_price=excluded.mark_price,
                    realized_pnl=excluded.realized_pnl,
                    updated_at=excluded.updated_at
                """,
                (symbol, new_qty, new_avg, fill_price, realized, timestamp),
            )
            connection.execute(
                "UPDATE account_state SET cash=cash-?-?, updated_at=? WHERE id=1",
                (delta * fill_price, fee, timestamp),
            )
            self._snapshot(connection, timestamp)
            return order_id, fill_id

    def _snapshot(self, connection: sqlite3.Connection, timestamp: str) -> None:
        account = connection.execute("SELECT * FROM account_state WHERE id=1").fetchone()
        assert account is not None
        positions = connection.execute("SELECT * FROM positions").fetchall()
        equity = float(account["cash"]) + sum(
            float(row["quantity"]) * float(row["mark_price"]) for row in positions
        )
        gross = sum(abs(float(row["quantity"]) * float(row["mark_price"])) for row in positions)
        high_water = max(float(account["high_water_mark"]), equity)
        drawdown = max(0.0, 1 - equity / max(high_water, 1e-12))
        connection.execute(
            "UPDATE account_state SET high_water_mark=?, updated_at=? WHERE id=1",
            (high_water, timestamp),
        )
        connection.execute(
            """
            INSERT INTO equity_curve (equity, cash, gross_exposure, drawdown, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (equity, account["cash"], gross, drawdown, timestamp),
        )

    def list_orders(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT o.*, f.fill_px, f.fee, f.slippage_bps, f.filled_at
                FROM paper_orders o
                LEFT JOIN paper_fills f ON f.order_id=o.id
                ORDER BY o.created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def equity_curve(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM equity_curve ORDER BY recorded_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in list(item):
            if key.endswith("_json"):
                item[key.removesuffix("_json")] = json.loads(item.pop(key) or "{}")
        return item


def source_quality(source: str) -> float:
    lowered = source.lower()
    if "official" in lowered or lowered.startswith(("okx", "bybit", "binance")):
        return 0.92
    if "security" in lowered or "news" in lowered:
        return 0.78
    if "unverified" in lowered or "anonymous" in lowered:
        return 0.35
    return 0.55


def seed_assets(repository: Repository, universe: Sequence[str]) -> None:
    known = {
        "BTC": Asset(
            "bitcoin",
            "BTC",
            "Bitcoin",
            ("BTC", "$BTC", "Bitcoin"),
            "bitcoin",
            {"okx": "BTC-USDT", "bybit": "BTCUSDT", "binance_futures": "BTCUSDT"},
        ),
        "ETH": Asset(
            "ethereum",
            "ETH",
            "Ethereum",
            ("ETH", "$ETH", "Ethereum", "Ether"),
            "ethereum",
            {"okx": "ETH-USDT", "bybit": "ETHUSDT", "binance_futures": "ETHUSDT"},
        ),
        "SOL": Asset(
            "solana",
            "SOL",
            "Solana",
            ("SOL", "$SOL", "Solana"),
            "solana",
            {"okx": "SOL-USDT", "bybit": "SOLUSDT", "binance_futures": "SOLUSDT"},
        ),
    }
    for symbol in universe:
        asset = known.get(symbol.upper())
        if asset:
            repository.upsert_asset(asset)
