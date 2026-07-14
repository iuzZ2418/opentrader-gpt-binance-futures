from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import JsonValue

from .audit import AuditRepository
from .contracts import CandleInterval, MarketBar
from .domain import MarketQuote
from .market_data import DerivativesRiskSnapshot

BAR_SERIES_SCHEMA = "binance-usdm-closed-bars-v1"
QUOTE_SCHEMA = "binance-usdm-execution-quote-v1"
DERIVATIVES_SCHEMA = "binance-usdm-derivatives-risk-v1"
LINEAGE_REFERENCE_SCHEMA = "market-input-lineage-ref-v1"

_SOURCE = "binance_futures"
_VENUE = "BINANCE_USDM"
_SYMBOL_PATTERN = re.compile(r"[A-Z0-9]{2,30}\Z")


@dataclass(frozen=True, slots=True)
class MarketEvidenceBundle:
    """Three bounded views of one immutable, content-addressed market input.

    ``gpt_evidence`` is intentionally compact. ``audit_evidence`` is the complete
    normalized payload stored in ``external_evidence``. ``feature_reference`` can be
    copied into a candidate's ``feature_snapshot`` without duplicating the raw series.
    """

    evidence_id: str
    evidence_record_id: str
    version: int
    digest_sha256: str
    gpt_evidence: dict[str, JsonValue]
    audit_evidence: dict[str, JsonValue]
    feature_reference: dict[str, JsonValue]


class MarketLineageRecorder:
    """Normalize and durably bind decision inputs to the append-only audit ledger."""

    def __init__(
        self,
        audit: AuditRepository,
        *,
        source: str = _SOURCE,
        venue: str = _VENUE,
    ) -> None:
        source = source.strip()
        venue = venue.strip().upper()
        if not source or not venue:
            raise ValueError("source and venue must be non-empty")
        self.audit = audit
        self.source = source
        self.venue = venue

    def record_closed_bars(
        self,
        bars: Sequence[MarketBar],
        *,
        collected_at: datetime | None = None,
    ) -> MarketEvidenceBundle:
        """Persist a complete closed-candle input window and return its bounded views."""

        collected = _utc(collected_at or datetime.now(UTC), "collected_at")
        normalized = self._normalize_bars(bars, collected_at=collected)
        first = normalized[0]
        last = normalized[-1]
        symbol = str(first["symbol"])
        interval = str(first["interval"])
        count = len(normalized)
        payload: dict[str, JsonValue] = {
            "schema": BAR_SERIES_SCHEMA,
            "venue": self.venue,
            "kind": "closed_bars",
            "symbol": symbol,
            "interval": interval,
            "all_closed": True,
            "count": count,
            "first_open_time": first["open_time"],
            "last_close_time": last["close_time"],
            "bars": normalized,
        }
        source_id = f"{symbol}:klines:{interval}:closed-window:{count}"
        evidence_id = f"binance:{source_id}"
        stored = self._store(
            source_id=source_id,
            evidence_id=evidence_id,
            occurred_at=str(last["close_time"]),
            payload=payload,
            collected_at=collected,
            source_url="https://fapi.binance.com/fapi/v1/klines",
        )
        latest_ohlcv: dict[str, JsonValue] = {
            key: last[key] for key in ("open", "high", "low", "close", "volume")
        }
        attributes: dict[str, JsonValue] = {
            "schema": BAR_SERIES_SCHEMA,
            "symbol": symbol,
            "interval": interval,
            "count": count,
            "all_closed": True,
            "first_open_time": first["open_time"],
            "last_close_time": last["close_time"],
            "latest_ohlcv": latest_ohlcv,
        }
        return self._bundle(
            stored,
            kind="closed_bars",
            payload=payload,
            summary=(
                f"{count} fully closed Binance USD-M {interval} bars for {symbol}; "
                f"window {first['open_time']} through {last['close_time']}."
            ),
            attributes=attributes,
            reference_fields={
                "symbol": symbol,
                "interval": interval,
                "count": count,
                "first_open_time": first["open_time"],
                "last_close_time": last["close_time"],
            },
        )

    def record_execution_quote(
        self,
        quote: MarketQuote,
        *,
        collected_at: datetime | None = None,
    ) -> MarketEvidenceBundle:
        """Persist the exact quote used for deterministic execution/risk checks."""

        collected = _utc(collected_at or datetime.now(UTC), "collected_at")
        symbol = _symbol(quote.symbol)
        timestamp = _utc(quote.timestamp, "quote.timestamp")
        if timestamp > collected:
            raise ValueError("quote.timestamp cannot be after collected_at")
        bid = _number("quote.bid", quote.bid, positive=True)
        ask = _number("quote.ask", quote.ask, positive=True)
        last = _number("quote.last", quote.last, positive=True)
        volume = _number("quote.volume_24h", quote.volume_24h, minimum=0)
        if bid > ask:
            raise ValueError("quote.bid cannot exceed quote.ask")
        payload: dict[str, JsonValue] = {
            "schema": QUOTE_SCHEMA,
            "venue": self.venue,
            "kind": "execution_quote",
            "symbol": symbol,
            "timestamp": _iso(timestamp),
            "bid": bid,
            "ask": ask,
            "last": last,
            "volume_24h": volume,
            "spread_bps": _canonical_number((ask - bid) / last * 10_000),
        }
        source_id = f"{symbol}:execution-quote"
        evidence_id = f"binance:{source_id}"
        stored = self._store(
            source_id=source_id,
            evidence_id=evidence_id,
            occurred_at=timestamp,
            payload=payload,
            collected_at=collected,
            source_url="https://fapi.binance.com/fapi/v1/ticker/bookTicker",
        )
        return self._bundle(
            stored,
            kind="execution_quote",
            payload=payload,
            summary=f"Binance USD-M execution quote for {symbol} at {_iso(timestamp)}.",
            attributes={
                key: payload[key]
                for key in ("symbol", "timestamp", "bid", "ask", "last", "spread_bps")
            },
            reference_fields={"symbol": symbol, "observed_at": _iso(timestamp)},
        )

    def record_derivatives_snapshot(
        self,
        snapshot: DerivativesRiskSnapshot,
        *,
        collected_at: datetime | None = None,
    ) -> MarketEvidenceBundle:
        """Persist the complete non-directional funding/OI/ADL/book-risk snapshot."""

        collected = _utc(collected_at or datetime.now(UTC), "collected_at")
        symbol = _symbol(snapshot.symbol)
        observed = _utc(snapshot.observed_at, "snapshot.observed_at")
        if observed > collected:
            raise ValueError("snapshot.observed_at cannot be after collected_at")
        mark_price = _number("snapshot.mark_price", snapshot.mark_price, positive=True)
        index_price = _number("snapshot.index_price", snapshot.index_price, positive=True)
        funding_rate = _number("snapshot.funding_rate", snapshot.funding_rate)
        open_interest = _number("snapshot.open_interest", snapshot.open_interest, minimum=0)
        spread_bps = _number("snapshot.spread_bps", snapshot.spread_bps, minimum=0)
        depth = _number(
            "snapshot.depth_within_20bps",
            snapshot.depth_within_20bps,
            minimum=0,
        )
        expected_notional = _number(
            "snapshot.expected_order_notional",
            snapshot.expected_order_notional,
            positive=True,
        )
        oi_change = (
            _number(
                "snapshot.open_interest_change_24h_fraction",
                snapshot.open_interest_change_24h_fraction,
            )
            if snapshot.open_interest_change_24h_fraction is not None
            else None
        )
        adl = snapshot.adl_quantile
        if adl is not None and (
            isinstance(adl, bool) or not isinstance(adl, int) or not 0 <= adl <= 4
        ):
            raise ValueError("snapshot.adl_quantile must be an integer from 0 to 4 or None")
        payload: dict[str, JsonValue] = {
            "schema": DERIVATIVES_SCHEMA,
            "venue": self.venue,
            "kind": "derivatives_risk",
            "symbol": symbol,
            "observed_at": _iso(observed),
            "mark_price": mark_price,
            "index_price": index_price,
            "basis_fraction": _canonical_number(mark_price / index_price - 1),
            "funding_rate": funding_rate,
            "open_interest": open_interest,
            "open_interest_change_24h_fraction": oi_change,
            "adl_quantile": adl,
            "spread_bps": spread_bps,
            "depth_within_20bps": depth,
            "expected_order_notional": expected_notional,
            "depth_multiple": _canonical_number(depth / expected_notional),
        }
        source_id = f"{symbol}:derivatives-risk"
        evidence_id = f"binance:{source_id}"
        stored = self._store(
            source_id=source_id,
            evidence_id=evidence_id,
            occurred_at=observed,
            payload=payload,
            collected_at=collected,
        )
        return self._bundle(
            stored,
            kind="derivatives_risk",
            payload=payload,
            summary=f"Binance USD-M derivatives risk snapshot for {symbol} at {_iso(observed)}.",
            attributes={
                key: payload[key]
                for key in (
                    "symbol",
                    "observed_at",
                    "mark_price",
                    "index_price",
                    "basis_fraction",
                    "funding_rate",
                    "open_interest",
                    "open_interest_change_24h_fraction",
                    "adl_quantile",
                    "spread_bps",
                    "depth_multiple",
                )
            },
            reference_fields={"symbol": symbol, "observed_at": _iso(observed)},
        )

    def _normalize_bars(
        self,
        bars: Sequence[MarketBar],
        *,
        collected_at: datetime,
    ) -> list[dict[str, JsonValue]]:
        materialized = tuple(bars)
        if not materialized:
            raise ValueError("bar series must not be empty")
        if not all(isinstance(bar, MarketBar) for bar in materialized):
            raise TypeError("bar series must contain only MarketBar values")
        symbol = _symbol(materialized[0].symbol)
        interval = materialized[0].interval
        if not isinstance(interval, CandleInterval):
            raise ValueError("bar interval must be a supported CandleInterval")
        normalized: list[dict[str, JsonValue]] = []
        prior_open: datetime | None = None
        prior_close: datetime | None = None
        for index, bar in enumerate(materialized):
            if _symbol(bar.symbol) != symbol:
                raise ValueError("all bars must have the same symbol")
            if bar.interval is not interval:
                raise ValueError("all bars must have the same interval")
            if bar.is_closed is not True:
                raise ValueError(f"bar[{index}] is not closed")
            open_time = _utc(bar.open_time, f"bar[{index}].open_time")
            close_time = _utc(bar.close_time, f"bar[{index}].close_time")
            if close_time > collected_at:
                raise ValueError(f"bar[{index}].close_time is after collected_at")
            if prior_open is not None and open_time <= prior_open:
                raise ValueError("bars must be strictly ordered by open_time")
            if prior_close is not None and close_time <= prior_close:
                raise ValueError("bars must be strictly ordered by close_time")
            if prior_close is not None and open_time < prior_close:
                raise ValueError("bars must not overlap")
            open_price = _number(f"bar[{index}].open", bar.open, positive=True)
            high = _number(f"bar[{index}].high", bar.high, positive=True)
            low = _number(f"bar[{index}].low", bar.low, positive=True)
            close = _number(f"bar[{index}].close", bar.close, positive=True)
            volume = _number(f"bar[{index}].volume", bar.volume, minimum=0)
            if close_time <= open_time:
                raise ValueError(f"bar[{index}].close_time must be after open_time")
            if low > min(open_price, close) or high < max(open_price, close) or low > high:
                raise ValueError(f"bar[{index}] has inconsistent OHLC values")
            normalized.append(
                {
                    "open_time": _iso(open_time),
                    "close_time": _iso(close_time),
                    "symbol": symbol,
                    "interval": interval.value,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "is_closed": True,
                }
            )
            prior_open = open_time
            prior_close = close_time
        return normalized

    def _store(
        self,
        *,
        source_id: str,
        evidence_id: str,
        occurred_at: datetime | str,
        payload: Mapping[str, JsonValue],
        collected_at: datetime,
        source_url: str | None = None,
    ) -> dict[str, Any]:
        latest = self.audit.latest_external_evidence(evidence_id)
        if latest is not None and latest.get("payload") != payload:
            latest_created = _utc_from_storage(latest.get("created_at"), "latest.created_at")
            if collected_at < latest_created:
                raise ValueError("changed evidence cannot be observed before its latest version")
        stored = self.audit.ensure_external_evidence(
            source=self.source,
            source_id=source_id,
            source_url=source_url,
            evidence_id=evidence_id,
            occurred_at=occurred_at,
            first_observed_at=collected_at,
            created_at=collected_at,
            payload=payload,
        )
        if stored.get("payload") != payload:
            raise RuntimeError("stored market evidence does not match the normalized input")
        return stored

    def _bundle(
        self,
        stored: Mapping[str, Any],
        *,
        kind: str,
        payload: Mapping[str, JsonValue],
        summary: str,
        attributes: Mapping[str, JsonValue],
        reference_fields: Mapping[str, JsonValue],
    ) -> MarketEvidenceBundle:
        evidence_id = str(stored["evidence_id"])
        record_id = str(stored["evidence_record_id"])
        digest = str(stored["content_hash"])
        version = int(stored["version"])
        version_observed_at = str(stored["created_at"])
        gpt_evidence: dict[str, JsonValue] = {
            "evidence_id": evidence_id,
            "evidence_record_id": record_id,
            "source": self.source,
            "source_type": kind,
            "source_id": str(stored["source_id"]),
            "occurred_at": str(stored["occurred_at"]),
            "first_observed_at": str(stored["first_observed_at"]),
            "observed_at": version_observed_at,
            "summary": summary,
            "confidence": 1.0,
            "content_hash": digest,
            "attributes": {
                "evidence_version": version,
                "input_digest_sha256": digest,
                **dict(attributes),
            },
        }
        feature_reference: dict[str, JsonValue] = {
            "schema": LINEAGE_REFERENCE_SCHEMA,
            "kind": kind,
            "evidence_id": evidence_id,
            "evidence_record_id": record_id,
            "evidence_version": version,
            "digest_sha256": digest,
            "version_observed_at": version_observed_at,
            **dict(reference_fields),
        }
        return MarketEvidenceBundle(
            evidence_id=evidence_id,
            evidence_record_id=record_id,
            version=version,
            digest_sha256=digest,
            gpt_evidence=gpt_evidence,
            audit_evidence=copy.deepcopy(dict(payload)),
            feature_reference=feature_reference,
        )


def _symbol(value: str) -> str:
    normalized = str(value).strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(normalized):
        raise ValueError("symbol must contain only 2-30 uppercase letters or digits")
    return normalized


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _utc_from_storage(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"{name} is missing from stored evidence")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{name} is not a valid timestamp") from exc
    return _utc(parsed, name)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _number(
    name: str,
    value: float,
    *,
    positive: bool = False,
    minimum: float | None = None,
) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if positive and numeric <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return _canonical_number(numeric)


def _canonical_number(value: float) -> float:
    return 0.0 if value == 0 else float(value)
