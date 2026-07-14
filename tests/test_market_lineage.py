from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.contracts import CandleInterval, MarketBar
from crypto_event_trader.domain import MarketQuote
from crypto_event_trader.market_data import DerivativesRiskSnapshot
from crypto_event_trader.market_lineage import (
    BAR_SERIES_SCHEMA,
    LINEAGE_REFERENCE_SCHEMA,
    MarketLineageRecorder,
)

NOW = datetime(2026, 7, 14, 12, tzinfo=UTC)


def _repository(tmp_path: Path) -> AuditRepository:
    repository = AuditRepository(f"sqlite:///{tmp_path / 'audit.db'}")
    repository.initialize()
    return repository


def _bars(
    *,
    count: int = 3,
    interval: CandleInterval = CandleInterval.ONE_HOUR,
    symbol: str = "BTCUSDT",
) -> tuple[MarketBar, ...]:
    hours = 1 if interval is CandleInterval.ONE_HOUR else 4
    start = NOW - timedelta(hours=hours * count)
    result: list[MarketBar] = []
    for index in range(count):
        open_price = 100.0 + index
        close = open_price + 0.5
        result.append(
            MarketBar(
                symbol=symbol,
                interval=interval,
                open_time=start + timedelta(hours=hours * index),
                close_time=start + timedelta(hours=hours * (index + 1)),
                open=open_price,
                high=close + 0.25,
                low=open_price - 0.25,
                close=close,
                volume=10.0 + index,
                is_closed=True,
            )
        )
    return tuple(result)


def _canonical_digest(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def test_closed_bars_persist_full_payload_but_return_bounded_gpt_view(
    tmp_path: Path,
) -> None:
    audit = _repository(tmp_path)
    recorder = MarketLineageRecorder(audit)

    bundle = recorder.record_closed_bars(_bars(), collected_at=NOW + timedelta(seconds=1))
    stored = audit.latest_external_evidence(bundle.evidence_id)

    assert stored is not None
    assert stored["evidence_record_id"] == bundle.evidence_record_id
    assert stored["content_hash"] == bundle.digest_sha256
    assert stored["payload"] == bundle.audit_evidence
    assert bundle.digest_sha256 == _canonical_digest(bundle.audit_evidence)
    assert bundle.audit_evidence["schema"] == BAR_SERIES_SCHEMA
    assert bundle.audit_evidence["all_closed"] is True
    assert bundle.audit_evidence["count"] == 3
    assert bundle.audit_evidence["bars"] == [
        {
            "open_time": "2026-07-14T09:00:00Z",
            "close_time": "2026-07-14T10:00:00Z",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "open": 100.0,
            "high": 100.75,
            "low": 99.75,
            "close": 100.5,
            "volume": 10.0,
            "is_closed": True,
        },
        {
            "open_time": "2026-07-14T10:00:00Z",
            "close_time": "2026-07-14T11:00:00Z",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "open": 101.0,
            "high": 101.75,
            "low": 100.75,
            "close": 101.5,
            "volume": 11.0,
            "is_closed": True,
        },
        {
            "open_time": "2026-07-14T11:00:00Z",
            "close_time": "2026-07-14T12:00:00Z",
            "symbol": "BTCUSDT",
            "interval": "1h",
            "open": 102.0,
            "high": 102.75,
            "low": 101.75,
            "close": 102.5,
            "volume": 12.0,
            "is_closed": True,
        },
    ]
    attributes = bundle.gpt_evidence["attributes"]
    assert isinstance(attributes, dict)
    assert "bars" not in attributes
    assert attributes["latest_ohlcv"] == {
        "open": 102.0,
        "high": 102.75,
        "low": 101.75,
        "close": 102.5,
        "volume": 12.0,
    }
    assert bundle.feature_reference == {
        "schema": LINEAGE_REFERENCE_SCHEMA,
        "kind": "closed_bars",
        "evidence_id": bundle.evidence_id,
        "evidence_record_id": bundle.evidence_record_id,
        "evidence_version": 1,
        "digest_sha256": bundle.digest_sha256,
        "version_observed_at": "2026-07-14T12:00:01Z",
        "symbol": "BTCUSDT",
        "interval": "1h",
        "count": 3,
        "first_open_time": "2026-07-14T09:00:00Z",
        "last_close_time": "2026-07-14T12:00:00Z",
    }


def test_exact_bar_window_reuses_record_and_changed_window_appends_version(
    tmp_path: Path,
) -> None:
    audit = _repository(tmp_path)
    recorder = MarketLineageRecorder(audit)
    bars = _bars()

    first = recorder.record_closed_bars(bars, collected_at=NOW + timedelta(seconds=1))
    duplicate = recorder.record_closed_bars(bars, collected_at=NOW + timedelta(seconds=2))
    revised_last = bars[-1].model_copy(update={"high": 103.0, "close": 102.75})
    changed = recorder.record_closed_bars(
        (*bars[:-1], revised_last),
        collected_at=NOW + timedelta(seconds=3),
    )
    latest = audit.latest_external_evidence(first.evidence_id)

    assert duplicate.evidence_record_id == first.evidence_record_id
    assert duplicate.version == 1
    assert changed.evidence_id == first.evidence_id
    assert changed.evidence_record_id != first.evidence_record_id
    assert changed.version == 2
    assert latest is not None
    assert latest["prior_evidence_record_id"] == first.evidence_record_id
    assert latest["payload"]["bars"][-1]["close"] == 102.75


def test_closed_bar_validation_fails_closed_before_writing(tmp_path: Path) -> None:
    audit = _repository(tmp_path)
    recorder = MarketLineageRecorder(audit)
    bars = _bars()
    mixed_symbol = (*bars[:1], bars[1].model_copy(update={"symbol": "ETHUSDT"}), bars[2])
    mixed_interval = (
        *bars[:1],
        bars[1].model_copy(update={"interval": CandleInterval.FOUR_HOURS}),
        bars[2],
    )
    overlapping = (
        bars[0],
        bars[1].model_copy(update={"open_time": bars[0].close_time - timedelta(minutes=1)}),
        bars[2],
    )
    future = (*bars[:-1], bars[-1].model_copy(update={"close_time": NOW + timedelta(seconds=1)}))
    inconsistent = (*bars[:-1], bars[-1].model_copy(update={"high": 100.0}))
    non_finite = (*bars[:-1], bars[-1].model_copy(update={"high": float("inf")}))
    cases = (
        ((), "must not be empty"),
        ((*bars[:-1], bars[-1].model_copy(update={"is_closed": False})), "not closed"),
        (mixed_symbol, "same symbol"),
        (mixed_interval, "same interval"),
        (tuple(reversed(bars)), "strictly ordered"),
        (overlapping, "must not overlap"),
        (future, "after collected_at"),
        (inconsistent, "inconsistent OHLC"),
        (non_finite, "must be finite"),
    )

    for invalid, message in cases:
        with pytest.raises((TypeError, ValueError), match=message):
            recorder.record_closed_bars(invalid, collected_at=NOW)

    assert audit.latest_external_evidence(
        "binance:BTCUSDT:klines:1h:closed-window:3"
    ) is None


def test_execution_quote_is_normalized_versioned_and_strict(tmp_path: Path) -> None:
    audit = _repository(tmp_path)
    recorder = MarketLineageRecorder(audit)
    quote = MarketQuote(
        symbol="btcusdt",
        bid=99.0,
        ask=101.0,
        last=100.0,
        volume_24h=5_000.0,
        timestamp=NOW,
    )

    first = recorder.record_execution_quote(quote, collected_at=NOW + timedelta(seconds=1))
    duplicate = recorder.record_execution_quote(
        quote, collected_at=NOW + timedelta(seconds=2)
    )

    assert duplicate.evidence_record_id == first.evidence_record_id
    assert first.audit_evidence == {
        "schema": "binance-usdm-execution-quote-v1",
        "venue": "BINANCE_USDM",
        "kind": "execution_quote",
        "symbol": "BTCUSDT",
        "timestamp": "2026-07-14T12:00:00Z",
        "bid": 99.0,
        "ask": 101.0,
        "last": 100.0,
        "volume_24h": 5_000.0,
        "spread_bps": 200.0,
    }
    with pytest.raises(ValueError, match="bid cannot exceed"):
        recorder.record_execution_quote(
            replace(quote, bid=102.0), collected_at=NOW + timedelta(seconds=3)
        )
    with pytest.raises(ValueError, match="must be finite"):
        recorder.record_execution_quote(
            replace(quote, ask=float("nan")), collected_at=NOW + timedelta(seconds=3)
        )
    with pytest.raises(ValueError, match="before its latest version"):
        recorder.record_execution_quote(
            replace(quote, ask=102.0), collected_at=NOW + timedelta(milliseconds=500)
        )


def test_derivatives_snapshot_has_full_raw_and_compact_lineage_views(
    tmp_path: Path,
) -> None:
    audit = _repository(tmp_path)
    recorder = MarketLineageRecorder(audit)
    snapshot = DerivativesRiskSnapshot(
        symbol="BTCUSDT",
        mark_price=101.0,
        index_price=100.0,
        funding_rate=0.001,
        open_interest=1_000.0,
        adl_quantile=3,
        spread_bps=4.0,
        depth_within_20bps=20_000.0,
        expected_order_notional=1_000.0,
        observed_at=NOW,
        open_interest_change_24h_fraction=0.12,
    )

    bundle = recorder.record_derivatives_snapshot(
        snapshot, collected_at=NOW + timedelta(seconds=1)
    )
    stored = audit.latest_external_evidence(bundle.evidence_id)

    assert stored is not None
    assert stored["payload"] == bundle.audit_evidence
    assert bundle.audit_evidence["basis_fraction"] == pytest.approx(0.01)
    assert bundle.audit_evidence["depth_multiple"] == 20.0
    assert bundle.feature_reference["digest_sha256"] == bundle.digest_sha256
    attributes = bundle.gpt_evidence["attributes"]
    assert isinstance(attributes, dict)
    assert attributes["funding_rate"] == 0.001
    assert attributes["open_interest_change_24h_fraction"] == 0.12
    with pytest.raises(ValueError, match="0 to 4"):
        recorder.record_derivatives_snapshot(
            replace(snapshot, adl_quantile=5),
            collected_at=NOW + timedelta(seconds=2),
        )
    with pytest.raises(ValueError, match="must be positive"):
        recorder.record_derivatives_snapshot(
            replace(snapshot, expected_order_notional=0),
            collected_at=NOW + timedelta(seconds=2),
        )
    with pytest.raises(ValueError, match="after collected_at"):
        recorder.record_derivatives_snapshot(
            replace(snapshot, observed_at=NOW + timedelta(seconds=3)),
            collected_at=NOW + timedelta(seconds=2),
        )
