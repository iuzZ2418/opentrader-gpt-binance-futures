from datetime import UTC, datetime

from crypto_event_trader.domain import Asset, EventType, Polarity
from crypto_event_trader.extraction import BaselineEventExtractor


def test_official_listing_is_structured() -> None:
    extractor = BaselineEventExtractor(
        [Asset("solana", "SOL", "Solana", ("SOL", "$SOL", "Solana"))]
    )
    events = extractor.extract(
        {
            "id": 1,
            "title": "Official exchange will list SOL/USDT",
            "text": "Trading for Solana starts today.",
            "source_key": "exchange_official",
            "quality_score": 0.92,
            "doc_type": "announcement",
            "published_at": datetime.now(UTC).isoformat(),
        }
    )
    assert len(events) == 1
    assert events[0].event_type == EventType.LISTING
    assert events[0].polarity == Polarity.POSITIVE
    assert events[0].confidence > 0.75


def test_symbol_matching_does_not_match_substrings() -> None:
    extractor = BaselineEventExtractor([Asset("solana", "SOL", "Solana", ("SOL",))])
    assert extractor.extract({"id": 1, "title": "A solution", "text": "No asset here"}) == []
