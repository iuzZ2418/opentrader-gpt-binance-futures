from crypto_event_trader.domain import EventType, ExtractedEvent, Polarity
from crypto_event_trader.signals import SignalScorer, threshold_bucket


def test_high_quality_listing_reaches_candidate_or_trade_bucket() -> None:
    event = ExtractedEvent(
        document_id=1,
        asset_id="solana",
        symbol="SOL",
        event_type=EventType.LISTING,
        polarity=Polarity.POSITIVE,
        factuality=0.96,
        urgency=0.88,
        novelty=0.90,
        sentiment=1.0,
        bot_score=0.05,
        source_quality=0.92,
        confidence=0.90,
        matched_entities=["SOL"],
        reasoning_tags=["official"],
    )
    signal = SignalScorer().score(1, event, market_confirmation=0.8)
    assert signal.score >= 0.82
    assert signal.threshold_bucket == "paper_trade"


def test_threshold_boundaries() -> None:
    assert threshold_bucket(0.82) == "paper_trade"
    assert threshold_bucket(0.72) == "candidate"
    assert threshold_bucket(0.60) == "alert"
    assert threshold_bucket(0.59) == "research_only"
