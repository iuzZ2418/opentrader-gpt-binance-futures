from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentType(StrEnum):
    POST = "post"
    NEWS = "news"
    ANNOUNCEMENT = "announcement"
    ONCHAIN = "onchain"


class EventType(StrEnum):
    LISTING = "listing"
    DELISTING = "delisting"
    HACK = "hack"
    EXPLOIT = "exploit"
    PARTNERSHIP = "partnership"
    REGULATION = "regulation"
    MAINTENANCE = "exchange_maintenance"
    TOKEN_UNLOCK = "token_unlock"
    WHALE_FLOW = "whale_flow"
    OPINION = "opinion"
    UNKNOWN = "unknown"


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

    @property
    def direction(self) -> int:
        return {self.POSITIVE: 1, self.NEGATIVE: -1, self.NEUTRAL: 0}[self]


@dataclass(slots=True)
class RawDocument:
    source: str
    source_id: str
    doc_type: DocumentType
    title: str
    text: str
    published_at: datetime
    url: str = ""
    author: str = ""
    engagement: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Asset:
    asset_id: str
    symbol: str
    name: str
    aliases: tuple[str, ...]
    coingecko_id: str | None = None
    exchange_symbols: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedEvent:
    document_id: int
    asset_id: str
    symbol: str
    event_type: EventType
    polarity: Polarity
    factuality: float
    urgency: float
    novelty: float
    sentiment: float
    bot_score: float
    source_quality: float
    confidence: float
    matched_entities: list[str]
    reasoning_tags: list[str]
    extracted_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class SignalInputs:
    event_strength: float
    source_quality: float
    novelty: float
    factuality: float
    sentiment_direction: float
    market_confirmation: float
    onchain_confirmation: float
    engagement_quality: float
    bot_score: float
    pre_move_penalty: float
    illiquidity_penalty: float


@dataclass(slots=True)
class Signal:
    event_id: int
    asset_id: str
    symbol: str
    direction: int
    score: float
    threshold_bucket: str
    inputs: SignalInputs
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class MarketQuote:
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class OrderIntent:
    signal_id: int
    symbol: str
    side: str
    quantity: float
    quote: MarketQuote
    venue: str = "internal-paper"
