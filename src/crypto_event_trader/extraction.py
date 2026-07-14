from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .domain import Asset, EventType, ExtractedEvent, Polarity


@dataclass(frozen=True, slots=True)
class EventRule:
    event_type: EventType
    terms: tuple[str, ...]
    polarity: Polarity
    strength: float
    urgency: float


RULES = (
    EventRule(EventType.DELISTING, ("delist", "remove trading"), Polarity.NEGATIVE, 0.96, 0.90),
    EventRule(EventType.HACK, ("hack", "hacked", "breach"), Polarity.NEGATIVE, 0.98, 0.98),
    EventRule(EventType.EXPLOIT, ("exploit", "vulnerability"), Polarity.NEGATIVE, 0.96, 0.96),
    EventRule(
        EventType.LISTING, ("will list", "new listing", "lists "), Polarity.POSITIVE, 0.95, 0.88
    ),
    EventRule(
        EventType.MAINTENANCE,
        ("maintenance", "suspend deposits", "pause withdrawals"),
        Polarity.NEGATIVE,
        0.75,
        0.82,
    ),
    EventRule(
        EventType.TOKEN_UNLOCK, ("token unlock", "vesting unlock"), Polarity.NEGATIVE, 0.78, 0.70
    ),
    EventRule(
        EventType.PARTNERSHIP,
        ("partnership", "partners with", "integration"),
        Polarity.POSITIVE,
        0.70,
        0.45,
    ),
    EventRule(
        EventType.REGULATION,
        ("regulator", "sec ", "approved etf", "regulation"),
        Polarity.NEUTRAL,
        0.82,
        0.72,
    ),
    EventRule(
        EventType.WHALE_FLOW,
        ("whale", "large transfer", "exchange inflow"),
        Polarity.NEUTRAL,
        0.72,
        0.68,
    ),
    EventRule(
        EventType.OPINION,
        ("i think", "prediction", "might moon", "bullish", "bearish"),
        Polarity.NEUTRAL,
        0.25,
        0.20,
    ),
)


class BaselineEventExtractor:
    """Deterministic, auditable baseline that can later be replaced by an LLM adapter."""

    def __init__(self, assets: list[Asset]) -> None:
        self.assets = assets

    def extract(self, document: dict[str, Any]) -> list[ExtractedEvent]:
        text = f"{document.get('title', '')} {document.get('text', '')}".strip()
        lowered = text.lower()
        rule = next((item for item in RULES if any(term in lowered for term in item.terms)), None)
        if rule is None:
            rule = EventRule(EventType.UNKNOWN, (), Polarity.NEUTRAL, 0.10, 0.10)

        source_quality = float(document.get("quality_score", 0.5))
        factuality = self._factuality(text, source_quality)
        bot_score = self._bot_score(document)
        novelty = 0.90 if rule.event_type not in {EventType.OPINION, EventType.UNKNOWN} else 0.25
        sentiment = self._sentiment(lowered, rule.polarity)
        results: list[ExtractedEvent] = []

        for asset in self.assets:
            matches = self._asset_matches(text, asset)
            if not matches:
                continue
            confidence = min(
                0.99,
                0.28 * rule.strength
                + 0.24 * source_quality
                + 0.22 * factuality
                + 0.16 * (1 - bot_score)
                + 0.10 * min(1, len(matches) / 2),
            )
            tags = [f"event:{rule.event_type.value}"]
            if source_quality >= 0.85:
                tags.append("official_or_primary_source")
            if factuality >= 0.8:
                tags.append("factual_language")
            if bot_score >= 0.65:
                tags.append("elevated_bot_risk")
            results.append(
                ExtractedEvent(
                    document_id=int(document["id"]),
                    asset_id=asset.asset_id,
                    symbol=asset.symbol,
                    event_type=rule.event_type,
                    polarity=rule.polarity,
                    factuality=factuality,
                    urgency=rule.urgency,
                    novelty=novelty,
                    sentiment=sentiment,
                    bot_score=bot_score,
                    source_quality=source_quality,
                    confidence=confidence,
                    matched_entities=matches,
                    reasoning_tags=tags,
                )
            )
        return results

    @staticmethod
    def _asset_matches(text: str, asset: Asset) -> list[str]:
        matches: list[str] = []
        for alias in asset.aliases:
            pattern = re.escape(alias)
            if alias.replace("$", "").isalnum():
                pattern = rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])"
            if re.search(pattern, text, flags=re.IGNORECASE):
                matches.append(alias)
        return sorted(set(matches), key=str.lower)

    @staticmethod
    def _factuality(text: str, source_quality: float) -> float:
        hedges = sum(text.lower().count(word) for word in ("rumor", "might", "maybe", "i think"))
        evidence_terms = sum(
            text.lower().count(word)
            for word in ("official", "announced", "disclosed", "verified", "starts", "reported")
        )
        return max(
            0.05, min(0.98, 0.52 + 0.35 * source_quality + 0.05 * evidence_terms - 0.14 * hedges)
        )

    @staticmethod
    def _sentiment(text: str, polarity: Polarity) -> float:
        positive = sum(
            text.count(word) for word in ("approved", "list", "partnership", "bullish", "moon")
        )
        negative = sum(
            text.count(word) for word in ("hack", "exploit", "delist", "pause", "bearish")
        )
        lexical = max(-1.0, min(1.0, (positive - negative) / max(1, positive + negative)))
        if lexical:
            return lexical
        return 0.65 * polarity.direction

    @staticmethod
    def _bot_score(document: dict[str, Any]) -> float:
        if document.get("doc_type") != "post":
            return 0.05
        engagement = json.loads(document.get("engagement_json") or "{}")
        text = str(document.get("text", ""))
        repeated = 1 if re.search(r"(.)\1{5,}", text) else 0
        low_engagement = 1 if sum(float(v) for v in engagement.values()) < 2 else 0
        anonymous = 1 if str(document.get("author", "")).lower() in {"", "anonymous"} else 0
        return min(0.95, 0.20 + 0.25 * repeated + 0.20 * low_engagement + 0.20 * anonymous)


def assets_from_rows(rows: list[dict[str, Any]]) -> list[Asset]:
    return [
        Asset(
            asset_id=item["asset_id"],
            coingecko_id=item.get("coingecko_id"),
            symbol=item["symbol"],
            name=item["name"],
            aliases=tuple(item["aliases"]),
            exchange_symbols=item.get("exchange_symbols", {}),
        )
        for item in rows
    ]
