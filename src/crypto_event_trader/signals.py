from __future__ import annotations

from collections.abc import Mapping

from .domain import EventType, ExtractedEvent, Signal, SignalInputs

EVENT_STRENGTH = {
    EventType.LISTING: 0.95,
    EventType.DELISTING: 0.96,
    EventType.HACK: 0.98,
    EventType.EXPLOIT: 0.96,
    EventType.MAINTENANCE: 0.75,
    EventType.REGULATION: 0.82,
    EventType.TOKEN_UNLOCK: 0.78,
    EventType.WHALE_FLOW: 0.72,
    EventType.PARTNERSHIP: 0.70,
    EventType.OPINION: 0.25,
    EventType.UNKNOWN: 0.10,
}


BASELINE_SIGNAL_WEIGHTS: dict[str, float] = {
    "event_strength": 0.22,
    "source_quality": 0.18,
    "novelty": 0.14,
    "factuality": 0.12,
    "sentiment_direction": 0.10,
    "market_confirmation": 0.10,
    "onchain_confirmation": 0.08,
    "engagement_quality": 0.06,
    "bot_score": -0.14,
    "pre_move_penalty": -0.10,
    "illiquidity_penalty": -0.08,
}


class SignalScorer:
    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        selected = dict(weights or BASELINE_SIGNAL_WEIGHTS)
        expected = set(BASELINE_SIGNAL_WEIGHTS)
        if set(selected) != expected:
            missing = sorted(expected - set(selected))
            extra = sorted(set(selected) - expected)
            raise ValueError(f"Invalid signal weights; missing={missing}, extra={extra}")
        self.weights = selected

    def score(
        self,
        event_id: int,
        event: ExtractedEvent,
        *,
        market_confirmation: float = 0.75,
        onchain_confirmation: float = 0.50,
        engagement_quality: float = 0.50,
        pre_move_penalty: float = 0.05,
        illiquidity_penalty: float = 0.02,
    ) -> Signal:
        inputs = SignalInputs(
            event_strength=EVENT_STRENGTH[event.event_type],
            source_quality=event.source_quality,
            novelty=event.novelty,
            factuality=event.factuality,
            sentiment_direction=abs(event.sentiment),
            market_confirmation=market_confirmation,
            onchain_confirmation=onchain_confirmation,
            engagement_quality=engagement_quality,
            bot_score=event.bot_score,
            pre_move_penalty=pre_move_penalty,
            illiquidity_penalty=illiquidity_penalty,
        )
        value = sum(
            self.weights[name] * float(getattr(inputs, name))
            for name in BASELINE_SIGNAL_WEIGHTS
        )
        value = round(max(0.0, min(1.0, value)), 6)
        return Signal(
            event_id=event_id,
            asset_id=event.asset_id,
            symbol=event.symbol,
            direction=event.polarity.direction,
            score=value,
            threshold_bucket=threshold_bucket(value),
            inputs=inputs,
        )


def threshold_bucket(score: float) -> str:
    if score >= 0.82:
        return "paper_trade"
    if score >= 0.72:
        return "candidate"
    if score >= 0.60:
        return "alert"
    return "research_only"
