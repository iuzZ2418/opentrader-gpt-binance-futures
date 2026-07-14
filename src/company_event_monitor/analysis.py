from __future__ import annotations

from collections.abc import Iterable

from .domain import ChangeType, EventStatus, EventType, FundamentalEvent, SourceTier

SOURCE_WEIGHTS = {
    SourceTier.A: 1.0,
    SourceTier.B: 0.86,
    SourceTier.C: 0.82,
    SourceTier.D: 0.58,
    SourceTier.E: 0.3,
}
IMPACT_WEIGHTS = {
    "revenue": 1.0,
    "gross_margin": 1.0,
    "profit": 1.0,
    "operating_cash_flow": 1.0,
    "accounts_receivable": 0.9,
    "orders": 0.9,
    "compliance": 0.9,
    "contingent_liability": 0.85,
    "inventory": 0.8,
    "capex": 0.75,
    "capacity": 0.7,
    "cost": 0.85,
    "product": 0.65,
    "financing": 0.7,
    "dilution": 0.65,
    "customer": 0.85,
    "supply_chain": 0.85,
    "production": 0.9,
    "quality": 0.85,
    "overseas": 0.8,
    "cash_flow": 1.0,
    "debt": 1.0,
    "impairment": 0.9,
    "governance": 0.7,
    "capital_market": 0.65,
}

EVENT_FAMILIES = {
    "revenue_growth": "revenue",
    "revenue_decline": "revenue",
    "order_growth": "orders",
    "order_decline": "orders",
    "price_increase": "price",
    "price_decline": "price",
    "capacity_utilization_up": "capacity_utilization",
    "capacity_utilization_down": "capacity_utilization",
    "capex_increase": "capex_direction",
    "capex_reduction": "capex_direction",
}

EVENT_SEVERITY = {
    EventType.REGULATORY_PENALTY: 1.0,
    EventType.LITIGATION: 0.92,
    EventType.DEBT_PRESSURE: 1.0,
    EventType.CASH_FLOW_PRESSURE: 0.96,
    EventType.GOODWILL_IMPAIRMENT: 0.9,
    EventType.PRODUCTION_SUSPENSION: 0.9,
    EventType.CUSTOMER_LOSS: 0.88,
    EventType.PROJECT_DELAY: 0.82,
    EventType.MAJOR_CONTRACT: 0.82,
    EventType.REVENUE_DECLINE: 0.9,
    EventType.REVENUE_GROWTH: 0.82,
    EventType.ORDER_DECLINE: 0.86,
    EventType.ORDER_GROWTH: 0.78,
    EventType.MARGIN_PRESSURE: 0.9,
    EventType.RECEIVABLE_PRESSURE: 0.88,
    EventType.INVENTORY_PRESSURE: 0.78,
    EventType.R_AND_D_PROGRESS: 0.66,
    EventType.NEW_PRODUCT: 0.68,
    EventType.SHARE_REPURCHASE: 0.58,
    EventType.SHAREHOLDER_REDUCTION: 0.62,
    EventType.RELATED_PARTY_TRANSACTION: 0.66,
    EventType.OTHER: 0.45,
}


def event_family(event: FundamentalEvent) -> str:
    return EVENT_FAMILIES.get(event.event_type.value, event.event_type.value)


def classify_change(event: FundamentalEvent, history: Iterable[FundamentalEvent]) -> None:
    previous = sorted(
        (
            item
            for item in history
            if item.company_id == event.company_id
            and event_family(item) == event_family(event)
            and item.published_at < event.published_at
        ),
        key=lambda item: item.published_at,
    )
    if not previous:
        event.change_type, event.novelty = ChangeType.NEW, 1.0
        return
    last = previous[-1]
    days_apart = abs((event.published_at - last.published_at).days)
    if (
        last.direction != event.direction
        and last.source_name != event.source_name
        and days_apart <= 30
    ):
        event.change_type, event.novelty = ChangeType.CONFLICT, 1.0
    elif last.direction != event.direction or last.status == EventStatus.RESOLVED != (
        event.status == EventStatus.RESOLVED
    ):
        event.change_type, event.novelty = ChangeType.REVERSAL, 0.95
    elif event.certainty - last.certainty >= 0.2 or (
        last.status in {EventStatus.POSSIBLE, EventStatus.EXPECTED}
        and event.status == EventStatus.OCCURRED
    ):
        event.change_type, event.novelty = ChangeType.ESCALATION, 0.9
    elif event.numeric_evidence and event.numeric_evidence != last.numeric_evidence:
        event.change_type, event.novelty = ChangeType.UPDATE, 0.75
    else:
        event.change_type, event.novelty = ChangeType.REPEAT, 0.2


def score_event(event: FundamentalEvent) -> FundamentalEvent:
    source = SOURCE_WEIGHTS[event.source_tier]
    impact = max((IMPACT_WEIGHTS.get(item, 0.5) for item in event.impact_dimensions), default=0.5)
    severity = EVENT_SEVERITY.get(event.event_type, 0.7)
    numeric = min(1.0, 0.52 + 0.15 * len(event.numeric_evidence))
    specificity = min(1.0, 0.42 + min(len(event.evidence_text), 240) / 420)
    evidence = 0.58 * numeric + 0.42 * specificity
    change = {
        ChangeType.NEW: 1.0,
        ChangeType.ESCALATION: 1.0,
        ChangeType.REVERSAL: 0.95,
        ChangeType.CONFLICT: 1.0,
        ChangeType.UPDATE: 0.75,
        ChangeType.REPEAT: 0.2,
    }[event.change_type]
    score = (
        0.16 * event.novelty
        + 0.15 * impact
        + 0.14 * severity
        + 0.11 * source
        + 0.15 * change
        + 0.11 * evidence
        + 0.1 * event.certainty
        + 0.08 * event.confidence
    )
    if event.change_type == ChangeType.REPEAT:
        score *= 0.72
    event.value_score = round(max(0.05, min(1.0, score)), 3)
    event.score_reasons = [
        f"变化:{event.change_type.value}",
        f"来源:{event.source_tier.value}",
        f"影响:{','.join(event.impact_dimensions)}",
        f"事件强度:{severity:.2f}",
        f"确定性:{event.certainty:.2f}",
    ]
    if event.numeric_evidence:
        event.score_reasons.append("包含量化证据")
    return event
