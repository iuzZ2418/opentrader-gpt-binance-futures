from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class SourceTier(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class EventType(StrEnum):
    REVENUE_GROWTH = "revenue_growth"
    REVENUE_DECLINE = "revenue_decline"
    ORDER_GROWTH = "order_growth"
    ORDER_DECLINE = "order_decline"
    PRICE_INCREASE = "price_increase"
    PRICE_DECLINE = "price_decline"
    MARGIN_PRESSURE = "margin_pressure"
    COST_INCREASE = "cost_increase"
    RECEIVABLE_PRESSURE = "receivable_pressure"
    INVENTORY_PRESSURE = "inventory_pressure"
    CASH_FLOW_PRESSURE = "cash_flow_pressure"
    CAPACITY_EXPANSION = "capacity_expansion"
    PROJECT_DELAY = "project_delay"
    MAJOR_CONTRACT = "major_contract"
    LITIGATION = "litigation"
    REGULATORY_PENALTY = "regulatory_penalty"
    FINANCING = "financing"
    SHAREHOLDER_CHANGE = "shareholder_change"
    R_AND_D_PROGRESS = "r_and_d_progress"
    CUSTOMER_CONCENTRATION = "customer_concentration"
    CUSTOMER_LOSS = "customer_loss"
    SUPPLIER_DISRUPTION = "supplier_disruption"
    RAW_MATERIAL_SHORTAGE = "raw_material_shortage"
    CAPACITY_UTILIZATION_UP = "capacity_utilization_up"
    CAPACITY_UTILIZATION_DOWN = "capacity_utilization_down"
    PRODUCTION_SUSPENSION = "production_suspension"
    NEW_FACTORY = "new_factory"
    NEW_PRODUCT = "new_product"
    PRODUCT_RECALL = "product_recall"
    QUALITY_ISSUE = "quality_issue"
    OVERSEAS_EXPANSION = "overseas_expansion"
    OVERSEAS_DEMAND_PRESSURE = "overseas_demand_pressure"
    TARIFF_PRESSURE = "tariff_pressure"
    FX_IMPACT = "fx_impact"
    CAPEX_INCREASE = "capex_increase"
    CAPEX_REDUCTION = "capex_reduction"
    DEBT_PRESSURE = "debt_pressure"
    GOODWILL_IMPAIRMENT = "goodwill_impairment"
    SUBSIDY = "subsidy"
    MANAGEMENT_CHANGE = "management_change"
    EQUITY_INCENTIVE = "equity_incentive"
    SHARE_REPURCHASE = "share_repurchase"
    SHAREHOLDER_REDUCTION = "shareholder_reduction"
    RELATED_PARTY_TRANSACTION = "related_party_transaction"
    OTHER = "other"


class EventStatus(StrEnum):
    OCCURRED = "occurred"
    EXPECTED = "expected"
    POSSIBLE = "possible"
    RESOLVED = "resolved"


class ChangeType(StrEnum):
    NEW = "new"
    REPEAT = "repeat"
    UPDATE = "update"
    ESCALATION = "escalation"
    REVERSAL = "reversal"
    CONFLICT = "conflict"


@dataclass(slots=True)
class Company:
    company_id: str
    name: str
    ticker: str
    aliases: tuple[str, ...] = ()
    industry: str = ""
    market: str = ""
    source_org_id: str = ""


@dataclass(slots=True)
class Document:
    source_id: str
    source_name: str
    source_tier: SourceTier
    doc_type: str
    title: str
    text: str
    published_at: datetime
    url: str = ""
    segments: tuple[EvidenceSegment, ...] = ()


@dataclass(slots=True)
class EvidenceSegment:
    text: str
    page: int | None = None
    section: str = ""


@dataclass(slots=True)
class FundamentalEvent:
    company_id: str
    ticker: str
    company_name: str
    event_type: EventType
    status: EventStatus
    direction: int
    standardized_text: str
    evidence_text: str
    evidence_page: int | None
    evidence_section: str
    source_id: str
    source_name: str
    source_tier: SourceTier
    published_at: datetime
    certainty: float
    impact_dimensions: tuple[str, ...]
    numeric_evidence: tuple[str, ...] = ()
    change_type: ChangeType = ChangeType.NEW
    novelty: float = 1.0
    confidence: float = 0.5
    value_score: float = 0.0
    score_reasons: list[str] = field(default_factory=list)
    cause_text: str = ""
    importance_reason: str = ""
    processing_method: str = "rule"
    model_version: str = ""
    review_status: str = "verified"

    @property
    def event_key(self) -> str:
        return f"{self.company_id}:{self.event_type.value}"
