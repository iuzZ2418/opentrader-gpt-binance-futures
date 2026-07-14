from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

THESIS_STATES = {
    "tracking": "跟踪中",
    "strengthened": "证据加强",
    "weakened": "证据削弱",
    "contested": "证据分歧",
    "confirmed": "人工确认",
    "invalidated": "人工失效",
}

STANCE_LABELS = {
    "supports": "支持",
    "contradicts": "反对",
    "neutral": "中性",
}

IMPACT_LABELS = {
    "revenue": "收入",
    "orders": "订单",
    "price": "价格",
    "cost": "成本",
    "gross_margin": "毛利率",
    "profit": "利润",
    "customer": "客户",
    "accounts_receivable": "应收账款",
    "inventory": "存货",
    "operating_cash_flow": "经营现金流",
    "cash_flow": "现金流",
    "capacity": "产能",
    "production": "生产",
    "capex": "资本开支",
    "product": "产品",
    "quality": "质量",
    "overseas": "海外业务",
    "compliance": "合规",
    "contingent_liability": "或有负债",
    "governance": "治理",
    "capital_market": "资本市场",
    "financing": "融资",
    "debt": "债务",
    "impairment": "减值",
    "supply_chain": "供应链",
}

COMMITMENT_TERMS = (
    "计划",
    "预计",
    "拟",
    "将",
    "力争",
    "目标",
    "有望",
    "争取",
    "承诺",
)


def match_thesis_event(thesis: dict[str, Any], event: dict[str, Any]) -> dict[str, Any] | None:
    """Return a deterministic, auditable thesis-event match."""
    thesis_impacts = {str(value) for value in thesis.get("impact_dimensions") or []}
    event_impacts = {str(value) for value in event.get("impact_dimensions") or []}
    overlap = thesis_impacts & event_impacts
    narrative = " ".join(
        str(thesis.get(key) or "") for key in ("title", "description", "invalidation_criteria")
    )
    event_text = " ".join(
        str(event.get(key) or "")
        for key in ("standardized_text", "evidence_text", "cause_text", "importance_reason")
    )
    keywords = _keywords(narrative)
    matched_terms = [term for term in keywords if term in event_text]
    if not overlap and not matched_terms:
        return None

    relevance = 0.42
    if overlap:
        relevance += min(0.38, 0.19 * len(overlap))
    relevance += min(0.2, 0.05 * len(matched_terms))
    relevance = round(min(1.0, relevance), 3)

    thesis_direction = int(thesis.get("thesis_direction") or 1)
    event_direction = int(event.get("direction") or 0)
    if event_direction == 0:
        stance = "neutral"
    elif event_direction == thesis_direction:
        stance = "supports"
    else:
        stance = "contradicts"
    reason_parts = []
    if overlap:
        reason_parts.append("共同影响维度：" + "、".join(IMPACT_LABELS.get(v, v) for v in overlap))
    if matched_terms:
        reason_parts.append("匹配关键词：" + "、".join(matched_terms[:5]))
    return {
        "stance": stance,
        "relevance": relevance,
        "rationale": "；".join(reason_parts) or "与观点主题相关",
    }


def evaluate_thesis(evidence: list[dict[str, Any]], manual_state: str = "") -> dict[str, Any]:
    if manual_state in {"confirmed", "invalidated"}:
        return {
            "state": manual_state,
            "score": 1.0 if manual_state == "confirmed" else -1.0,
            "support_count": sum(item.get("stance") == "supports" for item in evidence),
            "contradict_count": sum(item.get("stance") == "contradicts" for item in evidence),
        }

    support = [item for item in evidence if item.get("stance") == "supports"]
    contradict = [item for item in evidence if item.get("stance") == "contradicts"]

    def weight(item: dict[str, Any]) -> float:
        return (
            float(item.get("relevance") or 0)
            * float(item.get("value_score") or 0.5)
            * float(item.get("certainty") or 0.7)
        )

    raw_score = sum(weight(item) for item in support) - sum(weight(item) for item in contradict)
    score = round(max(-1.0, min(1.0, raw_score)), 3)
    if support and contradict and abs(score) < 0.35:
        state = "contested"
    elif score >= 0.35:
        state = "strengthened"
    elif score <= -0.35:
        state = "weakened"
    else:
        state = "tracking"
    return {
        "state": state,
        "score": score,
        "support_count": len(support),
        "contradict_count": len(contradict),
    }


def is_management_commitment(event: dict[str, Any]) -> bool:
    text = f"{event.get('standardized_text', '')} {event.get('evidence_text', '')}"
    return str(event.get("status")) in {"expected", "possible"} or any(
        term in text for term in COMMITMENT_TERMS
    )


def document_family(title: str, doc_type: str = "") -> str:
    families = (
        "年度报告",
        "半年度报告",
        "季度报告",
        "业绩预告",
        "业绩快报",
        "投资者关系",
        "调研",
        "问询回复",
        "问询",
        "重大合同",
        "诉讼",
        "处罚",
        "项目进展",
    )
    return next((item for item in families if item in title), doc_type or "其他公告")


def compare_document_text(previous_text: str, current_text: str) -> dict[str, Any]:
    previous = _paragraphs(previous_text)
    current = _paragraphs(current_text)
    previous_set = set(previous)
    current_set = set(current)
    added = _prioritize_changes([item for item in current if item not in previous_set])
    removed = _prioritize_changes([item for item in previous if item not in current_set])
    similarity = SequenceMatcher(
        None,
        "\n".join(previous)[:200_000],
        "\n".join(current)[:200_000],
        autojunk=True,
    ).ratio()
    if similarity >= 0.96:
        change_kind = "minor"
    elif similarity >= 0.75:
        change_kind = "updated"
    else:
        change_kind = "major"
    return {
        "similarity": round(similarity, 4),
        "change_kind": change_kind,
        "added_text": "\n".join(added[:8]),
        "removed_text": "\n".join(removed[:8]),
    }


def _paragraphs(text: str) -> list[str]:
    values = []
    for value in re.split(r"[\r\n]+", text):
        normalized = re.sub(r"\s+", " ", value).strip()
        if len(normalized) >= 12:
            values.append(normalized[:1000])
    return values


def _prioritize_changes(values: list[str]) -> list[str]:
    high_value = (
        "增长",
        "下降",
        "风险",
        "订单",
        "收入",
        "利润",
        "现金流",
        "应收",
        "存货",
        "产能",
        "客户",
        "项目",
        "预计",
        "承诺",
    )
    return sorted(
        values,
        key=lambda value: (sum(term in value for term in high_value), len(value)),
        reverse=True,
    )


def _keywords(text: str) -> list[str]:
    tokens = re.split(r"[\s,，。；;：:/、（）()]+", text)
    stop = {"公司", "可能", "预计", "关注", "研究", "观点", "未来", "相关", "情况"}
    return [token for token in tokens if 2 <= len(token) <= 16 and token not in stop][:30]
