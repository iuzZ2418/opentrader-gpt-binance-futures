from datetime import datetime

from company_event_monitor.domain import (
    ChangeType,
    Company,
    Document,
    EventStatus,
    EventType,
    SourceTier,
)
from company_event_monitor.pipeline import EventPipeline


def doc(
    source_id: str,
    text: str,
    date: str,
    tier: SourceTier = SourceTier.A,
    title: str = "示例公司公告",
) -> Document:
    return Document(
        source_id,
        "测试来源",
        tier,
        "announcement",
        title,
        text,
        datetime.fromisoformat(date),
    )


def test_extracts_standardized_event_with_evidence() -> None:
    pipeline = EventPipeline([Company("c1", "示例公司", "000001")])
    events = pipeline.process(
        [
            doc(
                "1",
                "示例公司客户回款周期延长至180天，应收账款增加45%。",
                "2026-01-01T10:00:00+08:00",
            )
        ]
    )
    assert len(events) == 1
    event = events[0]
    assert event.event_type == EventType.RECEIVABLE_PRESSURE
    assert event.status == EventStatus.OCCURRED
    assert event.evidence_page is None
    assert set(event.numeric_evidence) == {"180天", "45%"}
    assert event.evidence_text in "示例公司客户回款周期延长至180天，应收账款增加45%。"
    assert event.value_score > 0.8


def test_possible_event_becoming_occurred_is_escalation() -> None:
    pipeline = EventPipeline([Company("c1", "示例公司", "000001")])
    events = pipeline.process(
        [
            doc("1", "示例公司客户回款可能放缓。", "2026-01-01T10:00:00+08:00", SourceTier.B),
            doc("2", "示例公司客户回款周期延长至180天。", "2026-02-01T10:00:00+08:00"),
        ]
    )
    latest = next(item for item in events if item.source_id == "2")
    assert latest.change_type == ChangeType.ESCALATION
    assert latest.novelty == 0.9


def test_unrelated_company_is_not_extracted() -> None:
    pipeline = EventPipeline([Company("c1", "示例公司", "000001")])
    assert (
        pipeline.process(
            [doc("1", "另一家公司订单下降。", "2026-01-01T10:00:00+08:00", title="他司公告")]
        )
        == []
    )


def test_growth_followed_by_decline_is_reversal() -> None:
    pipeline = EventPipeline([Company("c1", "示例公司", "000001")])
    events = pipeline.process(
        [
            doc("1", "示例公司订单充足。", "2026-01-01T10:00:00+08:00"),
            doc("2", "示例公司订单下降。", "2026-02-01T10:00:00+08:00"),
        ]
    )
    latest = next(item for item in events if item.source_id == "2")
    assert latest.change_type == ChangeType.REVERSAL


def test_opposite_concurrent_sources_are_conflict() -> None:
    pipeline = EventPipeline([Company("c1", "示例公司", "000001")])
    first = doc("1", "示例公司订单充足。", "2026-01-01T10:00:00+08:00")
    first.source_name = "投资者关系记录"
    second = doc("2", "示例公司订单下降。", "2026-01-12T10:00:00+08:00")
    second.source_name = "问询回复"
    events = pipeline.process([first, second])
    latest = next(item for item in events if item.source_id == "2")
    assert latest.change_type == ChangeType.CONFLICT


def test_legal_reference_is_not_treated_as_real_event() -> None:
    pipeline = EventPipeline([Company("c1", "示例公司", "000001")])
    events = pipeline.process(
        [
            doc(
                "1",
                "示例公司遵守《上市公司股东减持股份管理暂行办法》，不存在违规事项。",
                "2026-01-01T10:00:00+08:00",
            )
        ]
    )
    assert events == []


def test_explicit_reduction_plan_is_still_extracted() -> None:
    pipeline = EventPipeline([Company("c1", "示例公司", "000001")])
    events = pipeline.process(
        [doc("1", "示例公司股东拟实施减持计划。", "2026-01-01T10:00:00+08:00")]
    )
    assert events[0].event_type == EventType.SHAREHOLDER_REDUCTION
