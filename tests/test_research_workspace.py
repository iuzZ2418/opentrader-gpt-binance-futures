from datetime import datetime

from fastapi.testclient import TestClient

from company_event_monitor.api import create_app
from company_event_monitor.domain import Company, Document, SourceTier
from company_event_monitor.service import MonitorService
from company_event_monitor.storage import EventRepository


def _document(source_id: str, title: str, text: str, published_at: str) -> Document:
    return Document(
        source_id=source_id,
        source_name="深圳证券交易所",
        source_tier=SourceTier.A,
        doc_type="exchange_announcement",
        title=title,
        text=text,
        published_at=datetime.fromisoformat(published_at),
        url=f"https://example.invalid/{source_id}.pdf",
    )


def _process(repository: EventRepository, document: Document) -> int:
    result = MonitorService(repository, backfill=False).process_document(document)
    repository.link_company_document("c1", result.document_id)
    return result.document_id


def test_thesis_is_linked_to_supporting_and_contradicting_evidence(tmp_path) -> None:
    repository = EventRepository(tmp_path / "research.db")
    repository.initialize()
    repository.upsert_company(Company("c1", "示例公司", "000001"))
    _process(
        repository,
        _document(
            "support",
            "示例公司订单进展公告",
            "示例公司订单增长30%，需求增长。",
            "2026-01-01T10:00:00+08:00",
        ),
    )
    thesis = repository.create_thesis(
        "c1",
        "订单增长能够支持收入改善",
        description="持续跟踪新签订单和客户需求",
        impact_dimensions=["orders"],
        invalidation_criteria="订单连续两个季度下降",
    )
    assert thesis["state"] == "strengthened"
    assert thesis["support_count"] >= 1
    assert thesis["evidence"][0]["evidence_text"]

    _process(
        repository,
        _document(
            "contradict",
            "示例公司经营风险公告",
            "示例公司订单下降，需求下降。",
            "2026-04-01T10:00:00+08:00",
        ),
    )
    workspace = repository.refresh_research_workspace("c1")
    updated = workspace["theses"][0]
    assert updated["support_count"] >= 1
    assert updated["contradict_count"] >= 1
    assert {item["stance"] for item in updated["evidence"]} >= {
        "supports",
        "contradicts",
    }


def test_commitment_and_document_change_are_tracked(tmp_path) -> None:
    repository = EventRepository(tmp_path / "commitments.db")
    repository.initialize()
    repository.upsert_company(Company("c1", "示例公司", "000001"))
    first_id = _process(
        repository,
        _document(
            "annual-2025",
            "示例公司2025年年度报告",
            "示例公司预计新增产能投产。\n公司订单增长。",
            "2026-03-01T10:00:00+08:00",
        ),
    )
    second_id = _process(
        repository,
        _document(
            "annual-2026",
            "示例公司2026年年度报告",
            "示例公司新增产能已经投产。\n公司订单下降。\n经营现金流承压。",
            "2027-03-01T10:00:00+08:00",
        ),
    )
    repository.register_document_change("c1", first_id)
    repository.register_document_change("c1", second_id)
    workspace = repository.refresh_research_workspace("c1")
    assert workspace["commitments"][0]["status"] == "fulfilled"
    assert workspace["document_changes"][0]["document_family"] == "年度报告"
    assert workspace["document_changes"][0]["added_text"]


def test_research_workspace_api_creates_and_updates_thesis(tmp_path) -> None:
    database = tmp_path / "api.db"
    repository = EventRepository(database)
    repository.initialize()
    repository.upsert_company(Company("c1", "示例公司", "000001"))
    with TestClient(create_app(database)) as client:
        created = client.post(
            "/research/theses",
            json={
                "company_id": "c1",
                "title": "现金流有望改善",
                "description": "跟踪经营现金流",
                "thesis_direction": 1,
                "impact_dimensions": ["operating_cash_flow"],
                "invalidation_criteria": "现金流继续转差",
            },
        )
        assert created.status_code == 200
        thesis_id = created.json()["id"]
        updated = client.patch(
            f"/research/theses/{thesis_id}", json={"manual_state": "confirmed"}
        )
        workspace = client.get("/companies/c1/research-workspace")
    assert updated.status_code == 200
    assert updated.json()["state"] == "confirmed"
    assert workspace.status_code == 200
    assert workspace.json()["theses"][0]["id"] == thesis_id
