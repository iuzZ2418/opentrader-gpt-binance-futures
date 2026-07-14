from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from company_event_monitor.annotations import export_annotations
from company_event_monitor.api import create_app
from company_event_monitor.cli import run_file_persistent
from company_event_monitor.domain import Company, Document, EvidenceSegment, SourceTier
from company_event_monitor.service import MonitorService
from company_event_monitor.storage import EventRepository


def sample_document(source_id: str = "doc-1") -> Document:
    return Document(
        source_id=source_id,
        source_name="交易所公告",
        source_tier=SourceTier.A,
        doc_type="announcement",
        title="示例公司公告",
        text="示例公司订单下降，应收账款增加45%。",
        published_at=datetime.fromisoformat("2026-01-01T10:00:00+08:00"),
        url="https://example.invalid/doc-1",
    )


def test_persistent_pipeline_is_idempotent(tmp_path) -> None:
    repository = EventRepository(tmp_path / "events.db")
    repository.initialize()
    repository.upsert_company(
        Company(
            "c1",
            "示例公司",
            "000001",
            market="szse",
            source_org_id="gssz0000001",
        )
    )
    service = MonitorService(repository)

    first = service.process_document(sample_document())
    second = service.process_document(sample_document())

    assert first.inserted_document is True
    assert first.inserted_events == 2
    assert second.inserted_document is False
    assert repository.counts() == {
        "companies": 1,
        "documents": 1,
        "fundamental_events": 2,
        "analyst_feedback": 0,
        "event_relations": 0,
        "document_segments": 1,
        "segment_annotations": 0,
    }
    assert repository.history()[0].company_name == "示例公司"
    assert repository.list_companies()[0].market == "szse"


def test_api_processes_lists_and_accepts_feedback(tmp_path) -> None:
    path = tmp_path / "api.db"
    client = TestClient(create_app(path))
    company_response = client.post(
        "/companies",
        json={"company_id": "c1", "name": "示例公司", "ticker": "000001"},
    )
    assert company_response.status_code == 200
    assert client.get("/companies").json()[0]["company_id"] == "c1"
    document = sample_document()
    payload = {
        "source_id": document.source_id,
        "source_name": document.source_name,
        "source_tier": document.source_tier.value,
        "doc_type": document.doc_type,
        "title": document.title,
        "text": document.text,
        "published_at": document.published_at.isoformat(),
        "url": document.url,
    }

    response = client.post("/documents", json=payload)
    assert response.status_code == 200
    assert response.json()["inserted_events"] == 2

    events = client.get("/events", params={"company_id": "c1"}).json()
    assert len(events) == 2
    assert events[0]["evidence_text"]
    event_id = events[0]["id"]

    feedback = client.post(
        f"/events/{event_id}/feedback",
        json={"label": "valuable", "note": "影响盈利预测"},
    )
    assert feedback.status_code == 200
    assert client.get("/health").json()["counts"]["analyst_feedback"] == 1
    assert len(client.get("/ontology").json()) >= 40
    digest = client.get("/digest", params={"day": "2026-01-01"}).json()
    assert digest["total"] == 2
    exported = client.get("/export/events.csv")
    assert exported.status_code == 200
    assert exported.content.startswith(b"\xef\xbb\xbf")
    assert "standardized_text" in exported.text
    queue = client.get("/annotation/queue", params={"annotator": "researcher-1"}).json()
    assert len(queue) == 1
    annotation = client.post(
        f"/annotation/segments/{queue[0]['segment_id']}",
        json={
            "label": "event",
            "event_type": "order_decline",
            "direction": -1,
            "status": "occurred",
            "annotator": "researcher-1",
        },
    )
    assert annotation.status_code == 200
    assert client.get("/annotation/queue", params={"annotator": "researcher-1"}).json() == []
    second_annotation = client.post(
        f"/annotation/segments/{queue[0]['segment_id']}",
        json={
            "label": "event",
            "event_type": "order_decline",
            "direction": -1,
            "status": "occurred",
            "annotator": "researcher-2",
        },
    )
    assert second_annotation.status_code == 200
    agreement = client.get(
        "/annotation/agreement",
        params={"annotator_a": "researcher-1", "annotator_b": "researcher-2"},
    ).json()
    assert agreement["overlap"] == 1
    assert agreement["exact_agreement"] == 1.0


def test_sample_file_can_be_rerun_without_duplicates(tmp_path) -> None:
    sample = Path(__file__).parents[1] / "data" / "company_event_sample.json"
    database = tmp_path / "sample.db"
    first = run_file_persistent(sample, database)
    second = run_file_persistent(sample, database)
    assert first["counts"]["documents"] == 3
    assert first["counts"]["fundamental_events"] == 4
    assert second["counts"] == first["counts"]
    assert all(not item["inserted_document"] for item in second["processed"])


def test_page_and_section_are_persisted_with_event(tmp_path) -> None:
    repository = EventRepository(tmp_path / "evidence.db")
    repository.initialize()
    repository.upsert_company(Company("c1", "示例公司", "000001"))
    document = sample_document()
    document.segments = (EvidenceSegment("示例公司订单下降。", page=12, section="三、经营情况"),)
    MonitorService(repository).process_document(document)
    event = repository.list_events()[0]
    assert event["evidence_page"] == 12
    assert event["evidence_section"] == "三、经营情况"


def test_cross_document_event_relation_is_persisted(tmp_path) -> None:
    repository = EventRepository(tmp_path / "relations.db")
    repository.initialize()
    repository.upsert_company(Company("c1", "示例公司", "000001"))
    service = MonitorService(repository)
    first = sample_document("first")
    first.text = "示例公司订单充足。"
    second = sample_document("second")
    second.source_name = "问询回复"
    second.text = "示例公司订单下降。"
    second.published_at = datetime.fromisoformat("2026-01-12T10:00:00+08:00")
    service.process_document(first)
    service.process_document(second)
    latest = next(item for item in repository.list_events() if item["source_id"] == "second")
    relations = repository.relations(latest["id"])
    assert len(relations) == 1
    assert relations[0]["relation_type"] == "conflicts"
    assert repository.counts()["event_relations"] == 1


def test_annotations_export_as_jsonl(tmp_path) -> None:
    database = tmp_path / "annotations.db"
    repository = EventRepository(database)
    repository.initialize()
    repository.upsert_company(Company("c1", "示例公司", "000001"))
    MonitorService(repository).process_document(sample_document())
    segment = repository.annotation_queue()[0]
    repository.add_annotation(
        segment["segment_id"],
        label="no_event",
        annotator="reviewer",
        note="测试",
    )
    output = tmp_path / "annotations.jsonl"
    assert export_annotations(database, output) == 1
    assert '"label": "no_event"' in output.read_text(encoding="utf-8")


def test_annotation_queue_matches_company_alias(tmp_path) -> None:
    repository = EventRepository(tmp_path / "alias.db")
    repository.initialize()
    repository.upsert_company(Company("c1", "示例汽车零部件", "000001", aliases=("示例公司",)))
    MonitorService(repository).process_document(sample_document())
    queue = repository.annotation_queue()
    assert len(queue) == 1
    assert queue[0]["company_id"] == "c1"
