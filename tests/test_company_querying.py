from __future__ import annotations

from datetime import UTC, datetime

import httpx
from fastapi.testclient import TestClient

from company_event_monitor.analysis import score_event
from company_event_monitor.api import create_app
from company_event_monitor.docx_reports import company_report
from company_event_monitor.domain import (
    Company,
    Document,
    EventStatus,
    EventType,
    FundamentalEvent,
    SourceTier,
)
from company_event_monitor.ingestion.cninfo import search_companies
from company_event_monitor.ingestion.exchange import query_exchange_announcements
from company_event_monitor.querying import _select_documents
from company_event_monitor.storage import EventRepository


def test_cninfo_search_resolves_all_a_share_markets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/topSearch/query")
        return httpx.Response(
            200,
            json=[
                {"code": "002050", "orgId": "sz-id", "zwjc": "三花智控", "category": "A股"},
                {"code": "600028", "orgId": "sh-id", "zwjc": "中国石化", "category": "A股"},
                {"code": "920718", "orgId": "bj-id", "zwjc": "合肥高科", "category": "A股"},
            ],
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        results = search_companies("公司", client=client)
    assert [(item.ticker, item.market) for item in results] == [
        ("002050", "szse"),
        ("600028", "sse"),
        ("920718", "bse"),
    ]


def test_snapshot_serializes_event_datetimes(tmp_path) -> None:
    repository = EventRepository(tmp_path / "events.db")
    repository.initialize()
    company = Company("szse-002050", "三花智控", "002050", market="szse")
    repository.upsert_company(company)
    repository.create_query_job("job-1", "002050", company.company_id)
    snapshot = repository.save_snapshot(
        company.company_id,
        "job-1",
        180,
        {"company": {"name": company.name}, "published_at": datetime.now(UTC)},
        "version-1",
    )
    assert "T" in snapshot["summary"]["published_at"]


def test_szse_official_source_returns_exchange_documents() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "www.szse.cn"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "doc-1",
                        "annId": 123,
                        "title": "示例公司：2026年一季度报告",
                        "publishTime": "2026-04-30 00:00:00",
                        "attachPath": "/disc/example.PDF",
                    }
                ]
            },
        )

    company = Company("szse-000001", "示例公司", "000001", market="szse")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        documents = query_exchange_announcements(
            company,
            datetime(2026, 1, 1).date(),
            datetime(2026, 7, 1).date(),
            client=client,
        )
    assert documents[0].source_name == "深圳证券交易所"
    assert documents[0].url.startswith("https://disc.static.szse.cn/download/")


def test_attention_score_varies_by_event_severity_and_evidence() -> None:
    common = {
        "company_id": "szse-000001",
        "ticker": "000001",
        "company_name": "示例公司",
        "status": EventStatus.OCCURRED,
        "direction": -1,
        "evidence_page": 1,
        "evidence_section": "经营情况",
        "source_id": "doc",
        "source_name": "深圳证券交易所",
        "source_tier": SourceTier.A,
        "published_at": datetime.now(UTC),
        "certainty": 0.9,
        "confidence": 0.85,
    }
    penalty = FundamentalEvent(
        **common,
        event_type=EventType.REGULATORY_PENALTY,
        standardized_text="公司受到监管处罚。",
        evidence_text="公司收到监管机构处罚决定，罚款100万元。",
        impact_dimensions=("compliance",),
        numeric_evidence=("100万元",),
    )
    other = FundamentalEvent(
        **common,
        event_type=EventType.OTHER,
        standardized_text="公司披露一般事项。",
        evidence_text="公司披露一项一般事项。",
        impact_dimensions=("other",),
    )
    score_event(penalty)
    score_event(other)
    assert penalty.value_score > other.value_score
    assert penalty.value_score != 0.94


def test_document_selection_expands_beyond_keyword_only_materials() -> None:
    company_documents = [
        Document(
            source_id=str(index),
            source_name="深圳证券交易所",
            source_tier=SourceTier.A,
            doc_type="exchange_announcement",
            title=f"示例公司业务进展公告{index}",
            text="",
            published_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        for index in range(25)
    ]
    # Coverage-first mode has no fixed 30-document ceiling.
    assert len(_select_documents(company_documents)) == 25


def test_batch_history_and_processed_disclosure_are_persistent(tmp_path) -> None:
    repository = EventRepository(tmp_path / "batch.db")
    repository.initialize()
    companies = [
        Company("szse-000001", "示例甲", "000001", market="szse"),
        Company("sse-600001", "示例乙", "600001", market="sse"),
    ]
    for company in companies:
        repository.upsert_company(company)
    document = Document(
        source_id="doc-1",
        source_name="交易所公告",
        source_tier=SourceTier.A,
        doc_type="announcement",
        title="经营进展公告",
        text="",
        published_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    repository.upsert_discoveries(
        companies[0].company_id,
        [document],
        {(document.source_name, document.source_id)},
    )
    repository.mark_discovery_processed(document.source_name, document.source_id, "processed")
    assert repository.processed_disclosure_keys(companies[0].company_id) == {
        (document.source_name, document.source_id)
    }

    batch = repository.create_batch_query(
        "batch-1",
        "半导体批次",
        "criteria",
        {"terms": ["半导体"]},
        [company.company_id for company in companies],
    )
    repository.update_batch_member("batch-1", companies[0].company_id, status="completed")
    repository.update_batch_query(
        "batch-1", status="completed", stage="completed", progress=1, completed_companies=2
    )
    saved = repository.batch_query(str(batch["id"]))
    assert saved["criteria"]["terms"] == ["半导体"]
    assert len(saved["members"]) == 2
    assert repository.recent_batch_queries(1)[0]["name"] == "半导体批次"


def test_new_library_and_report_endpoints(tmp_path) -> None:
    database = tmp_path / "api.db"
    repository = EventRepository(database)
    repository.initialize()
    company = Company("sse-600028", "中国石化", "600028", market="sse")
    repository.upsert_company(company)
    repository.create_query_job("job-1", company.ticker, company.company_id)
    repository.save_snapshot(
        company.company_id,
        "job-1",
        180,
        {
            "company": {
                "company_id": company.company_id,
                "name": company.name,
                "ticker": company.ticker,
            },
            "document_count": 0,
            "event_count": 0,
            "counts": {},
            "important_events": [],
        },
        "version-1",
    )
    with TestClient(create_app(database)) as client:
        library = client.get("/library/companies")
        report = client.get(f"/companies/{company.company_id}/report")
        exported = client.get("/reports/company.docx", params={"company_id": company.company_id})
    assert library.status_code == 200
    assert library.json()[0]["ticker"] == "600028"
    assert report.status_code == 200
    assert exported.status_code == 200
    assert exported.content.startswith(b"PK")


def test_company_report_is_valid_docx_container() -> None:
    content = company_report(
        {
            "summary": {
                "company": {"name": "示例公司", "ticker": "000001"},
                "counts": {},
                "important_events": [],
            }
        }
    )
    assert content.startswith(b"PK")
