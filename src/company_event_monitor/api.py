from __future__ import annotations

import csv
import io
import os
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

from .agreement import annotation_agreement
from .deepseek import DeepSeekClient, get_api_key, save_api_key
from .docx_reports import company_report, comparison_report
from .domain import Company, Document, SourceTier
from .extraction import RULES
from .querying import AmbiguousCompanyError, CompanyNotFoundError, QueryCoordinator
from .reporting import daily_digest
from .service import MonitorService
from .storage import EventRepository


class DocumentInput(BaseModel):
    source_id: str
    source_name: str
    source_tier: SourceTier
    doc_type: str
    title: str
    text: str
    published_at: datetime
    url: str = ""


class CompanyInput(BaseModel):
    company_id: str
    name: str
    ticker: str
    aliases: tuple[str, ...] = ()
    industry: str = ""
    market: str = ""
    source_org_id: str = ""


class FeedbackInput(BaseModel):
    label: str
    note: str = ""
    analyst_id: str = ""


class AnnotationInput(BaseModel):
    label: str
    event_type: str | None = None
    direction: int | None = None
    status: str | None = None
    annotator: str = ""
    note: str = ""


class QueryJobInput(BaseModel):
    query: str
    company_id: str = ""


class ComparisonJobInput(BaseModel):
    company_ids: list[str]
    window_days: int = 180


class BatchJobInput(BaseModel):
    name: str = ""
    queries: list[str] = []
    company_ids: list[str] = []
    markets: list[str] = []
    local_only: bool = False
    limit: int = 12
    window_days: int = 180


class DeepSeekKeyInput(BaseModel):
    api_key: str
    test_connection: bool = True


class ThesisInput(BaseModel):
    company_id: str
    title: str
    description: str = ""
    thesis_direction: int = 1
    impact_dimensions: list[str]
    invalidation_criteria: str = ""


class ThesisUpdateInput(BaseModel):
    title: str | None = None
    description: str | None = None
    thesis_direction: int | None = None
    impact_dimensions: list[str] | None = None
    invalidation_criteria: str | None = None
    manual_state: str | None = None
    archived: bool | None = None


def create_app(database_path: Path | str | None = None) -> FastAPI:
    path = database_path or os.getenv("COMPANY_EVENT_DB", "data/company_events.db")
    repository = EventRepository(path)
    service = MonitorService(repository)
    coordinator = QueryCoordinator(repository)
    app = FastAPI(title="A股研究证据与观点跟踪", version="0.8.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "counts": repository.counts()}

    @app.post("/companies")
    def upsert_company(payload: CompanyInput) -> dict:
        repository.upsert_company(Company(**payload.model_dump()))
        return {"status": "saved", "company_id": payload.company_id}

    @app.get("/companies")
    def companies() -> list[dict]:
        return [asdict(company) for company in repository.list_companies()]

    @app.get("/company-search")
    def company_search(q: str = Query(min_length=1, max_length=50)) -> list[dict]:
        try:
            return [asdict(company) for company in coordinator.search(q)]
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=f"官方公司目录访问失败：{error}") from error

    @app.post("/query-jobs")
    def start_query_job(payload: QueryJobInput) -> dict:
        try:
            return coordinator.start_query(payload.query, payload.company_id)
        except AmbiguousCompanyError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(error),
                    "candidates": [asdict(company) for company in error.candidates],
                },
            ) from error
        except CompanyNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=f"官方公司目录访问失败：{error}") from error

    @app.get("/query-jobs/{job_id}")
    def query_job(job_id: str) -> dict:
        try:
            return repository.query_job(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="查询任务不存在") from error

    @app.get("/library/companies")
    def company_library() -> list[dict]:
        return repository.library_companies()

    @app.get("/companies/{company_id}/report")
    def latest_company_report(company_id: str) -> dict:
        snapshot = repository.latest_snapshot(company_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="该公司尚无本地查询结果")
        return snapshot

    @app.get("/companies/{company_id}/documents")
    def company_documents(
        company_id: str,
        limit: int = Query(200, ge=1, le=1000),
    ) -> list[dict]:
        if repository.get_company(company_id) is None:
            raise HTTPException(status_code=404, detail="本地公司不存在")
        return repository.company_disclosures(company_id, limit)

    @app.get("/companies/{company_id}/market-analysis")
    def company_market_analysis(company_id: str) -> dict:
        result = repository.latest_market_analysis(company_id)
        if result is None:
            raise HTTPException(status_code=404, detail="该公司尚无本地行情分析")
        return result

    @app.get("/companies/{company_id}/research-workspace")
    def company_research_workspace(company_id: str) -> dict:
        try:
            return repository.research_workspace(company_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="本地公司不存在") from error

    @app.post("/companies/{company_id}/research-workspace/refresh")
    def refresh_company_research_workspace(company_id: str) -> dict:
        try:
            return repository.refresh_research_workspace(company_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="本地公司不存在") from error

    @app.get("/research/theses")
    def research_theses(company_id: str | None = None) -> list[dict]:
        return repository.list_theses(company_id)

    @app.post("/research/theses")
    def create_research_thesis(payload: ThesisInput) -> dict:
        try:
            return repository.create_thesis(**payload.model_dump())
        except KeyError as error:
            raise HTTPException(status_code=404, detail="本地公司不存在") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.patch("/research/theses/{thesis_id}")
    def update_research_thesis(thesis_id: str, payload: ThesisUpdateInput) -> dict:
        try:
            return repository.update_thesis(
                thesis_id, **payload.model_dump(exclude_unset=True)
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="研究观点不存在") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.delete("/library/companies/{company_id}")
    def remove_company(company_id: str) -> dict:
        if repository.get_company(company_id) is None:
            raise HTTPException(status_code=404, detail="本地公司不存在")
        repository.delete_company(company_id)
        return {"status": "removed", "company_id": company_id}

    @app.post("/comparison-jobs")
    def start_comparison_job(payload: ComparisonJobInput) -> dict:
        try:
            return coordinator.start_comparison(payload.company_ids, payload.window_days)
        except (ValueError, CompanyNotFoundError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/comparison-jobs/{comparison_id}")
    def comparison_job(comparison_id: str) -> dict:
        try:
            return repository.comparison(comparison_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="比较任务不存在") from error

    @app.get("/comparisons")
    def recent_comparisons(limit: int = Query(10, ge=1, le=50)) -> list[dict]:
        return repository.recent_comparisons(limit)

    @app.get("/comparisons/{comparison_id}")
    def comparison_result(comparison_id: str) -> dict:
        try:
            comparison = repository.comparison(comparison_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="比较结果不存在") from error
        if comparison["status"] != "completed":
            raise HTTPException(status_code=409, detail="比较尚未完成")
        return comparison

    @app.post("/batch-jobs")
    def start_batch_job(payload: BatchJobInput) -> dict:
        companies = [
            company
            for company_id in payload.company_ids
            if (company := repository.get_company(company_id)) is not None
        ]
        if not companies:
            companies = coordinator.resolve_batch_candidates(
                payload.queries,
                markets=payload.markets,
                local_only=payload.local_only,
                limit=payload.limit,
            )
        try:
            return coordinator.start_batch_query(
                companies,
                name=payload.name,
                query_mode="criteria" if payload.queries else "library",
                criteria={
                    "queries": payload.queries,
                    "markets": payload.markets,
                    "local_only": payload.local_only,
                },
                window_days=payload.window_days,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/batch-jobs")
    def batch_history(limit: int = Query(30, ge=1, le=100)) -> list[dict]:
        return repository.recent_batch_queries(limit)

    @app.get("/batch-jobs/{batch_id}")
    def batch_job(batch_id: str) -> dict:
        try:
            return repository.batch_query(batch_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="批量任务不存在") from error

    @app.get("/reports/company.docx")
    def export_company_docx(company_id: str) -> Response:
        snapshot = repository.latest_snapshot(company_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="该公司尚无报告")
        company = repository.get_company(company_id)
        filename = f"{company.ticker if company else 'company'}-report.docx"
        return Response(
            content=company_report(snapshot),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/reports/comparison.docx")
    def export_comparison_docx(comparison_id: str) -> Response:
        try:
            comparison = repository.comparison(comparison_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="比较结果不存在") from error
        if comparison["status"] != "completed":
            raise HTTPException(status_code=409, detail="比较尚未完成")
        return Response(
            content=comparison_report(comparison),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="comparison-{comparison_id[:8]}.docx"'
            },
        )

    @app.get("/settings/deepseek/status")
    def deepseek_status() -> dict:
        return {
            "configured": bool(get_api_key()),
            "flash_model": "deepseek-v4-flash",
            "pro_model": "deepseek-v4-pro",
        }

    @app.post("/settings/deepseek")
    def configure_deepseek(payload: DeepSeekKeyInput) -> dict:
        try:
            save_api_key(payload.api_key)
            if payload.test_connection:
                return DeepSeekClient(repository, api_key=payload.api_key).test_connection()
            return {"configured": True}
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/documents")
    def process_document(payload: DocumentInput) -> dict:
        return asdict(service.process_document(Document(**payload.model_dump())))

    @app.get("/events")
    def events(limit: int = Query(100, ge=1, le=1000), company_id: str | None = None) -> list[dict]:
        return repository.list_events(limit, company_id)

    @app.get("/ontology")
    def ontology() -> list[dict]:
        return [
            {
                "event_type": rule.event_type.value,
                "label": rule.label,
                "direction": rule.direction,
                "terms": rule.terms,
                "impact_dimensions": rule.impacts,
            }
            for rule in RULES
        ]

    @app.get("/digest")
    def digest(day: date | None = None, limit: int = Query(20, ge=1, le=200)) -> dict:
        return daily_digest(repository.list_events(10_000), day, limit)

    @app.get("/companies/{company_id}/timeline")
    def timeline(company_id: str, limit: int = Query(200, ge=1, le=1000)) -> list[dict]:
        return sorted(
            repository.list_events(limit, company_id), key=lambda item: item["published_at"]
        )

    @app.get("/events/{event_id}/relations")
    def relations(event_id: int) -> list[dict]:
        return repository.relations(event_id)

    @app.get("/export/events.csv")
    def export_events(
        company_id: str | None = None,
        limit: int = Query(1000, ge=1, le=10_000),
    ) -> Response:
        events = repository.list_events(limit, company_id)
        stream = io.StringIO()
        fields = [
            "ticker",
            "company_name",
            "published_at",
            "event_type",
            "direction",
            "change_type",
            "value_score",
            "standardized_text",
            "evidence_text",
            "evidence_page",
            "evidence_section",
            "source_name",
            "source_tier",
            "url",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
        return Response(
            content="\ufeff" + stream.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=company-events.csv"},
        )

    @app.get("/annotation/queue")
    def annotation_queue(
        limit: int = Query(100, ge=1, le=1000),
        company_id: str | None = None,
        annotator: str = "",
    ) -> list[dict]:
        return repository.annotation_queue(limit, company_id=company_id, annotator=annotator)

    @app.post("/annotation/segments/{segment_id}")
    def annotate(segment_id: int, payload: AnnotationInput) -> dict:
        try:
            annotation_id = repository.add_annotation(segment_id, **payload.model_dump())
            return {"id": annotation_id, "status": "saved"}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Segment not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/annotation/agreement")
    def agreement(annotator_a: str, annotator_b: str) -> dict:
        return annotation_agreement(repository.annotations(), annotator_a, annotator_b)

    @app.post("/events/{event_id}/feedback")
    def feedback(event_id: int, payload: FeedbackInput) -> dict:
        try:
            feedback_id = repository.add_feedback(event_id, **payload.model_dump())
            return {"id": feedback_id, "status": "created"}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Event not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


app = create_app()
