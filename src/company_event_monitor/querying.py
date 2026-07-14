from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from threading import Lock
from typing import Any
from uuid import uuid4

import httpx

from .analysis import score_event
from .deepseek import (
    DeepSeekClient,
    HybridDeepSeekExtractor,
    company_research_summary,
    comparison_summary,
)
from .domain import Company, Document, SourceTier
from .ingestion.cninfo import query_announcements, search_companies
from .ingestion.exchange import query_exchange_announcements
from .ingestion.web import enrich_document
from .market import update_market_analysis
from .service import MonitorService
from .storage import EventRepository

HIGH_VALUE_TERMS = (
    "年度报告",
    "半年度报告",
    "季度报告",
    "业绩预告",
    "业绩快报",
    "调研",
    "投资者关系",
    "问询",
    "回复",
    "合同",
    "中标",
    "诉讼",
    "仲裁",
    "处罚",
    "项目",
    "投资",
    "担保",
    "关联交易",
    "回购",
    "增持",
    "减持",
    "融资",
    "可转债",
    "定增",
    "研发",
    "产品",
    "风险",
    "更正",
    "补充",
)
LOW_VALUE_TERMS = (
    "法律意见书",
    "核查意见",
    "股东大会通知",
    "董事会决议",
    "监事会决议",
    "权益分派实施",
    "提示性公告",
    "翌日披露表格",
    "证券变动月报表",
    "前十大股东",
    "独立董事",
    "公司章程",
    "制度",
    "鉴证报告",
    "专项报告",
)

DIMENSIONS: dict[str, tuple[str, ...]] = {
    "收入与订单": ("revenue", "orders"),
    "价格、成本和毛利率": ("price", "cost", "gross_margin", "profit"),
    "客户与应收账款": ("customer", "accounts_receivable"),
    "存货与现金流": ("inventory", "operating_cash_flow", "cash_flow"),
    "产能、项目和资本开支": ("capacity", "production", "capex"),
    "产品与研发": ("product", "quality"),
    "海外业务": ("overseas",),
    "治理、诉讼和合规": (
        "governance",
        "contingent_liability",
        "compliance",
        "capital_market",
    ),
}

INITIAL_LOOKBACK_DAYS = 1095
EXTENDED_LOOKBACK_DAYS = 1825
INITIAL_MAX_PAGES = 24
EXTENDED_MAX_PAGES = 40


class CompanyNotFoundError(ValueError):
    pass


class AmbiguousCompanyError(ValueError):
    def __init__(self, candidates: list[Company]) -> None:
        super().__init__("公司名称存在歧义，请从候选公司中选择")
        self.candidates = candidates


class QueryCoordinator:
    def __init__(self, repository: EventRepository, max_workers: int = 2) -> None:
        self.repository = repository
        self.repository.initialize()
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="company-job",
        )
        self._company_locks: dict[str, Lock] = {}
        self._locks_guard = Lock()

    def search(self, query: str) -> list[Company]:
        cached = self.repository.search_cached_companies(query.strip())
        if cached:
            return cached
        companies = search_companies(query)
        self.repository.cache_companies(companies)
        return companies

    def start_query(self, query: str, company_id: str = "") -> dict[str, Any]:
        company = self._resolve(query, company_id)
        self.repository.upsert_company(company)
        job_id = uuid4().hex
        job = self.repository.create_query_job(job_id, query, company.company_id)
        self.executor.submit(self._execute_query, job_id, company)
        return job

    def start_comparison(self, company_ids: list[str], window_days: int = 180) -> dict[str, Any]:
        if not 2 <= len(company_ids) <= 12:
            raise ValueError("请选择 2—12 家公司")
        unique_ids = list(dict.fromkeys(company_ids))
        if len(unique_ids) != len(company_ids):
            raise ValueError("比较公司不能重复")
        for company_id in unique_ids:
            if self.repository.get_company(company_id) is None:
                raise CompanyNotFoundError(f"本地公司不存在：{company_id}")
        comparison_id = uuid4().hex
        result = self.repository.create_comparison(comparison_id, unique_ids, window_days)
        self.executor.submit(self._execute_comparison, comparison_id, unique_ids, window_days)
        return result

    def resolve_batch_candidates(
        self,
        queries: list[str],
        *,
        markets: list[str] | None = None,
        local_only: bool = False,
        limit: int = 20,
    ) -> list[Company]:
        """Resolve pasted codes/names and keyword filters into a previewable company set."""
        unique: dict[str, Company] = {}
        markets = markets or []
        for query in [item.strip() for item in queries if item.strip()]:
            local_matches = self.repository.search_company_pool(
                [query], markets, local_only=True, limit=limit
            )
            for company in local_matches:
                unique[company.company_id] = company
            if not local_only:
                for company in self.repository.search_company_pool(
                    [query], markets, local_only=False, limit=limit
                ):
                    unique[company.company_id] = company
            if not local_only and len(unique) < limit:
                try:
                    for company in self.search(query):
                        if markets and company.market not in markets:
                            continue
                        unique[company.company_id] = company
                except Exception:
                    continue
            if len(unique) >= limit:
                break
        if not queries:
            for company in self.repository.search_company_pool(
                [], markets, local_only=local_only, limit=limit
            ):
                unique[company.company_id] = company
        return sorted(unique.values(), key=lambda item: item.ticker)[:limit]

    def start_batch_query(
        self,
        companies: list[Company],
        *,
        name: str,
        query_mode: str,
        criteria: dict[str, Any],
        window_days: int = 180,
    ) -> dict[str, Any]:
        unique = list({company.company_id: company for company in companies}.values())
        if not 2 <= len(unique) <= 12:
            raise ValueError("批量研究请选择 2—12 家公司")
        for company in unique:
            self.repository.upsert_company(company)
        batch_id = uuid4().hex
        batch = self.repository.create_batch_query(
            batch_id,
            name.strip() or f"批量研究 {datetime.now():%Y-%m-%d %H:%M}",
            query_mode,
            criteria,
            [company.company_id for company in unique],
        )
        # A batch owns its orchestration thread; document extraction still remains capped at two.
        thread = __import__("threading").Thread(
            target=self._execute_batch_query,
            args=(batch_id, unique, window_days),
            daemon=True,
            name=f"batch-{batch_id[:8]}",
        )
        thread.start()
        return batch

    def _resolve(self, query: str, company_id: str) -> Company:
        if company_id:
            saved = self.repository.get_company(company_id)
            if saved is not None:
                return saved
            candidates = self.search(query)
            selected = next((item for item in candidates if item.company_id == company_id), None)
            if selected is not None:
                return selected
            raise CompanyNotFoundError("所选公司不存在，请重新搜索")
        local = self.repository.company_by_ticker(query.strip())
        if local is not None:
            return local
        candidates = self.search(query)
        if not candidates:
            raise CompanyNotFoundError("未在官方公司目录中找到该 A 股公司")
        exact = [item for item in candidates if query.strip() in {item.ticker, item.name}]
        if len(exact) == 1:
            return exact[0]
        if len(candidates) == 1:
            return candidates[0]
        raise AmbiguousCompanyError(candidates)

    def _lock_for(self, company_id: str) -> Lock:
        with self._locks_guard:
            return self._company_locks.setdefault(company_id, Lock())

    def _execute_query(self, job_id: str, company: Company) -> dict[str, Any]:
        with self._lock_for(company.company_id):
            return self._execute_query_locked(job_id, company)

    def _execute_batch_query(
        self,
        batch_id: str,
        companies: list[Company],
        requested_window: int,
    ) -> None:
        failures: list[dict[str, str]] = []
        completed_count = 0
        try:
            self.repository.update_batch_query(
                batch_id, status="running", stage="updating", progress=0.02
            )

            def update_company(company: Company) -> tuple[Company, str, str]:
                job_id = uuid4().hex
                self.repository.create_query_job(job_id, company.ticker, company.company_id)
                self.repository.update_batch_member(
                    batch_id,
                    company.company_id,
                    query_job_id=job_id,
                    status="running",
                )
                try:
                    self._execute_query(job_id, company)
                    latest = self.repository.latest_document_at(company.company_id)
                    as_of = latest.isoformat() if latest else ""
                    self.repository.update_batch_member(
                        batch_id,
                        company.company_id,
                        status="completed",
                        data_as_of=as_of,
                    )
                    return company, "", as_of
                except Exception as error:
                    latest = self.repository.latest_document_at(company.company_id)
                    as_of = latest.isoformat() if latest else ""
                    self.repository.update_batch_member(
                        batch_id,
                        company.company_id,
                        status="failed",
                        data_as_of=as_of,
                        error=str(error)[:1000],
                    )
                    return company, str(error), as_of

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="batch-company") as pool:
                futures = [pool.submit(update_company, company) for company in companies]
                for future in as_completed(futures):
                    company, error, _as_of = future.result()
                    completed_count += 1
                    if error:
                        failures.append({"company_id": company.company_id, "error": error})
                    self.repository.update_batch_query(
                        batch_id,
                        completed_companies=completed_count,
                        failed_companies=len(failures),
                        progress=0.04 + 0.66 * (completed_count / len(companies)),
                    )

            self.repository.update_batch_query(
                batch_id, stage="comparing", progress=0.72
            )
            company_ids = [company.company_id for company in companies]
            window_days = requested_window
            recent_since = datetime.now(UTC) - timedelta(days=requested_window)
            if any(
                len(self.repository.events_in_window(company_id, recent_since)) < 3
                for company_id in company_ids
            ):
                window_days = 365
            comparison_id = uuid4().hex
            self.repository.create_comparison(comparison_id, company_ids, window_days)
            for company in companies:
                latest = self.repository.latest_document_at(company.company_id)
                failure = next(
                    (item for item in failures if item["company_id"] == company.company_id), None
                )
                self.repository.update_comparison_member(
                    comparison_id,
                    company.company_id,
                    update_status="failed" if failure else "completed",
                    data_as_of=latest.isoformat() if latest else "",
                    error=(failure or {}).get("error", "")[:1000],
                )
            comparison = self._complete_saved_comparison(
                comparison_id, company_ids, window_days, failures
            )
            self.repository.update_batch_query(
                batch_id,
                status="completed",
                stage="completed",
                progress=1,
                comparison_id=comparison_id,
                result_json=comparison.get("result") or {},
                completed_companies=completed_count,
                failed_companies=len(failures),
                completed_at=datetime.now().astimezone().isoformat(),
            )
        except Exception as error:
            self.repository.update_batch_query(
                batch_id,
                status="failed",
                stage="failed",
                progress=1,
                completed_companies=completed_count,
                failed_companies=len(failures),
                error=str(error)[:4000],
                completed_at=datetime.now().astimezone().isoformat(),
            )

    def _complete_saved_comparison(
        self,
        comparison_id: str,
        company_ids: list[str],
        window_days: int,
        failures: list[dict[str, str]],
    ) -> dict[str, Any]:
        self.repository.update_comparison(
            comparison_id, status="running", stage="comparing", progress=0.72
        )
        result = self._build_comparison(company_ids, window_days, failures)
        try:
            deepseek = DeepSeekClient(self.repository)
            if not deepseek.configured:
                raise RuntimeError("未配置 DeepSeek API 密钥")
            result["deepseek_summary"] = comparison_summary(
                deepseek, _comparison_prompt_payload(result)
            )
        except RuntimeError as error:
            result["deepseek_summary"] = _fallback_comparison_summary(result)
            result["ai_warning"] = str(error)
        data_version = hashlib.sha256(
            json.dumps(
                {
                    item["company_id"]: {
                        "data_as_of": item["data_as_of"],
                        "event_count": item["event_count"],
                        "market_hash": (item.get("price_analysis") or {}).get("data_hash"),
                    }
                    for item in result["companies"]
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cached = self.repository.cached_comparison(
            data_version, company_ids, window_days, exclude_id=comparison_id
        )
        if cached is not None:
            result = cached["result"]
            result["cache_reused"] = True
        return self.repository.update_comparison(
            comparison_id,
            status="completed",
            stage="completed",
            progress=1,
            data_version=data_version,
            result_json=result,
            completed_at=datetime.now().astimezone().isoformat(),
        )

    def _execute_query_locked(self, job_id: str, company: Company) -> dict[str, Any]:
        errors: list[str] = []
        inserted_documents = 0
        inserted_events = 0
        try:
            self.repository.update_query_job(
                job_id,
                status="running",
                stage="fetching",
                progress=0.08,
                message="正在检查交易所与巨潮最新公告",
            )
            latest_discovery = self.repository.latest_discovery_at(company.company_id)
            first_query = latest_discovery is None
            latest = latest_discovery or self.repository.latest_document_at(company.company_id)
            end = date.today()
            start = (
                end - timedelta(days=INITIAL_LOOKBACK_DAYS)
                if first_query
                else latest.date() - timedelta(days=3)
            )
            documents, catalog_documents, source_errors = self._fetch_documents(
                company, start, end, max_pages=INITIAL_MAX_PAGES
            )
            documents = _merge_source_documents(
                [*documents, *self._failed_documents(company.company_id)],
                company,
            )
            errors.extend(source_errors)
            selected = _select_documents(documents)
            window_days = INITIAL_LOOKBACK_DAYS if first_query else 365
            if first_query and len(selected) < 10:
                window_days = EXTENDED_LOOKBACK_DAYS
                documents, catalog_documents, source_errors = self._fetch_documents(
                    company,
                    end - timedelta(days=EXTENDED_LOOKBACK_DAYS),
                    end,
                    max_pages=EXTENDED_MAX_PAGES,
                )
                documents = _merge_source_documents(
                    [*documents, *self._failed_documents(company.company_id)],
                    company,
                )
                errors.extend(source_errors)
                selected = _select_documents(documents)
            selected_ids = {(item.source_name, item.source_id) for item in selected}
            self.repository.upsert_discoveries(
                company.company_id,
                catalog_documents,
                selected_ids,
            )
            processed_keys = self.repository.processed_disclosure_keys(company.company_id)
            historical_skipped = sum(
                (item.source_name, item.source_id) in processed_keys for item in selected
            )
            selected = [
                item
                for item in selected
                if (item.source_name, item.source_id) not in processed_keys
            ]
            self.repository.update_query_job(
                job_id,
                stage="downloading",
                progress=0.18,
                discovered_documents=len(catalog_documents),
                skipped_documents=historical_skipped,
                message=(
                    f"从 {len(self.repository.source_counts(company.company_id))} 个来源发现 "
                    f"{len(catalog_documents)} 条材料；快速跳过历史材料 {historical_skipped} 份，"
                    f"本次分析 {len(selected)} 份"
                ),
            )
            deepseek = DeepSeekClient(self.repository)
            service = MonitorService(
                self.repository,
                HybridDeepSeekExtractor(company, deepseek),
                backfill=False,
            )
            total = max(1, len(selected))
            with httpx.Client(timeout=45, follow_redirects=True) as client:
                for index, metadata in enumerate(selected, start=1):
                    self.repository.update_query_job(
                        job_id,
                        stage="processing",
                        progress=0.18 + 0.68 * ((index - 1) / total),
                        processed_documents=index - 1,
                        skipped_documents=historical_skipped,
                        inserted_documents=inserted_documents,
                        inserted_events=inserted_events,
                        message=(
                            f"正在下载并分析第 {index}/{len(selected)} 份材料："
                            f"{metadata.title[:40]}"
                        ),
                    )
                    try:
                        document = enrich_document(metadata, client=client)
                        if not document.text.strip():
                            raise ValueError("未提取到正文")
                        result = service.process_document(document)
                        self.repository.link_company_document(
                            company.company_id, result.document_id
                        )
                        self.repository.register_document_change(
                            company.company_id, result.document_id
                        )
                        self.repository.mark_discovery_processed(
                            metadata.source_name,
                            metadata.source_id,
                            "processed",
                        )
                        inserted_documents += int(result.inserted_document)
                        inserted_events += result.inserted_events
                        # Keep the first partial result responsive, then throttle rescoring.
                        if index == 1 or index % 3 == 0:
                            self._save_snapshot(company, job_id, window_days)
                    except Exception as error:
                        self.repository.mark_discovery_processed(
                            metadata.source_name,
                            metadata.source_id,
                            "failed",
                        )
                        errors.append(f"{metadata.title[:50]}：{error}")
            self.repository.backfill_document_changes(company.company_id)
            self.repository.update_query_job(
                job_id,
                stage="market",
                progress=0.89,
                processed_documents=len(selected),
                skipped_documents=historical_skipped,
                inserted_documents=inserted_documents,
                inserted_events=inserted_events,
                message="正在更新行情并分析事件后的价格反应",
            )
            try:
                since = datetime.now(UTC) - timedelta(days=max(window_days, 365))
                market_events = _deduplicate_event_rows(
                    self.repository.events_in_window(company.company_id, since)
                )
                update_market_analysis(self.repository, company, market_events)
            except Exception as error:
                errors.append(f"行情更新失败：{error}")
            self.repository.update_query_job(
                job_id,
                stage="summarizing",
                progress=0.94,
                message="正在生成公司总结与研究问题",
            )
            snapshot = self._save_snapshot(
                company,
                job_id,
                window_days,
                generate_ai=True,
            )
            completed = datetime.now().astimezone().isoformat()
            if not selected:
                message = "已检查至当前时间，未发现新增公开文本；以下为本地最近结果。"
            elif inserted_documents == 0:
                message = "已检查至当前时间，未发现新增公开文本；以下为本地最近结果。"
            else:
                message = f"查询完成：新增 {inserted_documents} 份文档、{inserted_events} 个事件"
            return self.repository.update_query_job(
                job_id,
                status="completed",
                stage="completed",
                progress=1,
                processed_documents=len(selected),
                skipped_documents=historical_skipped,
                inserted_documents=inserted_documents,
                inserted_events=inserted_events,
                message=message,
                error="\n".join(errors)[:4000],
                completed_at=completed,
            ) | {"snapshot": snapshot}
        except Exception as error:
            completed = datetime.now().astimezone().isoformat()
            self.repository.update_query_job(
                job_id,
                status="failed",
                stage="failed",
                progress=1,
                message="查询失败，仍可查看本地历史结果",
                error=str(error)[:4000],
                completed_at=completed,
            )
            raise

    def _fetch_documents(
        self,
        company: Company,
        start: date,
        end: date,
        *,
        max_pages: int = 4,
    ) -> tuple[list[Document], list[Document], list[str]]:
        catalog: list[Document] = []
        errors: list[str] = []
        try:
            catalog.extend(
                query_exchange_announcements(
                    company,
                    start,
                    end,
                    page_size=50,
                    max_pages=max_pages,
                )
            )
        except Exception as error:
            errors.append(f"交易所官网查询失败：{error}")
        try:
            catalog.extend(
                query_announcements(
                    company,
                    start,
                    end,
                    page_size=50,
                    max_pages=max_pages,
                )
            )
        except Exception as error:
            errors.append(f"巨潮资讯查询失败：{error}")
        if not catalog:
            raise RuntimeError("所有公开数据源均暂时不可用")
        catalog = list(
            {(document.source_name, document.source_id): document for document in catalog}.values()
        )
        return _merge_source_documents(catalog, company), catalog, errors

    def _failed_documents(self, company_id: str) -> list[Document]:
        return [
            Document(
                source_id=str(item["source_id"]),
                source_name=str(item["source_name"]),
                source_tier=SourceTier.A,
                doc_type="retry_disclosure",
                title=str(item["title"]),
                text="",
                published_at=datetime.fromisoformat(str(item["published_at"])),
                url=str(item["url"]),
            )
            for item in self.repository.company_disclosures(company_id, 1000)
            if item["processing_status"] == "failed"
        ]

    def _save_snapshot(
        self,
        company: Company,
        job_id: str,
        window_days: int,
        *,
        generate_ai: bool = False,
    ) -> dict[str, Any]:
        self._rescore_company(company.company_id)
        research_workspace = self.repository.refresh_research_workspace(company.company_id)
        since = datetime.now(UTC) - timedelta(days=window_days)
        raw_events = self.repository.events_in_window(company.company_id, since)
        events = _deduplicate_event_rows(raw_events)
        latest = self.repository.latest_discovery_at(company.company_id)
        latest = latest or self.repository.latest_document_at(company.company_id)
        source_counts = self.repository.source_counts(company.company_id)
        market_saved = self.repository.latest_market_analysis(company.company_id)
        market_analysis = (market_saved or {}).get("analysis") or {}
        counts = {
            "positive": sum(item["direction"] > 0 for item in events),
            "negative": sum(item["direction"] < 0 for item in events),
            "escalation": sum(item["change_type"] == "escalation" for item in events),
            "conflict": sum(item["change_type"] == "conflict" for item in events),
        }
        research_summary = _deterministic_company_summary(company, events, counts)
        if market_analysis:
            returns = market_analysis.get("returns") or {}
            forecast = market_analysis.get("forecast_20d") or {}
            research_summary["summary"] += (
                f" 行情截至{market_analysis.get('as_of', '最近交易日')}，近20日收益"
                f"{float(returns.get('20d') or 0):+.1%}，20交易日模型为"
                f"{forecast.get('regime', '震荡情景')}。"
            )
            research_summary["research_questions"] = [
                *(research_summary.get("research_questions") or []),
                "跟踪后续公告是否验证当前价格情景，并关注相对基准强弱是否持续。",
            ][:5]
        if generate_ai and events:
            try:
                research_summary = company_research_summary(
                    DeepSeekClient(self.repository),
                    {
                        "company_id": company.company_id,
                        "company_name": company.name,
                        "window_days": window_days,
                        "counts": counts,
                        "market_analysis": {
                            key: value
                            for key, value in market_analysis.items()
                            if key not in {"recent_prices", "event_price_links"}
                        },
                        "events": events[:18],
                    },
                )
            except RuntimeError as error:
                research_summary["ai_warning"] = str(error)
        processed_count = self.repository.company_document_count(company.company_id)
        source_record_count = sum(source_counts.values())
        disclosures = self.repository.company_disclosures(company.company_id, 10_000)
        discovered_count = len(
            {
                (
                    str(item["published_at"])[:10],
                    _normalized_title(str(item["title"]), company),
                )
                for item in disclosures
            }
        )
        scores = [float(item["value_score"]) for item in events]
        summary = {
            "company": asdict(company),
            "updated_at": datetime.now().astimezone().isoformat(),
            "data_as_of": latest.isoformat() if latest else "",
            "window_days": window_days,
            "document_count": processed_count,
            "processed_document_count": processed_count,
            "discovered_document_count": discovered_count,
            "source_record_count": source_record_count,
            "source_counts": source_counts,
            "event_count": len(events),
            "raw_event_count": len(raw_events),
            "counts": counts,
            "attention_range": {
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "average": round(sum(scores) / len(scores), 3) if scores else 0,
            },
            "research_summary": research_summary,
            "market_analysis": market_analysis,
            "research_workspace": research_workspace,
            "important_events": events[:30],
        }
        version_source = (
            f"{company.company_id}|{summary['data_as_of']}|{len(events)}|"
            f"{processed_count}|{discovered_count}|{market_analysis.get('data_hash', '')}|"
            f"{len(research_workspace.get('theses') or [])}|"
            f"{len(research_workspace.get('commitments') or [])}|"
            f"{len(research_workspace.get('document_changes') or [])}"
        )
        data_version = hashlib.sha256(version_source.encode()).hexdigest()
        return self.repository.save_snapshot(
            company.company_id,
            job_id,
            window_days,
            summary,
            data_version,
        )

    def _rescore_company(self, company_id: str) -> None:
        for event in self.repository.history(company_id):
            score_event(event)
            self.repository.update_event_scoring(
                self.repository.event_id_for(event),
                event,
            )

    def _execute_comparison(
        self,
        comparison_id: str,
        company_ids: list[str],
        requested_window: int,
    ) -> None:
        failures: list[dict[str, str]] = []
        try:
            self.repository.update_comparison(
                comparison_id,
                status="running",
                stage="updating",
                progress=0.03,
            )
            companies = [self.repository.get_company(company_id) for company_id in company_ids]
            for index, company in enumerate(companies, start=1):
                assert company is not None
                job_id = uuid4().hex
                self.repository.create_query_job(job_id, company.ticker, company.company_id)
                try:
                    self._execute_query(job_id, company)
                    latest = self.repository.latest_document_at(company.company_id)
                    self.repository.update_comparison_member(
                        comparison_id,
                        company.company_id,
                        update_status="completed",
                        data_as_of=latest.isoformat() if latest else "",
                    )
                except Exception as error:
                    latest = self.repository.latest_document_at(company.company_id)
                    failures.append({"company_id": company.company_id, "error": str(error)})
                    self.repository.update_comparison_member(
                        comparison_id,
                        company.company_id,
                        update_status="failed",
                        data_as_of=latest.isoformat() if latest else "",
                        error=str(error)[:1000],
                    )
                self.repository.update_comparison(
                    comparison_id,
                    progress=0.05 + 0.55 * (index / len(companies)),
                )
            window_days = requested_window
            recent_since = datetime.now(UTC) - timedelta(days=requested_window)
            if any(
                len(self.repository.events_in_window(company_id, recent_since)) < 3
                for company_id in company_ids
            ):
                window_days = 365
            self.repository.update_comparison(
                comparison_id,
                stage="comparing",
                progress=0.7,
                window_days=window_days,
            )
            result = self._build_comparison(company_ids, window_days, failures)
            deepseek = DeepSeekClient(self.repository)
            try:
                if not deepseek.configured:
                    raise RuntimeError("未配置 DeepSeek API 密钥")
                result["deepseek_summary"] = comparison_summary(
                    deepseek,
                    _comparison_prompt_payload(result),
                )
            except RuntimeError as error:
                result["deepseek_summary"] = _fallback_comparison_summary(result)
                result["ai_warning"] = str(error)
            data_version = hashlib.sha256(
                json.dumps(
                    {
                        item["company_id"]: {
                            "data_as_of": item["data_as_of"],
                            "event_count": item["event_count"],
                            "market_as_of": (item.get("price_analysis") or {}).get("as_of"),
                            "market_hash": (item.get("price_analysis") or {}).get("data_hash"),
                        }
                        for item in result["companies"]
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            cached = self.repository.cached_comparison(
                data_version,
                company_ids,
                window_days,
                exclude_id=comparison_id,
            )
            if cached is not None:
                result = cached["result"]
                result["cache_reused"] = True
            self.repository.update_comparison(
                comparison_id,
                status="completed",
                stage="completed",
                progress=1,
                data_version=data_version,
                result_json=result,
                completed_at=datetime.now().astimezone().isoformat(),
            )
        except Exception as error:
            self.repository.update_comparison(
                comparison_id,
                status="failed",
                stage="failed",
                progress=1,
                error=str(error)[:4000],
                completed_at=datetime.now().astimezone().isoformat(),
            )

    def _build_comparison(
        self,
        company_ids: list[str],
        window_days: int,
        failures: list[dict[str, str]],
    ) -> dict[str, Any]:
        since = datetime.now(UTC) - timedelta(days=window_days)
        companies: list[dict[str, Any]] = []
        by_company: dict[str, list[dict[str, Any]]] = {}
        for company_id in company_ids:
            company = self.repository.get_company(company_id)
            assert company is not None
            events = _deduplicate_event_rows(self.repository.events_in_window(company_id, since))
            by_company[company_id] = events
            latest = self.repository.latest_document_at(company_id)
            snapshot = self.repository.latest_snapshot(company_id)
            saved_summary = (snapshot or {}).get("summary") or {}
            source_counts = self.repository.source_counts(company_id)
            market_saved = self.repository.latest_market_analysis(company_id)
            full_market = (market_saved or {}).get("analysis") or {}
            price_analysis = {
                key: value
                for key, value in full_market.items()
                if key not in {"recent_prices", "event_price_links"}
            }
            companies.append(
                {
                    **asdict(company),
                    "data_as_of": latest.isoformat() if latest else "",
                    "document_count": self.repository.company_document_count(company_id),
                    "discovered_document_count": saved_summary.get(
                        "discovered_document_count",
                        sum(source_counts.values()),
                    ),
                    "source_counts": source_counts,
                    "price_analysis": price_analysis,
                    "event_count": len(events),
                    "positive": sum(item["direction"] > 0 for item in events),
                    "negative": sum(item["direction"] < 0 for item in events),
                    "changes": sum(
                        item["change_type"] in {"new", "escalation", "reversal", "conflict"}
                        for item in events
                    ),
                    "top_events": events[:3],
                }
            )
        heatmap: list[dict[str, Any]] = []
        for dimension, impacts in DIMENSIONS.items():
            row: dict[str, Any] = {"dimension": dimension}
            for company in companies:
                events = [
                    event
                    for event in by_company[company["company_id"]]
                    if set(event["impact_dimensions"]) & set(impacts)
                ]
                row[company["ticker"]] = {
                    "count": len(events),
                    "positive": sum(event["direction"] > 0 for event in events),
                    "negative": sum(event["direction"] < 0 for event in events),
                    "attention": round(sum(event["value_score"] for event in events), 2),
                }
            heatmap.append(row)
        negative_types: dict[str, set[str]] = {}
        for company_id, events in by_company.items():
            for event in events:
                if event["direction"] < 0:
                    negative_types.setdefault(event["event_type"], set()).add(company_id)
        common_risks = sorted(
            event_type for event_type, members in negative_types.items() if len(members) >= 2
        )
        distinctive = sorted(
            event_type for event_type, members in negative_types.items() if len(members) == 1
        )
        narrative_changes = [
            event
            for events in by_company.values()
            for event in events
            if event["change_type"] in {"escalation", "reversal", "conflict"}
        ][:20]
        return {
            "window_days": window_days,
            "generated_at": datetime.now().astimezone().isoformat(),
            "partial_update_failure": bool(failures),
            "update_failures": failures,
            "companies": companies,
            "market_comparison": [
                {
                    "company_id": item["company_id"],
                    "name": item["name"],
                    "ticker": item["ticker"],
                    "price_analysis": item.get("price_analysis") or {},
                }
                for item in companies
            ],
            "heatmap": heatmap,
            "common_risks": common_risks,
            "distinctive_events": distinctive,
            "narrative_changes": narrative_changes,
            "disclaimer": "仅输出研究关注度与公开证据，不构成投资建议或自动评级。",
        }


def _select_documents(
    documents: list[Document],
    limit: int | None = None,
) -> list[Document]:
    unique: dict[str, Document] = {}
    for document in documents:
        unique[document.source_id] = document
    ranked = sorted(
        unique.values(),
        key=lambda document: (
            _document_score(document.title),
            document.published_at,
        ),
        reverse=True,
    )
    valuable = [item for item in ranked if _document_score(item.title) >= 0]
    return valuable if limit is None else valuable[:limit]


def _document_score(title: str) -> int:
    score = 1 + sum(3 for term in HIGH_VALUE_TERMS if term in title)
    score -= sum(6 for term in LOW_VALUE_TERMS if term in title)
    if "摘要" in title and "年度报告" in title:
        score -= 3
    return score


def _merge_source_documents(catalog: list[Document], company: Company) -> list[Document]:
    merged: dict[tuple[str, str], Document] = {}
    priority = {"上海证券交易所": 3, "深圳证券交易所": 3, "巨潮资讯": 2}
    for document in catalog:
        normalized = _normalized_title(document.title, company)
        key = (document.published_at.date().isoformat(), normalized)
        current = merged.get(key)
        if current is None or priority.get(document.source_name, 1) > priority.get(
            current.source_name, 1
        ):
            merged[key] = document
    return list(merged.values())


def _deduplicate_event_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes = {
        "conflict": 6,
        "escalation": 5,
        "reversal": 4,
        "new": 3,
        "update": 2,
        "repeat": 1,
    }
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        key = (str(event["event_type"]), str(event["standardized_text"]))
        current = unique.get(key)
        current_rank = changes.get(str((current or {}).get("change_type")), 0)
        event_rank = changes.get(str(event.get("change_type")), 0)
        if current is None or (event_rank, event["value_score"]) > (
            current_rank,
            current["value_score"],
        ):
            unique[key] = event
    return sorted(
        unique.values(),
        key=lambda item: (item["value_score"], item["published_at"]),
        reverse=True,
    )


def _normalized_title(title: str, company: Company) -> str:
    value = title
    for prefix in (company.name, company.ticker, *company.aliases):
        if prefix:
            value = value.replace(prefix, "")
    value = re.sub(r"[：:（）()\s·—_-]", "", value)
    return value


def _deterministic_company_summary(
    company: Company,
    events: list[dict[str, Any]],
    counts: dict[str, int],
) -> dict[str, Any]:
    if not events:
        return {
            "summary": f"当前窗口内尚未识别到 {company.name} 的高置信度经营事件。",
            "key_points": [],
            "research_questions": ["检查已发现但尚未分析的材料是否包含增量经营信息。"],
            "generated_by": "rule",
        }
    negative = [item for item in events if item["direction"] < 0]
    positive = [item for item in events if item["direction"] > 0]
    lead = negative[0] if negative else events[0]
    summary = (
        f"最近窗口内共识别 {len(events)} 项事件，其中正向 {counts['positive']} 项、"
        f"负向 {counts['negative']} 项、风险升级 {counts['escalation']} 项。"
        f"当前最值得关注的是：{lead['standardized_text']}"
    )
    key_points = [item["standardized_text"] for item in events[:5]]
    impact_order: list[str] = []
    for event in negative[:8]:
        for impact in event["impact_dimensions"]:
            if impact not in impact_order:
                impact_order.append(impact)
    questions = [f"后续核验 {impact} 指标是否与相关文本变化一致。" for impact in impact_order[:4]]
    if positive and negative:
        questions.append("核验正向进展能否抵消已披露负向风险对经营结果的影响。")
    return {
        "summary": summary,
        "key_points": key_points,
        "research_questions": questions[:5],
        "generated_by": "rule",
    }


def _comparison_prompt_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_days": result["window_days"],
        "companies": result["companies"],
        "market_comparison": result.get("market_comparison") or [],
        "heatmap": result["heatmap"],
        "common_risks": result["common_risks"],
        "distinctive_events": result["distinctive_events"],
        "narrative_changes": result["narrative_changes"],
    }


def _fallback_comparison_summary(result: dict[str, Any]) -> str:
    names = "、".join(company["name"] for company in result["companies"])
    risk = "、".join(result["common_risks"][:5]) or "暂无跨公司共同风险"
    priced = [
        company
        for company in result["companies"]
        if (company.get("price_analysis") or {}).get("returns")
    ]
    market_text = ""
    if priced:
        ranked = sorted(
            priced,
            key=lambda company: float(
                ((company.get("price_analysis") or {}).get("benchmark") or {}).get("excess_20d")
                or 0
            ),
            reverse=True,
        )
        leader = ranked[0]
        laggard = ranked[-1]
        lead_excess = float(leader["price_analysis"]["benchmark"].get("excess_20d") or 0)
        lag_excess = float(laggard["price_analysis"]["benchmark"].get("excess_20d") or 0)
        market_text = (
            f"近20日相对各自市场基准，{leader['name']}为{lead_excess:+.1%}，"
            f"{laggard['name']}为{lag_excess:+.1%}；该排序仅描述历史表现。"
        )
    return (
        f"本次在最近 {result['window_days']} 日统一窗口内比较了 {names}。"
        f"共同风险包括：{risk}。{market_text}各公司最重要的变化与实际数据截止日期已列于下表，"
        "请结合原文证据进一步核验。"
    )
