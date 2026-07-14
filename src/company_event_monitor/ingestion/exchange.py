from __future__ import annotations

import time
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from ..domain import Company, Document, SourceTier

SSE_QUERY_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletin.do"
SSE_BASE_URL = "https://www.sse.com.cn"
SZSE_QUERY_URL = "https://www.szse.cn/api/disc/announcement/annList"
SZSE_PDF_BASE_URL = "https://disc.static.szse.cn/download"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def query_exchange_announcements(
    company: Company,
    start: date,
    end: date,
    *,
    page_size: int = 50,
    max_pages: int = 4,
    client: httpx.Client | None = None,
) -> list[Document]:
    if company.market == "sse":
        return _query_sse(company, start, end, page_size, max_pages, client)
    if company.market == "szse":
        return _query_szse(company, start, end, page_size, max_pages, client)
    return []


def _query_sse(
    company: Company,
    start: date,
    end: date,
    page_size: int,
    max_pages: int,
    client: httpx.Client | None,
) -> list[Document]:
    owns_client = client is None
    session = client or httpx.Client(timeout=30, follow_redirects=True)
    results: list[Document] = []
    try:
        for page in range(1, max_pages + 1):
            response = session.get(
                SSE_QUERY_URL,
                params={
                    "isPagination": "true",
                    "productId": company.ticker,
                    "keyWord": "",
                    "securityType": "0101,120100,020100,020200,120200",
                    "reportType2": "DQGG",
                    "reportType": "ALL",
                    "beginDate": start.isoformat(),
                    "endDate": end.isoformat(),
                    "pageHelp.pageSize": str(page_size),
                    "pageHelp.pageNo": str(page),
                    "pageHelp.beginPage": str(page),
                    "pageHelp.cacheSize": "1",
                    "pageHelp.endPage": str(page),
                    "_": str(int(time.time() * 1000)),
                },
                headers={
                    "User-Agent": "Mozilla/5.0 CompanyEventMonitor/0.5",
                    "Referer": "https://www.sse.com.cn/assortment/stock/list/info/announcement/",
                },
            )
            response.raise_for_status()
            payload = response.json()
            items = (payload.get("pageHelp") or {}).get("data") or []
            results.extend(_sse_document(item) for item in items)
            if len(items) < page_size:
                break
        return results
    finally:
        if owns_client:
            session.close()


def _query_szse(
    company: Company,
    start: date,
    end: date,
    page_size: int,
    max_pages: int,
    client: httpx.Client | None,
) -> list[Document]:
    owns_client = client is None
    session = client or httpx.Client(timeout=30, follow_redirects=True)
    results: list[Document] = []
    try:
        for page in range(1, max_pages + 1):
            response = session.post(
                SZSE_QUERY_URL,
                params={"random": str(time.time())},
                json={
                    "seDate": [start.isoformat(), end.isoformat()],
                    "channelCode": ["listedNotice_disc"],
                    "pageSize": page_size,
                    "pageNum": page,
                    "stock": [company.ticker],
                    "searchKey": [],
                },
                headers={
                    "User-Agent": "Mozilla/5.0 CompanyEventMonitor/0.5",
                    "Referer": "https://www.szse.cn/disclosure/listed/notice/index.html",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            items = response.json().get("data") or []
            results.extend(_szse_document(item) for item in items)
            if len(items) < page_size:
                break
        return results
    finally:
        if owns_client:
            session.close()


def _sse_document(item: dict[str, Any]) -> Document:
    published = datetime.strptime(str(item["SSEDATE"]), "%Y-%m-%d").replace(tzinfo=SHANGHAI)
    path = str(item.get("URL") or "")
    return Document(
        source_id=f"SSE-{path.rsplit('/', 1)[-1]}",
        source_name="上海证券交易所",
        source_tier=SourceTier.A,
        doc_type="exchange_announcement",
        title=str(item.get("TITLE") or ""),
        text="",
        published_at=published.astimezone(UTC),
        url=f"{SSE_BASE_URL}{path}" if path.startswith("/") else path,
    )


def _szse_document(item: dict[str, Any]) -> Document:
    published = datetime.strptime(str(item["publishTime"]), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=SHANGHAI
    )
    path = str(item.get("attachPath") or "")
    return Document(
        source_id=f"SZSE-{item.get('annId') or item.get('id')}",
        source_name="深圳证券交易所",
        source_tier=SourceTier.A,
        doc_type="exchange_announcement",
        title=str(item.get("title") or ""),
        text="",
        published_at=published.astimezone(UTC),
        url=f"{SZSE_PDF_BASE_URL}{path}" if path.startswith("/") else path,
    )
