from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any

import httpx

from ..domain import Company, Document, SourceTier

QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
PDF_BASE_URL = "https://static.cninfo.com.cn/"
REFERER = "https://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/search"
SZSE_CATALOG_URL = "https://www.cninfo.com.cn/new/data/szse_stock.json"
TOP_SEARCH_URL = "https://www.cninfo.com.cn/new/information/topSearch/query"


class CninfoConfigurationError(ValueError):
    pass


def search_companies(
    query: str,
    *,
    max_results: int = 10,
    client: httpx.Client | None = None,
) -> list[Company]:
    """Resolve an A-share ticker or short name through CNINFO's public search."""
    keyword = query.strip()
    if not keyword:
        return []
    owns_client = client is None
    session = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        response = session.post(
            TOP_SEARCH_URL,
            params={"keyWord": keyword, "maxNum": str(max(1, min(max_results, 20)))},
            headers={
                "User-Agent": "CompanyEventMonitor/0.4 (+public-disclosure research)",
                "Referer": REFERER,
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        response.raise_for_status()
        seen: set[str] = set()
        results: list[Company] = []
        for item in response.json() or []:
            ticker = re.sub(r"\D", "", str(item.get("code") or ""))
            org_id = str(item.get("orgId") or "").strip()
            name = re.sub(r"<[^>]+>", "", str(item.get("zwjc") or "")).strip()
            market = _market_for_ticker(ticker)
            if not ticker or not org_id or not name or market is None or ticker in seen:
                continue
            category = str(item.get("category") or "")
            if category and "A股" not in category:
                continue
            seen.add(ticker)
            results.append(
                Company(
                    company_id=f"{market}-{ticker}",
                    name=name,
                    ticker=ticker,
                    aliases=(name,),
                    market=market,
                    source_org_id=org_id,
                )
            )
        results.sort(key=lambda company: (company.ticker != keyword, company.ticker))
        return results
    finally:
        if owns_client:
            session.close()


def _market_for_ticker(ticker: str) -> str | None:
    if ticker.startswith(("0", "2", "3")):
        return "szse"
    if ticker.startswith("6"):
        return "sse"
    if ticker.startswith(("4", "8", "9")):
        return "bse"
    return None


def lookup_szse_company(
    ticker: str,
    *,
    client: httpx.Client | None = None,
) -> Company | None:
    owns_client = client is None
    session = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        response = session.get(
            SZSE_CATALOG_URL,
            headers={"User-Agent": "CompanyEventMonitor/0.1", "Referer": REFERER},
        )
        response.raise_for_status()
        match = next(
            (
                item
                for item in response.json().get("stockList", [])
                if str(item.get("code")) == ticker
            ),
            None,
        )
        if not match:
            return None
        name = str(match.get("zwjc") or ticker)
        return Company(
            company_id=f"szse-{ticker}",
            name=name,
            ticker=ticker,
            aliases=(name,),
            market="szse",
            source_org_id=str(match["orgId"]),
        )
    finally:
        if owns_client:
            session.close()


def query_announcements(
    company: Company,
    start: date,
    end: date,
    *,
    page_size: int = 30,
    max_pages: int = 3,
    search_keyword: str = "",
    client: httpx.Client | None = None,
) -> list[Document]:
    market = company.market.lower()
    if market not in {"szse", "sse", "bse"}:
        raise CninfoConfigurationError("market must be 'szse', 'sse' or 'bse'")
    if not company.source_org_id:
        raise CninfoConfigurationError("source_org_id is required for company-level query")
    if start > end:
        raise ValueError("start date must not be after end date")
    page_size = max(1, min(page_size, 100))
    max_pages = max(1, min(max_pages, 20))
    owns_client = client is None
    session = client or httpx.Client(timeout=30, follow_redirects=True)
    documents: list[Document] = []
    try:
        for page in range(1, max_pages + 1):
            response = session.post(
                QUERY_URL,
                data=_query_payload(company, start, end, page, page_size, search_keyword),
                headers={
                    "User-Agent": "CompanyEventMonitor/0.1 (+public-disclosure research)",
                    "Referer": REFERER,
                    "X-Requested-With": "XMLHttpRequest",
                    "Accept": "application/json, text/plain, */*",
                },
            )
            response.raise_for_status()
            payload = response.json()
            announcements = payload.get("announcements") or []
            documents.extend(_to_document(item) for item in announcements)
            has_more = payload.get("hasMore")
            if has_more in {False, None, "0", "false", "False"} or not announcements:
                break
        return documents
    finally:
        if owns_client:
            session.close()


def _query_payload(
    company: Company,
    start: date,
    end: date,
    page: int,
    page_size: int,
    search_keyword: str,
) -> dict[str, str]:
    market = company.market.lower()
    plate = {"szse": "sz", "sse": "sh", "bse": "bj"}[market]
    return {
        "pageNum": str(page),
        "pageSize": str(page_size),
        "column": market,
        "tabName": "fulltext",
        "plate": plate,
        "stock": f"{company.ticker},{company.source_org_id}",
        "searchkey": search_keyword,
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": f"{start.isoformat()}~{end.isoformat()}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }


def _to_document(item: dict[str, Any]) -> Document:
    timestamp = datetime.fromtimestamp(int(item["announcementTime"]) / 1000, tz=UTC)
    relative_url = str(item.get("adjunctUrl") or "").lstrip("/")
    return Document(
        source_id=str(item["announcementId"]),
        source_name="巨潮资讯",
        source_tier=SourceTier.A,
        doc_type="announcement",
        title=str(item.get("announcementTitle") or item.get("shortTitle") or ""),
        text=str(item.get("announcementContent") or ""),
        published_at=timestamp,
        url=f"{PDF_BASE_URL}{relative_url}" if relative_url else "",
    )
