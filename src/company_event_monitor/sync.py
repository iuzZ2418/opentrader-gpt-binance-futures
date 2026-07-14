from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import httpx

from .ingestion.cninfo import query_announcements
from .ingestion.web import enrich_document
from .service import MonitorService
from .storage import EventRepository


def sync_cninfo(
    database: Path,
    start: date,
    end: date,
    *,
    max_documents: int = 20,
    fetch_fulltext: bool = True,
) -> dict:
    repository = EventRepository(database)
    service = MonitorService(repository)
    summaries = []
    with httpx.Client(timeout=40, follow_redirects=True) as client:
        for company in repository.list_companies():
            if not company.market or not company.source_org_id:
                continue
            documents = query_announcements(company, start, end, client=client)
            for document in documents[:max_documents]:
                if fetch_fulltext and document.url:
                    document = enrich_document(document, client=client)
                summaries.append(asdict(service.process_document(document)))
    return {"counts": repository.counts(), "processed": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description="同步巨潮资讯公开公告")
    parser.add_argument("--database", type=Path, default=Path("data/company_events.db"))
    parser.add_argument(
        "--start", type=date.fromisoformat, default=date.today() - timedelta(days=7)
    )
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--max-documents", type=int, default=20)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    result = sync_cninfo(
        args.database,
        args.start,
        args.end,
        max_documents=max(1, min(args.max_documents, 100)),
        fetch_fulltext=not args.metadata_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
