from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .domain import Company, Document, SourceTier
from .pipeline import EventPipeline
from .service import MonitorService
from .storage import EventRepository


def run_file(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    companies = [Company(**item) for item in payload["companies"]]
    documents = [
        Document(
            **{
                **item,
                "source_tier": SourceTier(item["source_tier"]),
                "published_at": datetime.fromisoformat(item["published_at"]),
            }
        )
        for item in payload["documents"]
    ]
    pipeline = EventPipeline(companies)
    pipeline.process(documents)
    return pipeline.feed()


def run_file_persistent(path: Path, database: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    repository = EventRepository(database)
    repository.initialize()
    for item in payload["companies"]:
        repository.upsert_company(Company(**item))
    service = MonitorService(repository)
    results = []
    for item in payload["documents"]:
        document = Document(
            **{
                **item,
                "source_tier": SourceTier(item["source_tier"]),
                "published_at": datetime.fromisoformat(item["published_at"]),
            }
        )
        results.append(service.process_document(document))
    return {
        "counts": repository.counts(),
        "processed": [asdict(result) for result in results],
        "events": repository.list_events(1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="上市公司基本面事件监测")
    parser.add_argument("input", type=Path, help="包含companies和documents的UTF-8 JSON文件")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--database", type=Path, help="可选SQLite数据库；启用增量幂等处理")
    args = parser.parse_args()
    payload = (
        run_file_persistent(args.input, args.database) if args.database else run_file(args.input)
    )
    result = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
