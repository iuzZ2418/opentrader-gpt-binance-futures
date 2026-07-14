from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ..domain import DocumentType, RawDocument


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def load_documents(path: Path | str) -> list[RawDocument]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        RawDocument(
            source=item["source"],
            source_id=item["source_id"],
            doc_type=DocumentType(item["doc_type"]),
            title=item["title"],
            text=item["text"],
            published_at=_datetime(item["published_at"]),
            url=item.get("url", ""),
            author=item.get("author", ""),
            engagement=item.get("engagement", {}),
            raw=item,
        )
        for item in payload
    ]
