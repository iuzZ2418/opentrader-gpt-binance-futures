from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from ..domain import DocumentType, RawDocument


def _text(element: ElementTree.Element, *names: str) -> str:
    for name in names:
        found = element.find(name)
        if found is not None and found.text:
            return found.text.strip()
    return ""


def _published(value: str) -> datetime:
    if not value:
        return datetime.now(UTC)
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class RssIngestor:
    """Generic RSS/Atom adapter for configured official announcement and news feeds."""

    def __init__(self, source: str, url: str, doc_type: DocumentType) -> None:
        self.source = source
        self.url = url
        self.doc_type = doc_type

    def fetch(self, limit: int = 50) -> list[RawDocument]:
        request = Request(self.url, headers={"User-Agent": "crypto-event-trader/0.1"})
        with urlopen(request, timeout=15) as response:  # noqa: S310
            root = ElementTree.fromstring(response.read())

        entries = root.findall(".//item") or root.findall("{*}entry")
        result: list[RawDocument] = []
        for entry in entries[:limit]:
            title = _text(entry, "title", "{*}title")
            body = _text(
                entry,
                "description",
                "content",
                "summary",
                "{*}content",
                "{*}summary",
            )
            url = _text(entry, "link")
            if not url:
                link = entry.find("{*}link")
                url = link.attrib.get("href", "") if link is not None else ""
            guid = _text(entry, "guid", "id", "{*}id") or url
            source_id = guid or hashlib.sha256(f"{title}{body}".encode()).hexdigest()
            published = _text(
                entry, "pubDate", "published", "updated", "{*}published", "{*}updated"
            )
            result.append(
                RawDocument(
                    source=self.source,
                    source_id=source_id,
                    doc_type=self.doc_type,
                    title=title,
                    text=body,
                    published_at=_published(published),
                    url=url,
                    raw={"feed": self.url},
                )
            )
        return result
