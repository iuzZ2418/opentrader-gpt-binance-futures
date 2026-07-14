from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from ..domain import Document, SourceTier

TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class FeedConfig:
    url: str
    source_name: str
    source_tier: SourceTier
    doc_type: str = "announcement"


@dataclass(slots=True)
class FeedState:
    etag: str = ""
    last_modified: str = ""


def clean_html(value: str) -> str:
    return " ".join(html.unescape(TAG_PATTERN.sub(" ", value or "")).split())


def parse_feed(xml_text: str, config: FeedConfig) -> list[Document]:
    root = ET.fromstring(xml_text)
    items = root.findall(".//item")
    if items:
        return [_rss_document(item, config) for item in items]
    namespace = "{http://www.w3.org/2005/Atom}"
    return [
        _atom_document(item, config, namespace) for item in root.findall(f".//{namespace}entry")
    ]


def fetch_feed(
    config: FeedConfig,
    state: FeedState | None = None,
    client: httpx.Client | None = None,
) -> tuple[list[Document], FeedState, bool]:
    current = state or FeedState()
    headers = {"User-Agent": "CompanyEventMonitor/0.1 (+research ingestion)"}
    if current.etag:
        headers["If-None-Match"] = current.etag
    if current.last_modified:
        headers["If-Modified-Since"] = current.last_modified
    owns_client = client is None
    session = client or httpx.Client(timeout=20, follow_redirects=True)
    try:
        response = session.get(config.url, headers=headers)
        if response.status_code == 304:
            return [], current, False
        response.raise_for_status()
        next_state = FeedState(
            etag=response.headers.get("etag", ""),
            last_modified=response.headers.get("last-modified", ""),
        )
        return parse_feed(response.text, config), next_state, True
    finally:
        if owns_client:
            session.close()


def _text(element: ET.Element, *names: str) -> str:
    for name in names:
        child = element.find(name)
        if child is not None and child.text:
            return child.text.strip()
    return ""


def _published(value: str) -> datetime:
    if not value:
        return datetime.now(UTC)
    try:
        result = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result.replace(tzinfo=result.tzinfo or UTC)


def _source_id(guid: str, url: str, title: str) -> str:
    return guid or hashlib.sha256(f"{url}\n{title}".encode()).hexdigest()


def _rss_document(item: ET.Element, config: FeedConfig) -> Document:
    title = clean_html(_text(item, "title"))
    url = _text(item, "link")
    description = clean_html(
        _text(item, "description", "{http://purl.org/rss/1.0/modules/content/}encoded")
    )
    return Document(
        source_id=_source_id(_text(item, "guid"), url, title),
        source_name=config.source_name,
        source_tier=config.source_tier,
        doc_type=config.doc_type,
        title=title,
        text=description,
        published_at=_published(_text(item, "pubDate", "date")),
        url=url,
    )


def _atom_document(item: ET.Element, config: FeedConfig, namespace: str) -> Document:
    title = clean_html(_text(item, f"{namespace}title"))
    link = item.find(f"{namespace}link")
    url = link.attrib.get("href", "") if link is not None else ""
    body = clean_html(_text(item, f"{namespace}content", f"{namespace}summary"))
    return Document(
        source_id=_source_id(_text(item, f"{namespace}id"), url, title),
        source_name=config.source_name,
        source_tier=config.source_tier,
        doc_type=config.doc_type,
        title=title,
        text=body,
        published_at=_published(_text(item, f"{namespace}published", f"{namespace}updated")),
        url=url,
    )
