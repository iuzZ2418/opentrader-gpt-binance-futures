from __future__ import annotations

from dataclasses import replace

import httpx

from ..domain import Document
from ..parsing import parse_html, parse_pdf


class UnsupportedContentType(ValueError):
    pass


def enrich_document(
    document: Document,
    *,
    client: httpx.Client | None = None,
    max_bytes: int = 30 * 1024 * 1024,
) -> Document:
    """Fetch the linked source and replace feed summary with page-aware full text."""
    if not document.url:
        return document
    owns_client = client is None
    session = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        response = session.get(
            document.url,
            headers={"User-Agent": "CompanyEventMonitor/0.1 (+research ingestion)"},
        )
        response.raise_for_status()
        if len(response.content) > max_bytes:
            raise ValueError(f"Document exceeds {max_bytes} bytes")
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" in content_type or response.content.startswith(b"%PDF"):
            text, segments = parse_pdf(response.content)
        elif "html" in content_type or response.text.lstrip().startswith(("<", "<!")):
            text, segments = parse_html(response.text)
        elif content_type.startswith("text/"):
            text = response.text.strip()
            segments = ()
        else:
            raise UnsupportedContentType(content_type or "unknown")
        return replace(document, text=text or document.text, segments=segments)
    finally:
        if owns_client:
            session.close()
