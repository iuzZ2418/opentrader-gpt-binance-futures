from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx

from ..domain import DocumentType, RawDocument

X_ACCOUNT_ID = re.compile(r"^[0-9]{1,30}$")


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class XFilteredStreamClient:
    """Read-only X filtered-stream adapter.

    Raw text is kept in the local RawDocument. ``external_llm_payload`` omits it
    unless the deployment has explicitly confirmed that sending X content to an
    external model is allowed.
    """

    endpoint = "https://api.x.com/2/tweets/search/stream"

    def __init__(
        self,
        bearer_token: str,
        *,
        allowed_account_ids: Iterable[str] | None = None,
        timeout: float = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not bearer_token:
            raise ValueError("X bearer token is required")
        if allowed_account_ids is None:
            allowed_account_ids = os.getenv("X_ALLOWED_ACCOUNT_IDS", "").split(",")
        self.allowed_account_ids = frozenset(
            str(item).strip() for item in allowed_account_ids if str(item).strip()
        )
        if not self.allowed_account_ids:
            raise ValueError("a non-empty immutable X account-ID allowlist is required")
        if any(not X_ACCOUNT_ID.fullmatch(item) for item in self.allowed_account_ids):
            raise ValueError("X allowlist entries must be immutable numeric account IDs")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {bearer_token}"},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> XFilteredStreamClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def stream(self, *, limit: int | None = None) -> Iterator[RawDocument]:
        seen = 0
        for payload in self.stream_payloads():
            document = self.parse_payload(payload)
            if document is None:
                continue
            yield document
            seen += 1
            if limit is not None and seen >= limit:
                return

    def stream_payloads(self) -> Iterator[dict[str, Any]]:
        """Yield inert API payloads so a caller can also record edit/delete events.

        This is deliberately a read-only transport.  It never follows URLs or executes
        content embedded in a post.
        """

        params = {
            "tweet.fields": (
                "author_id,created_at,public_metrics,lang,edit_history_tweet_ids"
            ),
            "expansions": "author_id",
            "user.fields": "username,verified",
        }
        with self.client.stream("GET", self.endpoint, params=params) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload

    def parse_lines(self, lines: Iterable[str]) -> list[RawDocument]:
        documents: list[RawDocument] = []
        for line in lines:
            if not line.strip():
                continue
            document = self.parse_payload(json.loads(line))
            if document is not None:
                documents.append(document)
        return documents

    def parse_payload(self, payload: dict[str, Any]) -> RawDocument | None:
        data = payload.get("data")
        if not isinstance(data, dict) or not data.get("id"):
            return None
        users = {
            str(item.get("id")): item
            for item in payload.get("includes", {}).get("users", [])
            if isinstance(item, dict)
        }
        author_id = str(data.get("author_id") or "").strip()
        if author_id not in self.allowed_account_ids:
            return None
        user = users.get(author_id, {})
        username = str(user.get("username") or author_id)
        metrics = {
            str(key): float(value)
            for key, value in (data.get("public_metrics") or {}).items()
            if isinstance(value, (int, float))
        }
        text = str(data.get("text") or "")
        observed_post_id = str(data["id"])
        edit_history = tuple(
            str(item)
            for item in (data.get("edit_history_tweet_ids") or ())
            if str(item).strip()
        )
        stable_post_id = edit_history[0] if edit_history else observed_post_id
        return RawDocument(
            source="x_official_api",
            source_id=stable_post_id,
            doc_type=DocumentType.POST,
            title=text[:160],
            text=text,
            published_at=_parse_datetime(data.get("created_at")),
            url=f"https://x.com/{username}/status/{observed_post_id}",
            author=author_id,
            engagement=metrics,
            raw={
                "author_id": author_id,
                "username": username,
                "verified": bool(user.get("verified", False)),
                "lang": data.get("lang"),
                "observed_post_id": observed_post_id,
                "edit_history_post_ids": list(edit_history),
                "ingested_at": datetime.now(UTC).isoformat(),
            },
        )

    @staticmethod
    def external_llm_payload(
        document: RawDocument, *, allow_content: bool = False
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": "x_official_api",
            "source_id": document.source_id,
            "account_id": str(document.raw.get("author_id") or document.author),
            "published_at": document.published_at.isoformat(),
            "engagement_total": sum(document.engagement.values()),
            "verified_source": bool(document.raw.get("verified", False)),
        }
        if allow_content:
            payload["text"] = document.text
            payload["author_account_id"] = document.author
            payload["display_username"] = str(document.raw.get("username") or "")
            payload["url"] = document.url
        return payload


def deleted_post_ids(payload: dict[str, Any]) -> set[str]:
    """Extract deletion targets from an X compliance stream event."""

    data = payload.get("data") or {}
    delete = data.get("delete") or {}
    post = delete.get("tweet") or delete.get("post") or {}
    identifier = post.get("id")
    return {str(identifier)} if identifier else set()
