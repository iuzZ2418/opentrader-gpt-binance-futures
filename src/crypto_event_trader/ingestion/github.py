from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from ..domain import DocumentType, RawDocument
from ..security import validate_service_base_url

GITHUB_REPOSITORY = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,99})/[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,99})$"
)


@dataclass(frozen=True, slots=True)
class GitHubPollResult:
    documents: list[RawDocument]
    etag: str | None
    not_modified: bool
    remaining_requests: int | None


@dataclass(frozen=True, slots=True)
class GitHubWebhookDeletion:
    source: str
    source_id: str


@dataclass(frozen=True, slots=True)
class GitHubWebhookBatch:
    repository: str
    documents: tuple[RawDocument, ...]
    deletions: tuple[GitHubWebhookDeletion, ...]


class GitHubReadOnlyClient:
    """ETag-aware reader for allow-listed public upstream repositories."""

    def __init__(
        self,
        token: str | None = None,
        *,
        allowed_repositories: Iterable[str] | None = None,
        base_url: str = "https://api.github.com",
        timeout: float = 15,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if allowed_repositories is None:
            allowed_repositories = os.getenv("GITHUB_ALLOWED_REPOSITORIES", "").split(",")
        self.allowed_repositories = frozenset(
            _canonical_repository(item)
            for item in allowed_repositories
            if str(item).strip()
        )
        normalized_base_url = validate_service_base_url(
            base_url.rstrip("/"),
            service="github",
            scheme="https",
            allowed_hosts=frozenset({"api.github.com"}),
            allowed_paths=frozenset({"", "/"}),
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "crypto-event-trader/0.3",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.Client(
            base_url=normalized_base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def poll_releases(
        self, repository: str, *, etag: str | None = None, limit: int = 30
    ) -> GitHubPollResult:
        repository = self._require_allowed_repository(repository)
        _validate_limit(limit)
        payload, response = self._get(
            f"/repos/{repository}/releases", etag=etag, params={"per_page": limit}
        )
        if response.status_code == 304:
            return self._result([], response, not_modified=True)
        documents = [self._release_document(repository, item) for item in payload]
        return self._result(documents, response)

    def poll_security_advisories(
        self, repository: str, *, etag: str | None = None, limit: int = 30
    ) -> GitHubPollResult:
        repository = self._require_allowed_repository(repository)
        _validate_limit(limit)
        payload, response = self._get(
            f"/repos/{repository}/security-advisories",
            etag=etag,
            params={"per_page": limit},
        )
        if response.status_code == 304:
            return self._result([], response, not_modified=True)
        documents = [self._advisory_document(repository, item) for item in payload]
        return self._result(documents, response)

    def poll_commits(
        self, repository: str, *, etag: str | None = None, limit: int = 30
    ) -> GitHubPollResult:
        repository = self._require_allowed_repository(repository)
        _validate_limit(limit)
        payload, response = self._get(
            f"/repos/{repository}/commits", etag=etag, params={"per_page": limit}
        )
        if response.status_code == 304:
            return self._result([], response, not_modified=True)
        documents = [self._commit_document(repository, item) for item in payload]
        return self._result(documents, response)

    def _require_allowed_repository(self, repository: str) -> str:
        canonical = _canonical_repository(repository)
        if canonical not in self.allowed_repositories:
            raise PermissionError("GitHub repository is not in the configured allowlist")
        return canonical

    def _get(
        self, path: str, *, etag: str | None, params: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], httpx.Response]:
        headers = {"If-None-Match": etag} if etag else None
        response = self.client.get(path, params=params, headers=headers)
        if response.status_code == 304:
            return [], response
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("unexpected GitHub response")
        return payload, response

    @staticmethod
    def _result(
        documents: list[RawDocument], response: httpx.Response, *, not_modified: bool = False
    ) -> GitHubPollResult:
        remaining = response.headers.get("X-RateLimit-Remaining")
        return GitHubPollResult(
            documents=documents,
            etag=response.headers.get("ETag"),
            not_modified=not_modified,
            remaining_requests=int(remaining) if remaining is not None else None,
        )

    @staticmethod
    def _release_document(repository: str, item: dict[str, Any]) -> RawDocument:
        published = item.get("published_at") or item.get("created_at")
        return _github_document(
            repository,
            source_id=f"release:{item.get('id')}",
            title=str(item.get("name") or item.get("tag_name") or "release"),
            text=str(item.get("body") or ""),
            url=str(item.get("html_url") or ""),
            published_at=published,
            raw={
                "kind": "release",
                "tag_name": item.get("tag_name"),
                "updated_at": item.get("updated_at"),
            },
        )

    @staticmethod
    def _advisory_document(repository: str, item: dict[str, Any]) -> RawDocument:
        identifier = item.get("ghsa_id") or item.get("id")
        return _github_document(
            repository,
            source_id=f"advisory:{identifier}",
            title=str(item.get("summary") or identifier or "security advisory"),
            text=str(item.get("description") or ""),
            url=str(item.get("html_url") or ""),
            published_at=item.get("published_at") or item.get("created_at"),
            raw={
                "kind": "security_advisory",
                "severity": item.get("severity"),
                "updated_at": item.get("updated_at"),
            },
        )

    @staticmethod
    def _commit_document(repository: str, item: dict[str, Any]) -> RawDocument:
        commit = item.get("commit") or {}
        committer = commit.get("committer") or {}
        sha = str(item.get("sha") or "")
        message = str(commit.get("message") or "")
        return _github_document(
            repository,
            source_id=f"commit:{sha}",
            title=message.splitlines()[0][:160] if message else sha,
            text=message,
            url=str(item.get("html_url") or ""),
            published_at=committer.get("date"),
            raw={
                "kind": "commit",
                "sha": sha,
                # Never use the commit date as the historical availability time.
                "first_observed_at": datetime.now(UTC).isoformat(),
            },
        )


def _github_document(
    repository: str,
    *,
    source_id: str,
    title: str,
    text: str,
    url: str,
    published_at: str | None,
    raw: dict[str, Any],
) -> RawDocument:
    parsed = (
        datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if published_at
        else datetime.now(UTC)
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    raw = {**raw, "repository": repository, "ingested_at": datetime.now(UTC).isoformat()}
    return RawDocument(
        source=f"github:{repository}",
        source_id=source_id,
        doc_type=DocumentType.ANNOUNCEMENT,
        title=title,
        text=text,
        published_at=parsed,
        url=url,
        author=repository,
        raw=raw,
    )


def verify_webhook_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    if not secret or not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def parse_github_webhook(
    event_type: str,
    payload: Mapping[str, Any],
    *,
    allowed_repositories: Sequence[str],
) -> GitHubWebhookBatch:
    """Convert an authenticated, allow-listed webhook into inert evidence documents.

    Only release, push, and repository-advisory facts are accepted.  The parser never follows a
    URL, downloads an artifact, checks out a commit, imports a module, or executes source code.
    """

    repository_payload = payload.get("repository")
    if not isinstance(repository_payload, Mapping):
        raise ValueError("GitHub webhook has no repository identity")
    repository = _canonical_repository(repository_payload.get("full_name"))
    allowed = frozenset(_canonical_repository(item) for item in allowed_repositories)
    if repository not in allowed:
        raise PermissionError("GitHub repository is not in the configured allowlist")

    normalized_event = str(event_type).strip().lower()
    action = str(payload.get("action") or "").strip().lower()
    source = f"github:{repository}"
    documents: list[RawDocument] = []
    deletions: list[GitHubWebhookDeletion] = []

    if normalized_event == "ping":
        pass
    elif normalized_event == "release":
        item = payload.get("release")
        if not isinstance(item, Mapping) or item.get("id") is None:
            raise ValueError("GitHub release webhook is incomplete")
        source_id = f"release:{item.get('id')}"
        if action == "deleted":
            deletions.append(GitHubWebhookDeletion(source, source_id))
        else:
            documents.append(GitHubReadOnlyClient._release_document(repository, dict(item)))
    elif normalized_event == "push":
        commits = payload.get("commits")
        if not isinstance(commits, list) or len(commits) > 100:
            raise ValueError("GitHub push webhook commits must be a bounded list")
        for item in commits:
            if not isinstance(item, Mapping):
                raise ValueError("GitHub push webhook contains an invalid commit")
            sha = str(item.get("id") or item.get("sha") or "").strip()
            if not sha:
                raise ValueError("GitHub push commit has no immutable ID")
            message = str(item.get("message") or "")
            documents.append(
                _github_document(
                    repository,
                    source_id=f"commit:{sha}",
                    title=message.splitlines()[0][:160] if message else sha,
                    text=message,
                    url=str(item.get("url") or ""),
                    published_at=str(item.get("timestamp") or "") or None,
                    raw={
                        "kind": "commit",
                        "sha": sha,
                        "first_observed_at": datetime.now(UTC).isoformat(),
                    },
                )
            )
    elif normalized_event in {"repository_advisory", "security_advisory"}:
        item = payload.get("repository_advisory") or payload.get("security_advisory")
        if not isinstance(item, Mapping):
            raise ValueError("GitHub advisory webhook is incomplete")
        identifier = item.get("ghsa_id") or item.get("id")
        if identifier is None:
            raise ValueError("GitHub advisory webhook has no immutable ID")
        source_id = f"advisory:{identifier}"
        if action in {"deleted", "withdrawn"}:
            deletions.append(GitHubWebhookDeletion(source, source_id))
        else:
            documents.append(GitHubReadOnlyClient._advisory_document(repository, dict(item)))
    else:
        raise ValueError("unsupported GitHub webhook event")

    return GitHubWebhookBatch(repository, tuple(documents), tuple(deletions))


def _canonical_repository(repository: object) -> str:
    value = str(repository).strip()
    if not GITHUB_REPOSITORY.fullmatch(value):
        raise ValueError("GitHub repository must be an owner/name identifier")
    return value.lower()


def _validate_limit(limit: int) -> None:
    if not 1 <= limit <= 100:
        raise ValueError("GitHub page limit must be between 1 and 100")
