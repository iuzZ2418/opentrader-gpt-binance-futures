from __future__ import annotations

import hashlib
import hmac
import json

import httpx
import pytest

from crypto_event_trader.ingestion.github import (
    GitHubReadOnlyClient,
    parse_github_webhook,
    verify_webhook_signature,
)
from crypto_event_trader.ingestion.x import XFilteredStreamClient, deleted_post_ids


def test_x_payload_is_allowlisted_and_redacted_for_external_llm() -> None:
    client = XFilteredStreamClient("token", allowed_account_ids={"1"})
    document = client.parse_payload(
        {
            "data": {
                "id": "123",
                "author_id": "1",
                "text": "Scheduled futures maintenance",
                "created_at": "2026-07-14T01:02:03Z",
                "public_metrics": {"like_count": 10, "retweet_count": 2},
            },
            "includes": {"users": [{"id": "1", "username": "Binance", "verified": True}]},
        }
    )
    assert document is not None
    redacted = client.external_llm_payload(document)
    assert "text" not in redacted
    assert "author" not in redacted
    assert redacted["account_id"] == "1"
    assert redacted["engagement_total"] == 12
    assert client.external_llm_payload(document, allow_content=True)["text"].startswith("Scheduled")
    client.close()


def test_x_non_allowlisted_payload_and_compliance_delete() -> None:
    client = XFilteredStreamClient("token", allowed_account_ids={"1"})
    assert (
        client.parse_payload(
            {"data": {"id": "9", "author_id": "2", "text": "noise"}}
        )
        is None
    )
    assert deleted_post_ids({"data": {"delete": {"tweet": {"id": "9"}}}}) == {"9"}
    client.close()


def test_github_poll_uses_etag_and_first_observed_time() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers.get("If-None-Match") == '"old"'
        return httpx.Response(
            200,
            headers={"ETag": '"new"', "X-RateLimit-Remaining": "4999"},
            json=[
                {
                    "sha": "abc",
                    "html_url": "https://github.com/acme/protocol/commit/abc",
                    "commit": {
                        "message": "Security hardening\nDetails",
                        "committer": {"date": "2026-07-13T00:00:00Z"},
                    },
                }
            ],
        )

    client = GitHubReadOnlyClient(
        "token",
        allowed_repositories={"acme/protocol"},
        transport=httpx.MockTransport(handler),
    )
    result = client.poll_commits("acme/protocol", etag='"old"')
    assert calls == 1
    assert result.etag == '"new"'
    assert result.remaining_requests == 4999
    assert result.documents[0].raw["first_observed_at"]
    client.close()


def test_github_304_and_webhook_signature() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(304, headers={"ETag": '"same"'}))
    client = GitHubReadOnlyClient(
        allowed_repositories={"acme/protocol"}, transport=transport
    )
    result = client.poll_releases("acme/protocol", etag='"same"')
    assert result.not_modified is True
    assert result.documents == []
    client.close()

    body = json.dumps({"ok": True}).encode()
    digest = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
    assert verify_webhook_signature("secret", body, f"sha256={digest}") is True
    assert verify_webhook_signature("secret", body, "sha256=bad") is False


def test_x_allowlist_is_immutable_account_id_not_mutable_username() -> None:
    client = XFilteredStreamClient("token", allowed_account_ids={"42"})
    first = client.parse_payload(
        {
            "data": {"id": "1", "author_id": "42", "text": "one"},
            "includes": {"users": [{"id": "42", "username": "old_name"}]},
        }
    )
    renamed = client.parse_payload(
        {
            "data": {"id": "2", "author_id": "42", "text": "two"},
            "includes": {"users": [{"id": "42", "username": "new_name"}]},
        }
    )
    assert first is not None and renamed is not None
    assert first.author == renamed.author == "42"
    assert renamed.raw["username"] == "new_name"
    client.close()

    with pytest.raises(ValueError, match="numeric account IDs"):
        XFilteredStreamClient("token", allowed_account_ids={"mutable_username"})


def test_github_repository_allowlist_blocks_request_before_token_can_leave() -> None:
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=[])

    client = GitHubReadOnlyClient(
        "github-secret",
        allowed_repositories={"trusted/project"},
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(PermissionError, match="allowlist"):
        client.poll_releases("attacker/project")
    assert called is False
    client.close()


def test_github_webhook_parser_accepts_bounded_facts_and_deletions() -> None:
    release = parse_github_webhook(
        "release",
        {
            "action": "published",
            "repository": {"full_name": "Acme/Protocol"},
            "release": {
                "id": 7,
                "name": "v1.2.3",
                "body": "Audited release notes",
                "published_at": "2026-07-14T00:00:00Z",
                "html_url": "https://github.com/acme/protocol/releases/tag/v1.2.3",
            },
        },
        allowed_repositories=("acme/protocol",),
    )
    assert release.repository == "acme/protocol"
    assert release.documents[0].source_id == "release:7"
    assert release.deletions == ()

    deletion = parse_github_webhook(
        "repository_advisory",
        {
            "action": "withdrawn",
            "repository": {"full_name": "acme/protocol"},
            "repository_advisory": {"ghsa_id": "GHSA-1234"},
        },
        allowed_repositories=("acme/protocol",),
    )
    assert deletion.documents == ()
    assert deletion.deletions[0].source_id == "advisory:GHSA-1234"

    with pytest.raises(PermissionError, match="allowlist"):
        parse_github_webhook(
            "ping",
            {"repository": {"full_name": "attacker/project"}},
            allowed_repositories=("acme/protocol",),
        )
