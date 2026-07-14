from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace
from types import SimpleNamespace

from fastapi.testclient import TestClient

from crypto_event_trader.config import Settings
from crypto_event_trader.intelligence_api import create_intelligence_webhook_app


class FakeWorker:
    def __init__(self) -> None:
        self.documents = []
        self.deletions = []
        self.started = False

    def startup_check(self):
        self.started = True
        return {"openai": "required:test"}

    def ingest_document(self, document):
        self.documents.append(document)
        return SimpleNamespace(evidence_record_id="evidence-release-7")

    def ingest_deletion(self, *, source, source_id):
        self.deletions.append((source, source_id))
        return SimpleNamespace(evidence_record_id="evidence-delete-7")


def _settings(tmp_path, **changes) -> Settings:
    values = {
        "app_env": "test",
        "database_url": f"sqlite:///{tmp_path / 'legacy.db'}",
        "audit_database_url": f"sqlite:///{tmp_path / 'audit.db'}",
        "github_webhook_secret": "webhook-secret",
        "github_allowed_repositories": ("acme/protocol",),
    }
    values.update(changes)
    return replace(Settings.from_env(), **values)


def _signed_headers(body: bytes, *, event: str = "release") -> dict[str, str]:
    digest = hmac.new(b"webhook-secret", body, hashlib.sha256).hexdigest()
    return {
        "Content-Type": "application/json",
        "X-Hub-Signature-256": f"sha256={digest}",
        "X-GitHub-Event": event,
        "X-GitHub-Delivery": "delivery-001",
    }


def test_signed_allowlisted_github_webhook_enters_intelligence_ledger(tmp_path) -> None:
    worker = FakeWorker()
    app = create_intelligence_webhook_app(_settings(tmp_path), worker=worker)
    body = json.dumps(
        {
            "action": "published",
            "repository": {"full_name": "acme/protocol"},
            "release": {
                "id": 7,
                "name": "v1.2.3",
                "body": "release notes",
                "published_at": "2026-07-14T00:00:00Z",
            },
        },
        separators=(",", ":"),
    ).encode()

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/github",
            content=body,
            headers=_signed_headers(body),
        )
    assert response.status_code == 202
    assert response.json()["evidence_record_ids"] == ["evidence-release-7"]
    assert worker.started is True
    assert worker.documents[0].source == "github:acme/protocol"


def test_github_webhook_rejects_bad_signature_and_repository(tmp_path) -> None:
    worker = FakeWorker()
    app = create_intelligence_webhook_app(_settings(tmp_path), worker=worker)
    invalid_body = b'{}'
    with TestClient(app) as client:
        invalid = client.post(
            "/webhooks/github",
            content=invalid_body,
            headers={
                **_signed_headers(invalid_body),
                "X-Hub-Signature-256": "sha256=bad",
            },
        )
        assert invalid.status_code == 401

        disallowed_body = json.dumps(
            {
                "repository": {"full_name": "attacker/project"},
            },
            separators=(",", ":"),
        ).encode()
        disallowed = client.post(
            "/webhooks/github",
            content=disallowed_body,
            headers=_signed_headers(disallowed_body, event="ping"),
        )
    assert disallowed.status_code == 403
    assert worker.documents == []
