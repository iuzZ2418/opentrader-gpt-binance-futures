from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from crypto_event_trader.audit import AuditRepository
from crypto_event_trader.config import Settings
from crypto_event_trader.domain import DocumentType, RawDocument
from crypto_event_trader.ingestion.github import GitHubPollResult
from crypto_event_trader.ingestion.x import XFilteredStreamClient
from crypto_event_trader.intelligence_worker import (
    EvidenceInbox,
    EvidenceOperation,
    EvidencePublisher,
    ExternalEvidenceNotification,
    InMemoryIntelligenceState,
    IntelligenceWorker,
    NormalizationStatus,
)
from crypto_event_trader.openai_research import (
    AuxiliaryIntelligenceError,
    EventAggregates,
    EventExtraction,
    EventType,
)
from crypto_event_trader.task_queue import InMemoryTaskQueue
from crypto_event_trader.worker import (
    _external_evidence_packet,
    _is_high_impact_intelligence,
)

NOW = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


class FakeExtractor:
    model = "gpt-exact"
    prompt_version = "event-extraction-test-v1"

    def __init__(self, *, event_type: EventType = EventType.SECURITY) -> None:
        self.event_type = event_type
        self.calls: list[tuple[dict[str, Any], ...]] = []
        self.checks = 0
        self.last_response_id: str | None = None
        self.last_latency_ms: int | None = None

    def check_model_access(self) -> bool:
        self.checks += 1
        return True

    def extract(self, documents: tuple[dict[str, Any], ...]) -> EventExtraction:
        self.calls.append(documents)
        self.last_response_id = f"resp-fake-{len(self.calls)}"
        self.last_latency_ms = 12
        source_ids = tuple(str(item["source_id"]) for item in documents)
        return EventExtraction(
            event_type=self.event_type,
            sentiment=-0.8 if self.event_type is EventType.SECURITY else 0.2,
            confidence=0.91,
            source_ids=source_ids,
            aggregates=EventAggregates(
                document_count=len(documents),
                source_count=len(source_ids),
                engagement_total=12,
                weighted_sentiment=-0.8,
                verified_source_share=1,
                novelty_score=0.9,
            ),
        )


def make_worker(
    *,
    extractor: FakeExtractor | None = None,
    x_client: XFilteredStreamClient | None = None,
    queue: Any | None = None,
    github_client: Any | None = None,
    github_repositories: tuple[str, ...] = (),
) -> tuple[IntelligenceWorker, AuditRepository, Any, InMemoryIntelligenceState]:
    audit = AuditRepository("sqlite:///:memory:")
    audit.initialize()
    queue = queue or InMemoryTaskQueue()
    state = InMemoryIntelligenceState()
    worker = IntelligenceWorker(
        audit=audit,
        extractor=extractor or FakeExtractor(),  # type: ignore[arg-type]
        publisher=EvidencePublisher(queue),
        state=state,
        github_client=github_client,
        github_repositories=github_repositories,
        x_client=x_client,
        symbols=("BTCUSDT", "ETHUSDT"),
        evidence_ttl_seconds=60,
        exact_model="gpt-exact",
        clock=lambda: NOW,
    )
    return worker, audit, queue, state


def test_startup_checks_exact_model_and_reports_unconfigured_sources() -> None:
    extractor = FakeExtractor()
    worker, _, _, _ = make_worker(extractor=extractor)
    assert worker.startup_check() == {
        "openai": "required:gpt-exact",
        "github": "skipped:not_configured",
        "x": "skipped:not_configured",
    }
    assert extractor.checks == 1
    worker.exact_model = "different"
    with pytest.raises(AuxiliaryIntelligenceError, match="openai_model_mismatch"):
        worker.startup_check()
    assert extractor.checks == 1


def test_intelligence_settings_are_explicit_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTELLIGENCE_STREAM_NAME", "trader:test-evidence")
    monkeypatch.setenv("INTELLIGENCE_POLL_SECONDS", "45")
    monkeypatch.setenv("INTELLIGENCE_EVIDENCE_TTL_SECONDS", "900")
    monkeypatch.setenv("GITHUB_POLL_LIMIT", "20")
    settings = Settings.from_env()
    assert settings.intelligence_stream_name == "trader:test-evidence"
    assert settings.intelligence_poll_seconds == 45
    assert settings.intelligence_evidence_ttl_seconds == 900
    assert settings.github_poll_limit == 20
    with pytest.raises(ValueError, match="GITHUB_POLL_LIMIT"):
        replace(settings, github_poll_limit=0)


def test_x_redacted_path_uses_local_features_and_versions_edits_and_deletes() -> None:
    extractor = FakeExtractor(event_type=EventType.SECURITY)
    x_client = XFilteredStreamClient("token", allowed_account_ids={"42"})
    worker, audit, queue, _ = make_worker(extractor=extractor, x_client=x_client)
    original = {
        "data": {
            "id": "100",
            "author_id": "42",
            "text": "Official BTC protocol hack reported",
            "created_at": "2026-07-14T07:00:00Z",
            "edit_history_tweet_ids": ["100"],
            "public_metrics": {"like_count": 10, "retweet_count": 2},
        },
        "includes": {"users": [{"id": "42", "username": "source", "verified": True}]},
    }
    first = worker.ingest_x_payload(original)
    assert first is not None and first.operation is EvidenceOperation.ORIGINAL
    sent = extractor.calls[0][0]
    assert "text" not in sent and "title" not in sent and "url" not in sent
    assert "account_id" not in sent and "author" not in sent
    assert sent["event_type"] == "SECURITY"
    assert sent["source_id"] == "100"
    assert sent["aggregates"]["asset_count"] == 1

    edited = {
        **original,
        "data": {
            **original["data"],
            "id": "101",
            "text": "Official BTC protocol hack reported and contained",
            "edit_history_tweet_ids": ["100", "101"],
        },
    }
    second = worker.ingest_x_payload(edited)
    assert second is not None and second.operation is EvidenceOperation.EDIT
    latest = audit.latest_external_evidence("x_official_api:100")
    assert latest is not None and latest["version"] == 2
    assert latest["payload"]["raw_document"]["text"].startswith("Official BTC")

    before_delete = EvidenceInbox()
    for _, task in queue.pending:
        before_delete.accept_task(task)
    assert before_delete.for_symbol("BTCUSDT", now=NOW)[0].version == 2

    deleted = worker.ingest_x_payload(
        {"data": {"delete": {"tweet": {"id": "101"}}}}
    )
    assert deleted is not None and deleted.operation is EvidenceOperation.DELETE
    latest = audit.latest_external_evidence("x_official_api:100")
    assert latest is not None and latest["version"] == 3 and latest["deleted_at"]
    inbox = EvidenceInbox()
    for _, task in queue.pending:
        inbox.accept_task(task)
    assert inbox.for_symbol("BTCUSDT", now=NOW) == ()
    x_client.close()


def test_x_unknown_is_low_confidence_local_none_and_never_sent_to_model() -> None:
    extractor = FakeExtractor(event_type=EventType.LISTING)
    x_client = XFilteredStreamClient("token", allowed_account_ids={"42"})
    worker, audit, queue, _ = make_worker(extractor=extractor, x_client=x_client)
    outcome = worker.ingest_x_payload(
        {
            "data": {
                "id": "200",
                "author_id": "42",
                "text": "Good morning everyone",
                "created_at": "2026-07-14T07:00:00Z",
            }
        }
    )
    assert outcome is not None
    assert extractor.calls == []
    latest = audit.latest_external_evidence("x_official_api:200")
    assert latest is not None
    assert latest["payload"]["raw_document"]["text"] == "Good morning everyone"
    task = queue.pending[0][1]
    notification = ExternalEvidenceNotification.model_validate(task.payload)
    assert notification.event_type is EventType.NONE
    assert notification.confidence == 0.10
    assert notification.usable_for_trading is False
    assert notification.extractor_model == "local-deterministic-v1"
    assert "Good morning everyone" not in task.model_dump_json()
    x_client.close()


class FakeGitHub:
    def __init__(self, document: RawDocument) -> None:
        self.document = document
        self.seen_etags: list[tuple[str, str | None]] = []

    def _result(self, kind: str, etag: str | None) -> GitHubPollResult:
        self.seen_etags.append((kind, etag))
        return GitHubPollResult(
            documents=[self.document] if kind == "releases" else [],
            etag=f'"{kind}-new"',
            not_modified=False,
            remaining_requests=100,
        )

    def poll_releases(self, _: str, *, etag: str | None, limit: int) -> GitHubPollResult:
        assert limit == 30
        return self._result("releases", etag)

    def poll_security_advisories(
        self, _: str, *, etag: str | None, limit: int
    ) -> GitHubPollResult:
        assert limit == 30
        return self._result("security-advisories", etag)

    def poll_commits(self, _: str, *, etag: str | None, limit: int) -> GitHubPollResult:
        assert limit == 30
        return self._result("commits", etag)


def test_github_three_endpoint_poll_commits_etags_after_publication() -> None:
    document = RawDocument(
        source="github:acme/protocol",
        source_id="release:1",
        doc_type=DocumentType.ANNOUNCEMENT,
        title="BTC security release",
        text="BTC protocol security hardening",
        published_at=NOW - timedelta(hours=1),
        url="https://github.com/acme/protocol/releases/tag/v1",
        raw={"kind": "release", "repository": "acme/protocol"},
    )
    github = FakeGitHub(document)
    worker, audit, queue, state = make_worker(
        github_client=github,
        github_repositories=("acme/protocol",),
    )
    assert worker.poll_github_once() == 1
    assert len(queue.pending) == 1
    assert audit.latest_external_evidence("github:acme/protocol:release:1") is not None
    assert state.etags == {
        "acme/protocol:releases": '"releases-new"',
        "acme/protocol:security-advisories": '"security-advisories-new"',
        "acme/protocol:commits": '"commits-new"',
    }


class FailingOnceQueue(InMemoryTaskQueue):
    def __init__(self) -> None:
        super().__init__()
        self.should_fail = True

    def publish(self, task: Any) -> str:
        if self.should_fail:
            self.should_fail = False
            raise ConnectionError("redis unavailable")
        return super().publish(task)


def test_audit_record_replays_as_outbox_after_redis_failure() -> None:
    queue = FailingOnceQueue()
    extractor = FakeExtractor(event_type=EventType.SECURITY)
    worker, audit, _, _ = make_worker(extractor=extractor, queue=queue)
    document = RawDocument(
        source="github:acme/protocol",
        source_id="commit:abc",
        doc_type=DocumentType.ANNOUNCEMENT,
        title="BTC exploit mitigation",
        text="BTC exploit mitigation",
        published_at=NOW,
    )
    with pytest.raises(ConnectionError, match="redis unavailable"):
        worker.ingest_document(document)
    latest = audit.latest_external_evidence("github:acme/protocol:commit:abc")
    assert latest is not None and latest["version"] == 1

    assert worker.replay_unpublished() == 1
    assert len(queue.pending) == 1
    assert len(extractor.calls) == 1
    latest = audit.latest_external_evidence("github:acme/protocol:commit:abc")
    assert latest is not None and latest["version"] == 1

    replay = worker.ingest_document(document)
    assert replay.replayed is True
    assert len(queue.pending) == 1


def test_evidence_inbox_keeps_latest_usable_version_and_enforces_ttl() -> None:
    queue = InMemoryTaskQueue()
    worker, _, _, _ = make_worker(queue=queue)
    document = RawDocument(
        source="github:acme/protocol",
        source_id="release:2",
        doc_type=DocumentType.ANNOUNCEMENT,
        title="ETH exploit fix",
        text="ETH exploit fix",
        published_at=NOW,
    )
    worker.ingest_document(document)
    inbox = EvidenceInbox()
    assert inbox.consume_once(queue) == 1
    current = inbox.for_symbol("ETHUSDT", now=NOW + timedelta(seconds=59))
    assert len(current) == 1
    assert current[0].normalization_status is NormalizationStatus.COMPLETED
    assert current[0].extractor_response_id == "resp-fake-1"
    assert current[0].extractor_latency_ms == 12
    assert inbox.for_symbol("ETHUSDT", now=NOW + timedelta(seconds=60)) == ()


def test_evidence_inbox_rehydrates_unexpired_latest_versions_from_audit() -> None:
    worker, audit, _, _ = make_worker()
    worker.ingest_document(
        RawDocument(
            source="github:acme/protocol",
            source_id="release:restart",
            doc_type=DocumentType.ANNOUNCEMENT,
            title="BTC exploit fix",
            text="BTC exploit fix",
            published_at=NOW,
        )
    )

    restarted = EvidenceInbox()
    assert restarted.hydrate_from_audit(audit, now=NOW) == 1
    evidence = restarted.for_symbol("BTCUSDT", now=NOW)
    assert len(evidence) == 1
    assert evidence[0].source_id == "release:restart"
    packet = _external_evidence_packet(evidence[0])
    assert packet["evidence_record_id"] == evidence[0].evidence_record_id
    assert packet["content_hash"] == evidence[0].content_hash
    assert "BTC exploit fix" not in str(packet)
    assert _is_high_impact_intelligence(evidence[0]) is True
