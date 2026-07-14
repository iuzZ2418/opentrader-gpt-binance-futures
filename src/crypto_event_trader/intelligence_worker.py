from __future__ import annotations

import hashlib
import json
import logging
import re
import signal
import socket
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .audit import AuditRepository
from .config import Settings
from .domain import RawDocument
from .extraction import RULES
from .ingestion.github import GitHubPollResult, GitHubReadOnlyClient
from .ingestion.x import XFilteredStreamClient, deleted_post_ids
from .openai_research import (
    AuxiliaryIntelligenceError,
    EventAggregates,
    EventExtraction,
    EventType,
    OpenAIEventExtractor,
)
from .security import SecurityBoundaryError, sanitize_untrusted_json
from .task_queue import RedisStreamQueue, TaskEnvelope, task_payload

LOGGER = logging.getLogger("crypto_event_trader.intelligence")
NOTIFICATION_TASK_TYPE = "external_evidence.normalized.v1"
NOTIFICATION_SCHEMA_VERSION = "external-evidence.v1"
EVIDENCE_CONSUMER_GROUP = "external-evidence-inbox"
MIN_USABLE_CONFIDENCE = 0.50


class EvidenceOperation(StrEnum):
    ORIGINAL = "ORIGINAL"
    EDIT = "EDIT"
    DELETE = "DELETE"


class NormalizationStatus(StrEnum):
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    DELETED = "DELETED"


class ExternalEvidenceNotification(BaseModel):
    """Content-free Redis message referencing one immutable audit-ledger version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external-evidence.v1"] = NOTIFICATION_SCHEMA_VERSION
    evidence_record_id: str = Field(min_length=1, max_length=160)
    evidence_id: str = Field(min_length=1, max_length=512)
    version: int = Field(ge=1)
    operation: EvidenceOperation
    source: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=256)
    symbols: tuple[str, ...] = Field(min_length=1, max_length=100)
    event_type: EventType
    sentiment: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    source_ids: tuple[str, ...] = Field(max_length=100)
    aggregates: dict[str, JsonValue]
    normalization_status: NormalizationStatus
    usable_for_trading: bool
    extractor_model: str = Field(min_length=1, max_length=160)
    extractor_prompt_version: str = Field(min_length=1, max_length=160)
    extractor_response_id: str | None = Field(default=None, max_length=256)
    extractor_latency_ms: int | None = Field(default=None, ge=0)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    occurred_at: datetime
    observed_at: datetime
    expires_at: datetime
    deleted_at: datetime | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> ExternalEvidenceNotification:
        if self.expires_at <= self.observed_at:
            raise ValueError("evidence expiry must be after observation")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("evidence symbols must be unique")
        for symbol in self.symbols:
            if symbol != "*" and not re.fullmatch(r"[A-Z0-9]{2,30}", symbol):
                raise ValueError("invalid evidence symbol")
        if self.normalization_status is not NormalizationStatus.COMPLETED:
            if self.usable_for_trading:
                raise ValueError("rejected/deleted evidence cannot be usable")
        if self.operation is EvidenceOperation.DELETE:
            if (
                self.deleted_at is None
                or self.normalization_status is not NormalizationStatus.DELETED
            ):
                raise ValueError("delete notifications require a deletion timestamp")
        elif self.deleted_at is not None:
            raise ValueError("only delete notifications may carry deleted_at")
        return self


class EvidenceInbox:
    """Latest-version, TTL-aware cache consumed by the trading-cycle approval packet."""

    def __init__(self, *, max_items: int = 20_000) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self.max_items = max_items
        self._latest: dict[str, ExternalEvidenceNotification] = {}
        self._lock = threading.RLock()

    def accept(self, value: ExternalEvidenceNotification | Mapping[str, Any]) -> bool:
        notification = (
            value
            if isinstance(value, ExternalEvidenceNotification)
            else ExternalEvidenceNotification.model_validate(value)
        )
        with self._lock:
            previous = self._latest.get(notification.evidence_id)
            if previous is not None and notification.version <= previous.version:
                return False
            self._latest[notification.evidence_id] = notification
            if len(self._latest) > self.max_items:
                oldest = min(self._latest.values(), key=lambda item: item.observed_at)
                self._latest.pop(oldest.evidence_id, None)
            return True

    def hydrate_from_audit(
        self,
        audit: AuditRepository,
        *,
        now: datetime | None = None,
        limit: int = 5_000,
    ) -> int:
        """Rebuild the cache after a consumer restart from the durable source of truth."""

        reference = (now or datetime.now(UTC)).astimezone(UTC)
        accepted = 0
        for row in reversed(audit.latest_external_evidence_batch(limit=limit)):
            payload = row.get("payload") or {}
            raw_notification = payload.get("notification")
            if not isinstance(raw_notification, Mapping):
                continue
            notification = ExternalEvidenceNotification.model_validate(raw_notification)
            if notification.expires_at <= reference:
                continue
            accepted += int(self.accept(notification))
        return accepted

    def accept_task(self, task: TaskEnvelope) -> bool:
        if task.task_type != NOTIFICATION_TASK_TYPE:
            raise ValueError(f"unexpected intelligence task type: {task.task_type}")
        return self.accept(task.payload)

    def consume_once(
        self,
        queue: Any,
        *,
        count: int = 100,
        block_ms: int = 0,
    ) -> int:
        processed = 0
        for message_id, task in queue.read(count=count, block_ms=block_ms):
            try:
                self.accept_task(task)
            except Exception as error:
                queue.fail(message_id, task, error)
            else:
                queue.ack(message_id)
            processed += 1
        return processed

    def for_symbol(
        self,
        symbol: str,
        *,
        now: datetime | None = None,
    ) -> tuple[ExternalEvidenceNotification, ...]:
        reference = (now or datetime.now(UTC)).astimezone(UTC)
        normalized_symbol = symbol.strip().upper()
        with self._lock:
            expired = [
                evidence_id
                for evidence_id, item in self._latest.items()
                if item.expires_at <= reference
            ]
            for evidence_id in expired:
                self._latest.pop(evidence_id, None)
            result = [
                item
                for item in self._latest.values()
                if item.usable_for_trading
                and item.deleted_at is None
                and (normalized_symbol in item.symbols or "*" in item.symbols)
            ]
            return tuple(sorted(result, key=lambda item: item.observed_at, reverse=True))


class IntelligenceState(Protocol):
    def get_etag(self, key: str) -> str | None: ...

    def set_etag(self, key: str, value: str) -> None: ...

    def resolve_x_post_id(self, post_id: str) -> str: ...

    def remember_x_post_ids(self, stable_id: str, post_ids: Iterable[str]) -> None: ...

    def was_notification_published(self, evidence_record_id: str) -> bool: ...

    def mark_notification_published(
        self, evidence_record_id: str, *, ttl_seconds: int
    ) -> None: ...


class RedisIntelligenceState:
    def __init__(self, client: Any, *, namespace: str = "trader:intelligence-state") -> None:
        self.client = client
        self.etag_key = f"{namespace}:github-etags"
        self.x_alias_key = f"{namespace}:x-post-aliases"
        self.published_prefix = f"{namespace}:published"

    def get_etag(self, key: str) -> str | None:
        value = self.client.hget(self.etag_key, key)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value) if value else None

    def set_etag(self, key: str, value: str) -> None:
        self.client.hset(self.etag_key, key, value)

    def resolve_x_post_id(self, post_id: str) -> str:
        value = self.client.hget(self.x_alias_key, post_id)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value) if value else post_id

    def remember_x_post_ids(self, stable_id: str, post_ids: Iterable[str]) -> None:
        aliases = {str(item): stable_id for item in post_ids if str(item).strip()}
        if aliases:
            self.client.hset(self.x_alias_key, mapping=aliases)

    def was_notification_published(self, evidence_record_id: str) -> bool:
        return bool(self.client.exists(f"{self.published_prefix}:{evidence_record_id}"))

    def mark_notification_published(
        self, evidence_record_id: str, *, ttl_seconds: int
    ) -> None:
        self.client.set(
            f"{self.published_prefix}:{evidence_record_id}",
            "1",
            ex=max(60, ttl_seconds),
        )


class InMemoryIntelligenceState:
    def __init__(self) -> None:
        self.etags: dict[str, str] = {}
        self.x_aliases: dict[str, str] = {}
        self.published: set[str] = set()

    def get_etag(self, key: str) -> str | None:
        return self.etags.get(key)

    def set_etag(self, key: str, value: str) -> None:
        self.etags[key] = value

    def resolve_x_post_id(self, post_id: str) -> str:
        return self.x_aliases.get(post_id, post_id)

    def remember_x_post_ids(self, stable_id: str, post_ids: Iterable[str]) -> None:
        self.x_aliases.update(
            {str(item): stable_id for item in post_ids if str(item).strip()}
        )

    def was_notification_published(self, evidence_record_id: str) -> bool:
        return evidence_record_id in self.published

    def mark_notification_published(
        self, evidence_record_id: str, *, ttl_seconds: int
    ) -> None:
        del ttl_seconds
        self.published.add(evidence_record_id)


class EvidencePublisher:
    def __init__(self, queue: Any) -> None:
        self.queue = queue

    def publish(self, notification: ExternalEvidenceNotification) -> str:
        correlation_id = "evidence-" + hashlib.sha256(
            notification.evidence_id.encode("utf-8")
        ).hexdigest()[:32]
        task = TaskEnvelope(
            task_id=f"evidence-{notification.evidence_record_id}",
            task_type=NOTIFICATION_TASK_TYPE,
            correlation_id=correlation_id,
            payload=task_payload(notification.model_dump(mode="json")),
            created_at=notification.observed_at,
        )
        return self.queue.publish(task)


@dataclass(frozen=True, slots=True)
class IngestionOutcome:
    evidence_record_id: str
    operation: EvidenceOperation
    published: bool
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class _LocalXFeatures:
    event_type: EventType
    sentiment: float
    confidence: float
    engagement_total: float
    verified_source_share: float
    novelty_score: float
    factuality: float
    urgency: float
    bot_score: float
    asset_count: int

    @property
    def is_unknown(self) -> bool:
        return self.event_type is EventType.NONE


class IntelligenceWorker:
    """Read-only external collector and structured evidence normalizer.

    GitHub responses and X posts remain inert JSON.  This module has no subprocess,
    module-loading, repository checkout, archive download, or execution path.
    """

    GITHUB_POLLS: tuple[
        tuple[str, Callable[..., GitHubPollResult]], ...
    ] = ()

    def __init__(
        self,
        *,
        audit: AuditRepository,
        extractor: OpenAIEventExtractor,
        publisher: EvidencePublisher,
        state: IntelligenceState,
        github_client: GitHubReadOnlyClient | None = None,
        github_repositories: Sequence[str] = (),
        github_poll_limit: int = 30,
        x_client: XFilteredStreamClient | None = None,
        x_content_to_openai_allowed: bool = False,
        symbols: Sequence[str] = (),
        evidence_ttl_seconds: int = 3_600,
        exact_model: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not 1 <= github_poll_limit <= 100:
            raise ValueError("github_poll_limit must be between 1 and 100")
        if evidence_ttl_seconds <= 0:
            raise ValueError("evidence_ttl_seconds must be positive")
        self.audit = audit
        self.extractor = extractor
        self.publisher = publisher
        self.state = state
        self.github_client = github_client
        self.github_repositories = tuple(github_repositories)
        self.github_poll_limit = github_poll_limit
        self.x_client = x_client
        self.x_content_to_openai_allowed = x_content_to_openai_allowed
        self.symbols = tuple(
            dict.fromkeys(item.strip().upper() for item in symbols if item.strip())
        )
        self.evidence_ttl = timedelta(seconds=evidence_ttl_seconds)
        self.exact_model = (exact_model or extractor.model).strip()
        self.clock = clock or (lambda: datetime.now(UTC))
        self._extract_lock = threading.Lock()

    def source_status(self) -> dict[str, str]:
        return {
            "openai": f"required:{self.exact_model}",
            "github": (
                "enabled"
                if self.github_client and self.github_repositories
                else "skipped:not_configured"
            ),
            "x": "enabled" if self.x_client else "skipped:not_configured",
        }

    def startup_check(self) -> dict[str, str]:
        if self.extractor.model != self.exact_model:
            raise AuxiliaryIntelligenceError("openai_model_mismatch")
        self.extractor.check_model_access()
        return self.source_status()

    def close(self) -> None:
        if self.x_client is not None:
            self.x_client.close()
        if self.github_client is not None:
            self.github_client.close()
        self.extractor.close()
        self.audit.close()
        queue_client = getattr(self.publisher.queue, "client", None)
        close = getattr(queue_client, "close", None)
        if callable(close):
            close()

    def poll_github_once(self) -> int:
        if self.github_client is None or not self.github_repositories:
            LOGGER.info("github_collector_skipped reason=not_configured")
            return 0
        processed = 0
        polls = (
            ("releases", self.github_client.poll_releases),
            ("security-advisories", self.github_client.poll_security_advisories),
            ("commits", self.github_client.poll_commits),
        )
        for repository in self.github_repositories:
            for kind, poll in polls:
                state_key = f"{repository.lower()}:{kind}"
                result = poll(
                    repository,
                    etag=self.state.get_etag(state_key),
                    limit=self.github_poll_limit,
                )
                for document in result.documents:
                    self.ingest_document(document)
                    processed += 1
                # Advancing the ETag is the commit point: audit and Redis publication
                # for every returned document have already succeeded.
                if result.etag:
                    self.state.set_etag(state_key, result.etag)
                if result.remaining_requests == 0:
                    LOGGER.warning("github_rate_limit_exhausted repository=%s", repository)
                    return processed
        return processed

    def replay_unpublished(self, *, limit: int = 1_000) -> int:
        """Recover audit-committed notifications after Redis/restart failures."""

        replayed = 0
        now = self.clock().astimezone(UTC)
        for row in reversed(self.audit.latest_external_evidence_batch(limit=limit)):
            payload = row.get("payload") or {}
            raw_notification = payload.get("notification")
            if not isinstance(raw_notification, Mapping):
                continue
            notification = ExternalEvidenceNotification.model_validate(raw_notification)
            if notification.expires_at <= now:
                continue
            if self.state.was_notification_published(notification.evidence_record_id):
                continue
            self._publish(notification)
            replayed += 1
        return replayed

    def ingest_x_payload(self, payload: Mapping[str, Any]) -> IngestionOutcome | None:
        if self.x_client is None:
            return None
        deleted_ids = deleted_post_ids(dict(payload))
        if deleted_ids:
            observed_id = next(iter(deleted_ids))
            stable_id = self.state.resolve_x_post_id(observed_id)
            return self.ingest_deletion(
                source="x_official_api",
                source_id=stable_id,
                observed_source_id=observed_id,
            )
        document = self.x_client.parse_payload(dict(payload))
        if document is None:
            return None
        raw_ids = document.raw.get("edit_history_post_ids") or ()
        observed_id = str(document.raw.get("observed_post_id") or document.source_id)
        self.state.remember_x_post_ids(
            document.source_id,
            (document.source_id, observed_id, *(str(item) for item in raw_ids)),
        )
        return self.ingest_document(document)

    def ingest_document(self, document: RawDocument) -> IngestionOutcome:
        observed_at = self.clock().astimezone(UTC)
        raw_document = _raw_document(document)
        source_hash = _source_content_hash(raw_document)
        evidence_id = f"{document.source}:{document.source_id}"
        latest = self.audit.latest_external_evidence(evidence_id)
        if latest is not None:
            latest_payload = latest.get("payload") or {}
            if latest_payload.get("source_content_hash") == source_hash and not latest.get(
                "deleted_at"
            ):
                notification_payload = latest_payload.get("notification")
                if isinstance(notification_payload, Mapping):
                    notification = ExternalEvidenceNotification.model_validate(
                        notification_payload
                    )
                    if not self.state.was_notification_published(
                        notification.evidence_record_id
                    ):
                        self._publish(notification)
                    return IngestionOutcome(
                        evidence_record_id=notification.evidence_record_id,
                        operation=notification.operation,
                        published=True,
                        replayed=True,
                    )
                raise RuntimeError("latest evidence has no replayable notification")
        operation = EvidenceOperation.EDIT if latest is not None else EvidenceOperation.ORIGINAL
        symbols = _infer_symbols(document, self.symbols)
        (
            extraction,
            status,
            reason,
            normalization_model,
            response_id,
            latency_ms,
        ) = self._extract(document)
        record_id = "evidence_" + hashlib.sha256(
            f"{evidence_id}:{source_hash}:{observed_at.isoformat()}".encode()
        ).hexdigest()[:32]
        notification = _notification(
            record_id=record_id,
            evidence_id=evidence_id,
            version=int(latest["version"]) + 1 if latest else 1,
            operation=operation,
            source=document.source,
            source_id=document.source_id,
            symbols=symbols,
            extraction=extraction,
            status=status,
            extractor_model=normalization_model,
            extractor_prompt_version=self.extractor.prompt_version,
            extractor_response_id=response_id,
            extractor_latency_ms=latency_ms,
            content_hash=source_hash,
            occurred_at=_document_occurred_at(document),
            observed_at=observed_at,
            expires_at=observed_at + self.evidence_ttl,
        )
        payload: dict[str, Any] = {
            "operation": operation.value,
            "source_content_hash": source_hash,
            "raw_document": raw_document,
            "normalization": {
                "status": status.value,
                "reason": reason,
                "extractor_model": normalization_model,
                "extractor_prompt_version": self.extractor.prompt_version,
                "extractor_response_id": response_id,
                "extractor_latency_ms": latency_ms,
            },
            "notification": notification.model_dump(mode="json"),
        }
        first_observed = document.raw.get("first_observed_at") or observed_at
        self.audit.append_external_evidence(
            source=document.source,
            source_id=document.source_id,
            evidence_id=evidence_id,
            evidence_record_id=record_id,
            occurred_at=notification.occurred_at,
            first_observed_at=first_observed,
            source_url=document.url or None,
            payload=payload,
            created_at=observed_at,
        )
        self._publish(notification)
        return IngestionOutcome(record_id, operation, True)

    def ingest_deletion(
        self,
        *,
        source: str,
        source_id: str,
        observed_source_id: str | None = None,
    ) -> IngestionOutcome:
        observed_at = self.clock().astimezone(UTC)
        evidence_id = f"{source}:{source_id}"
        latest = self.audit.latest_external_evidence(evidence_id)
        if latest is not None and latest.get("deleted_at"):
            return IngestionOutcome(
                evidence_record_id=str(latest["evidence_record_id"]),
                operation=EvidenceOperation.DELETE,
                published=False,
                replayed=True,
            )
        previous_payload = (latest or {}).get("payload") or {}
        previous_notification = previous_payload.get("notification") or {}
        symbols = tuple(previous_notification.get("symbols") or ("*",))
        marker = {
            "deleted_source_id": observed_source_id or source_id,
            "prior_content_hash": previous_payload.get("source_content_hash"),
        }
        content_hash = hashlib.sha256(
            json.dumps(marker, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        record_id = "evidence_" + hashlib.sha256(
            f"{evidence_id}:delete:{observed_at.isoformat()}".encode()
        ).hexdigest()[:32]
        notification = _notification(
            record_id=record_id,
            evidence_id=evidence_id,
            version=int(latest["version"]) + 1 if latest else 1,
            operation=EvidenceOperation.DELETE,
            source=source,
            source_id=source_id,
            symbols=symbols,
            extraction=None,
            status=NormalizationStatus.DELETED,
            extractor_model="none:source-deletion",
            extractor_prompt_version=self.extractor.prompt_version,
            extractor_response_id=None,
            extractor_latency_ms=None,
            content_hash=content_hash,
            occurred_at=observed_at,
            observed_at=observed_at,
            expires_at=observed_at + self.evidence_ttl,
            deleted_at=observed_at,
        )
        self.audit.append_external_evidence(
            source=source,
            source_id=source_id,
            evidence_id=evidence_id,
            evidence_record_id=record_id,
            occurred_at=observed_at,
            first_observed_at=(latest or {}).get("first_observed_at") or observed_at,
            deleted_at=observed_at,
            payload={
                "operation": EvidenceOperation.DELETE.value,
                "source_content_hash": content_hash,
                "deletion": marker,
                "normalization": {
                    "status": NormalizationStatus.DELETED.value,
                    "reason": "source_deleted",
                    "extractor_model": "none:source-deletion",
                    "extractor_prompt_version": self.extractor.prompt_version,
                },
                "notification": notification.model_dump(mode="json"),
            },
            created_at=observed_at,
        )
        self._publish(notification)
        return IngestionOutcome(record_id, EvidenceOperation.DELETE, True)

    def _publish(self, notification: ExternalEvidenceNotification) -> None:
        self.publisher.publish(notification)
        remaining = max(
            60,
            int((notification.expires_at - self.clock().astimezone(UTC)).total_seconds()),
        )
        self.state.mark_notification_published(
            notification.evidence_record_id,
            ttl_seconds=remaining + 60,
        )

    def _extract(
        self, document: RawDocument
    ) -> tuple[
        EventExtraction | None,
        NormalizationStatus,
        str | None,
        str,
        str | None,
        int | None,
    ]:
        local_x: _LocalXFeatures | None = None
        if document.source == "x_official_api" and not self.x_content_to_openai_allowed:
            local_x = _local_x_features(document, self.symbols)
            if local_x.is_unknown:
                extraction = EventExtraction(
                    event_type=EventType.NONE,
                    sentiment=local_x.sentiment,
                    confidence=local_x.confidence,
                    source_ids=(document.source_id,),
                    aggregates=EventAggregates(
                        document_count=1,
                        source_count=1,
                        engagement_total=local_x.engagement_total,
                        weighted_sentiment=local_x.sentiment,
                        verified_source_share=local_x.verified_source_share,
                        novelty_score=local_x.novelty_score,
                    ),
                )
                return (
                    extraction,
                    NormalizationStatus.COMPLETED,
                    "local_unknown_x_content_redacted",
                    "local-deterministic-v1",
                    None,
                    None,
                )
        packet = _openai_document(
            document,
            allow_x_content=self.x_content_to_openai_allowed,
            local_x=local_x,
        )
        try:
            with self._extract_lock:
                extraction = self.extractor.extract((packet,))
                response_id = getattr(self.extractor, "last_response_id", None)
                latency_ms = getattr(self.extractor, "last_latency_ms", None)
        except (AuxiliaryIntelligenceError, SecurityBoundaryError) as error:
            return (
                None,
                NormalizationStatus.REJECTED,
                str(error)[:160],
                self.exact_model,
                None,
                None,
            )
        except Exception:
            LOGGER.exception(
                "intelligence_extractor_internal_error source=%s source_id=%s",
                document.source,
                document.source_id,
            )
            return (
                None,
                NormalizationStatus.REJECTED,
                "extractor_internal_error",
                self.exact_model,
                None,
                None,
            )
        if local_x is not None:
            if extraction.event_type not in {local_x.event_type, EventType.NONE}:
                return (
                    None,
                    NormalizationStatus.REJECTED,
                    "x_model_conflicts_local_event_type",
                    self.exact_model,
                    response_id,
                    latency_ms,
                )
            if (
                local_x.sentiment > 0.05
                and extraction.sentiment < -0.05
                or local_x.sentiment < -0.05
                and extraction.sentiment > 0.05
            ):
                return (
                    None,
                    NormalizationStatus.REJECTED,
                    "x_model_conflicts_local_sentiment",
                    self.exact_model,
                    response_id,
                    latency_ms,
                )
        return (
            extraction,
            NormalizationStatus.COMPLETED,
            None,
            self.exact_model,
            response_id,
            latency_ms,
        )


class IntelligenceRuntime:
    def __init__(
        self,
        worker: IntelligenceWorker,
        *,
        poll_seconds: float,
        x_reconnect_seconds: float,
    ) -> None:
        self.worker = worker
        self.poll_seconds = poll_seconds
        self.x_reconnect_seconds = x_reconnect_seconds
        self.stop_event = threading.Event()

    def stop(self, *_: object) -> None:
        self.stop_event.set()

    def run(self) -> None:
        x_thread: threading.Thread | None = None
        try:
            status = self.worker.startup_check()
            LOGGER.info("intelligence_startup %s", json.dumps(status, sort_keys=True))
            replayed = self.worker.replay_unpublished()
            LOGGER.info("intelligence_outbox_replayed count=%d", replayed)
            if self.worker.x_client is not None:
                x_thread = threading.Thread(
                    target=self._run_x_stream,
                    name="x-filtered-stream",
                    daemon=True,
                )
                x_thread.start()
            else:
                LOGGER.info("x_collector_skipped reason=not_configured")
            while not self.stop_event.is_set():
                try:
                    replayed = self.worker.replay_unpublished()
                    processed = self.worker.poll_github_once()
                    LOGGER.info(
                        "github_poll_completed processed=%d outbox_replayed=%d",
                        processed,
                        replayed,
                    )
                except Exception as error:
                    LOGGER.error("github_poll_failed error=%s", type(error).__name__)
                self.stop_event.wait(self.poll_seconds)
        finally:
            self.stop_event.set()
            if x_thread is not None:
                x_thread.join(timeout=min(5.0, self.x_reconnect_seconds + 1))
            self.worker.close()

    def _run_x_stream(self) -> None:
        assert self.worker.x_client is not None
        while not self.stop_event.is_set():
            try:
                self.worker.replay_unpublished()
                for payload in self.worker.x_client.stream_payloads():
                    if self.stop_event.is_set():
                        return
                    self.worker.ingest_x_payload(payload)
            except Exception as error:
                LOGGER.error("x_stream_disconnected error=%s", type(error).__name__)
            self.stop_event.wait(self.x_reconnect_seconds)


def build_intelligence_runtime(
    settings: Settings | None = None,
    *,
    webhook_only: bool = False,
) -> IntelligenceRuntime:
    settings = settings or Settings.from_env()
    audit = AuditRepository(settings.audit_database_url)
    audit.initialize()
    extractor = OpenAIEventExtractor(
        api_key=settings.openai_api_key,
        model=settings.openai_extraction_model,
        project=settings.openai_project,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.openai_request_timeout_seconds,
        x_content_to_openai_allowed=settings.x_content_to_openai_allowed,
    )
    consumer = f"intelligence-{socket.gethostname()}"
    queue = RedisStreamQueue(
        settings.redis_url,
        stream=settings.intelligence_stream_name,
        group=EVIDENCE_CONSUMER_GROUP,
        consumer=consumer,
    )
    state = RedisIntelligenceState(queue.client)
    github_client: GitHubReadOnlyClient | None = None
    if settings.github_allowed_repositories and not webhook_only:
        github_client = GitHubReadOnlyClient(
            settings.github_token,
            allowed_repositories=settings.github_allowed_repositories,
        )
    x_configured = not webhook_only and bool(
        settings.x_bearer_token or settings.x_allowed_account_ids
    )
    if x_configured and not (
        settings.x_bearer_token and settings.x_allowed_account_ids
    ):
        raise ValueError("X collector requires both bearer token and account-ID allowlist")
    x_client = (
        XFilteredStreamClient(
            settings.x_bearer_token,
            allowed_account_ids=settings.x_allowed_account_ids,
            timeout=settings.openai_request_timeout_seconds,
        )
        if settings.x_bearer_token and not webhook_only
        else None
    )
    worker = IntelligenceWorker(
        audit=audit,
        extractor=extractor,
        publisher=EvidencePublisher(queue),
        state=state,
        github_client=github_client,
        github_repositories=settings.github_allowed_repositories,
        github_poll_limit=settings.github_poll_limit,
        x_client=x_client,
        x_content_to_openai_allowed=settings.x_content_to_openai_allowed,
        symbols=settings.futures_universe,
        evidence_ttl_seconds=settings.intelligence_evidence_ttl_seconds,
        exact_model=settings.openai_extraction_model,
    )
    return IntelligenceRuntime(
        worker,
        poll_seconds=settings.intelligence_poll_seconds,
        x_reconnect_seconds=settings.x_stream_reconnect_seconds,
    )


def build_evidence_inbox_queue(
    settings: Settings,
    *,
    consumer: str,
) -> RedisStreamQueue:
    """Create the queue adapter consumed by ``EvidenceInbox.consume_once``."""

    return RedisStreamQueue(
        settings.redis_url,
        stream=settings.intelligence_stream_name,
        group=EVIDENCE_CONSUMER_GROUP,
        consumer=consumer,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runtime = build_intelligence_runtime()
    signal.signal(signal.SIGINT, runtime.stop)
    signal.signal(signal.SIGTERM, runtime.stop)
    runtime.run()


def _raw_document(document: RawDocument) -> dict[str, JsonValue]:
    value = {
        "source": document.source,
        "source_id": document.source_id,
        "doc_type": document.doc_type.value,
        "title": document.title,
        "text": document.text,
        "published_at": document.published_at.astimezone(UTC).isoformat(),
        "url": document.url,
        "author": document.author,
        "engagement": document.engagement,
        "raw": document.raw,
    }
    sanitized = sanitize_untrusted_json(
        value,
        max_depth=8,
        max_mapping_items=100,
        max_sequence_items=100,
        max_string_chars=50_000,
        max_nodes=5_000,
        max_bytes=256_000,
    )
    if not isinstance(sanitized, dict):  # pragma: no cover - fixed local shape
        raise SecurityBoundaryError("external_document_shape_invalid")
    return sanitized


def _source_content_hash(document: Mapping[str, JsonValue]) -> str:
    stable = dict(document)
    raw = stable.get("raw")
    if isinstance(raw, dict):
        stable["raw"] = {
            key: value
            for key, value in raw.items()
            if key not in {"ingested_at", "first_observed_at"}
        }
    encoded = json.dumps(
        stable,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _openai_document(
    document: RawDocument,
    *,
    allow_x_content: bool,
    local_x: _LocalXFeatures | None = None,
) -> dict[str, Any]:
    if document.source == "x_official_api" and not allow_x_content:
        payload = XFilteredStreamClient.external_llm_payload(
            document,
            allow_content=False,
        )
        payload.pop("account_id", None)
        if local_x is None:  # pragma: no cover - guarded by IntelligenceWorker
            raise SecurityBoundaryError("local_x_classification_missing")
        payload.update(
            {
                "event_type": local_x.event_type.value,
                "sentiment": local_x.sentiment,
                "confidence": local_x.confidence,
                "aggregates": {
                    "engagement_total": local_x.engagement_total,
                    "verified_source_share": local_x.verified_source_share,
                    "novelty_score": local_x.novelty_score,
                    "factuality": local_x.factuality,
                    "urgency": local_x.urgency,
                    "bot_score": local_x.bot_score,
                    "asset_count": local_x.asset_count,
                },
                "language": str(document.raw.get("lang") or "unknown")[:32],
            }
        )
        return payload
    return {
        "source": document.source,
        "source_id": document.source_id,
        "title": document.title,
        "text": document.text,
        "published_at": document.published_at.astimezone(UTC).isoformat(),
        "url": document.url,
        "author": document.author,
        "engagement": document.engagement,
        "repository": document.raw.get("repository"),
        "kind": document.raw.get("kind"),
        "severity": document.raw.get("severity"),
    }


def _infer_symbols(document: RawDocument, universe: Sequence[str]) -> tuple[str, ...]:
    text = f"{document.title}\n{document.text}".upper()
    result: list[str] = []
    for symbol in universe:
        base = symbol.removesuffix("USDT")
        variants = (symbol, base)
        if any(
            re.search(rf"(?<![A-Z0-9])\$?{re.escape(item)}(?![A-Z0-9])", text)
            for item in variants
        ):
            result.append(symbol)
    return tuple(result) or ("*",)


def _local_x_features(
    document: RawDocument,
    universe: Sequence[str],
) -> _LocalXFeatures:
    text = f"{document.title} {document.text}".strip()
    lowered = text.lower()
    rule = next(
        (item for item in RULES if any(term in lowered for term in item.terms)),
        None,
    )
    engagement_total = sum(
        max(0.0, float(value)) for value in document.engagement.values()
    )
    verified_share = 1.0 if bool(document.raw.get("verified")) else 0.0
    inferred_symbols = _infer_symbols(document, universe)
    asset_count = 0 if inferred_symbols == ("*",) else len(inferred_symbols)
    hedges = sum(lowered.count(word) for word in ("rumor", "might", "maybe", "i think"))
    evidence_terms = sum(
        lowered.count(word)
        for word in ("official", "announced", "verified", "starts", "reported")
    )
    source_quality = 0.85 if verified_share else 0.60
    factuality = max(
        0.05,
        min(0.98, 0.52 + 0.35 * source_quality + 0.05 * evidence_terms - 0.14 * hedges),
    )
    bot_score = min(0.95, 0.15 + (0.20 if engagement_total < 2 else 0.0))
    if rule is None:
        return _LocalXFeatures(
            event_type=EventType.NONE,
            sentiment=0.0,
            confidence=0.10,
            engagement_total=engagement_total,
            verified_source_share=verified_share,
            novelty_score=0.10,
            factuality=factuality,
            urgency=0.10,
            bot_score=bot_score,
            asset_count=asset_count,
        )
    event_type = {
        "listing": EventType.LISTING,
        "delisting": EventType.DELISTING,
        "hack": EventType.SECURITY,
        "exploit": EventType.SECURITY,
        "partnership": EventType.OTHER,
        "regulation": EventType.REGULATION,
        "exchange_maintenance": EventType.EXCHANGE_OUTAGE,
        "token_unlock": EventType.MARKET_STRUCTURE,
        "whale_flow": EventType.MARKET_STRUCTURE,
        "opinion": EventType.SOCIAL_SENTIMENT,
        "unknown": EventType.NONE,
    }[rule.event_type.value]
    positive = sum(
        lowered.count(word)
        for word in ("approved", "list", "partnership", "bullish", "moon")
    )
    negative = sum(
        lowered.count(word)
        for word in ("hack", "exploit", "delist", "pause", "bearish")
    )
    lexical = (positive - negative) / max(1, positive + negative)
    sentiment = max(-1.0, min(1.0, lexical or 0.65 * rule.polarity.direction))
    confidence = min(
        0.95,
        0.50 * rule.strength + 0.30 * source_quality + 0.20 * factuality,
    )
    return _LocalXFeatures(
        event_type=event_type,
        sentiment=sentiment,
        confidence=confidence,
        engagement_total=engagement_total,
        verified_source_share=verified_share,
        novelty_score=(
            0.90
            if event_type not in {EventType.SOCIAL_SENTIMENT, EventType.NONE}
            else 0.20
        ),
        factuality=factuality,
        urgency=rule.urgency,
        bot_score=bot_score,
        asset_count=asset_count,
    )


def _document_occurred_at(document: RawDocument) -> datetime:
    updated = document.raw.get("updated_at")
    if isinstance(updated, str) and updated:
        try:
            value = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            return (value if value.tzinfo else value.replace(tzinfo=UTC)).astimezone(UTC)
        except ValueError:
            pass
    return document.published_at.astimezone(UTC)


def _notification(
    *,
    record_id: str,
    evidence_id: str,
    version: int,
    operation: EvidenceOperation,
    source: str,
    source_id: str,
    symbols: Sequence[str],
    extraction: EventExtraction | None,
    status: NormalizationStatus,
    extractor_model: str,
    extractor_prompt_version: str,
    extractor_response_id: str | None,
    extractor_latency_ms: int | None,
    content_hash: str,
    occurred_at: datetime,
    observed_at: datetime,
    expires_at: datetime,
    deleted_at: datetime | None = None,
) -> ExternalEvidenceNotification:
    if extraction is None:
        event_type = EventType.NONE
        sentiment = 0.0
        confidence = 0.0
        source_ids: tuple[str, ...] = ()
        aggregates: dict[str, JsonValue] = {
            "document_count": 0,
            "source_count": 0,
            "engagement_total": 0.0,
            "weighted_sentiment": 0.0,
            "verified_source_share": 0.0,
            "novelty_score": 0.0,
        }
    else:
        event_type = extraction.event_type
        sentiment = extraction.sentiment
        confidence = extraction.confidence
        source_ids = extraction.source_ids
        aggregates = task_payload(extraction.aggregates.model_dump(mode="json"))
    return ExternalEvidenceNotification(
        evidence_record_id=record_id,
        evidence_id=evidence_id,
        version=version,
        operation=operation,
        source=source,
        source_id=source_id,
        symbols=tuple(symbols),
        event_type=event_type,
        sentiment=sentiment,
        confidence=confidence,
        source_ids=source_ids,
        aggregates=aggregates,
        normalization_status=status,
        usable_for_trading=(
            status is NormalizationStatus.COMPLETED
            and event_type is not EventType.NONE
            and confidence >= MIN_USABLE_CONFIDENCE
        ),
        extractor_model=extractor_model,
        extractor_prompt_version=extractor_prompt_version,
        extractor_response_id=extractor_response_id,
        extractor_latency_ms=extractor_latency_ms,
        content_hash=content_hash,
        occurred_at=occurred_at,
        observed_at=observed_at,
        expires_at=expires_at,
        deleted_at=deleted_at,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
