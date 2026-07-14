from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Protocol

from .analysis import classify_change, event_family, score_event
from .domain import ChangeType, Document, FundamentalEvent
from .extraction import BaselineChineseExtractor
from .parsing import segment_text
from .storage import EventRepository


@dataclass(slots=True)
class ProcessResult:
    inserted_document: bool
    document_id: int
    extracted: int
    inserted_events: int


class EventExtractor(Protocol):
    def extract(self, document: Document) -> list[FundamentalEvent]: ...


class MonitorService:
    def __init__(
        self,
        repository: EventRepository,
        extractor: EventExtractor | None = None,
        *,
        backfill: bool = True,
    ) -> None:
        self.repository = repository
        self.extractor = extractor
        self.repository.initialize()
        if backfill:
            self._backfill_segments()
            self._backfill_relations()

    def process_document(self, document: Document) -> ProcessResult:
        if not document.segments:
            document = replace(document, segments=segment_text(document.text))
        document_id, inserted = self.repository.insert_document(document)
        if not inserted and self.repository.document_event_count(document_id) > 0:
            return ProcessResult(False, document_id, 0, 0)
        extractor = self.extractor or BaselineChineseExtractor(self.repository.list_companies())
        events = extractor.extract(document)
        history = self.repository.history(events[0].company_id) if events else []
        inserted_events = 0
        for event in events:
            previous = self._latest_related(event, history)
            classify_change(event, history)
            score_event(event)
            event_id, event_inserted = self.repository.insert_event(document_id, event)
            inserted_events += int(event_inserted)
            if event_inserted and previous is not None:
                self.repository.add_relation(
                    self.repository.event_id_for(previous),
                    event_id,
                    self._relation_type(event.change_type),
                    SequenceMatcher(None, previous.evidence_text, event.evidence_text).ratio(),
                )
            history.append(event)
        return ProcessResult(True, document_id, len(events), inserted_events)

    @staticmethod
    def _latest_related(
        event: FundamentalEvent,
        history: list[FundamentalEvent],
    ) -> FundamentalEvent | None:
        matches = [
            item
            for item in history
            if item.company_id == event.company_id
            and event_family(item) == event_family(event)
            and item.published_at < event.published_at
        ]
        return max(matches, key=lambda item: item.published_at, default=None)

    @staticmethod
    def _relation_type(change: ChangeType) -> str:
        return {
            ChangeType.NEW: "supports",
            ChangeType.REPEAT: "duplicates",
            ChangeType.UPDATE: "updates",
            ChangeType.ESCALATION: "escalates",
            ChangeType.REVERSAL: "reverses",
            ChangeType.CONFLICT: "conflicts",
        }[change]

    def _backfill_relations(self) -> None:
        history: list[FundamentalEvent] = []
        for event in self.repository.history():
            previous = self._latest_related(event, history)
            if previous is not None:
                self.repository.add_relation(
                    self.repository.event_id_for(previous),
                    self.repository.event_id_for(event),
                    self._relation_type(event.change_type),
                    SequenceMatcher(None, previous.evidence_text, event.evidence_text).ratio(),
                )
            history.append(event)

    def _backfill_segments(self) -> None:
        for document_id, text in self.repository.documents_without_segments():
            self.repository.insert_segments(document_id, segment_text(text))
