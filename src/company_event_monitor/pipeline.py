from __future__ import annotations

from dataclasses import asdict

from .analysis import classify_change, score_event
from .domain import Company, Document, FundamentalEvent
from .extraction import BaselineChineseExtractor


class EventPipeline:
    def __init__(self, companies: list[Company]) -> None:
        self.extractor = BaselineChineseExtractor(companies)
        self.events: list[FundamentalEvent] = []

    def process(self, documents: list[Document]) -> list[FundamentalEvent]:
        added: list[FundamentalEvent] = []
        for document in sorted(documents, key=lambda item: item.published_at):
            for event in self.extractor.extract(document):
                classify_change(event, self.events)
                score_event(event)
                self.events.append(event)
                added.append(event)
        return sorted(added, key=lambda item: (item.value_score, item.published_at), reverse=True)

    def feed(self, limit: int = 100) -> list[dict]:
        return [
            asdict(item)
            for item in sorted(
                self.events, key=lambda item: (item.value_score, item.published_at), reverse=True
            )[:limit]
        ]
