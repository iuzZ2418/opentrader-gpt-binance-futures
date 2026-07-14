from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from typing import Any


def daily_digest(events: list[dict[str, Any]], day: date | None = None, limit: int = 20) -> dict:
    target = day or datetime.now().astimezone().date()
    selected = [event for event in events if _date(event["published_at"]) == target]
    selected.sort(key=lambda event: (event["value_score"], event["published_at"]), reverse=True)
    changes = Counter(str(event["change_type"]) for event in selected)
    event_types = Counter(str(event["event_type"]) for event in selected)
    companies = Counter(str(event["company_name"]) for event in selected)
    return {
        "date": target.isoformat(),
        "total": len(selected),
        "change_counts": dict(changes),
        "top_event_types": event_types.most_common(10),
        "top_companies": companies.most_common(10),
        "events": selected[:limit],
    }


def _date(value: datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(value).date()
