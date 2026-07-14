from __future__ import annotations

from collections import Counter
from typing import Any


def annotation_agreement(
    annotations: list[dict[str, Any]],
    annotator_a: str,
    annotator_b: str,
) -> dict[str, Any]:
    by_segment: dict[int, dict[str, dict[str, Any]]] = {}
    for annotation in annotations:
        by_segment.setdefault(int(annotation["segment_id"]), {})[str(annotation["annotator"])] = (
            annotation
        )
    pairs = [
        (segment_id, values[annotator_a], values[annotator_b])
        for segment_id, values in by_segment.items()
        if annotator_a in values and annotator_b in values
    ]
    labels_a = [_category(first) for _, first, _ in pairs]
    labels_b = [_category(second) for _, _, second in pairs]
    observed = _divide(sum(a == b for a, b in zip(labels_a, labels_b, strict=True)), len(pairs))
    expected = _expected_agreement(labels_a, labels_b)
    kappa = _divide(observed - expected, 1 - expected) if expected < 1 else 1.0
    disagreements = [
        {
            "segment_id": segment_id,
            "text": first["text"],
            "annotator_a": _category(first),
            "annotator_b": _category(second),
        }
        for segment_id, first, second in pairs
        if _category(first) != _category(second)
    ]
    return {
        "annotator_a": annotator_a,
        "annotator_b": annotator_b,
        "overlap": len(pairs),
        "exact_agreement": observed,
        "cohen_kappa": round(kappa, 4),
        "disagreements": disagreements,
    }


def _category(annotation: dict[str, Any]) -> str:
    label = str(annotation["label"])
    if label != "event":
        return label
    return ":".join(
        (
            "event",
            str(annotation.get("event_type") or ""),
            str(annotation.get("direction") if annotation.get("direction") is not None else ""),
            str(annotation.get("status") or ""),
        )
    )


def _expected_agreement(labels_a: list[str], labels_b: list[str]) -> float:
    if not labels_a:
        return 0.0
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    categories = set(counts_a) | set(counts_b)
    total = len(labels_a)
    return sum((counts_a[item] / total) * (counts_b[item] / total) for item in categories)


def _divide(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
