from company_event_monitor.agreement import annotation_agreement


def annotation(segment_id: int, annotator: str, label: str, event_type: str | None = None) -> dict:
    return {
        "segment_id": segment_id,
        "annotator": annotator,
        "label": label,
        "event_type": event_type,
        "direction": -1 if event_type else None,
        "status": "occurred" if event_type else None,
        "text": f"片段{segment_id}",
    }


def test_agreement_reports_overlap_kappa_and_disagreements() -> None:
    rows = [
        annotation(1, "a", "event", "order_decline"),
        annotation(1, "b", "event", "order_decline"),
        annotation(2, "a", "no_event"),
        annotation(2, "b", "event", "order_decline"),
        annotation(3, "a", "no_event"),
        annotation(3, "b", "no_event"),
    ]
    result = annotation_agreement(rows, "a", "b")
    assert result["overlap"] == 3
    assert result["exact_agreement"] == 0.6667
    assert len(result["disagreements"]) == 1
    assert result["disagreements"][0]["segment_id"] == 2


def test_agreement_without_overlap_is_explicit() -> None:
    result = annotation_agreement([], "a", "b")
    assert result["overlap"] == 0
    assert result["exact_agreement"] == 0.0
    assert result["cohen_kappa"] == 0.0
