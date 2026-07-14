from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .domain import Company, Document, SourceTier
from .extraction import BaselineChineseExtractor


def evaluate(path: Path) -> dict[str, Any]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    company = Company("evaluation", "示例公司", "000001", aliases=("示例企业",))
    extractor = BaselineChineseExtractor([company])
    true_positive = false_positive = false_negative = 0
    case_results = []
    numeric_total = numeric_preserved = 0
    for index, case in enumerate(cases, start=1):
        document = Document(
            source_id=f"case-{index}",
            source_name="人工评测样本",
            source_tier=SourceTier.A,
            doc_type="evaluation",
            title="示例公司披露",
            text=case["text"],
            published_at=datetime.fromisoformat("2026-01-01T00:00:00+08:00"),
        )
        events = extractor.extract(document)
        predicted = {event.event_type.value for event in events}
        expected = set(case["expected_event_types"])
        true_positive += len(predicted & expected)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
        expected_numbers = set(case.get("expected_numbers", []))
        predicted_numbers = {value for event in events for value in event.numeric_evidence}
        numeric_total += len(expected_numbers)
        numeric_preserved += len(expected_numbers & predicted_numbers)
        case_results.append(
            {
                "id": case.get("id", str(index)),
                "passed": predicted == expected and expected_numbers <= predicted_numbers,
                "expected": sorted(expected),
                "predicted": sorted(predicted),
                "missing_numbers": sorted(expected_numbers - predicted_numbers),
            }
        )
    precision = _divide(true_positive, true_positive + false_positive)
    recall = _divide(true_positive, true_positive + false_negative)
    return {
        "cases": len(cases),
        "passed_cases": sum(item["passed"] for item in case_results),
        "precision": precision,
        "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall),
        "numeric_preservation": _divide(numeric_preserved, numeric_total),
        "details": case_results,
    }


def _divide(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="评测上市公司事件抽取基线")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/evaluation_seed.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(evaluate(args.dataset), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
