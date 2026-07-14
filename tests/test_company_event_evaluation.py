from pathlib import Path

from company_event_monitor.evaluation import evaluate


def test_seed_evaluation_is_reproducible() -> None:
    dataset = Path(__file__).parents[1] / "data" / "evaluation_seed.json"
    result = evaluate(dataset)
    assert result["cases"] == 12
    assert result["passed_cases"] == 12
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["numeric_preservation"] == 1.0
