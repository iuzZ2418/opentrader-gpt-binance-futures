from pathlib import Path

from company_event_monitor.bootstrap import bootstrap


def test_bootstrap_seeds_only_empty_database(tmp_path) -> None:
    sample = Path(__file__).parents[1] / "data" / "company_event_sample.json"
    database = tmp_path / "demo.db"
    first = bootstrap(database, sample)
    second = bootstrap(database, sample)
    assert first["seeded"] is True
    assert first["counts"]["documents"] == 3
    assert second["seeded"] is False
    assert second["counts"] == first["counts"]
