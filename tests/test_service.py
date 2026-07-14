from dataclasses import replace
from pathlib import Path

from crypto_event_trader.config import Settings
from crypto_event_trader.database import Repository
from crypto_event_trader.service import TradingService


def test_sample_cycle_is_idempotent_and_closes_the_loop(tmp_path: Path) -> None:
    settings = replace(
        Settings.from_env(),
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        execution_venue="internal",
    )
    repository = Repository(settings.sqlite_path())
    service = TradingService(settings, repository)
    sample = Path(__file__).parents[1] / "data" / "sample_documents.json"

    first = service.run_sample_cycle(sample)
    second = service.run_sample_cycle(sample)

    assert first.documents_inserted == 3
    assert first.events_created == 3
    assert first.signals_created == 3
    assert first.orders_filled >= 1
    assert second.documents_inserted == 0
    assert second.events_created == 0
    assert len(repository.list_orders()) == first.orders_filled
    assert repository.portfolio()["positions"]
