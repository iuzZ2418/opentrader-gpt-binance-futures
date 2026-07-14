from company_event_monitor.registry import register_company
from company_event_monitor.storage import EventRepository


def test_register_explicit_sse_company(tmp_path) -> None:
    database = tmp_path / "registry.db"
    company = register_company(
        database,
        "600785",
        market="sse",
        name="新华百货",
        source_org_id="gssh0600785",
        aliases=("银川新华百货",),
    )
    saved = EventRepository(database).list_companies()[0]
    assert company.company_id == "sse-600785"
    assert saved.source_org_id == "gssh0600785"
    assert saved.aliases == ("银川新华百货",)
