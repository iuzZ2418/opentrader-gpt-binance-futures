from datetime import date, timedelta

import pytest

from company_event_monitor.domain import Company
from company_event_monitor.market import PriceBar, analyze_market
from company_event_monitor.storage import EventRepository


def _bars(*, drift: float, count: int = 420, base: float = 20.0) -> list[PriceBar]:
    start = date(2024, 1, 1)
    result: list[PriceBar] = []
    close = base
    for index in range(count):
        cycle = ((index % 17) - 8) / 4000
        close *= 1 + drift + cycle
        result.append(
            PriceBar(
                trade_date=start + timedelta(days=index),
                open=close * 0.995,
                close=close,
                high=close * 1.01,
                low=close * 0.99,
                volume=1_000_000 + index * 1000,
                amount=close * 1_000_000,
                pct_change=(drift + cycle) * 100,
                turnover=1.2,
            )
        )
    return result


def test_market_analysis_has_auditable_forecast() -> None:
    company = Company("szse:002050", "三花智控", "002050", market="深市")
    analysis = analyze_market(
        company,
        _bars(drift=0.001),
        "深证成指",
        _bars(drift=0.0004, base=10_000),
        [
            {
                "id": 1,
                "published_at": "2024-10-01T08:00:00+08:00",
                "standardized_text": "公司订单增长。",
                "direction": 1,
                "value_score": 0.87,
                "change_type": "new",
            }
        ],
    )

    forecast = analysis["forecast_20d"]
    probabilities = forecast["probabilities"]
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.001)
    assert forecast["price_range"]["downside_p10"] < forecast["price_range"]["upside_p90"]
    assert 0 <= forecast["confidence"] <= 0.85
    assert forecast["backtest"]["sample_count"] > 0
    assert forecast["method"]
    assert analysis["event_price_links"]
    assert "不" in analysis["disclaimer"]


def test_market_data_persists_locally(tmp_path) -> None:
    company = Company("szse:002050", "三花智控", "002050", market="深市")
    bars = _bars(drift=0.0005, count=40)
    repository = EventRepository(tmp_path / "events.db")
    repository.initialize()
    repository.upsert_company(company)

    assert repository.upsert_price_bars(company.company_id, bars) == 40
    analysis = {
        "as_of": bars[-1].trade_date.isoformat(),
        "data_hash": "market-v1",
        "latest_price": bars[-1].close,
    }
    repository.save_market_analysis(company.company_id, analysis)

    assert len(repository.price_bars(company.company_id)) == 40
    saved = repository.latest_market_analysis(company.company_id)
    assert saved is not None
    assert saved["analysis"]["data_hash"] == "market-v1"
