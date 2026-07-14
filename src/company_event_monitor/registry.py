from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .domain import Company
from .ingestion.cninfo import lookup_szse_company
from .storage import EventRepository


def register_company(
    database: Path,
    ticker: str,
    *,
    market: str,
    name: str = "",
    source_org_id: str = "",
    industry: str = "汽车零部件",
    aliases: tuple[str, ...] = (),
) -> Company:
    market = market.lower()
    if market == "szse" and not source_org_id:
        discovered = lookup_szse_company(ticker)
        if discovered is None:
            raise ValueError(f"Ticker not found in official SZSE catalog: {ticker}")
        company = replace(
            discovered,
            name=name or discovered.name,
            aliases=tuple(dict.fromkeys((discovered.name, *aliases))),
            industry=industry,
        )
    else:
        if not name or not source_org_id:
            raise ValueError("name and source_org_id are required outside automatic SZSE lookup")
        company = Company(
            company_id=f"{market}-{ticker}",
            name=name,
            ticker=ticker,
            aliases=aliases,
            industry=industry,
            market=market,
            source_org_id=source_org_id,
        )
    repository = EventRepository(database)
    repository.initialize()
    repository.upsert_company(company)
    return company


def main() -> None:
    parser = argparse.ArgumentParser(description="登记公告监测公司")
    parser.add_argument("ticker")
    parser.add_argument("--market", choices=("szse", "sse"), required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--source-org-id", default="")
    parser.add_argument("--industry", default="汽车零部件")
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--database", type=Path, default=Path("data/company_events.db"))
    args = parser.parse_args()
    company = register_company(
        args.database,
        args.ticker,
        market=args.market,
        name=args.name,
        source_org_id=args.source_org_id,
        industry=args.industry,
        aliases=tuple(args.alias),
    )
    print(json.dumps(asdict(company), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
