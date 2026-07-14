from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli import run_file_persistent
from .storage import EventRepository


def bootstrap(database: Path, sample: Path | None = None) -> dict:
    repository = EventRepository(database)
    repository.initialize()
    before = repository.counts()
    seeded = False
    if sample is not None and before["companies"] == 0:
        run_file_persistent(sample, database)
        seeded = True
    return {"seeded": seeded, "counts": repository.counts(), "database": str(database)}


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化上市公司事件数据库")
    parser.add_argument("--database", type=Path, default=Path("data/company_events.db"))
    parser.add_argument("--sample", type=Path)
    args = parser.parse_args()
    print(json.dumps(bootstrap(args.database, args.sample), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
