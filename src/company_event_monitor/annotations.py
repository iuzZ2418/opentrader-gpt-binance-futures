from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import EventRepository


def export_annotations(database: Path, output: Path) -> int:
    repository = EventRepository(database)
    repository.initialize()
    rows = repository.annotations()
    content = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content + ("\n" if content else ""), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出研究员片段标注为JSONL")
    parser.add_argument("--database", type=Path, default=Path("data/company_events.db"))
    parser.add_argument("--output", type=Path, default=Path("outputs/annotations.jsonl"))
    args = parser.parse_args()
    count = export_annotations(args.database, args.output)
    print(json.dumps({"annotations": count, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
