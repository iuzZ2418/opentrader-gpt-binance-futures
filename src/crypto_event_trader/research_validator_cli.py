from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from .audit import AuditRepository
from .research_manifest import (
    ResearchManifestError,
    load_verified_research_manifest,
    validate_and_append_research_manifest,
)
from .research_validation import ResearchValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a read-only, SHA-256-pinned research manifest and append its "
            "evidence to PostgreSQL"
        )
    )
    parser.add_argument("--manifest", required=True, help="read-only JSON manifest path")
    parser.add_argument(
        "--sha256",
        required=True,
        help="operator-verified SHA-256 of the exact manifest bytes",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # Verify the operator-pinned bytes before opening a database connection or applying a
        # migration.  The manifest is never imported and no field is treated as executable code
        # or a path to executable code.
        loaded = load_verified_research_manifest(
            args.manifest,
            expected_sha256=args.sha256,
        )
        database_url = os.getenv("AUDIT_DATABASE_URL", "").strip()
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ResearchManifestError("POSTGRES_AUDIT_DATABASE_REQUIRED")
        audit = AuditRepository(database_url)
        try:
            audit.initialize()
            result = validate_and_append_research_manifest(loaded, audit)
        finally:
            audit.close()
    except ResearchManifestError as error:
        print(
            json.dumps(
                {"status": "REJECTED", "reason_code": error.code},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except (ResearchValidationError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "REJECTED",
                    "reason_code": f"VALIDATION_{type(error).__name__.upper()}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception:  # noqa: BLE001 - the production CLI must fail closed without a traceback
        # Database drivers and migration adapters have optional exception hierarchies.  Do not
        # let an unexpected adapter failure print a connection URL or any manifest content.
        print(
            json.dumps(
                {"status": "REJECTED", "reason_code": "VALIDATION_INTERNAL_ERROR"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 3 if result.get("status") == "NOT_MATURE" else 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":  # pragma: no cover - console script owns the normal entrypoint
    main()
