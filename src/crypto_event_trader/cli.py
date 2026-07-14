from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

from .audit import AuditRepository
from .binance import BinanceApiError, BinanceFuturesClient
from .binance_runtime import build_binance_approval_runtime
from .config import Settings
from .research import performance_summary
from .service import TradingService


def _service(settings: Settings | None = None) -> TradingService:
    return TradingService(settings or Settings.from_env())


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto event research and paper trader")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="initialize schema and seed the asset universe")
    subparsers.add_parser("init-audit", help="initialize the append-only audit schema")
    sample = subparsers.add_parser("run-sample", help="run the deterministic sample")
    sample.add_argument(
        "--offline",
        action="store_true",
        help="use the internal matcher instead of submitting Binance Demo orders",
    )
    subparsers.add_parser("status", help="show portfolio and pipeline counts")
    subparsers.add_parser(
        "binance-check", help="read-only check of configured Binance Futures connectivity"
    )
    subparsers.add_parser(
        "external-readiness",
        help="probe the explicitly configured Demo/Live account and exact OpenAI model",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.command == "run-sample" and args.offline:
        settings = replace(settings, execution_venue="internal")
    if args.command == "init-db":
        service = _service(settings)
        print(json.dumps({"status": "initialized", "database": str(service.repository.path)}))
    elif args.command == "init-audit":
        audit = AuditRepository(settings.audit_database_url)
        audit.initialize()
        audit.close()
        print(json.dumps({"status": "initialized", "audit": settings.audit_database_url}))
    elif args.command == "run-sample":
        if settings.execution_venue != "internal":
            raise SystemExit(
                "Legacy sample execution is internal-paper only; use crypto-trader-worker "
                "for the audited GPT Demo/Live path."
            )
        service = _service(settings)
        print(json.dumps(asdict(service.run_sample_cycle()), indent=2))
    elif args.command == "status":
        service = _service(replace(settings, execution_venue="internal"))
        repository = service.repository
        account = repository.account()
        output = {
            "portfolio": repository.portfolio(),
            "research": performance_summary(
                repository.list_signals(10_000),
                repository.list_orders(10_000),
                repository.equity_curve(100_000),
                account["initial_cash"],
            ),
        }
        print(json.dumps(output, indent=2))
    elif args.command == "binance-check":
        environment = (
            "production"
            if settings.execution_venue == "binance_futures_live"
            else "demo"
        )
        base_url = (
            settings.binance_futures_live_url
            if environment == "production"
            else settings.binance_futures_demo_url
        )
        client = BinanceFuturesClient(
            settings.binance_api_key,
            settings.binance_api_secret,
            base_url=base_url,
            environment=environment,
            allow_production_trading=False,
            max_leverage=settings.max_leverage,
            recv_window_ms=settings.binance_recv_window_ms,
        )
        try:
            offset = client.sync_time()
            output = {
                "status": "ok",
                "environment": f"USD-M Futures {environment}",
                "base_url": base_url,
                "server_time_offset_ms": offset,
                "credentials_configured": settings.binance_credentials_ready,
                "quotes": {
                    symbol: asdict(quote)
                    for symbol, quote in client.fetch_quotes(
                        {symbol: symbol for symbol in settings.futures_universe}
                    ).items()
                },
            }
            if settings.binance_credentials_ready:
                output["usdt_balance"] = next(
                    (
                        item
                        for item in client.account_balance()
                        if item.get("asset") == "USDT"
                    ),
                    None,
                )
            print(json.dumps(output, indent=2, default=str))
        except BinanceApiError as error:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "environment": f"USD-M Futures {environment}",
                        "base_url": base_url,
                        "detail": str(error),
                    },
                    indent=2,
                )
            )
            raise SystemExit(1) from None
        finally:
            client.close()
    elif args.command == "external-readiness":
        runtime = build_binance_approval_runtime(settings)
        try:
            print(json.dumps(runtime.startup_check(), indent=2, default=str))
        finally:
            runtime.close()


if __name__ == "__main__":
    main()
