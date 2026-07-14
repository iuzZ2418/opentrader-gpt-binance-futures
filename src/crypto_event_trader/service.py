from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .binance import BinanceFuturesDemoClient
from .config import Settings
from .database import Repository, seed_assets
from .domain import MarketQuote, RawDocument, Signal, SignalInputs
from .execution import BinanceFuturesDemoExecutor, PaperExecutor
from .extraction import BaselineEventExtractor, assets_from_rows
from .ingestion import CoinGeckoClient, load_documents
from .signals import SignalScorer


@dataclass(slots=True)
class CycleResult:
    documents_seen: int = 0
    documents_inserted: int = 0
    events_created: int = 0
    signals_created: int = 0
    orders_filled: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0


class TradingService:
    def __init__(
        self,
        settings: Settings,
        repository: Repository | None = None,
        binance_client: BinanceFuturesDemoClient | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository or Repository(settings.sqlite_path())
        self.repository.initialize(settings.initial_cash)
        seed_assets(self.repository, settings.asset_universe)
        self.scorer = SignalScorer()
        asset_rows = self.repository.list_assets()
        self.binance_client = binance_client or BinanceFuturesDemoClient(
            settings.binance_api_key,
            settings.binance_api_secret,
            base_url=settings.binance_futures_demo_url,
            recv_window_ms=settings.binance_recv_window_ms,
        )
        if settings.execution_venue == "binance_futures_demo":
            symbol_map = {
                item["symbol"]: item["exchange_symbols"].get("binance_futures", "")
                for item in asset_rows
            }
            self.executor = BinanceFuturesDemoExecutor(
                self.repository, settings, self.binance_client, symbol_map
            )
        elif settings.execution_venue == "internal":
            self.executor = PaperExecutor(self.repository, settings)
        else:
            raise ValueError(f"Unsupported EXECUTION_VENUE: {settings.execution_venue}")

    def ingest(
        self, documents: list[RawDocument], result: CycleResult | None = None
    ) -> CycleResult:
        result = result or CycleResult()
        for document in documents:
            result.documents_seen += 1
            _, inserted = self.repository.insert_document(document)
            result.documents_inserted += int(inserted)
        return result

    def process(
        self, quotes: dict[str, MarketQuote], result: CycleResult | None = None
    ) -> CycleResult:
        result = result or CycleResult()
        assets = assets_from_rows(self.repository.list_assets())
        extractor = BaselineEventExtractor(assets)
        for document in self.repository.unprocessed_documents():
            for event in extractor.extract(document):
                event_id = self.repository.insert_event(event)
                result.events_created += 1
                quote = quotes.get(event.symbol)
                market_confirmation = 0.80 if quote and quote.volume_24h >= 10_000_000 else 0.45
                illiquidity = 0.02 if quote and quote.volume_24h >= 10_000_000 else 0.35
                signal = self.scorer.score(
                    event_id,
                    event,
                    market_confirmation=market_confirmation,
                    illiquidity_penalty=illiquidity,
                )
                self.repository.insert_signal(signal)
                result.signals_created += 1
        self.execute_pending(quotes, result)
        return result

    def execute_pending(
        self, quotes: dict[str, MarketQuote], result: CycleResult | None = None
    ) -> CycleResult:
        result = result or CycleResult()
        for item in self.repository.pending_signals(self.settings.min_signal_score):
            quote = quotes.get(item["symbol"])
            if not quote:
                continue
            signal = Signal(
                event_id=item["event_id"],
                asset_id=item["asset_id"],
                symbol=item["symbol"],
                direction=item["direction"],
                score=item["score"],
                threshold_bucket=item["threshold_bucket"],
                inputs=SignalInputs(**item["reason"]),
            )
            outcome = self.executor.execute(item["id"], signal, quote)
            if outcome["status"] == "filled":
                result.orders_filled += 1
            elif outcome["status"] == "submitted":
                result.orders_submitted += 1
            elif outcome["status"] == "rejected":
                result.orders_rejected += 1
        return result

    def run_sample_cycle(self, sample_path: Path | str | None = None) -> CycleResult:
        path = Path(sample_path) if sample_path else Path("data/sample_documents.json")
        result = self.ingest(load_documents(path))
        return self.process(sample_quotes(), result)

    def fetch_live_quotes(self) -> dict[str, MarketQuote]:
        assets = {
            item["symbol"]: item["exchange_symbols"]["binance_futures"]
            for item in self.repository.list_assets()
            if item["exchange_symbols"].get("binance_futures")
        }
        return self.binance_client.fetch_quotes(assets)

    def fetch_coingecko_quotes(self) -> dict[str, MarketQuote]:
        assets = {
            item["symbol"]: item["coingecko_id"]
            for item in self.repository.list_assets()
            if item.get("coingecko_id")
        }
        return CoinGeckoClient(self.settings.coingecko_api_key).fetch_quotes(assets)


def sample_quotes() -> dict[str, MarketQuote]:
    return {
        "BTC": MarketQuote("BTC", 64_993.5, 65_006.5, 65_000, 25_000_000_000),
        "ETH": MarketQuote("ETH", 3_499.3, 3_500.7, 3_500, 12_000_000_000),
        "SOL": MarketQuote("SOL", 149.97, 150.03, 150, 2_500_000_000),
    }
