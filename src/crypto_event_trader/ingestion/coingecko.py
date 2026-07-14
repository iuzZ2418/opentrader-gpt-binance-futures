from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..domain import MarketQuote


class CoinGeckoClient:
    """Small read-only client used for market confirmation and paper fills."""

    endpoint = "https://api.coingecko.com/api/v3/simple/price"

    def __init__(self, api_key: str | None = None, timeout: float = 10) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def fetch_quotes(self, assets: dict[str, str]) -> dict[str, MarketQuote]:
        params = urlencode(
            {
                "ids": ",".join(assets.values()),
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_last_updated_at": "true",
            }
        )
        headers = {"Accept": "application/json", "User-Agent": "crypto-event-trader/0.1"}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        request = Request(f"{self.endpoint}?{params}", headers=headers)
        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))

        quotes: dict[str, MarketQuote] = {}
        for symbol, asset_id in assets.items():
            item = payload.get(asset_id)
            if not item or not item.get("usd"):
                continue
            price = float(item["usd"])
            timestamp = datetime.fromtimestamp(item.get("last_updated_at", 0), UTC)
            if timestamp.year == 1970:
                timestamp = datetime.now(UTC)
            # CoinGecko simple/price has no top-of-book; model a conservative 2 bps spread.
            quotes[symbol] = MarketQuote(
                symbol=symbol,
                bid=price * 0.9999,
                ask=price * 1.0001,
                last=price,
                volume_24h=float(item.get("usd_24h_vol", 0)),
                timestamp=timestamp,
            )
        return quotes
