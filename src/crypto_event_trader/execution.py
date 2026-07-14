from __future__ import annotations

from dataclasses import dataclass

from .binance import BinanceApiError, BinanceFuturesDemoClient
from .config import Settings
from .database import Repository
from .domain import MarketQuote, Signal


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reason: str
    quantity: float = 0.0
    notional: float = 0.0


class RiskManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self,
        signal: Signal,
        quote: MarketQuote,
        portfolio: dict,
        *,
        assumed_stop_distance: float = 0.03,
    ) -> RiskDecision:
        if signal.direction == 0:
            return RiskDecision(False, "neutral_signal")
        if signal.score < self.settings.min_signal_score:
            return RiskDecision(False, "score_below_threshold")
        if not portfolio["trading_enabled"]:
            return RiskDecision(False, "trading_disabled")
        if portfolio["drawdown"] >= self.settings.daily_drawdown_limit:
            return RiskDecision(False, "drawdown_limit")
        spread_bps = (quote.ask - quote.bid) / max(quote.last, 1e-12) * 10_000
        if spread_bps > 50:
            return RiskDecision(False, "spread_too_wide")
        if quote.volume_24h < 1_000_000:
            return RiskDecision(False, "insufficient_liquidity")

        open_symbols = {position["symbol"] for position in portfolio["positions"]}
        if (
            signal.symbol not in open_symbols
            and len(open_symbols) >= self.settings.max_open_positions
        ):
            return RiskDecision(False, "position_limit")

        equity = float(portfolio["equity"])
        risk_notional = equity * self.settings.risk_per_trade / assumed_stop_distance
        exposure_cap = equity * self.settings.max_asset_exposure
        existing = next(
            (
                position
                for position in portfolio["positions"]
                if position["symbol"] == signal.symbol
            ),
            None,
        )
        existing_exposure = abs(float(existing["quantity"]) * quote.last) if existing else 0.0
        available_exposure = max(0.0, exposure_cap - existing_exposure)
        notional = min(risk_notional, available_exposure)
        if notional < 10:
            return RiskDecision(False, "asset_exposure_limit")
        quantity = round(notional / quote.last, 8)
        return RiskDecision(True, "approved", quantity, quantity * quote.last)


class PaperExecutor:
    def __init__(self, repository: Repository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings
        self.risk = RiskManager(settings)

    def execute(self, signal_id: int, signal: Signal, quote: MarketQuote) -> dict:
        if self.repository.signal_has_order(signal_id):
            return {"status": "skipped", "reason": "signal_already_executed"}
        decision = self.risk.evaluate(signal, quote, self.repository.portfolio())
        if not decision.approved:
            return {"status": "rejected", "reason": decision.reason}

        side = "buy" if signal.direction > 0 else "sell"
        slip = self.settings.base_slippage_bps / 10_000
        intent_price = quote.ask if side == "buy" else quote.bid
        fill_price = intent_price * (1 + slip if side == "buy" else 1 - slip)
        fee = fill_price * decision.quantity * self.settings.taker_fee_bps / 10_000
        order_id, fill_id = self.repository.record_fill(
            signal_id=signal_id,
            venue="internal-paper",
            symbol=signal.symbol,
            side=side,
            intent_price=intent_price,
            fill_price=fill_price,
            quantity=decision.quantity,
            fee=fee,
            slippage_bps=self.settings.base_slippage_bps,
        )
        return {
            "status": "filled",
            "order_id": order_id,
            "fill_id": fill_id,
            "side": side,
            "quantity": decision.quantity,
            "fill_price": round(fill_price, 8),
            "fee": round(fee, 8),
        }


class BinanceFuturesDemoExecutor:
    def __init__(
        self,
        repository: Repository,
        settings: Settings,
        client: BinanceFuturesDemoClient,
        symbol_map: dict[str, str],
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.client = client
        self.symbol_map = symbol_map
        self.risk = RiskManager(settings)

    def execute(self, signal_id: int, signal: Signal, quote: MarketQuote) -> dict:
        if self.repository.signal_has_order(signal_id):
            return {"status": "skipped", "reason": "signal_already_executed"}
        decision = self.risk.evaluate(signal, quote, self.repository.portfolio())
        if not decision.approved:
            return {"status": "rejected", "reason": decision.reason}
        exchange_symbol = self.symbol_map.get(signal.symbol)
        if not exchange_symbol:
            return {"status": "rejected", "reason": "binance_symbol_not_mapped"}
        if not self.settings.binance_credentials_ready:
            return {"status": "rejected", "reason": "binance_credentials_missing"}

        side = "buy" if signal.direction > 0 else "sell"
        client_order_id = f"cet-{signal_id}"
        try:
            remote = self.client.place_market_order(
                symbol=exchange_symbol,
                side=side,
                quantity=decision.quantity,
                client_order_id=client_order_id,
            )
        except BinanceApiError as error:
            return {
                "status": "rejected",
                "reason": "binance_api_error",
                "detail": str(error),
                "code": error.code,
            }

        executed_quantity = float(remote.get("executedQty", 0))
        if str(remote.get("status", "")).upper() != "FILLED" or executed_quantity <= 0:
            return {
                "status": "submitted",
                "reason": "awaiting_binance_fill",
                "external_order_id": str(remote.get("orderId", "")),
                "external_client_order_id": client_order_id,
            }

        intent_price = quote.ask if side == "buy" else quote.bid
        fill_price = float(remote.get("avgPrice", 0))
        if fill_price <= 0 and float(remote.get("cumQuote", 0)) > 0:
            fill_price = float(remote["cumQuote"]) / executed_quantity
        if fill_price <= 0:
            fill_price = intent_price
        fee = fill_price * executed_quantity * self.settings.taker_fee_bps / 10_000
        slippage_bps = abs(fill_price / intent_price - 1) * 10_000
        order_id, fill_id = self.repository.record_fill(
            signal_id=signal_id,
            venue="binance-futures-demo",
            symbol=signal.symbol,
            side=side,
            intent_price=intent_price,
            fill_price=fill_price,
            quantity=executed_quantity,
            fee=fee,
            slippage_bps=slippage_bps,
            external_order_id=str(remote.get("orderId", "")),
            external_client_order_id=client_order_id,
            raw_response=remote,
        )
        return {
            "status": "filled",
            "order_id": order_id,
            "fill_id": fill_id,
            "external_order_id": str(remote.get("orderId", "")),
            "external_client_order_id": client_order_id,
            "side": side,
            "quantity": executed_quantity,
            "fill_price": round(fill_price, 8),
            "fee_estimate": round(fee, 8),
        }
