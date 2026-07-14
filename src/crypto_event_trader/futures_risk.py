from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .config import Settings
from .contracts import PositionThesis, TradeAction, TradeCandidate, TradeDecision, TradeDirection
from .control import TradingControlSnapshot
from .domain import MarketQuote
from .futures_portfolio import FuturesAccountSnapshot


@dataclass(frozen=True, slots=True)
class ExecutionIntent:
    approved: bool
    reason: str
    action: TradeAction
    symbol: str
    side: str | None = None
    quantity: float = 0.0
    notional: float = 0.0
    reduce_only: bool = False
    protective_stop_price: float | None = None
    correlation_cluster: str | None = None


class FuturesHardRisk:
    """Deterministic final authority after a model decision."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self,
        *,
        decision: TradeDecision,
        candidate: TradeCandidate | None,
        quote: MarketQuote,
        account: FuturesAccountSnapshot,
        control: TradingControlSnapshot,
        thesis: PositionThesis | None = None,
        signal_strengthening: bool = False,
        existing_protective_stop: float | None = None,
        now: datetime | None = None,
    ) -> ExecutionIntent:
        reference = now or datetime.now(UTC)
        action = decision.action
        symbol = decision.symbol.upper()
        current = next(
            (item for item in account.positions if item["symbol"].upper() == symbol), None
        )

        increases_exposure = action in {TradeAction.OPEN, TradeAction.ADD}
        if increases_exposure and not symbol.endswith("USDT"):
            return self._reject(action, symbol, "only_usdt_margined_perpetuals_are_allowed")
        if increases_exposure:
            if quote.timestamp.tzinfo is None:
                return self._reject(action, symbol, "quote_timestamp_not_timezone_aware")
            age = (reference - quote.timestamp.astimezone(UTC)).total_seconds()
            if age < -2 or age > self.settings.market_data_max_age_seconds:
                return self._reject(action, symbol, "market_data_stale")
            if quote.last <= 0 or quote.bid <= 0 or quote.ask <= quote.bid:
                return self._reject(action, symbol, "invalid_market_quote")
            spread_bps = (quote.ask - quote.bid) / quote.last * 10_000
            if spread_bps > self.settings.max_spread_bps:
                return self._reject(action, symbol, "spread_too_wide")

        if increases_exposure:
            if account.timestamp.tzinfo is None:
                return self._reject(action, symbol, "account_timestamp_not_timezone_aware")
            account_age = (reference - account.timestamp.astimezone(UTC)).total_seconds()
            if account_age < -2 or account_age > self.settings.market_data_max_age_seconds:
                return self._reject(action, symbol, "account_state_stale")

        if action in {TradeAction.REJECT, TradeAction.HOLD}:
            return self._reject(action, symbol, "no_execution_required")
        if action in {TradeAction.REDUCE, TradeAction.CLOSE}:
            if not current or not float(current["quantity"]):
                return self._reject(action, symbol, "position_not_found")
            current_quantity = abs(float(current["quantity"]))
            quantity = (
                current_quantity
                if action is TradeAction.CLOSE
                else current_quantity * decision.position_multiplier
            )
            side = "SELL" if float(current["quantity"]) > 0 else "BUY"
            return ExecutionIntent(
                True,
                "risk_reducing_order",
                action,
                symbol,
                side,
                quantity,
                quantity * quote.last,
                True,
            )

        if not control.new_positions_enabled:
            return self._reject(action, symbol, "new_positions_disabled")
        if account.daily_pnl_fraction <= -self.settings.daily_drawdown_limit:
            return self._reject(action, symbol, "daily_loss_limit")
        if account.drawdown >= self.settings.total_drawdown_limit:
            return self._reject(action, symbol, "total_drawdown_limit")
        if candidate is None:
            return self._reject(action, symbol, "candidate_required")
        if not candidate.is_valid(reference):
            return self._reject(action, symbol, "candidate_expired")
        if decision.candidate_id != candidate.candidate_id:
            return self._reject(action, symbol, "candidate_id_mismatch")
        if candidate.symbol != symbol or decision.direction is not candidate.direction:
            return self._reject(action, symbol, "candidate_direction_mismatch")

        if action is TradeAction.OPEN:
            if current is not None and float(current["quantity"]):
                return self._reject(action, symbol, "existing_position_requires_add")
            if decision.confidence < self.settings.decision_open_confidence:
                return self._reject(action, symbol, "open_confidence_below_threshold")

        if action is TradeAction.ADD:
            if not current or thesis is None:
                return self._reject(action, symbol, "open_position_required_for_add")
            expected_sign = 1 if candidate.direction is TradeDirection.LONG else -1
            current_quantity = float(current["quantity"])
            if current_quantity * expected_sign <= 0:
                return self._reject(action, symbol, "add_remote_position_direction_mismatch")
            if thesis.direction is not candidate.direction:
                return self._reject(action, symbol, "add_thesis_direction_mismatch")
            if decision.confidence < self.settings.decision_add_confidence:
                return self._reject(action, symbol, "add_confidence_below_threshold")
            if thesis.add_count >= 1:
                return self._reject(action, symbol, "add_limit_reached")
            if thesis.pnl_r < 1:
                return self._reject(action, symbol, "add_requires_one_r_profit")
            if not signal_strengthening:
                return self._reject(action, symbol, "add_requires_strengthening_signal")
        elif current is None and len(account.positions) >= self.settings.max_open_positions:
            return self._reject(action, symbol, "maximum_open_positions_reached")

        atr = float(
            candidate.feature_snapshot.get("atr_1h")
            or candidate.feature_snapshot.get("atr_14_1h")
            or 0
        )
        stop_distance = atr * 2
        if stop_distance <= 0:
            return self._reject(action, symbol, "atr_stop_missing")
        risk_fraction = (
            self.settings.initial_position_risk
            if action is TradeAction.OPEN
            else self.settings.add_position_risk
        )
        risk_fraction = min(risk_fraction, candidate.max_risk_fraction)
        allocated_equity = account.equity * self.settings.capital_allocation_fraction
        quantity_by_risk = allocated_equity * risk_fraction / stop_distance
        requested = candidate.max_quantity * decision.position_multiplier
        quantity = min(requested, quantity_by_risk)

        direction_sign = 1 if candidate.direction is TradeDirection.LONG else -1
        proposed_stop = quote.last - direction_sign * stop_distance
        if proposed_stop <= 0:
            return self._reject(action, symbol, "invalid_protective_stop")
        stop_price = proposed_stop
        if action is TradeAction.ADD:
            if existing_protective_stop is None:
                return self._reject(action, symbol, "existing_protective_stop_missing")
            try:
                existing_stop = float(existing_protective_stop)
            except (TypeError, ValueError):
                return self._reject(action, symbol, "existing_protective_stop_invalid")
            if existing_stop <= 0:
                return self._reject(action, symbol, "existing_protective_stop_invalid")
            # An add may tighten protection but may never increase the old loss distance.
            stop_price = (
                max(existing_stop, proposed_stop)
                if direction_sign > 0
                else min(existing_stop, proposed_stop)
            )

            current_quantity = abs(float(current["quantity"])) if current else 0.0
            current_entry = float(current["entry_price"]) if current else 0.0
            slippage = self.settings.base_slippage_bps / 10_000
            estimated_add_entry = (
                quote.ask * (1 + slippage)
                if direction_sign > 0
                else quote.bid * (1 - slippage)
            )
            existing_loss_per_unit = max(
                0.0, direction_sign * (current_entry - stop_price)
            )
            add_loss_per_unit = max(
                0.0, direction_sign * (estimated_add_entry - stop_price)
            )
            total_risk_budget = account.equity * self.settings.risk_per_trade
            existing_risk = current_quantity * existing_loss_per_unit
            tolerance = max(1e-9, total_risk_budget * 1e-10)
            if existing_risk > total_risk_budget + tolerance:
                return self._reject(action, symbol, "existing_position_risk_above_one_percent")
            if add_loss_per_unit > 0:
                quantity = min(
                    quantity,
                    max(0.0, total_risk_budget - existing_risk) / add_loss_per_unit,
                )

        current_symbol_notional = abs(float(current["notional"])) if current else 0.0
        symbol_room = max(
            0.0,
            allocated_equity * self.settings.max_asset_exposure - current_symbol_notional,
        )
        gross_room = max(
            0.0,
            allocated_equity * self.settings.max_gross_exposure - account.gross_notional,
        )
        candidate_cluster = str(
            candidate.feature_snapshot.get("correlation_cluster")
            or _default_correlation_cluster(symbol)
        )
        cluster_notional = sum(
            abs(float(item["notional"]))
            for item in account.positions
            if str(
                item.get("correlation_cluster")
                or _default_correlation_cluster(str(item["symbol"]))
            )
            == candidate_cluster
        )
        cluster_room = max(
            0.0,
            allocated_equity * self.settings.max_correlation_cluster_exposure
            - cluster_notional,
        )
        net_limit = allocated_equity * self.settings.max_net_exposure
        if direction_sign > 0:
            net_room = max(0.0, net_limit - account.net_notional)
        else:
            net_room = max(0.0, net_limit + account.net_notional)
        quantity = min(
            quantity,
            symbol_room / quote.last,
            gross_room / quote.last,
            net_room / quote.last,
            cluster_room / quote.last,
        )
        if quantity <= 0 or quantity * quote.last < 5:
            return self._reject(action, symbol, "exposure_or_minimum_notional_limit")

        if action is TradeAction.ADD and current is not None:
            current_quantity = abs(float(current["quantity"]))
            current_entry = float(current["entry_price"])
            slippage = self.settings.base_slippage_bps / 10_000
            estimated_add_entry = (
                quote.ask * (1 + slippage)
                if direction_sign > 0
                else quote.bid * (1 - slippage)
            )
            post_quantity = current_quantity + quantity
            post_entry = (
                current_quantity * current_entry + quantity * estimated_add_entry
            ) / post_quantity
            total_stop_risk = post_quantity * max(
                0.0, direction_sign * (post_entry - stop_price)
            )
            if total_stop_risk > account.equity * self.settings.risk_per_trade + 1e-8:
                return self._reject(action, symbol, "combined_position_risk_above_one_percent")

        side = "BUY" if candidate.direction is TradeDirection.LONG else "SELL"
        return ExecutionIntent(
            True,
            "approved",
            action,
            symbol,
            side,
            quantity,
            quantity * quote.last,
            False,
            stop_price,
            candidate_cluster,
        )

    @staticmethod
    def _reject(action: TradeAction, symbol: str, reason: str) -> ExecutionIntent:
        return ExecutionIntent(False, reason, action, symbol)


def emergency_exit_reason(account: FuturesAccountSnapshot, settings: Settings) -> str | None:
    if account.drawdown >= settings.total_drawdown_limit:
        return "total_drawdown_limit"
    if account.daily_pnl_fraction <= -settings.daily_drawdown_limit:
        return "daily_loss_limit"
    return None


def _default_correlation_cluster(symbol: str) -> str:
    normalized = symbol.upper()
    base = normalized.removesuffix("USDT")
    if base in {"BTC", "ETH"}:
        return "majors"
    if base in {"SOL", "BNB", "ADA", "AVAX", "DOT", "NEAR", "SUI", "APT"}:
        return "layer1"
    if base in {"DOGE", "SHIB", "PEPE", "BONK", "FLOKI", "WIF"}:
        return "memes"
    return f"other:{base}"
