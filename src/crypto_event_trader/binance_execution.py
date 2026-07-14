from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from typing import Any

from .audit import AuditRepository
from .binance import BinanceFuturesClient
from .futures_portfolio import FuturesAccountSnapshot


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class BinanceFuturesAccountSource:
    """Convert reconciled Binance REST state into the hard-risk account contract.

    Live startup does not silently reset daily-loss or high-water baselines. It remains closed to
    new positions until an account risk baseline exists in the audit database or an authenticated
    operator explicitly confirms a new baseline.
    """

    def __init__(
        self,
        client: BinanceFuturesClient,
        *,
        audit: AuditRepository,
        source: str,
        private_stream_ready: Callable[[], bool] | None = None,
        require_persisted_risk_state: bool | None = None,
    ) -> None:
        self.client = client
        self.audit = audit
        self.source = source
        self.private_stream_ready = private_stream_ready or (lambda: True)
        self.require_persisted_risk_state = (
            client.is_production
            if require_persisted_risk_state is None
            else require_persisted_risk_state
        )
        self._high_water: float | None = None
        self._day_start: float | None = None
        self._risk_day: date | None = None
        self._risk_context_ready = False
        self._restore_risk_state()

    @property
    def ready_for_new_orders(self) -> bool:
        if not self._risk_context_ready and self.require_persisted_risk_state:
            self._restore_risk_state()
        return self._risk_context_ready and bool(self.private_stream_ready())

    def snapshot(self, *, timestamp: datetime | None = None) -> FuturesAccountSnapshot:
        remote = self.client.rest_snapshot()
        account = remote.account
        wallet = _number(account.get("totalWalletBalance"))
        unrealized = _number(account.get("totalUnrealizedProfit"))
        equity = _number(account.get("totalMarginBalance"), wallet + unrealized)
        if equity <= 0:
            equity = wallet + unrealized
        observed = datetime.fromtimestamp(remote.observed_at_ms / 1_000, UTC)
        if not remote.observed_at_ms and timestamp is not None:
            observed = timestamp
        observed = observed.astimezone(UTC)
        self._ensure_risk_day(observed.date(), equity)

        positions: list[dict[str, Any]] = []
        gross = 0.0
        net = 0.0
        initial_margin = 0.0
        for item in remote.positions:
            quantity = float(item.quantity)
            if not quantity:
                continue
            mark = float(item.mark_price)
            entry = float(item.entry_price)
            notional = quantity * mark
            margin = abs(quantity * entry) / max(item.leverage, 1)
            gross += abs(notional)
            net += notional
            initial_margin += margin
            positions.append(
                {
                    "symbol": item.symbol,
                    "quantity": quantity,
                    "entry_price": entry,
                    "mark_price": mark,
                    "leverage": item.leverage,
                    "margin_type": item.margin_type,
                    "notional": notional,
                    "unrealized_pnl": float(item.unrealized_pnl),
                    "isolated_margin": margin,
                    "position_side": item.position_side,
                }
            )
        high_water = max(self._high_water or equity, equity)
        self._high_water = high_water
        day_start = self._day_start or equity
        maintenance = _number(account.get("totalMaintMargin"), gross * 0.005)
        accounting = self.audit.authoritative_accounting_totals(
            venue=self.source,
            quote_asset="USDT",
            as_of=observed,
        )
        return FuturesAccountSnapshot(
            wallet_balance=wallet,
            equity=equity,
            unrealized_pnl=unrealized,
            realized_pnl=accounting["realized_pnl"],
            funding_pnl=accounting["funding_pnl"],
            gross_notional=gross,
            net_notional=net,
            initial_margin=_number(account.get("totalInitialMargin"), initial_margin),
            maintenance_margin=maintenance,
            margin_ratio=maintenance / max(equity, 1e-12),
            daily_pnl_fraction=equity / max(day_start, 1e-12) - 1,
            drawdown=max(0.0, 1 - equity / max(high_water, 1e-12)),
            positions=tuple(positions),
            timestamp=observed,
        )

    def confirm_risk_baseline(self) -> FuturesAccountSnapshot:
        """Called only behind the authenticated control API during live bootstrap."""

        snapshot = self.snapshot()
        self._high_water = max(self._high_water or snapshot.equity, snapshot.equity)
        self._day_start = snapshot.equity
        self._risk_day = snapshot.timestamp.date()
        self._risk_context_ready = True
        self.audit.append_account_snapshot(
            equity=snapshot.equity,
            cash=snapshot.wallet_balance,
            gross_exposure=snapshot.gross_notional,
            net_exposure=snapshot.net_notional,
            daily_pnl=snapshot.daily_pnl_fraction,
            drawdown=snapshot.drawdown,
            positions=snapshot.positions,
            source=self.source,
            observed_at=snapshot.timestamp,
        )
        return snapshot

    def _restore_risk_state(self) -> None:
        getter = getattr(self.audit, "account_risk_state", None)
        state = getter(source=self.source) if callable(getter) else None
        if state:
            self._high_water = (
                _number(state.get("historical_high_water_equity") or state.get("high_water_equity"))
                or None
            )
            self._day_start = (
                _number(state.get("utc_day_start_equity") or state.get("day_start_equity")) or None
            )
            latest = state.get("latest")
            if isinstance(latest, Mapping):
                observed = latest.get("observed_at")
                if isinstance(observed, str):
                    self._risk_day = datetime.fromisoformat(observed.replace("Z", "+00:00")).date()
            self._risk_context_ready = bool(self._high_water and self._day_start)
        elif not self.require_persisted_risk_state:
            # Demo may bootstrap its risk frame from the first reconciled snapshot.
            self._risk_context_ready = True

    def _ensure_risk_day(self, current_day: date, equity: float) -> None:
        if self._risk_day is None:
            self._risk_day = current_day
        if self._high_water is None:
            self._high_water = equity
        if self._day_start is None:
            self._day_start = equity
        if self._risk_day != current_day:
            self._risk_day = current_day
            self._day_start = equity
            if self.require_persisted_risk_state:
                # A live UTC rollover baseline must be recorded by the reconciler/operator.
                self._risk_context_ready = False


# The safety-hardened gateway is isolated from the account snapshot adapter so it can evolve and
# be fault-replayed independently.  Re-export it here to preserve the package's public API.
from .binance_execution_gateway import (  # noqa: E402
    BinanceFuturesExecutionGateway as BinanceFuturesExecutionGateway,
)
from .binance_execution_gateway import ProtectiveOrderState as ProtectiveOrderState  # noqa: E402
