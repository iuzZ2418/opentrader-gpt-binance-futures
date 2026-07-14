from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from math import copysign


@dataclass(slots=True)
class FuturesPosition:
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    leverage: int = 1
    realized_pnl: float = 0.0
    funding_pnl: float = 0.0
    correlation_cluster: str | None = None

    @property
    def side(self) -> str:
        if self.quantity > 0:
            return "LONG"
        if self.quantity < 0:
            return "SHORT"
        return "FLAT"

    @property
    def notional(self) -> float:
        return self.quantity * self.mark_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.mark_price - self.entry_price)

    @property
    def isolated_margin(self) -> float:
        return abs(self.quantity * self.entry_price) / max(self.leverage, 1)


@dataclass(frozen=True, slots=True)
class FuturesAccountSnapshot:
    wallet_balance: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    funding_pnl: float
    gross_notional: float
    net_notional: float
    initial_margin: float
    maintenance_margin: float
    margin_ratio: float
    daily_pnl_fraction: float
    drawdown: float
    positions: tuple[dict, ...]
    timestamp: datetime


class FuturesPortfolio:
    """One-way, isolated-margin accounting model for paper and replay tests."""

    def __init__(
        self,
        initial_balance: float,
        *,
        default_leverage: int = 3,
        maintenance_margin_rate: float = 0.005,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("initial balance must be positive")
        self.initial_balance = float(initial_balance)
        self.wallet_balance = float(initial_balance)
        self.default_leverage = default_leverage
        self.maintenance_margin_rate = maintenance_margin_rate
        self.positions: dict[str, FuturesPosition] = {}
        self.high_water_equity = float(initial_balance)
        self.daily_start_equity = float(initial_balance)
        self.current_day = datetime.now(UTC).date()

    def position(self, symbol: str) -> FuturesPosition:
        return self.positions.setdefault(
            symbol,
            FuturesPosition(symbol=symbol, leverage=self.default_leverage),
        )

    def apply_fill(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
        leverage: int | None = None,
        correlation_cluster: str | None = None,
    ) -> FuturesPosition:
        if quantity <= 0 or price <= 0 or fee < 0:
            raise ValueError("fill quantity/price must be positive and fee non-negative")
        direction = 1.0 if side.upper() == "BUY" else -1.0 if side.upper() == "SELL" else 0.0
        if not direction:
            raise ValueError("side must be BUY or SELL")

        position = self.position(symbol)
        if leverage is not None:
            if leverage < 1:
                raise ValueError("leverage must be positive")
            position.leverage = leverage
        if correlation_cluster is not None:
            position.correlation_cluster = correlation_cluster
        signed_fill = direction * quantity
        previous_quantity = position.quantity

        if previous_quantity == 0 or previous_quantity * signed_fill > 0:
            new_quantity = previous_quantity + signed_fill
            previous_cost = abs(previous_quantity) * position.entry_price
            added_cost = quantity * price
            position.entry_price = (previous_cost + added_cost) / abs(new_quantity)
            position.quantity = new_quantity
        else:
            closed_quantity = min(abs(previous_quantity), quantity)
            closing_sign = copysign(1.0, previous_quantity)
            realized = closed_quantity * (price - position.entry_price) * closing_sign
            position.realized_pnl += realized
            self.wallet_balance += realized
            new_quantity = previous_quantity + signed_fill
            position.quantity = new_quantity
            if new_quantity == 0:
                position.entry_price = 0.0
            elif previous_quantity * new_quantity < 0:
                position.entry_price = price

        position.mark_price = price
        self.wallet_balance -= fee
        self._update_high_water()
        return position

    def mark(self, symbol: str, mark_price: float) -> None:
        if mark_price <= 0:
            raise ValueError("mark price must be positive")
        self.position(symbol).mark_price = mark_price
        self._update_high_water()

    def apply_funding(self, symbol: str, amount: float) -> None:
        """Apply a signed funding cash flow; positive means received."""

        position = self.position(symbol)
        position.funding_pnl += amount
        self.wallet_balance += amount
        self._update_high_water()

    def roll_day(self, new_day: date | None = None) -> None:
        target = new_day or datetime.now(UTC).date()
        if target != self.current_day:
            self.current_day = target
            self.daily_start_equity = self.snapshot().equity

    def snapshot(self, *, timestamp: datetime | None = None) -> FuturesAccountSnapshot:
        active = tuple(
            asdict(item)
            | {
                "side": item.side,
                "notional": item.notional,
                "unrealized_pnl": item.unrealized_pnl,
                "isolated_margin": item.isolated_margin,
            }
            for item in sorted(self.positions.values(), key=lambda value: value.symbol)
            if item.quantity
        )
        unrealized = sum(item["unrealized_pnl"] for item in active)
        equity = self.wallet_balance + unrealized
        gross = sum(abs(item["notional"]) for item in active)
        net = sum(item["notional"] for item in active)
        initial_margin = sum(item["isolated_margin"] for item in active)
        maintenance = gross * self.maintenance_margin_rate
        margin_ratio = maintenance / max(equity, 1e-12)
        realized = sum(item.realized_pnl for item in self.positions.values())
        funding = sum(item.funding_pnl for item in self.positions.values())
        daily_pnl = equity / max(self.daily_start_equity, 1e-12) - 1
        drawdown = max(0.0, 1 - equity / max(self.high_water_equity, 1e-12))
        return FuturesAccountSnapshot(
            wallet_balance=self.wallet_balance,
            equity=equity,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            funding_pnl=funding,
            gross_notional=gross,
            net_notional=net,
            initial_margin=initial_margin,
            maintenance_margin=maintenance,
            margin_ratio=margin_ratio,
            daily_pnl_fraction=daily_pnl,
            drawdown=drawdown,
            positions=active,
            timestamp=timestamp or datetime.now(UTC),
        )

    def _update_high_water(self) -> None:
        equity = self.wallet_balance + sum(
            position.unrealized_pnl for position in self.positions.values()
        )
        self.high_water_equity = max(self.high_water_equity, equity)
