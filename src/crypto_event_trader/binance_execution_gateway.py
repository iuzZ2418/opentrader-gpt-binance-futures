from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, TypeVar

from .approval import (
    GatewayFill,
    GatewayOrderEvent,
    GatewaySubmission,
    GatewaySubmissionUnresolved,
)
from .binance import (
    BinanceApiError,
    BinanceFuturesClient,
    BinanceSafetyError,
    ReconciliationResult,
)
from .config import Settings
from .contracts import TradeAction
from .control import TradingControl
from .domain import MarketQuote
from .futures_risk import ExecutionIntent


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class _ObservedOrder:
    client_id: str
    response: Mapping[str, Any]
    executed_quantity: float
    average_price: float
    status: str


@dataclass(frozen=True, slots=True)
class ProtectiveOrderState:
    symbol: str
    client_algo_id: str
    side: str
    trigger_price: float
    quantity: float
    external_order_id: str | None = None


_STANDARD_TERMINAL = frozenset({"FILLED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"})
_ALGO_ACTIVE = frozenset({"NEW", "PENDING_NEW", "WORKING", "ACCEPTED"})
_ALGO_TERMINAL = frozenset(
    {"CANCELED", "CANCELLED", "EXPIRED", "REJECTED", "FINISHED", "TRIGGERED"}
)
T = TypeVar("T")


class BinanceGatewaySubmissionUnresolved(GatewaySubmissionUnresolved, BinanceSafetyError):
    """An auditable uncertain submission that is also a venue safety failure."""


class BinanceEntryCancellationUnresolved(BinanceSafetyError):
    """A kill-switch cancellation that did not reach a verified terminal barrier."""

    def __init__(self, message: str, order_events: Sequence[GatewayOrderEvent]) -> None:
        super().__init__(message)
        self.order_events = tuple(order_events)


class BinanceFuturesExecutionGateway:
    """Fail-closed USDⓈ-M execution with terminal child-order barriers.

    A replacement entry is allowed only after the preceding child is observed terminal following
    its cancellation.  Unknown state engages the shared kill switch, blocks further exposure,
    removes remote ordinary entry orders, preserves/verifies algo protection, attempts a
    reduce-only emergency exit where quantity is known, and carries every observation to audit.
    """

    venue: str

    def __init__(
        self,
        client: BinanceFuturesClient,
        *,
        settings: Settings,
        control: TradingControl,
        quote_provider: Callable[[str], MarketQuote] | None = None,
        stream_ready: Callable[[], bool] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.settings = settings
        self.control = control
        self.quote_provider = quote_provider or self._fetch_quote
        self.stream_ready = stream_ready or (lambda: True)
        self.sleeper = sleeper
        self.venue = "binance_futures_live" if client.is_production else "binance_futures_demo"
        self._configured_symbols: set[str] = set()
        self._position_mode_configured = False
        self._active_symbols: set[str] = set()
        self._active_protective: dict[str, ProtectiveOrderState] = {}
        self._execution_uncertain = False

    @property
    def execution_uncertain(self) -> bool:
        return self._execution_uncertain

    def submit(
        self,
        *,
        intent: ExecutionIntent,
        quote: MarketQuote,
        client_order_id: str,
    ) -> GatewaySubmission:
        self._authorize(intent)
        if intent.reduce_only:
            return self._submit_exit(intent, quote, client_order_id)
        if self._execution_uncertain:
            raise BinanceSafetyError("execution uncertainty requires reconciliation")
        self._assert_entry_state(intent, quote)
        self._configure_symbol(intent)
        self._active_symbols.add(intent.symbol)
        return self._submit_entry(intent, quote, client_order_id)

    def cancel_all(self) -> tuple[GatewayOrderEvent, ...]:
        """Cancel every ordinary entry and prove terminal state before returning.

        Protective algo orders are deliberately retained.  A cancel acknowledgement is never
        treated as a barrier: every discovered entry is queried afterwards and the account is
        enumerated again.  Any unknown state fails loudly and keeps the shared kill switch set.
        """

        events: list[GatewayOrderEvent] = []
        try:
            self._cancel_remote_entry_orders(
                set(self._active_symbols), events, require_terminal=True
            )
        except Exception:
            self._execution_uncertain = True
            self.control.engage_kill_switch("entry_cancellation_unresolved")
            raise
        return tuple(events)

    def register_protective_order(self, state: ProtectiveOrderState) -> None:
        self._active_symbols.add(state.symbol)
        self._active_protective[state.symbol] = state

    def protective_stop_price(self, symbol: str) -> float | None:
        state = self._active_protective.get(symbol.strip().upper())
        return state.trigger_price if state is not None else None

    def planned_order_events(
        self,
        *,
        intent: ExecutionIntent,
        quote: MarketQuote,
        client_order_id: str,
    ) -> tuple[GatewayOrderEvent, ...]:
        """Describe deterministic child IDs for audit before any network mutation."""

        del quote
        if intent.reduce_only:
            active = self._active_protective.get(intent.symbol)
            if active is None:
                return ()
            return (
                self._algo_event(
                    active,
                    event_type="CANCEL_PLANNED",
                    status="PLANNED",
                    response={"planned_by": client_order_id},
                ),
            )
        side = intent.side or "BUY"
        exit_side = "SELL" if side == "BUY" else "BUY"

        def planned(
            role: str,
            child_id: str,
            child_side: str,
            order_type: str,
            *,
            reduce_only: bool,
            trigger_price: float | None = None,
        ) -> GatewayOrderEvent:
            return GatewayOrderEvent(
                role=role,
                event_type="PLANNED",
                status="PLANNED",
                client_order_id=child_id,
                symbol=intent.symbol,
                side=child_side,
                order_type=order_type,
                quantity=intent.quantity,
                reduce_only=reduce_only,
                trigger_price=trigger_price,
                source_event_id=f"gateway-plan:{role}:{child_id}",
                raw_response={"parent_client_order_id": client_order_id},
            )

        events = [
            planned(
                "ENTRY_ATTEMPT_RESERVED",
                self._child_id(client_order_id, "r"),
                side,
                "LIMIT",
                reduce_only=False,
            ),
            planned(
                "PROTECTIVE_STOP",
                self._child_id(client_order_id, "s"),
                exit_side,
                "STOP_MARKET_ALGO",
                reduce_only=True,
                trigger_price=intent.protective_stop_price,
            ),
            planned(
                "EMERGENCY_EXIT",
                self._child_id(client_order_id, "x"),
                exit_side,
                "MARKET",
                reduce_only=True,
            ),
        ]
        active = self._active_protective.get(intent.symbol)
        if active is not None:
            events.append(
                self._algo_event(
                    active,
                    event_type="CANCEL_PLANNED",
                    status="PLANNED",
                    response={"planned_by": client_order_id},
                )
            )
        return tuple(events)

    def reconcile(
        self,
        *,
        expected_open_client_ids: tuple[str, ...] | None = None,
        expected_positions: Mapping[str, float | Decimal] | None = None,
    ) -> ReconciliationResult:
        result = self.client.reconcile(
            expected_open_client_ids=expected_open_client_ids,
            expected_positions=expected_positions,
        )
        if not result.consistent:
            self._execution_uncertain = True
            self.control.engage_kill_switch("binance_reconciliation_mismatch")
        return result

    def _authorize(self, intent: ExecutionIntent) -> None:
        if self.settings.execution_venue != self.venue:
            raise BinanceSafetyError(
                f"configured venue {self.settings.execution_venue!r} does not match {self.venue}"
            )
        if self.client.is_production and not self.settings.production_trading_unlocked:
            raise BinanceSafetyError("live static safety gates are not all enabled")
        control = self.control.snapshot()
        if (
            self.client.is_production
            and not intent.reduce_only
            and (not control.runtime_live_unlocked or not control.new_positions_enabled)
        ):
            raise BinanceSafetyError("live runtime unlock is required for new exposure")
        if not intent.reduce_only and not control.new_positions_enabled:
            raise BinanceSafetyError("shared trading control blocks new exposure")

    def _mutate(self, intent: ExecutionIntent, operation: Callable[[], T]) -> T:
        """Authorize and execute exactly one account mutation without leaking permission."""

        self._authorize(intent)
        authorize = getattr(self.client, "mutation_authorization", None)
        if callable(authorize):
            with authorize():
                return operation()
        # Protocol-compatible test/dummy clients still receive a strictly scoped permission.
        previous = self.client.allow_production_trading
        self.client.allow_production_trading = True
        try:
            return operation()
        finally:
            self.client.allow_production_trading = previous

    def _configure_symbol(self, intent: ExecutionIntent) -> None:
        if not self._position_mode_configured:
            self._ignore_already_configured(
                lambda: self._mutate(
                    intent,
                    lambda: self.client.set_position_mode(dual_side_position=False),
                )
            )
            self._position_mode_configured = True
        if intent.symbol not in self._configured_symbols:
            self._ignore_already_configured(
                lambda: self._mutate(
                    intent,
                    lambda: self.client.set_margin_type(
                        symbol=intent.symbol, margin_type="ISOLATED"
                    ),
                )
            )
            self._mutate(
                intent,
                lambda: self.client.set_leverage(
                    symbol=intent.symbol, leverage=self.settings.max_leverage
                ),
            )
            self._configured_symbols.add(intent.symbol)

    @staticmethod
    def _ignore_already_configured(operation: Callable[[], Any]) -> None:
        try:
            operation()
        except BinanceApiError as error:
            if error.code not in {-4046, -4059}:
                raise

    def _assert_entry_state(self, intent: ExecutionIntent, quote: MarketQuote) -> None:
        self._authorize(intent)
        if self._execution_uncertain:
            raise BinanceSafetyError("execution uncertainty blocks entry child")
        if not self.stream_ready():
            raise BinanceSafetyError("market/private stream state is stale or incomplete")
        age = (datetime.now(UTC) - quote.timestamp.astimezone(UTC)).total_seconds()
        if age < -5 or age > self.settings.market_data_max_age_seconds:
            raise BinanceSafetyError(f"entry quote is stale or future-dated: age={age:.3f}s")

    def _submit_entry(
        self, intent: ExecutionIntent, quote: MarketQuote, client_order_id: str
    ) -> GatewaySubmission:
        attempts: list[_ObservedOrder] = []
        events: list[GatewayOrderEvent] = []
        remaining = intent.quantity
        working_quote = quote
        for attempt_index in range(2):
            if remaining <= 1e-12:
                break
            try:
                self._assert_entry_state(intent, working_quote)
            except Exception:
                if attempts:
                    break  # Preserve the terminal partial fill; protection is installed below.
                raise
            child_id = (
                client_order_id if attempt_index == 0 else self._child_id(client_order_id, "r")
            )
            side = intent.side or "BUY"
            limit_price = working_quote.ask if side == "BUY" else working_quote.bid
            submitted = self._place_entry_child(
                intent, child_id, side, remaining, limit_price, events
            )
            if submitted.status in _STANDARD_TERMINAL:
                observed = submitted
            else:
                self.sleeper(self.settings.entry_order_wait_seconds)
                observed = self._query_standard(
                    symbol=intent.symbol,
                    child_id=child_id,
                    fallback=submitted,
                    fallback_price=limit_price,
                    role="ENTRY_ATTEMPT",
                    event_type="OBSERVED",
                    side=side,
                    order_type="LIMIT",
                    quantity=remaining,
                    reduce_only=False,
                    events=events,
                )
            if observed.status not in _STANDARD_TERMINAL:
                resolved = self._cancel_and_resolve_entry(
                    intent=intent,
                    observed=observed,
                    requested_quantity=remaining,
                    fallback_price=limit_price,
                    events=events,
                )
                if resolved is None:
                    self._raise_unresolved(
                        "entry child did not reach a terminal state after cancellation",
                        intent=intent,
                        parent_id=client_order_id,
                        events=events,
                        attempts=(*attempts, observed),
                        emergency_quantity=(
                            self._current_position_quantity(intent.symbol)
                            or observed.executed_quantity
                        ),
                        fallback_price=quote.last,
                    )
                observed = resolved
            attempts.append(observed)
            remaining = max(0.0, remaining - observed.executed_quantity)
            if observed.status == "FILLED" or remaining <= 1e-12:
                break
            if attempt_index == 0:
                refreshed = self.quote_provider(intent.symbol)
                try:
                    self._assert_entry_state(intent, refreshed)
                except Exception:
                    if any(item.executed_quantity > 0 for item in attempts):
                        break
                    raise
                move_bps = abs(refreshed.last - quote.last) / quote.last * 10_000
                if move_bps > self.settings.entry_price_protection_bps:
                    break
                working_quote = refreshed

        fills = self._fills(attempts, side=intent.side or "BUY", role="ENTRY_ATTEMPT")
        total_filled = sum(fill.quantity for fill in fills)
        protective: ProtectiveOrderState | None = None
        if total_filled > 0:
            protective = self._install_complete_protection(
                intent=intent,
                parent_id=client_order_id,
                quote=quote,
                total_filled=total_filled,
                attempts=attempts,
                events=events,
            )
        status = (
            "FILLED"
            if total_filled + 1e-12 >= intent.quantity
            else "PARTIALLY_FILLED"
            if total_filled > 0
            else "CANCELED"
        )
        raw = {
            "attempts": [dict(item.response) for item in attempts],
            "protective_order": (
                {
                    "clientAlgoId": protective.client_algo_id,
                    "algoId": protective.external_order_id,
                    "quantity": protective.quantity,
                    "triggerPrice": protective.trigger_price,
                }
                if protective
                else {}
            ),
        }
        return GatewaySubmission(
            status=status,
            client_order_id=client_order_id,
            external_order_id=(str(attempts[-1].response.get("orderId", "")) if attempts else None)
            or None,
            fills=tuple(fills),
            protective_order_id=(
                protective.external_order_id or protective.client_algo_id if protective else None
            ),
            order_events=tuple(events),
            raw_response=raw,
        )

    def _place_entry_child(
        self,
        intent: ExecutionIntent,
        child_id: str,
        side: str,
        quantity: float,
        price: float,
        events: list[GatewayOrderEvent],
    ) -> _ObservedOrder:
        try:
            response = self._mutate(
                intent,
                lambda: self.client.place_limit_order(
                    symbol=intent.symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    client_order_id=child_id,
                ),
            )
            observed = self._observed(child_id, response, fallback_price=price)
        except Exception as error:
            observed = _ObservedOrder(child_id, {"error": str(error)}, 0.0, price, "UNKNOWN")
            resolved = self._cancel_and_resolve_entry(
                intent=intent,
                observed=observed,
                requested_quantity=quantity,
                fallback_price=price,
                events=events,
            )
            if resolved is None:
                self._raise_unresolved(
                    "entry placement status is unknown",
                    intent=intent,
                    parent_id=child_id,
                    events=events,
                    attempts=(observed,),
                    emergency_quantity=self._current_position_quantity(intent.symbol) or 0,
                    fallback_price=price,
                )
            observed = resolved
        events.append(
            self._standard_event(
                observed,
                role="ENTRY_ATTEMPT",
                event_type="SUBMITTED",
                symbol=intent.symbol,
                side=side,
                order_type="LIMIT",
                quantity=quantity,
                reduce_only=False,
            )
        )
        return observed

    def _cancel_and_resolve_entry(
        self,
        *,
        intent: ExecutionIntent,
        observed: _ObservedOrder,
        requested_quantity: float,
        fallback_price: float,
        events: list[GatewayOrderEvent],
    ) -> _ObservedOrder | None:
        try:
            response = self._mutate(
                self._risk_reducing_intent(intent.symbol),
                lambda: self.client.cancel_order(
                    symbol=intent.symbol, client_order_id=observed.client_id
                ),
            )
            cancel_status = str(response.get("status", "CANCEL_REQUESTED")).upper()
            cancel_observed = self._observed(
                observed.client_id, response, fallback_price=fallback_price
            )
            events.append(
                self._standard_event(
                    cancel_observed,
                    role="ENTRY_ATTEMPT",
                    event_type="CANCEL_REQUESTED",
                    symbol=intent.symbol,
                    side=intent.side or "BUY",
                    order_type="LIMIT",
                    quantity=requested_quantity,
                    reduce_only=False,
                    status_override=cancel_status,
                )
            )
        except Exception as error:
            events.append(
                self._standard_event(
                    observed,
                    role="ENTRY_ATTEMPT",
                    event_type="CANCEL_UNRESOLVED",
                    symbol=intent.symbol,
                    side=intent.side or "BUY",
                    order_type="LIMIT",
                    quantity=requested_quantity,
                    reduce_only=False,
                    status_override="UNKNOWN",
                    raw_override={"error": str(error)},
                )
            )
        # The cancellation response is never the barrier.  A fresh order query must expose the
        # final cumulative executedQty, including fills that raced the cancel request.
        final = self._query_standard(
            symbol=intent.symbol,
            child_id=observed.client_id,
            fallback=observed,
            fallback_price=fallback_price,
            role="ENTRY_ATTEMPT",
            event_type="POST_CANCEL_REQUERY",
            side=intent.side or "BUY",
            order_type="LIMIT",
            quantity=requested_quantity,
            reduce_only=False,
            events=events,
        )
        return final if final.status in _STANDARD_TERMINAL else None

    def _submit_exit(
        self, intent: ExecutionIntent, quote: MarketQuote, client_order_id: str
    ) -> GatewaySubmission:
        events: list[GatewayOrderEvent] = []
        response = self._mutate(
            intent,
            lambda: self.client.place_reduce_only_market_order(
                symbol=intent.symbol,
                side=intent.side or "BUY",
                quantity=intent.quantity,
                client_order_id=client_order_id,
            ),
        )
        observed = self._observed(client_order_id, response, fallback_price=quote.last)
        events.append(
            self._standard_event(
                observed,
                role="EXIT",
                event_type="SUBMITTED",
                symbol=intent.symbol,
                side=intent.side or "BUY",
                order_type="MARKET",
                quantity=intent.quantity,
                reduce_only=True,
            )
        )
        if observed.status not in _STANDARD_TERMINAL:
            observed = self._query_standard(
                symbol=intent.symbol,
                child_id=client_order_id,
                fallback=observed,
                fallback_price=quote.last,
                role="EXIT",
                event_type="REST_REQUERY",
                side=intent.side or "BUY",
                order_type="MARKET",
                quantity=intent.quantity,
                reduce_only=True,
                events=events,
            )
        if observed.status not in _STANDARD_TERMINAL:
            self._raise_unresolved(
                "reduce-only exit status is unknown",
                intent=intent,
                parent_id=client_order_id,
                events=events,
                attempts=(observed,),
                emergency_quantity=0,
                fallback_price=quote.last,
            )
        fills = self._fills((observed,), side=intent.side or "BUY", role="EXIT")
        remaining = self._current_position_quantity(intent.symbol)
        active = self._active_protective.get(intent.symbol)
        if remaining is None:
            if (
                intent.action is TradeAction.CLOSE
                and observed.executed_quantity + 1e-12 >= intent.quantity
            ):
                remaining = 0.0
            elif active is not None:
                remaining = max(0.0, active.quantity - observed.executed_quantity)
        if remaining is None:
            self._raise_unresolved(
                "cannot determine residual position after reduce-only exit",
                intent=intent,
                parent_id=client_order_id,
                events=events,
                attempts=(observed,),
                emergency_quantity=0,
                fallback_price=quote.last,
            )
        if active is not None:
            if not self._cancel_protective(active, events):
                self._raise_unresolved(
                    "protective stop cancellation is unresolved after exit",
                    intent=intent,
                    parent_id=client_order_id,
                    events=events,
                    attempts=(observed,),
                    emergency_quantity=remaining,
                    fallback_price=quote.last,
                )
            self._active_protective.pop(intent.symbol, None)
        replacement: ProtectiveOrderState | None = None
        if remaining > 1e-12:
            if active is None:
                self._raise_unresolved(
                    "residual position has no auditable protective stop",
                    intent=intent,
                    parent_id=client_order_id,
                    events=events,
                    attempts=(observed,),
                    emergency_quantity=remaining,
                    fallback_price=quote.last,
                )
            replacement = self._place_protective(
                symbol=intent.symbol,
                side=active.side,
                trigger_price=active.trigger_price,
                quantity=remaining,
                client_algo_id=self._replacement_algo_id(active.client_algo_id, remaining),
                events=events,
            )
            if replacement is None:
                self._raise_unresolved(
                    "resized protective stop could not be verified",
                    intent=intent,
                    parent_id=client_order_id,
                    events=events,
                    attempts=(observed,),
                    emergency_quantity=remaining,
                    fallback_price=quote.last,
                )
        return GatewaySubmission(
            status=observed.status,
            client_order_id=client_order_id,
            external_order_id=str(observed.response.get("orderId", "")) or None,
            fills=tuple(fills),
            protective_order_id=(
                replacement.external_order_id or replacement.client_algo_id if replacement else None
            ),
            order_events=tuple(events),
            raw_response=dict(response),
        )

    def _install_complete_protection(
        self,
        *,
        intent: ExecutionIntent,
        parent_id: str,
        quote: MarketQuote,
        total_filled: float,
        attempts: Sequence[_ObservedOrder],
        events: list[GatewayOrderEvent],
    ) -> ProtectiveOrderState:
        active = self._active_protective.get(intent.symbol)
        position_quantity = self._current_position_quantity(intent.symbol)
        if position_quantity is None:
            if intent.action is TradeAction.OPEN:
                position_quantity = total_filled
            elif active is not None:
                position_quantity = active.quantity + total_filled
        if position_quantity is None or position_quantity <= 1e-12:
            self._raise_unresolved(
                "cannot determine complete position size for protective stop",
                intent=intent,
                parent_id=parent_id,
                events=events,
                attempts=attempts,
                emergency_quantity=total_filled,
                fallback_price=quote.last,
            )
        if active is not None:
            if not self._cancel_protective(active, events):
                self._raise_unresolved(
                    "existing protective stop cancellation is unresolved",
                    intent=intent,
                    parent_id=parent_id,
                    events=events,
                    attempts=attempts,
                    emergency_quantity=position_quantity,
                    fallback_price=quote.last,
                )
            self._active_protective.pop(intent.symbol, None)
        protective = self._place_protective(
            symbol=intent.symbol,
            side="SELL" if intent.side == "BUY" else "BUY",
            trigger_price=intent.protective_stop_price or quote.last,
            quantity=position_quantity,
            client_algo_id=self._child_id(parent_id, "s"),
            events=events,
        )
        if protective is None:
            self._raise_unresolved(
                "protective stop failed; emergency reduce-only close submitted",
                intent=intent,
                parent_id=parent_id,
                events=events,
                attempts=attempts,
                emergency_quantity=position_quantity,
                fallback_price=quote.last,
            )
        return protective

    def _place_protective(
        self,
        *,
        symbol: str,
        side: str,
        trigger_price: float,
        quantity: float,
        client_algo_id: str,
        events: list[GatewayOrderEvent],
    ) -> ProtectiveOrderState | None:
        normalized_trigger = trigger_price
        normalize_price = getattr(self.client, "normalize_price", None)
        if callable(normalize_price):
            normalized_trigger = float(normalize_price(symbol, trigger_price))
        normalized_quantity = quantity
        normalize_quantity = getattr(self.client, "normalize_quantity", None)
        if callable(normalize_quantity):
            normalized_quantity = float(
                normalize_quantity(symbol, quantity, market=True)
            )
        expected = ProtectiveOrderState(
            symbol=symbol,
            client_algo_id=client_algo_id,
            side=side,
            trigger_price=normalized_trigger,
            quantity=normalized_quantity,
        )
        response: Mapping[str, Any]
        try:
            response = self._mutate(
                self._risk_reducing_intent(symbol),
                lambda: self.client.place_stop_market_algo_order(
                    symbol=symbol,
                    side=side,
                    trigger_price=trigger_price,
                    quantity=quantity,
                    client_algo_id=client_algo_id,
                    price_protect=True,
                ),
            )
        except Exception as error:
            events.append(
                self._algo_event(
                    expected,
                    event_type="SUBMISSION_UNRESOLVED",
                    status="UNKNOWN",
                    response={"error": str(error)},
                )
            )
        else:
            events.append(
                self._algo_event(
                    expected,
                    event_type="SUBMITTED",
                    status=self._algo_status(response),
                    response=response,
                )
            )
        # ACK and unknown mutation responses are both non-authoritative. Query exactly once by
        # deterministic client ID; never place a replacement until this state is resolved.
        try:
            verified = self.client.query_algo_order(client_algo_id=client_algo_id)
        except Exception as error:
            events.append(
                self._algo_event(
                    expected,
                    event_type="REST_VERIFICATION_FAILED",
                    status="UNKNOWN",
                    response={"error": str(error)},
                )
            )
            return None
        try:
            state = self._verified_protective_state(
                verified,
                expected_symbol=symbol,
                expected_client_id=client_algo_id,
                expected_side=side,
                expected_quantity=normalized_quantity,
                expected_trigger=normalized_trigger,
            )
        except BinanceSafetyError as error:
            events.append(
                self._algo_event(
                    expected,
                    event_type="REST_VERIFICATION_REJECTED",
                    status="INVALID",
                    response={**dict(verified), "verification_error": str(error)},
                )
            )
            return None
        events.append(
            self._algo_event(
                state,
                event_type="REST_VERIFIED",
                status=self._algo_status(verified),
                response=verified,
            )
        )
        self.register_protective_order(state)
        return state

    @staticmethod
    def _verified_protective_state(
        response: Mapping[str, Any],
        *,
        expected_symbol: str,
        expected_client_id: str,
        expected_side: str,
        expected_quantity: float,
        expected_trigger: float,
    ) -> ProtectiveOrderState:
        client_id = str(
            response.get("clientAlgoId") or response.get("clientOrderId") or ""
        )
        symbol = str(response.get("symbol") or "").upper()
        side = str(response.get("side") or "").upper()
        quantity = _number(response.get("quantity") or response.get("origQty"))
        trigger = _number(response.get("triggerPrice") or response.get("stopPrice"))
        algo_type = str(response.get("algoType") or "").upper()
        order_type = str(response.get("orderType") or response.get("type") or "").upper()
        working_type = str(response.get("workingType") or "").upper()
        position_side = str(response.get("positionSide") or "BOTH").upper()
        status = BinanceFuturesExecutionGateway._algo_status(response)
        errors: list[str] = []
        if client_id != expected_client_id:
            errors.append("client_id")
        if symbol != expected_symbol.upper():
            errors.append("symbol")
        if side != expected_side.upper():
            errors.append("side")
        tolerance = max(1e-12, abs(expected_quantity) * 1e-9)
        if quantity <= 0 or abs(quantity - expected_quantity) > tolerance:
            errors.append("quantity")
        trigger_tolerance = max(1e-12, abs(expected_trigger) * 1e-9)
        if trigger <= 0 or abs(trigger - expected_trigger) > trigger_tolerance:
            errors.append("trigger_price")
        if algo_type != "CONDITIONAL":
            errors.append("algo_type")
        if order_type != "STOP_MARKET":
            errors.append("order_type")
        if working_type != "MARK_PRICE":
            errors.append("working_type")
        if position_side != "BOTH":
            errors.append("position_side")
        if not _flag(response.get("reduceOnly")):
            errors.append("reduce_only")
        if _flag(response.get("closePosition")):
            errors.append("close_position")
        if not _flag(response.get("priceProtect")):
            errors.append("price_protect")
        if status not in _ALGO_ACTIVE:
            errors.append("status")
        if errors:
            raise BinanceSafetyError(
                "protective algo failed remote semantic verification: "
                + ",".join(errors)
            )
        return ProtectiveOrderState(
            symbol=symbol,
            client_algo_id=client_id,
            side=side,
            trigger_price=trigger,
            quantity=quantity,
            external_order_id=str(response.get("algoId", "")) or None,
        )

    def _cancel_protective(
        self, state: ProtectiveOrderState, events: list[GatewayOrderEvent]
    ) -> bool:
        try:
            response = self._mutate(
                self._risk_reducing_intent(state.symbol),
                lambda: self.client.cancel_algo_order(client_algo_id=state.client_algo_id),
            )
            events.append(
                self._algo_event(
                    state,
                    event_type="CANCEL_REQUESTED",
                    status=self._algo_status(response),
                    response=response,
                )
            )
        except Exception as error:
            events.append(
                self._algo_event(
                    state,
                    event_type="CANCEL_UNRESOLVED",
                    status="UNKNOWN",
                    response={"error": str(error)},
                )
            )
        try:
            response = self.client.query_algo_order(client_algo_id=state.client_algo_id)
        except Exception as error:
            events.append(
                self._algo_event(
                    state,
                    event_type="POST_CANCEL_REQUERY_FAILED",
                    status="UNKNOWN",
                    response={"error": str(error)},
                )
            )
            return False
        status = self._algo_status(response)
        events.append(
            self._algo_event(
                state,
                event_type="POST_CANCEL_REQUERY",
                status=status,
                response=response,
            )
        )
        return status in _ALGO_TERMINAL

    def reconcile_protective_orders(
        self,
        *,
        actual_positions: Mapping[str, float | Decimal],
        known_orders: Sequence[ProtectiveOrderState],
    ) -> tuple[GatewayOrderEvent, ...]:
        """Cancel stale algos and restore exactly-sized protection after restart."""

        events: list[GatewayOrderEvent] = []
        audited_ids = {state.client_algo_id for state in known_orders}
        try:
            remote_algos = self.client.open_algo_orders()
        except Exception as error:
            self._raise_global_algo_reconciliation_unavailable(
                actual_positions=actual_positions,
                events=events,
                error=error,
            )
        unknown_bot_algos: list[tuple[ProtectiveOrderState, Mapping[str, Any]]] = []
        for response in remote_algos:
            client_id = str(response.get("clientAlgoId") or response.get("clientOrderId") or "")
            # Never adopt a remote conditional order solely because it uses our namespace. A
            # leaked or collided client ID is not proof of an immutable audit owner.
            if not client_id.startswith("gpt-") or client_id in audited_ids:
                continue
            symbol = str(response.get("symbol", ""))
            trigger = _number(response.get("triggerPrice"))
            quantity = _number(response.get("quantity") or response.get("origQty"))
            side = str(response.get("side", "")).upper()
            fallback_symbol = symbol or next(iter(actual_positions), "UNKNOWN")
            signed = float(actual_positions.get(fallback_symbol, 0))
            unknown_bot_algos.append(
                (
                    ProtectiveOrderState(
                        symbol=fallback_symbol,
                        client_algo_id=client_id,
                        side=(
                            side
                            if side in {"BUY", "SELL"}
                            else ("SELL" if signed >= 0 else "BUY")
                        ),
                        trigger_price=max(trigger, 0.0),
                        quantity=max(quantity, abs(signed)),
                        external_order_id=str(response.get("algoId", "")) or None,
                    ),
                    response,
                )
            )
        if unknown_bot_algos:
            unresolved_ids: list[str] = []
            for state, _response in unknown_bot_algos:
                if not self._cancel_protective(state, events):
                    unresolved_ids.append(state.client_algo_id)
            detail = ",".join(state.client_algo_id for state, _ in unknown_bot_algos)
            if unresolved_ids:
                detail += "; cancellation_unresolved=" + ",".join(unresolved_ids)
            self._raise_global_algo_reconciliation_unavailable(
                actual_positions=actual_positions,
                events=events,
                error=BinanceSafetyError(
                    "unaudited remote bot algo order(s) cannot be adopted: " + detail
                ),
            )
        by_symbol: dict[str, list[ProtectiveOrderState]] = {}
        for state in known_orders:
            by_symbol.setdefault(state.symbol, []).append(state)
        symbols = set(by_symbol) | {
            symbol for symbol, quantity in actual_positions.items() if float(quantity)
        }
        for symbol in sorted(symbols):
            signed = float(actual_positions.get(symbol, 0))
            desired_quantity = abs(signed)
            desired_side = "SELL" if signed > 0 else "BUY"
            records = by_symbol.get(symbol, [])
            active: list[ProtectiveOrderState] = []
            for state in records:
                try:
                    response = self.client.query_algo_order(client_algo_id=state.client_algo_id)
                    status = self._algo_status(response)
                    events.append(
                        self._algo_event(
                            state,
                            event_type="REST_RECONCILIATION",
                            status=status,
                            response=response,
                        )
                    )
                    remote_state = self._verified_protective_state(
                        response,
                        expected_symbol=state.symbol,
                        expected_client_id=state.client_algo_id,
                        expected_side=state.side,
                        expected_quantity=state.quantity,
                        expected_trigger=state.trigger_price,
                    )
                except Exception as error:
                    events.append(
                        self._algo_event(
                            state,
                            event_type="REST_RECONCILIATION_FAILED",
                            status="UNKNOWN",
                            response={"error": str(error)},
                        )
                    )
                    self._raise_unresolved(
                        "protective algo query failed during reconciliation",
                        intent=self._signed_exit_intent(symbol, signed),
                        parent_id=state.client_algo_id,
                        events=events,
                        attempts=(),
                        emergency_quantity=desired_quantity,
                        fallback_price=1.0,
                    )
                if status in _ALGO_ACTIVE:
                    active.append(remote_state)
            keeper = next(
                (
                    state
                    for state in reversed(active)
                    if desired_quantity > 0
                    and state.side == desired_side
                    and abs(state.quantity - desired_quantity) <= 1e-12
                ),
                None,
            )
            for state in active:
                if state is keeper:
                    continue
                if not self._cancel_protective(state, events):
                    self._raise_unresolved(
                        "stale protective algo cancellation is unresolved",
                        intent=self._signed_exit_intent(symbol, signed),
                        parent_id=state.client_algo_id,
                        events=events,
                        attempts=(),
                        emergency_quantity=desired_quantity,
                        fallback_price=1.0,
                    )
            if desired_quantity <= 1e-12:
                self._active_protective.pop(symbol, None)
                continue
            if keeper is not None:
                self.register_protective_order(keeper)
                continue
            seed = records[-1] if records else None
            if seed is None or seed.trigger_price <= 0:
                self._raise_unresolved(
                    "open position has no auditable protective stop",
                    intent=self._signed_exit_intent(symbol, signed),
                    parent_id=f"reconcile-{symbol.lower()}",
                    events=events,
                    attempts=(),
                    emergency_quantity=desired_quantity,
                    fallback_price=1.0,
                )
            replacement = self._place_protective(
                symbol=symbol,
                side=desired_side,
                trigger_price=seed.trigger_price,
                quantity=desired_quantity,
                client_algo_id=self._replacement_algo_id(seed.client_algo_id, desired_quantity),
                events=events,
            )
            if replacement is None:
                self._raise_unresolved(
                    "protective algo replacement failed during reconciliation",
                    intent=self._signed_exit_intent(symbol, signed),
                    parent_id=seed.client_algo_id,
                    events=events,
                    attempts=(),
                    emergency_quantity=desired_quantity,
                    fallback_price=1.0,
                )
        return tuple(events)

    def _raise_global_algo_reconciliation_unavailable(
        self,
        *,
        actual_positions: Mapping[str, float | Decimal],
        events: list[GatewayOrderEvent],
        error: Exception,
    ) -> None:
        self._execution_uncertain = True
        self.control.engage_kill_switch("open_algo_order_reconciliation_unavailable")
        self._cancel_remote_entry_orders(set(actual_positions), events)
        fills: list[GatewayFill] = []
        for symbol, raw_quantity in actual_positions.items():
            signed = float(raw_quantity)
            if not signed:
                continue
            emergency = self._emergency_close(
                symbol=symbol,
                side="SELL" if signed > 0 else "BUY",
                quantity=abs(signed),
                parent_id=f"reconcile-{symbol.lower()}",
                fallback_price=1.0,
                events=events,
            )
            if emergency is not None and emergency.executed_quantity > 0:
                fills.extend(
                    self._fills(
                        (emergency,),
                        side="SELL" if signed > 0 else "BUY",
                        role="EMERGENCY_EXIT",
                    )
                )
        submission = GatewaySubmission(
            status="SUBMISSION_UNRESOLVED",
            client_order_id="algo-reconciliation",
            fills=tuple(fills),
            order_events=tuple(events),
            raw_response={"error": str(error), "requires_reconciliation": True},
        )
        raise BinanceGatewaySubmissionUnresolved(
            "open algo order reconciliation failed", submission
        ) from error

    def _raise_unresolved(
        self,
        message: str,
        *,
        intent: ExecutionIntent,
        parent_id: str,
        events: list[GatewayOrderEvent],
        attempts: Sequence[_ObservedOrder],
        emergency_quantity: float,
        fallback_price: float,
    ) -> None:
        self._engage_uncertainty(intent.symbol, events)
        emergency: _ObservedOrder | None = None
        if intent.reduce_only:
            emergency_side = intent.side or "SELL"
        else:
            emergency_side = "SELL" if intent.side == "BUY" else "BUY"
        if emergency_quantity > 1e-12:
            emergency = self._emergency_close(
                symbol=intent.symbol,
                side=emergency_side,
                quantity=emergency_quantity,
                parent_id=parent_id,
                fallback_price=fallback_price,
                events=events,
            )
        fills = self._fills(attempts, side=intent.side or "BUY", role="ENTRY_ATTEMPT")
        if emergency is not None and emergency.executed_quantity > 0:
            fills.extend(
                self._fills(
                    (emergency,),
                    side=emergency_side,
                    role="EMERGENCY_EXIT",
                )
            )
        submission = GatewaySubmission(
            status="SUBMISSION_UNRESOLVED",
            client_order_id=parent_id,
            external_order_id=None,
            fills=tuple(fills),
            order_events=tuple(events),
            raw_response={"error": message, "requires_reconciliation": True},
        )
        raise BinanceGatewaySubmissionUnresolved(message, submission)

    def _engage_uncertainty(self, symbol: str, events: list[GatewayOrderEvent]) -> None:
        self._execution_uncertain = True
        self.control.engage_kill_switch("binance_execution_status_unknown")
        self._active_symbols.add(symbol)
        self._cancel_remote_entry_orders(set(self._active_symbols), events)
        # Algo stops are never included in ordinary-order cancellation.  Verify every known stop
        # and retain it while reconciliation/flattening proceeds.
        for state in tuple(self._active_protective.values()):
            try:
                response = self.client.query_algo_order(client_algo_id=state.client_algo_id)
                status = self._algo_status(response)
            except Exception as error:
                response = {"error": str(error)}
                status = "UNKNOWN"
            events.append(
                self._algo_event(
                    state,
                    event_type="KILL_SWITCH_VERIFICATION",
                    status=status,
                    response=response,
                )
            )

    def _legacy_cancel_remote_entry_orders(
        self, symbols: set[str], events: list[GatewayOrderEvent]
    ) -> None:
        # Kept only for compatibility with any out-of-tree diagnostic caller. It delegates to
        # the same strict terminal barrier and cannot restore the former best-effort behavior.
        self._cancel_remote_entry_orders(symbols, events, require_terminal=True)
        return
        remote: list[tuple[str | None, Any]] = []
        try:
            # No symbol means all ordinary USDⓈ-M orders.  This is critical after a process
            # restart, when the in-memory active-symbol set is necessarily incomplete.
            remote = [(None, order) for order in self.client.open_orders()]
        except TypeError:
            for symbol in sorted(symbols):
                try:
                    remote.extend((symbol, order) for order in self.client.open_orders(symbol))
                except Exception:
                    continue
        except Exception:
            return
        seen: set[tuple[str, str]] = set()
        for fallback_symbol, order in remote:
            if isinstance(order, Mapping):
                reduce_only = bool(order.get("reduceOnly", order.get("reduce_only", False)))
                client_id = str(order.get("clientOrderId") or order.get("client_order_id") or "")
                symbol = str(order.get("symbol") or fallback_symbol or "")
                side = str(order.get("side", "BUY"))
                quantity = _number(order.get("origQty") or order.get("quantity"))
                price = _number(order.get("price"), 1.0)
            else:
                reduce_only = bool(order.reduce_only)
                client_id = str(order.client_order_id)
                symbol = str(order.symbol or fallback_symbol or "")
                side = str(order.side)
                quantity = float(order.original_quantity)
                price = float(order.price) or 1.0
            key = (symbol, client_id)
            if reduce_only or not symbol or not client_id or key in seen:
                continue
            seen.add(key)
            try:
                response = self._mutate(
                    self._risk_reducing_intent(symbol),
                    lambda s=symbol, cid=client_id: self.client.cancel_order(
                        symbol=s, client_order_id=cid
                    ),
                )
                # A cancel acknowledgement is not an execution barrier.  Reconciliation must
                # still query the order and observe a terminal cumulative executedQty.
                status = "CANCEL_REQUESTED"
            except Exception as error:
                response = {"error": str(error)}
                status = "UNKNOWN"
            observed = self._observed(client_id, response, fallback_price=price)
            events.append(
                self._standard_event(
                    observed,
                    role="REMOTE_ENTRY",
                    event_type="KILL_SWITCH_CANCEL",
                    symbol=symbol,
                    side=side,
                    order_type="LIMIT",
                    quantity=quantity,
                    reduce_only=False,
                    status_override=status,
                )
            )

    def _cancel_remote_entry_orders(
        self,
        symbols: set[str],
        events: list[GatewayOrderEvent],
        *,
        require_terminal: bool = False,
    ) -> None:
        try:
            remote = self._open_ordinary_orders(symbols)
        except Exception as error:
            if require_terminal:
                raise BinanceEntryCancellationUnresolved(
                    "ordinary-order enumeration failed during kill-switch cancellation",
                    events,
                ) from error
            return

        unresolved: list[str] = []
        seen: set[tuple[str, str]] = set()
        for fallback_symbol, order in remote:
            fields = self._ordinary_order_fields(order, fallback_symbol=fallback_symbol)
            if fields is None:
                if require_terminal:
                    unresolved.append("malformed_remote_order")
                continue
            symbol, client_id, side, quantity, price, reduce_only = fields
            if reduce_only:
                continue
            key = (symbol, client_id)
            if key in seen:
                continue
            seen.add(key)
            try:
                response = self._mutate(
                    self._risk_reducing_intent(symbol),
                    lambda s=symbol, cid=client_id: self.client.cancel_order(
                        symbol=s, client_order_id=cid
                    ),
                )
                cancel_observed = self._observed(
                    client_id, response, fallback_price=price
                )
                cancel_status = str(response.get("status", "CANCEL_REQUESTED")).upper()
            except Exception as error:
                cancel_observed = _ObservedOrder(
                    client_id,
                    {"error": str(error)},
                    0.0,
                    price,
                    "UNKNOWN",
                )
                cancel_status = "UNKNOWN"
            events.append(
                self._standard_event(
                    cancel_observed,
                    role="REMOTE_ENTRY",
                    event_type="KILL_SWITCH_CANCEL_REQUESTED",
                    symbol=symbol,
                    side=side,
                    order_type="LIMIT",
                    quantity=quantity,
                    reduce_only=False,
                    status_override=cancel_status,
                )
            )
            # Never resend a cancellation after an unknown response. A fresh status read is the
            # only resolution barrier and captures fills racing the cancellation.
            final = self._query_standard(
                symbol=symbol,
                child_id=client_id,
                fallback=cancel_observed,
                fallback_price=price,
                role="REMOTE_ENTRY",
                event_type="KILL_SWITCH_POST_CANCEL_REQUERY",
                side=side,
                order_type="LIMIT",
                quantity=quantity,
                reduce_only=False,
                events=events,
            )
            if final.status not in _STANDARD_TERMINAL:
                unresolved.append(f"{symbol}:{client_id}:{final.status}")

        if require_terminal and not unresolved:
            try:
                remaining = self._open_ordinary_orders(symbols)
            except Exception as error:
                raise BinanceEntryCancellationUnresolved(
                    "post-cancel ordinary-order enumeration failed",
                    events,
                ) from error
            for fallback_symbol, order in remaining:
                fields = self._ordinary_order_fields(
                    order, fallback_symbol=fallback_symbol
                )
                if fields is None:
                    unresolved.append("malformed_post_cancel_order")
                    continue
                symbol, client_id, _side, _quantity, _price, reduce_only = fields
                if not reduce_only:
                    unresolved.append(f"{symbol}:{client_id}:STILL_OPEN")

        if require_terminal and unresolved:
            raise BinanceEntryCancellationUnresolved(
                "ordinary entry cancellation unresolved: " + ",".join(unresolved[:20]),
                events,
            )

    def _open_ordinary_orders(self, symbols: set[str]) -> list[tuple[str | None, Any]]:
        try:
            # No symbol means the entire ordinary USD-M order book. This remains necessary after
            # a restart because the in-memory active-symbol set is necessarily incomplete.
            return [(None, order) for order in self.client.open_orders()]
        except TypeError:
            if not symbols:
                raise BinanceSafetyError(
                    "the order client cannot enumerate all ordinary orders"
                ) from None
            remote: list[tuple[str | None, Any]] = []
            failures: list[str] = []
            for symbol in sorted(symbols):
                try:
                    remote.extend(
                        (symbol, order) for order in self.client.open_orders(symbol)
                    )
                except Exception as error:
                    failures.append(f"{symbol}:{type(error).__name__}")
            if failures:
                raise BinanceSafetyError(
                    "ordinary-order enumeration failed for " + ",".join(failures)
                ) from None
            return remote

    @staticmethod
    def _ordinary_order_fields(
        order: Any, *, fallback_symbol: str | None
    ) -> tuple[str, str, str, float, float, bool] | None:
        try:
            if isinstance(order, Mapping):
                reduce_only = _flag(
                    order.get("reduceOnly", order.get("reduce_only", False))
                )
                client_id = str(
                    order.get("clientOrderId") or order.get("client_order_id") or ""
                )
                symbol = str(order.get("symbol") or fallback_symbol or "")
                side = str(order.get("side", "BUY")).upper()
                quantity = _number(order.get("origQty") or order.get("quantity"))
                price = _number(order.get("price"), 1.0) or 1.0
            else:
                reduce_only = bool(order.reduce_only)
                client_id = str(order.client_order_id)
                symbol = str(order.symbol or fallback_symbol or "")
                side = str(order.side).upper()
                quantity = float(order.original_quantity)
                price = float(order.price) or 1.0
        except (AttributeError, TypeError, ValueError):
            return None
        if not symbol or not client_id or side not in {"BUY", "SELL"} or quantity <= 0:
            return None
        return symbol, client_id, side, quantity, price, reduce_only

    def _emergency_close(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        parent_id: str,
        fallback_price: float,
        events: list[GatewayOrderEvent],
    ) -> _ObservedOrder | None:
        intent = self._risk_reducing_intent(symbol, side=side, quantity=quantity)
        emergency_id = self._child_id(parent_id, "x")
        try:
            response = self._mutate(
                intent,
                lambda: self.client.place_reduce_only_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    client_order_id=emergency_id,
                ),
            )
            observed = self._observed(emergency_id, response, fallback_price=fallback_price)
        except Exception as error:
            observed = _ObservedOrder(
                emergency_id,
                {"error": str(error)},
                0.0,
                fallback_price,
                "UNKNOWN",
            )
        events.append(
            self._standard_event(
                observed,
                role="EMERGENCY_EXIT",
                event_type="SUBMITTED",
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=quantity,
                reduce_only=True,
            )
        )
        if observed.status not in _STANDARD_TERMINAL:
            observed = self._query_standard(
                symbol=symbol,
                child_id=emergency_id,
                fallback=observed,
                fallback_price=fallback_price,
                role="EMERGENCY_EXIT",
                event_type="REST_REQUERY",
                side=side,
                order_type="MARKET",
                quantity=quantity,
                reduce_only=True,
                events=events,
            )
        return observed

    def _query_standard(
        self,
        *,
        symbol: str,
        child_id: str,
        fallback: _ObservedOrder,
        fallback_price: float,
        role: str,
        event_type: str,
        side: str,
        order_type: str,
        quantity: float,
        reduce_only: bool,
        events: list[GatewayOrderEvent],
    ) -> _ObservedOrder:
        try:
            response = self.client.query_order(symbol=symbol, client_order_id=child_id)
            observed = self._observed(child_id, response, fallback_price=fallback_price)
        except Exception as error:
            observed = _ObservedOrder(
                child_id,
                {"error": str(error), "last_observation": dict(fallback.response)},
                fallback.executed_quantity,
                fallback.average_price,
                "UNKNOWN",
            )
        events.append(
            self._standard_event(
                observed,
                role=role,
                event_type=event_type,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                reduce_only=reduce_only,
            )
        )
        return observed

    def _current_position_quantity(self, symbol: str) -> float | None:
        try:
            snapshot = self.client.rest_snapshot(symbol=symbol)
        except (Exception, AssertionError):
            try:
                snapshot = self.client.rest_snapshot()
            except (Exception, AssertionError):
                return None
        quantity = sum(
            float(position.quantity) for position in snapshot.positions if position.symbol == symbol
        )
        return abs(quantity)

    def _fetch_quote(self, symbol: str) -> MarketQuote:
        return self.client.fetch_quotes({symbol: symbol})[symbol]

    @staticmethod
    def _child_id(parent: str, suffix: str) -> str:
        return f"{parent[:34]}-{suffix}"[:36]

    @staticmethod
    def _replacement_algo_id(parent: str, quantity: float) -> str:
        digest = hashlib.sha256(f"{parent}:{quantity:.12g}".encode()).hexdigest()[:8]
        return f"{parent[:24]}-rs-{digest}"[:36]

    @staticmethod
    def _observed(
        client_id: str, response: Mapping[str, Any], *, fallback_price: float
    ) -> _ObservedOrder:
        quantity = _number(response.get("executedQty"))
        average = _number(response.get("avgPrice"))
        if average <= 0 and quantity > 0:
            average = _number(response.get("cumQuote")) / quantity
        if average <= 0:
            average = fallback_price
        return _ObservedOrder(
            client_id=client_id,
            response=dict(response),
            executed_quantity=quantity,
            average_price=average,
            status=str(response.get("status", "UNKNOWN")).upper(),
        )

    @staticmethod
    def _standard_event(
        observed: _ObservedOrder,
        *,
        role: str,
        event_type: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        reduce_only: bool,
        status_override: str | None = None,
        raw_override: Mapping[str, Any] | None = None,
    ) -> GatewayOrderEvent:
        status = status_override or observed.status
        external = str(observed.response.get("orderId", "")) or None
        marker = observed.response.get("updateTime", observed.response.get("time", 0))
        return GatewayOrderEvent(
            role=role,
            event_type=event_type,
            status=status,
            client_order_id=observed.client_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            reduce_only=reduce_only,
            external_order_id=external,
            executed_quantity=observed.executed_quantity,
            average_price=(observed.average_price if observed.executed_quantity > 0 else None),
            source_event_id=(
                f"gateway:{role}:{observed.client_id}:{event_type}:{status}:"
                f"{observed.executed_quantity:.12g}:{marker}"
            ),
            raw_response=dict(raw_override or observed.response),
        )

    @staticmethod
    def _algo_status(response: Mapping[str, Any]) -> str:
        return str(response.get("algoStatus") or response.get("status") or "UNKNOWN").upper()

    @staticmethod
    def _algo_event(
        state: ProtectiveOrderState,
        *,
        event_type: str,
        status: str,
        response: Mapping[str, Any],
    ) -> GatewayOrderEvent:
        marker = response.get("updateTime", response.get("time", 0))
        return GatewayOrderEvent(
            role="PROTECTIVE_STOP",
            event_type=event_type,
            status=status,
            client_order_id=state.client_algo_id,
            symbol=state.symbol,
            side=state.side,
            order_type="STOP_MARKET_ALGO",
            quantity=state.quantity,
            reduce_only=True,
            external_order_id=(str(response.get("algoId", "")) or state.external_order_id),
            trigger_price=state.trigger_price,
            source_event_id=(
                f"gateway:PROTECTIVE_STOP:{state.client_algo_id}:{event_type}:{status}:{marker}"
            ),
            raw_response=dict(response),
        )

    @staticmethod
    def _fills(attempts: Sequence[_ObservedOrder], *, side: str, role: str) -> list[GatewayFill]:
        fills: list[GatewayFill] = []
        for item in attempts:
            if item.executed_quantity <= 0:
                continue
            order_id = str(item.response.get("orderId", item.client_id))
            fills.append(
                GatewayFill(
                    fill_id=f"order-{order_id}",
                    price=item.average_price,
                    quantity=item.executed_quantity,
                    fee=0,
                    realized_pnl=(
                        _number(item.response.get("realizedPnl"))
                        if "realizedPnl" in item.response
                        else None
                    ),
                    client_order_id=item.client_id,
                    side=side,
                    role=role,
                    authoritative=False,
                    raw_response=dict(item.response),
                )
            )
        return fills

    @staticmethod
    def _risk_reducing_intent(
        symbol: str, *, side: str = "SELL", quantity: float = 0
    ) -> ExecutionIntent:
        return ExecutionIntent(
            approved=True,
            reason="execution_safety_reduction",
            action=TradeAction.CLOSE,
            symbol=symbol,
            side=side,
            quantity=quantity,
            notional=0,
            reduce_only=True,
        )

    @classmethod
    def _signed_exit_intent(cls, symbol: str, signed_quantity: float) -> ExecutionIntent:
        return cls._risk_reducing_intent(
            symbol,
            side="SELL" if signed_quantity > 0 else "BUY",
            quantity=abs(signed_quantity),
        )
