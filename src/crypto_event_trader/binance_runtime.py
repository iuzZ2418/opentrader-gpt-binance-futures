from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .approval import ApprovalTradingService, GatewayOrderEvent, GatewaySubmission
from .audit import AuditRepository
from .binance import (
    BinanceFuturesClient,
    BinanceSafetyError,
    FuturesRestSnapshot,
    ReconciliationResult,
)
from .binance_execution import (
    BinanceFuturesAccountSource,
    BinanceFuturesExecutionGateway,
    ProtectiveOrderState,
)
from .binance_streams import (
    AlgoOrderUpdate,
    ConditionalOrderTriggerReject,
    FuturesStreamEvent,
    FuturesStreamState,
    OrderTradeUpdate,
)
from .binance_ws_runtime import EventHandler, FuturesWebSocketRuntime, ReconcileHandler
from .config import Settings
from .control import TradingControl
from .market_data import BinanceFuturesMarketDataProvider
from .openai_decision import OpenAIResponsesDecisionProvider
from .security import (
    SecurityBoundaryError,
    validate_binance_runtime_urls,
    validate_openai_base_url,
)


@dataclass(slots=True)
class BinanceApprovalRuntime:
    settings: Settings
    control: TradingControl
    audit: AuditRepository
    client: BinanceFuturesClient
    streams: FuturesStreamState
    market_data: BinanceFuturesMarketDataProvider
    account_source: BinanceFuturesAccountSource
    gateway: BinanceFuturesExecutionGateway
    decision_provider: OpenAIResponsesDecisionProvider
    approvals: ApprovalTradingService

    def handle_stream_event(self, event: FuturesStreamEvent) -> None:
        """Persist authoritative private order events before any in-memory reaction.

        The private stream may repeat observations across reconnects. Both order observations
        and trades use venue-derived idempotency keys. An event for a client ID that cannot be
        traced to an audited parent or child order is treated as an external-state divergence,
        never as harmless noise.
        """

        if isinstance(event, AlgoOrderUpdate):
            self._handle_algo_order_update(event)
            return
        if isinstance(event, ConditionalOrderTriggerReject):
            self._handle_conditional_trigger_reject(event)
            return
        if not isinstance(event, OrderTradeUpdate):
            return
        try:
            owner = self.audit.resolve_venue_order_client_id(
                event.client_order_id, venue=self.gateway.venue
            )
        except Exception as error:
            self._private_stream_failure("private_order_audit_lookup_failed", error)
        if owner is None:
            self._private_stream_failure(
                "unknown_private_order_client_id",
                BinanceSafetyError(
                    "Private order update has no audited parent/child client ID: "
                    f"{event.client_order_id!r}"
                ),
            )
        assert owner is not None
        if str(owner["symbol"]).upper() != event.symbol.upper():
            self._private_stream_failure(
                "private_order_symbol_mismatch",
                BinanceSafetyError(
                    "Private order symbol does not match its audited owner: "
                    f"{event.client_order_id!r}"
                ),
            )
        observed_at = self._milliseconds_datetime(event.transaction_time or event.event_time)
        trade_marker = (
            str(event.trade_id)
            if event.trade_id >= 0
            else ":".join(
                (
                    str(event.trade_time),
                    format(event.last_filled_quantity, "f"),
                    format(event.last_filled_price, "f"),
                    format(event.accumulated_filled_quantity, "f"),
                )
            )
        )
        source_marker = (
            trade_marker
            if event.execution_type.upper() == "TRADE"
            else ":".join(
                (
                    str(event.transaction_time),
                    format(event.accumulated_filled_quantity, "f"),
                )
            )
        )
        source_event_id = ":".join(
            (
                "binance-ws",
                "ORDER_TRADE_UPDATE",
                str(event.order_id),
                event.execution_type.upper(),
                event.order_status.upper(),
                source_marker,
            )
        )
        raw = {
            **event.raw,
            "audit_resolution": {
                "matched_client_order_id": event.client_order_id,
                "client_id_role": owner["client_id_role"],
                "parent_client_order_id": owner["client_order_id"],
            },
        }
        try:
            self.audit.append_venue_order_event(
                trace_id=str(owner["trace_id"]),
                venue_order_id=str(owner["venue_order_id"]),
                event_type=f"PRIVATE_WS_{event.execution_type}",
                status=event.order_status,
                source_event_id=source_event_id,
                external_order_id=str(event.order_id),
                executed_quantity=float(event.accumulated_filled_quantity),
                average_price=(float(event.average_price) if event.average_price > 0 else None),
                raw_response=raw,
                observed_at=observed_at,
            )
            if event.execution_type.upper() == "TRADE":
                self._append_private_trade_fill(event, owner, raw, observed_at)
        except Exception as error:
            self._private_stream_failure("private_order_accounting_failed", error)

    def _handle_algo_order_update(self, event: AlgoOrderUpdate) -> None:
        status = event.status.upper()
        if (
            not event.client_algo_id
            or not event.symbol
            or event.algo_id <= 0
            or event.side.upper() not in {"BUY", "SELL"}
            or event.quantity <= 0
            or event.trigger_price <= 0
            or not event.reduce_only
        ):
            self._private_stream_failure(
                "invalid_protective_algo_update",
                BinanceSafetyError("Invalid or non-reduce-only Binance ALGO_UPDATE"),
            )
        try:
            owner = self.audit.resolve_venue_order_client_id(
                event.client_algo_id, venue=self.gateway.venue
            )
        except Exception as error:
            self._private_stream_failure("private_algo_audit_lookup_failed", error)
        if owner is None or str(owner.get("client_id_role", "")).upper() != "PROTECTIVE_STOP":
            self._private_stream_failure(
                "unknown_protective_algo_client_id",
                BinanceSafetyError(
                    "ALGO_UPDATE has no audited protective owner: "
                    f"{event.client_algo_id!r}"
                ),
            )
        assert owner is not None
        if str(owner["symbol"]).upper() != event.symbol.upper():
            self._private_stream_failure(
                "private_algo_symbol_mismatch",
                BinanceSafetyError("ALGO_UPDATE symbol does not match its audited owner"),
            )
        observed_at = self._milliseconds_datetime(event.transaction_time or event.event_time)
        source_event_id = ":".join(
            (
                "binance-ws",
                "ALGO_UPDATE",
                str(event.algo_id),
                status,
                str(event.transaction_time),
            )
        )
        raw = {
            **event.raw,
            "gateway_order_event": {
                "role": "PROTECTIVE_STOP",
                "child_client_order_id": event.client_algo_id,
                "symbol": event.symbol,
                "side": event.side.upper(),
                "order_type": event.order_type,
                "quantity": float(event.quantity),
                "reduce_only": True,
                "trigger_price": float(event.trigger_price),
            },
            "audit_resolution": {
                "matched_client_order_id": event.client_algo_id,
                "parent_client_order_id": owner["client_order_id"],
            },
        }
        try:
            self.audit.append_venue_order_event(
                trace_id=str(owner["trace_id"]),
                venue_order_id=str(owner["venue_order_id"]),
                event_type="PROTECTIVE_STOP_PRIVATE_WS_ALGO_UPDATE",
                status=status,
                source_event_id=source_event_id,
                external_order_id=str(event.algo_id),
                raw_response=raw,
                observed_at=observed_at,
            )
        except Exception as error:
            self._private_stream_failure("private_algo_accounting_failed", error)
        active = {"NEW", "PENDING_NEW", "WORKING", "ACCEPTED"}
        successful_terminal = {"TRIGGERED", "FINISHED"}
        unsafe_terminal = {"CANCELED", "CANCELLED", "EXPIRED", "REJECTED"}
        if status in active or status in successful_terminal:
            return
        reason = (
            "protective_algo_became_unsafe"
            if status in unsafe_terminal
            else "protective_algo_status_unknown"
        )
        self._private_stream_failure(
            reason,
            BinanceSafetyError(
                f"Protective algo {event.client_algo_id} entered unsafe status {status}: "
                f"{event.reject_reason or 'no reason supplied'}"
            ),
        )

    def _handle_conditional_trigger_reject(
        self, event: ConditionalOrderTriggerReject
    ) -> None:
        if not event.symbol or event.algo_id <= 0 or not event.reason.strip():
            self._private_stream_failure(
                "invalid_conditional_trigger_rejection",
                BinanceSafetyError("Invalid CONDITIONAL_ORDER_TRIGGER_REJECT payload"),
            )
        try:
            owner = self.audit.resolve_protective_algo_id(
                event.algo_id, venue=self.gateway.venue
            )
        except Exception as error:
            self._private_stream_failure("private_algo_audit_lookup_failed", error)
        if owner is None:
            self._private_stream_failure(
                "unknown_conditional_trigger_rejection",
                BinanceSafetyError(
                    "Conditional trigger rejection has no audited protective owner: "
                    f"{event.algo_id}"
                ),
            )
        assert owner is not None
        if str(owner["symbol"]).upper() != event.symbol.upper():
            self._private_stream_failure(
                "private_algo_symbol_mismatch",
                BinanceSafetyError(
                    "Conditional trigger rejection symbol does not match its audited owner"
                ),
            )
        child_id = str(owner["matched_client_order_id"])
        gateway_event = dict(owner["gateway_order_event"])
        observed_at = self._milliseconds_datetime(event.transaction_time or event.event_time)
        raw = {
            **event.raw,
            "gateway_order_event": gateway_event,
            "audit_resolution": {
                "matched_client_order_id": child_id,
                "parent_client_order_id": owner["client_order_id"],
            },
        }
        try:
            self.audit.append_venue_order_event(
                trace_id=str(owner["trace_id"]),
                venue_order_id=str(owner["venue_order_id"]),
                event_type="PROTECTIVE_STOP_CONDITIONAL_TRIGGER_REJECT",
                status="REJECTED",
                source_event_id=(
                    f"binance-ws:CONDITIONAL_ORDER_TRIGGER_REJECT:{event.algo_id}:"
                    f"{event.transaction_time}:{event.reason}"
                ),
                external_order_id=str(event.algo_id),
                raw_response=raw,
                observed_at=observed_at,
            )
        except Exception as error:
            self._private_stream_failure("private_algo_accounting_failed", error)
        self._private_stream_failure(
            "protective_algo_trigger_rejected",
            BinanceSafetyError(
                f"Protective algo {child_id} trigger was rejected: {event.reason}"
            ),
        )

    def _append_private_trade_fill(
        self,
        event: OrderTradeUpdate,
        owner: dict[str, Any],
        raw: dict[str, Any],
        observed_at: datetime,
    ) -> None:
        if event.last_filled_quantity <= 0 or event.last_filled_price <= 0:
            raise BinanceSafetyError("TRADE update has no positive last fill quantity/price")
        if event.commission < 0:
            raise BinanceSafetyError("TRADE update contains a negative commission")
        if event.commission > 0 and not event.commission_asset:
            raise BinanceSafetyError("TRADE update commission asset is missing")
        fee_asset = event.commission_asset or "USDT"
        trade_marker = (
            str(event.trade_id)
            if event.trade_id >= 0
            else ":".join(
                (
                    str(event.trade_time),
                    format(event.last_filled_quantity, "f"),
                    format(event.last_filled_price, "f"),
                )
            )
        )
        external_fill_id = f"binance-trade:{event.symbol}:{trade_marker}"
        self.audit.append_venue_fill(
            trace_id=str(owner["trace_id"]),
            venue_order_id=str(owner["venue_order_id"]),
            external_fill_id=external_fill_id,
            price=float(event.last_filled_price),
            quantity=float(event.last_filled_quantity),
            fee=float(event.commission),
            fee_asset=fee_asset,
            realized_pnl=float(event.realized_profit),
            raw_response=raw,
            filled_at=self._milliseconds_datetime(
                event.trade_time or event.transaction_time or event.event_time
            ),
        )
        if event.commission:
            self.audit.append_venue_accounting_event(
                trace_id=str(owner["trace_id"]),
                venue_order_id=str(owner["venue_order_id"]),
                venue=self.gateway.venue,
                external_income_id=f"trade-commission:{event.symbol}:{trade_marker}",
                symbol=event.symbol,
                income_type="COMMISSION",
                asset=fee_asset,
                amount=-float(event.commission),
                transaction_time=observed_at,
                trade_id=event.trade_id if event.trade_id >= 0 else trade_marker,
                raw_response=raw,
            )

    def _private_stream_failure(self, reason: str, error: Exception) -> None:
        self.control.engage_kill_switch(reason)
        try:
            self.reconcile()
        except Exception:
            # The original private-stream violation remains the primary error. The shared kill
            # latch is already engaged and the websocket supervisor will also reconcile again.
            pass
        raise error

    def startup_check(self) -> dict[str, Any]:
        """Synchronize time and reconcile REST before the worker enables stream readiness."""

        offset = self.client.sync_time()
        snapshot, dual_side_position = self._account_safety_snapshot()
        self._assert_safe_account_configuration(snapshot, dual_side_position=dual_side_position)
        model_access = self.decision_provider.check_model_access()
        reconciliation = self.reconcile()
        return {
            "server_time_offset_ms": offset,
            "rest_observed_at_ms": snapshot.observed_at_ms,
            "open_positions": len([item for item in snapshot.positions if item.quantity]),
            "open_orders": len(snapshot.open_orders),
            "model_access_verified": model_access,
            "one_way_position_mode_verified": True,
            "open_position_safety_verified": True,
            "reconciliation_consistent": reconciliation.consistent,
            "risk_context_ready": self.account_source.ready_for_new_orders,
            "stream_ready": self.streams.health().ready_for_new_orders,
        }

    def cancel_all_entries(self) -> tuple[GatewayOrderEvent, ...]:
        """Apply the kill-switch entry cancellation barrier and audit every observation."""

        try:
            events = self.gateway.cancel_all()
        except Exception as error:
            partial = getattr(error, "order_events", ())
            if isinstance(partial, Sequence):
                self._audit_entry_cancellation_events(tuple(partial))
            self._audit_cancellation_incident(
                f"kill_switch_cancel_unresolved:{type(error).__name__}"
            )
            raise
        unknown = self._audit_entry_cancellation_events(events)
        if unknown:
            self.control.engage_kill_switch("unaudited_remote_entry_order")
            self._audit_cancellation_incident("kill_switch_cancel_unknown_remote_order")
            raise BinanceSafetyError(
                "kill-switch cancellation found unaudited remote entry IDs: "
                + ",".join(unknown[:20])
            )
        reconciliation = self.reconcile()
        if not reconciliation.consistent:
            self._audit_cancellation_incident("kill_switch_cancel_reconciliation_mismatch")
            raise BinanceSafetyError(
                "kill-switch cancellation did not pass full account reconciliation"
            )
        self._audit_cancellation_incident("kill_switch_cancel_verified")
        return events

    def reconcile(self) -> ReconciliationResult:
        dual_side_position = self._read_position_mode_fail_closed()
        expected_positions: dict[str, float] = {}
        latest_account = self.audit.latest_account_snapshot(source=self.gateway.venue)
        if latest_account is not None:
            expected_positions = {
                str(item["symbol"]): float(item["quantity"])
                for item in latest_account.get("positions", [])
                if float(item.get("quantity", 0))
            }
        expected_open_ids: set[str] = set()
        child_states: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
        protective_states: dict[
            str, tuple[ProtectiveOrderState, dict[str, Any], dict[str, Any]]
        ] = {}
        owner_by_symbol: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for trace_id in self.audit.list_venue_order_trace_ids(venue=self.gateway.venue):
            trace = self.audit.get_trace(trace_id)
            for order in trace["venue_orders"]:
                if order["venue"] != self.gateway.venue:
                    continue
                events = [
                    event
                    for event in trace["venue_order_events"]
                    if event["venue_order_id"] == order["venue_order_id"]
                ]
                owner_by_symbol[str(order["symbol"])] = (trace, order)
                parent_events: list[dict[str, Any]] = []
                for event in events:
                    raw = event.get("raw_response")
                    child = raw.get("gateway_order_event") if isinstance(raw, dict) else None
                    if not isinstance(child, dict):
                        parent_events.append(event)
                        continue
                    child_id = str(child.get("child_client_order_id", ""))
                    role = str(child.get("role", "")).upper()
                    if not child_id:
                        continue
                    if role in {"ENTRY_ATTEMPT", "REMOTE_ENTRY"}:
                        child_states[child_id] = (trace, order, event)
                    elif role == "PROTECTIVE_STOP":
                        trigger = float(child.get("trigger_price") or 0)
                        quantity = float(child.get("quantity") or 0)
                        child_status = str(event.get("status", "UNKNOWN")).upper()
                        if child_status in {
                            "CANCELED",
                            "CANCELLED",
                            "EXPIRED",
                            "REJECTED",
                            "FINISHED",
                            "TRIGGERED",
                        }:
                            protective_states.pop(child_id, None)
                        elif child_status != "PLANNED" and trigger > 0 and quantity > 0:
                            protective_states[child_id] = (
                                ProtectiveOrderState(
                                    symbol=str(child.get("symbol") or order["symbol"]),
                                    client_algo_id=child_id,
                                    side=str(child.get("side", "SELL")),
                                    trigger_price=trigger,
                                    quantity=quantity,
                                    external_order_id=(
                                        str(event.get("external_order_id"))
                                        if event.get("external_order_id")
                                        else None
                                    ),
                                ),
                                trace,
                                order,
                            )
                latest = max(
                    parent_events,
                    key=lambda item: item["event_sequence"],
                    default=None,
                )
                status = str((latest or {}).get("status", order["status"])).upper()
                has_entry_children = any(
                    item[1]["venue_order_id"] == order["venue_order_id"]
                    for item in child_states.values()
                )
                if (
                    status in {"PREPARED", "SUBMITTING", "SUBMISSION_UNRESOLVED"}
                    and not has_entry_children
                ):
                    try:
                        response = self.client.query_order(
                            symbol=order["symbol"],
                            client_order_id=order["client_order_id"],
                        )
                    except Exception:
                        self.control.engage_kill_switch("unresolved_order_reconciliation_failed")
                        raise
                    status = str(response.get("status", "UNKNOWN")).upper()
                    executed = float(response.get("executedQty", 0) or 0)
                    average = float(response.get("avgPrice", 0) or 0) or None
                    source_event_id = ":".join(
                        (
                            "rest",
                            str(response.get("orderId", order["client_order_id"])),
                            status,
                            str(response.get("updateTime", response.get("time", 0))),
                            str(executed),
                        )
                    )
                    self.audit.append_venue_order_event(
                        trace_id=order["trace_id"],
                        venue_order_id=order["venue_order_id"],
                        event_type="REST_RECONCILIATION",
                        status=status,
                        source_event_id=source_event_id,
                        external_order_id=str(response.get("orderId", "")) or None,
                        executed_quantity=executed,
                        average_price=average,
                        raw_response=response,
                    )
                    self._append_reconciled_fill_delta(trace, order, response)
                if not has_entry_children and status in {"NEW", "PARTIALLY_FILLED", "PENDING_NEW"}:
                    expected_open_ids.add(str(order["client_order_id"]))

        for child_id, (trace, order, event) in child_states.items():
            child_status = str(event.get("status", "UNKNOWN")).upper()
            if child_status not in {
                "NEW",
                "PARTIALLY_FILLED",
                "PENDING_NEW",
                "UNKNOWN",
                "CANCEL_REQUESTED",
            }:
                continue
            try:
                response = self.client.query_order(
                    symbol=str(order["symbol"]), client_order_id=child_id
                )
            except Exception:
                self.control.engage_kill_switch("unresolved_child_order_reconciliation_failed")
                raise
            status = str(response.get("status", "UNKNOWN")).upper()
            executed = float(response.get("executedQty", 0) or 0)
            average = float(response.get("avgPrice", 0) or 0) or None
            self.audit.append_venue_order_event(
                trace_id=str(order["trace_id"]),
                venue_order_id=str(order["venue_order_id"]),
                event_type="ENTRY_ATTEMPT_REST_RECONCILIATION",
                status=status,
                source_event_id=(
                    f"rest-child:{child_id}:{status}:"
                    f"{response.get('updateTime', response.get('time', 0))}:{executed}"
                ),
                external_order_id=str(response.get("orderId", "")) or None,
                executed_quantity=executed,
                average_price=average,
                raw_response={
                    "gateway_order_event": {
                        "role": "ENTRY_ATTEMPT",
                        "child_client_order_id": child_id,
                        "symbol": order["symbol"],
                    },
                    "venue_response": response,
                },
            )
            self._append_reconciled_fill_delta(trace, order, response, child_id=child_id)
            if status in {"NEW", "PARTIALLY_FILLED", "PENDING_NEW"}:
                expected_open_ids.add(child_id)
            elif status not in {
                "FILLED",
                "CANCELED",
                "EXPIRED",
                "EXPIRED_IN_MATCH",
                "REJECTED",
            }:
                self.control.engage_kill_switch("child_order_terminal_state_unknown")
                raise BinanceSafetyError(
                    f"child order {child_id} returned non-terminal unknown status {status}"
                )
        try:
            result = self.gateway.reconcile(
                expected_open_client_ids=tuple(sorted(expected_open_ids)),
                expected_positions=expected_positions,
            )
        except Exception:
            self.control.engage_kill_switch("binance_reconciliation_unavailable")
            raise
        self._assert_safe_account_configuration(
            result.snapshot, dual_side_position=dual_side_position
        )
        actual_positions = {
            position.symbol: float(position.quantity)
            for position in result.snapshot.positions
            if position.quantity
        }
        known = [item[0] for item in protective_states.values()]
        try:
            protection_events = self.gateway.reconcile_protective_orders(
                actual_positions=actual_positions,
                known_orders=known,
            )
        except Exception as error:
            partial = getattr(error, "submission", None)
            if isinstance(partial, GatewaySubmission):
                self._audit_protective_reconciliation_events(
                    partial.order_events, protective_states, owner_by_symbol
                )
            self.control.engage_kill_switch("protective_order_reconciliation_failed")
            raise
        self._audit_protective_reconciliation_events(
            protection_events, protective_states, owner_by_symbol
        )
        self._reconcile_account_income()
        if not result.consistent:
            self.control.engage_kill_switch("startup_or_stream_reconciliation_error")
        return result

    def _audit_entry_cancellation_events(
        self, events: Sequence[GatewayOrderEvent]
    ) -> tuple[str, ...]:
        unknown: list[str] = []
        for event in events:
            owner = self.audit.resolve_venue_order_client_id(
                event.client_order_id, venue=self.gateway.venue
            )
            if owner is None:
                if event.client_order_id not in unknown:
                    unknown.append(event.client_order_id)
                continue
            self.audit.append_venue_order_event(
                trace_id=str(owner["trace_id"]),
                venue_order_id=str(owner["venue_order_id"]),
                event_type=f"{event.role}_{event.event_type}",
                status=event.status,
                source_event_id=event.source_event_id,
                external_order_id=event.external_order_id,
                executed_quantity=event.executed_quantity,
                average_price=event.average_price,
                raw_response={
                    "gateway_order_event": {
                        "role": event.role,
                        "child_client_order_id": event.client_order_id,
                        "symbol": event.symbol,
                        "side": event.side,
                        "order_type": event.order_type,
                        "quantity": event.quantity,
                        "reduce_only": event.reduce_only,
                    },
                    "venue_response": dict(event.raw_response),
                },
                observed_at=event.observed_at,
            )
        return tuple(unknown)

    def _audit_cancellation_incident(self, source: str) -> None:
        try:
            snapshot = self.account_source.snapshot()
            self.audit.append_account_snapshot(
                equity=snapshot.equity,
                cash=snapshot.wallet_balance,
                gross_exposure=snapshot.gross_notional,
                net_exposure=snapshot.net_notional,
                daily_pnl=snapshot.daily_pnl_fraction,
                drawdown=snapshot.drawdown,
                positions=snapshot.positions,
                source=source,
                observed_at=snapshot.timestamp,
            )
        except Exception:
            # Cancellation and reconciliation failures remain primary. The shared kill switch is
            # already engaged, and a later watchdog iteration retries the authoritative snapshot.
            return

    def _append_reconciled_fill_delta(
        self,
        trace: dict[str, Any],
        order: dict[str, Any],
        response: dict[str, Any],
        *,
        child_id: str | None = None,
    ) -> None:
        executed = float(response.get("executedQty", 0) or 0)
        if executed <= 0:
            return
        target_id = child_id or str(order["client_order_id"])
        raw_order_id = response.get("orderId")
        try:
            order_id = int(raw_order_id)
        except (TypeError, ValueError):
            self.control.engage_kill_switch("fill_accounting_order_id_missing")
            raise BinanceSafetyError(
                "Executed order is missing a valid authoritative Binance orderId"
            ) from None
        if order_id <= 0:
            self.control.engage_kill_switch("fill_accounting_order_id_missing")
            raise BinanceSafetyError(
                "Executed order is missing a valid authoritative Binance orderId"
            )
        try:
            trades = self.client.user_trades(
                str(order["symbol"]),
                order_id=order_id,
                limit=1_000,
            )
        except Exception:
            self.control.engage_kill_switch("fill_accounting_reconciliation_failed")
            raise
        if not trades:
            self.control.engage_kill_switch("fill_accounting_reconciliation_incomplete")
            raise BinanceSafetyError(
                "Executed order has no authoritative user-trade accounting records"
            )
        expected_symbol = str(order["symbol"]).upper()
        normalized_trades: list[dict[str, Any]] = []
        accounted_quantity = 0.0
        for item in trades:
            item_symbol = str(item.get("symbol") or "").upper()
            try:
                item_order_id = int(item.get("orderId"))
            except (TypeError, ValueError):
                item_order_id = -1
            quantity = float(item.get("qty", 0) or 0)
            price = float(item.get("price", 0) or 0)
            commission = float(item.get("commission", 0) or 0)
            commission_asset = str(item.get("commissionAsset") or "")
            trade_id = item.get("id")
            trade_time = int(item.get("time", 0) or 0)
            if (
                item_symbol != expected_symbol
                or item_order_id != order_id
                or quantity <= 0
                or price <= 0
                or commission < 0
                or (commission > 0 and not commission_asset)
                or trade_id in (None, "")
                or trade_time <= 0
            ):
                self.control.engage_kill_switch("invalid_user_trade_accounting")
                raise BinanceSafetyError(
                    "Binance user-trade accounting is not bound to the reconciled order"
                )
            accounted_quantity += quantity
            normalized_trades.append(dict(item))
        if not math.isclose(accounted_quantity, executed, rel_tol=1e-9, abs_tol=1e-12):
            self.control.engage_kill_switch("fill_accounting_reconciliation_incomplete")
            raise BinanceSafetyError(
                "Authoritative user-trade quantity does not exactly match the order's "
                "executed quantity"
            )
        for item in normalized_trades:
            quantity = float(item["qty"])
            price = float(item["price"])
            commission = float(item.get("commission", 0) or 0)
            commission_asset = str(item.get("commissionAsset") or "")
            trade_id = item["id"]
            trade_time = int(item.get("time", 0) or 0)
            raw = {
                **item,
                "client_order_id": target_id,
                "role": "REST_USER_TRADE_RECONCILIATION",
            }
            self.audit.append_venue_fill(
                trace_id=str(order["trace_id"]),
                venue_order_id=str(order["venue_order_id"]),
                external_fill_id=(f"binance-trade:{order['symbol']}:{trade_id}"),
                price=price,
                quantity=quantity,
                fee=commission,
                fee_asset=commission_asset or "USDT",
                realized_pnl=float(item.get("realizedPnl", 0) or 0),
                raw_response=raw,
                filled_at=self._milliseconds_datetime(trade_time),
            )
            if commission:
                self.audit.append_venue_accounting_event(
                    trace_id=str(order["trace_id"]),
                    venue_order_id=str(order["venue_order_id"]),
                    venue=self.gateway.venue,
                    external_income_id=(f"trade-commission:{order['symbol']}:{trade_id}"),
                    symbol=str(order["symbol"]),
                    income_type="COMMISSION",
                    asset=commission_asset,
                    amount=-commission,
                    transaction_time=self._milliseconds_datetime(trade_time),
                    trade_id=trade_id,
                    raw_response=raw,
                )

    def _reconcile_account_income(self) -> None:
        income_history = getattr(self.client, "income_history", None)
        latest_reader = getattr(self.audit, "latest_venue_accounting_event", None)
        append = getattr(self.audit, "append_venue_accounting_event", None)
        if not callable(income_history) or not callable(latest_reader) or not callable(append):
            # Lightweight protocol doubles used by unit tests may omit the production APIs.
            return
        latest = latest_reader(venue=self.gateway.venue, income_type="FUNDING_FEE")
        start_time: int | None = None
        if latest is not None:
            parsed = datetime.fromisoformat(str(latest["transaction_time"]).replace("Z", "+00:00"))
            start_time = int(parsed.timestamp() * 1_000)
        try:
            records = income_history(income_type="FUNDING_FEE", start_time=start_time, limit=1_000)
            for item in sorted(records, key=lambda row: int(row.get("time", 0) or 0)):
                income_type = str(item.get("incomeType") or "").upper()
                transaction_id = item.get("tranId")
                asset = str(item.get("asset") or "").upper()
                transaction_time = int(item.get("time", 0) or 0)
                if (
                    income_type != "FUNDING_FEE"
                    or transaction_id in (None, "")
                    or not asset
                    or transaction_time <= 0
                ):
                    raise BinanceSafetyError("Invalid Binance funding-income record")
                symbol = str(item.get("symbol") or "").upper()
                transaction_at = self._milliseconds_datetime(transaction_time)
                if symbol:
                    try:
                        attribution = self.audit.resolve_funding_attribution(
                            venue=self.gateway.venue,
                            symbol=symbol,
                            transaction_time=transaction_at,
                        )
                    except Exception as error:
                        attribution_status = "UNATTRIBUTED"
                        attribution_reason = f"ATTRIBUTION_RESOLUTION_ERROR:{type(error).__name__}"
                        attribution_trace = None
                        attribution_order = None
                    else:
                        attribution_status = attribution.status
                        attribution_reason = attribution.reason
                        attribution_trace = attribution.trace_id
                        attribution_order = attribution.venue_order_id
                else:
                    attribution_status = "UNATTRIBUTED"
                    attribution_reason = "MISSING_SYMBOL"
                    attribution_trace = None
                    attribution_order = None
                raw = {
                    **item,
                    "audit_attribution": {
                        "status": attribution_status,
                        "reason": attribution_reason,
                        "trace_id": attribution_trace,
                        "venue_order_id": attribution_order,
                    },
                }
                accounting_event_id = append(
                    venue=self.gateway.venue,
                    external_income_id=f"FUNDING_FEE:{transaction_id}",
                    symbol=symbol or None,
                    income_type=income_type,
                    asset=asset,
                    amount=float(item.get("income", 0) or 0),
                    transaction_time=transaction_at,
                    trace_id=attribution_trace,
                    venue_order_id=attribution_order,
                    trade_id=(str(item.get("tradeId")) if item.get("tradeId") else None),
                    raw_response=raw,
                )
                self.audit.append_venue_accounting_attribution(
                    accounting_event_id=accounting_event_id,
                    status=attribution_status,
                    reason=attribution_reason,
                    trace_id=attribution_trace,
                    venue_order_id=attribution_order,
                    resolved_at=datetime.now(UTC),
                )
                if attribution_status != "ATTRIBUTED":
                    self.control.engage_kill_switch("unattributed_funding_accounting")
                    raise BinanceSafetyError(
                        "Funding income could not be reliably attributed to an audited "
                        f"position: {attribution_reason}"
                    )
        except Exception:
            if not self.control.snapshot().kill_switch_active:
                self.control.engage_kill_switch("funding_accounting_reconciliation_failed")
            raise

    @staticmethod
    def _milliseconds_datetime(value: int | float) -> datetime:
        numeric = float(value or 0)
        return datetime.fromtimestamp(numeric / 1_000, UTC) if numeric > 0 else datetime.now(UTC)

    def _audit_protective_reconciliation_events(
        self,
        events: Sequence[GatewayOrderEvent],
        states: dict[str, tuple[ProtectiveOrderState, dict[str, Any], dict[str, Any]]],
        owners: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        for event in events:
            owned = states.get(event.client_order_id)
            trace_order = (owned[1], owned[2]) if owned is not None else owners.get(event.symbol)
            if trace_order is None:
                continue
            _trace, order = trace_order
            self.audit.append_venue_order_event(
                trace_id=str(order["trace_id"]),
                venue_order_id=str(order["venue_order_id"]),
                event_type=f"{event.role}_{event.event_type}",
                status=event.status,
                source_event_id=event.source_event_id,
                external_order_id=event.external_order_id,
                executed_quantity=event.executed_quantity,
                average_price=event.average_price,
                raw_response={
                    "gateway_order_event": {
                        "role": event.role,
                        "child_client_order_id": event.client_order_id,
                        "symbol": event.symbol,
                        "side": event.side,
                        "order_type": event.order_type,
                        "quantity": event.quantity,
                        "reduce_only": event.reduce_only,
                        "trigger_price": event.trigger_price,
                    },
                    "venue_response": dict(event.raw_response),
                },
                observed_at=event.observed_at,
            )

    def _account_safety_snapshot(self) -> tuple[FuturesRestSnapshot, bool]:
        dual_side_position = self._read_position_mode_fail_closed()
        try:
            snapshot = self.client.rest_snapshot()
        except Exception as error:
            self.control.engage_kill_switch("account_safety_snapshot_unavailable")
            raise BinanceSafetyError(
                "Binance account safety snapshot is unavailable; startup remains locked"
            ) from error
        return snapshot, dual_side_position

    def _read_position_mode_fail_closed(self) -> bool:
        try:
            return self.client.get_position_mode()
        except Exception as error:
            self.control.engage_kill_switch("position_mode_verification_failed")
            raise BinanceSafetyError(
                "Binance position mode could not be verified; startup remains locked"
            ) from error

    def _assert_safe_account_configuration(
        self,
        snapshot: FuturesRestSnapshot,
        *,
        dual_side_position: bool,
    ) -> None:
        violations: list[str] = []
        if dual_side_position:
            violations.append("hedge_mode_enabled")
        for position in snapshot.positions:
            if not position.quantity:
                continue
            symbol = position.symbol or "UNKNOWN"
            if position.position_side.strip().upper() != "BOTH":
                violations.append(f"{symbol}:position_side={position.position_side or 'missing'}")
            if position.margin_type.strip().lower() != "isolated":
                violations.append(f"{symbol}:margin_type={position.margin_type or 'missing'}")
            if not 1 <= position.leverage <= self.settings.max_leverage:
                violations.append(
                    f"{symbol}:leverage={position.leverage}:allowed=1-{self.settings.max_leverage}"
                )
        if violations:
            self.control.engage_kill_switch("unsafe_binance_account_configuration")
            detail = ", ".join(violations)
            raise BinanceSafetyError(
                "Binance account violates one-way/isolated/leverage safety policy: "
                f"{detail}; startup remains locked"
            )

    def websocket_runtime(
        self,
        *,
        symbols: Sequence[str] | None = None,
        on_event: EventHandler | None = None,
        on_reconcile_required: ReconcileHandler | None = None,
    ) -> FuturesWebSocketRuntime:
        async def reconcile(reason: str) -> None:
            self.reconcile()
            if on_reconcile_required is not None:
                result = on_reconcile_required(reason)
                if hasattr(result, "__await__"):
                    await result

        subscribed = tuple(symbols or self.settings.futures_universe)
        self.streams.set_required_market_symbols(subscribed)
        return FuturesWebSocketRuntime(
            client=self.client,
            ws_base_url=self.settings.binance_ws_base_url,
            symbols=subscribed,
            state=self.streams,
            on_event=on_event,
            on_reconcile_required=reconcile,
        )

    def close(self) -> None:
        self.decision_provider.close()
        self.client.close()
        self.audit.close()


def build_binance_approval_runtime(
    settings: Settings,
    *,
    control: TradingControl | None = None,
    audit: AuditRepository | None = None,
    stream_state: FuturesStreamState | None = None,
) -> BinanceApprovalRuntime:
    """Explicit external-venue factory; merely importing the API never calls it."""

    if settings.execution_venue == "binance_futures_demo":
        if settings.trading_stage != "demo":
            raise BinanceSafetyError("Binance Demo requires TRADING_STAGE=demo")
        environment = "demo"
        base_url = settings.binance_futures_demo_url
        ws_url = settings.binance_futures_demo_ws_url
    elif settings.execution_venue == "binance_futures_live":
        if settings.trading_stage not in {"canary", "scaled", "live"}:
            raise BinanceSafetyError("Binance Live requires TRADING_STAGE=canary, scaled, or live")
        if not settings.production_trading_unlocked:
            raise BinanceSafetyError("all three static live gates must be enabled")
        if not settings.audit_database_url.startswith(("postgresql://", "postgres://")):
            raise BinanceSafetyError("live trading requires PostgreSQL audit storage")
        if not settings.control_api_token:
            raise BinanceSafetyError("live trading requires CONTROL_API_TOKEN")
        if not settings.openai_project:
            raise BinanceSafetyError("live trading requires a dedicated OPENAI_PROJECT")
        environment = "production"
        base_url = settings.binance_futures_live_url
        ws_url = settings.binance_futures_live_ws_url
    else:
        raise BinanceSafetyError(
            "external runtime requires binance_futures_demo or binance_futures_live"
        )
    try:
        base_url, _ = validate_binance_runtime_urls(
            rest_url=base_url,
            ws_url=ws_url,
            environment=environment,
        )
        validate_openai_base_url(settings.openai_base_url)
    except SecurityBoundaryError as error:
        raise BinanceSafetyError(str(error)) from error
    if not settings.binance_credentials_ready:
        raise BinanceSafetyError("Binance API credentials are required")
    if not settings.openai_credentials_ready:
        raise BinanceSafetyError("OpenAI API credentials are required")

    runtime_control = control or TradingControl(settings)
    runtime_audit = audit or AuditRepository(settings.audit_database_url)
    runtime_audit.initialize()
    client = BinanceFuturesClient(
        settings.binance_api_key,
        settings.binance_api_secret,
        base_url=base_url,
        environment=environment,
        # Runtime unlock is applied per intent by BinanceFuturesExecutionGateway.
        allow_production_trading=False,
        max_leverage=settings.max_leverage,
        recv_window_ms=settings.binance_recv_window_ms,
    )
    streams = stream_state or FuturesStreamState(
        required_market_symbols=settings.futures_universe,
        market_stale_after_seconds=settings.market_data_max_age_seconds,
    )

    def stream_ready() -> bool:
        return streams.health().ready_for_new_orders

    source_name = "binance_futures_live" if environment == "production" else "binance_futures_demo"
    account_source = BinanceFuturesAccountSource(
        client,
        audit=runtime_audit,
        source=source_name,
        private_stream_ready=stream_ready,
    )
    market_data = BinanceFuturesMarketDataProvider(client)
    gateway = BinanceFuturesExecutionGateway(
        client,
        settings=settings,
        control=runtime_control,
        quote_provider=market_data.quote,
        stream_ready=stream_ready,
    )
    provider = OpenAIResponsesDecisionProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_decision_model,
        base_url=settings.openai_base_url,
        project=settings.openai_project,
        timeout_seconds=settings.openai_request_timeout_seconds,
        allow_web_search=False,
        x_content_to_openai_allowed=settings.x_content_to_openai_allowed,
    )
    approvals = ApprovalTradingService(
        settings=settings,
        decision_provider=provider,
        account_source=account_source,
        gateway=gateway,
        audit=runtime_audit,
        control=runtime_control,
    )
    runtime = BinanceApprovalRuntime(
        settings=settings,
        control=runtime_control,
        audit=runtime_audit,
        client=client,
        streams=streams,
        market_data=market_data,
        account_source=account_source,
        gateway=gateway,
        decision_provider=provider,
        approvals=approvals,
    )
    approvals.reconciliation_hook = runtime.reconcile
    return runtime
