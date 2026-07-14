from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from pydantic import JsonValue

from .approval import ApprovalResult, ApprovalTradingService
from .config import Settings
from .contracts import CandleInterval, PositionThesis, TradeCandidate, TradeDirection
from .market_data import (
    BinanceFuturesMarketDataProvider,
    DerivativesRiskOverlay,
    DerivativesRiskSnapshot,
)
from .market_lineage import MarketEvidenceBundle, MarketLineageRecorder
from .strategy import TrendBreakoutStrategy, average_true_range


@dataclass(frozen=True, slots=True)
class SymbolCycleResult:
    symbol: str
    status: str
    approval: ApprovalResult | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TradingCycleResult:
    started_at: datetime
    completed_at: datetime
    symbols: tuple[SymbolCycleResult, ...]
    emergency_exits: tuple[ApprovalResult, ...] = ()


class FuturesTradingCycle:
    """Fifteen-minute champion cycle using closed 1h/4h data only."""

    def __init__(
        self,
        *,
        settings: Settings,
        market_data: BinanceFuturesMarketDataProvider,
        strategy: TrendBreakoutStrategy,
        approvals: ApprovalTradingService,
        overlay: DerivativesRiskOverlay | None = None,
    ) -> None:
        self.settings = settings
        self.market_data = market_data
        self.strategy = strategy
        self.approvals = approvals
        self.overlay = overlay or DerivativesRiskOverlay()
        self.market_lineage = MarketLineageRecorder(approvals.audit)
        self._last_vote_strength: dict[str, int] = {}

    def run_once(
        self,
        symbols: Sequence[str],
        *,
        external_evidence: Mapping[str, Sequence[Mapping[str, JsonValue]]] | None = None,
        now: datetime | None = None,
    ) -> TradingCycleResult:
        started = (now or datetime.now(UTC)).astimezone(UTC)
        normalized = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        quotes: dict[str, Any] = {}
        symbol_results: list[SymbolCycleResult] = []
        for symbol in normalized:
            try:
                quotes[symbol] = self.market_data.quote(symbol)
            except Exception as error:
                symbol_results.append(
                    SymbolCycleResult(symbol, "MARKET_DATA_UNAVAILABLE", reason=str(error))
                )

        emergency = self.approvals.emergency_close_all(quotes, now=now)
        if emergency:
            return TradingCycleResult(
                started_at=started,
                completed_at=datetime.now(UTC),
                symbols=tuple(symbol_results),
                emergency_exits=emergency,
            )

        account = self.approvals.account_source.snapshot()
        positions = {
            str(item["symbol"]).upper(): item for item in account.positions
        }
        for symbol in normalized:
            if symbol not in quotes:
                continue
            try:
                quote = quotes[symbol]
                if now is None:
                    # The emergency pass above used a coherent snapshot. Refetch immediately
                    # before each approval so a slow ten-symbol cycle cannot reuse an old quote.
                    quote = self.market_data.quote(symbol)
                result = self._run_symbol(
                    symbol,
                    quote=quote,
                    equity=account.equity,
                    raw_position=positions.get(symbol),
                    external_evidence=(external_evidence or {}).get(symbol, ()),
                    now=now,
                )
            except Exception as error:
                result = SymbolCycleResult(symbol, "FAIL_CLOSED", reason=str(error))
            symbol_results.append(result)
        return TradingCycleResult(
            started_at=started,
            completed_at=datetime.now(UTC),
            symbols=tuple(symbol_results),
        )

    def run_forever(
        self,
        symbols: Sequence[str],
        *,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            self.run_once(symbols)
            stop_event.wait(self.settings.decision_cycle_seconds)

    def _run_symbol(
        self,
        symbol: str,
        *,
        quote: Any,
        equity: float,
        raw_position: Mapping[str, Any] | None,
        external_evidence: Sequence[Mapping[str, JsonValue]],
        now: datetime | None,
    ) -> SymbolCycleResult:
        hourly = self.market_data.closed_bars(symbol, CandleInterval.ONE_HOUR, 722)
        # Binance includes the current open candle; request one extra so 126-day Donchian
        # still has 127 fully closed four-hour bars after filtering.
        four_hour = self.market_data.closed_bars(symbol, CandleInterval.FOUR_HOURS, 128)
        expected_notional = max(5.0, equity * min(0.10, self.settings.max_asset_exposure))
        derivatives = self.market_data.derivatives_snapshot(
            symbol, expected_order_notional=expected_notional
        )
        reference = (now or datetime.now(UTC)).astimezone(UTC)
        overlay = self.overlay.classify(derivatives)
        quantity_cap = equity * self.settings.max_asset_exposure / quote.last
        candidate = self.strategy.generate_candidate(
            symbol=symbol,
            hourly_bars=hourly,
            four_hour_bars=four_hour,
            quantity_cap=quantity_cap,
            risk_regime=overlay.regime,
            now=reference,
        )
        thesis = self.approvals.current_thesis(symbol, mark_price=quote.last)
        signal_strengthening = False
        if candidate is not None:
            strength = max(
                int(candidate.feature_snapshot.get("long_votes", 0)),
                int(candidate.feature_snapshot.get("short_votes", 0)),
            )
            prior_strength = self._last_vote_strength.get(symbol)
            signal_strengthening = prior_strength is not None and strength > prior_strength
            self._last_vote_strength[symbol] = strength

        if raw_position is not None:
            direction = (
                TradeDirection.LONG
                if float(raw_position["quantity"]) > 0
                else TradeDirection.SHORT
            )
            if thesis is None:
                # A recovered live position may be reduced/closed, but never added without its
                # append-only thesis and initial R state.
                thesis = PositionThesis(
                    symbol=symbol,
                    direction=direction,
                    entry_reason="Recovered position; add disabled until thesis reconciliation",
                    expected_horizon_minutes=15,
                    add_count=1,
                )
            opposing_candidate = (
                candidate
                if candidate is not None and candidate.direction is not direction
                else None
            )
            if candidate is None or opposing_candidate is not None:
                candidate = self._management_candidate(
                    symbol=symbol,
                    direction=direction,
                    quantity=abs(float(raw_position["quantity"])),
                    hourly=hourly,
                    derivatives=derivatives,
                    overlay_reasons=overlay.reason_codes,
                    opposing_candidate=opposing_candidate,
                    now=reference,
                )
                signal_strengthening = False
            elif candidate.direction is direction:
                candidate = candidate.model_copy(
                    update={
                        "max_risk_fraction": min(
                            candidate.max_risk_fraction,
                            self.settings.add_position_risk,
                        )
                    }
                )
        elif candidate is None:
            return SymbolCycleResult(symbol, "NO_SIGNAL")

        market_inputs = (
            self.market_lineage.record_closed_bars(hourly, collected_at=reference),
            self.market_lineage.record_closed_bars(four_hour, collected_at=reference),
            self.market_lineage.record_execution_quote(quote, collected_at=reference),
            self.market_lineage.record_derivatives_snapshot(
                derivatives, collected_at=reference
            ),
        )
        candidate = candidate.model_copy(
            update={
                "feature_snapshot": {
                    **candidate.feature_snapshot,
                    "market_input_lineage": {
                        "hourly_bars": market_inputs[0].feature_reference,
                        "four_hour_bars": market_inputs[1].feature_reference,
                        "approval_quote": market_inputs[2].feature_reference,
                        "derivatives_risk": market_inputs[3].feature_reference,
                    },
                }
            }
        )
        market_model_evidence = [item.gpt_evidence for item in market_inputs]
        candidate = self._bind_approval_context(
            candidate,
            derivatives=derivatives,
            overlay_reasons=overlay.reason_codes,
            external_evidence=(*market_model_evidence, *external_evidence),
        )
        evidence = [
            self._strategy_evidence(candidate),
            *market_model_evidence,
            *external_evidence,
        ]
        raw_evidence = [
            self._strategy_evidence(candidate),
            *(self._raw_market_evidence(item) for item in market_inputs),
            *external_evidence,
        ]
        approval = self.approvals.review_candidate(
            candidate,
            quote=quote,
            evidence=evidence,
            raw_evidence=raw_evidence,
            position=thesis,
            signal_strengthening=signal_strengthening,
            now=now,
        )
        return SymbolCycleResult(symbol, approval.status, approval=approval)

    @staticmethod
    def _raw_market_evidence(bundle: MarketEvidenceBundle) -> dict[str, JsonValue]:
        raw = dict(bundle.gpt_evidence)
        raw["payload"] = bundle.audit_evidence
        return raw

    @staticmethod
    def _bind_approval_context(
        candidate: TradeCandidate,
        *,
        derivatives: DerivativesRiskSnapshot,
        overlay_reasons: Sequence[str],
        external_evidence: Sequence[Mapping[str, JsonValue]],
    ) -> TradeCandidate:
        """Bind the cache/idempotency key to every material approval input.

        Candidate creation time and ``bar_cutoff`` are deliberately excluded: they move on
        each retry inside the same decision bucket even when the closed bars and evidence are
        identical. Evidence object order is also non-semantic. Everything that can change the
        model's view -- source version/content, derived features, or derivatives state -- is
        canonicalized into the digest.
        """

        feature_snapshot = {
            key: value
            for key, value in candidate.feature_snapshot.items()
            if key
            not in {
                "approval_context_digest",
                "base_candidate_id",
                "bar_cutoff",
            }
        }
        base_candidate_id = str(
            candidate.feature_snapshot.get("base_candidate_id") or candidate.candidate_id
        )
        canonical_external_evidence = sorted(
            {
                json.dumps(
                    dict(item),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                for item in external_evidence
            }
        )
        context = {
            "schema": "approval-context-v1",
            "candidate": {
                "base_candidate_id": base_candidate_id,
                "strategy_version": candidate.strategy_version,
                "symbol": candidate.symbol,
                "direction": candidate.direction.value,
                "max_quantity": candidate.max_quantity,
                "max_risk_fraction": candidate.max_risk_fraction,
                "feature_snapshot": feature_snapshot,
            },
            "derivatives": {
                "symbol": derivatives.symbol,
                "mark_price": derivatives.mark_price,
                "index_price": derivatives.index_price,
                "funding_rate": derivatives.funding_rate,
                "open_interest": derivatives.open_interest,
                "open_interest_change_24h_fraction": (
                    derivatives.open_interest_change_24h_fraction
                ),
                "adl_quantile": derivatives.adl_quantile,
                "spread_bps": derivatives.spread_bps,
                "depth_within_20bps": derivatives.depth_within_20bps,
                "expected_order_notional": derivatives.expected_order_notional,
                "observed_at": derivatives.observed_at.isoformat(),
                "overlay_reasons": sorted(set(overlay_reasons)),
            },
            "external_evidence": canonical_external_evidence,
        }
        serialized = json.dumps(
            context,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        digest = sha256(serialized.encode("utf-8")).hexdigest()
        approval_identity = f"{base_candidate_id}|approval-context-v1|{digest}"
        bound_snapshot = dict(candidate.feature_snapshot)
        bound_snapshot.update(
            {
                "base_candidate_id": base_candidate_id,
                "approval_context_digest": digest,
            }
        )
        return candidate.model_copy(
            update={
                "candidate_id": (
                    f"cand_{sha256(approval_identity.encode('utf-8')).hexdigest()[:40]}"
                ),
                "feature_snapshot": bound_snapshot,
            }
        )

    def _management_candidate(
        self,
        *,
        symbol: str,
        direction: TradeDirection,
        quantity: float,
        hourly: Sequence[Any],
        derivatives: DerivativesRiskSnapshot,
        overlay_reasons: Sequence[str],
        opposing_candidate: TradeCandidate | None = None,
        now: datetime,
    ) -> TradeCandidate:
        atr = average_true_range(hourly, 14)
        return TradeCandidate(
            candidate_id=self._management_candidate_id(
                symbol=symbol,
                direction=direction,
                hourly_close=hourly[-1].close_time,
                now=now,
            ),
            strategy_version=f"{self.strategy.spec.version}-position-management",
            symbol=symbol,
            direction=direction,
            max_quantity=quantity,
            max_risk_fraction=self.settings.add_position_risk,
            feature_snapshot={
                "position_management_only": True,
                "atr_1h": atr,
                "mark_price": derivatives.mark_price,
                "funding_rate": derivatives.funding_rate,
                "basis_fraction": derivatives.basis_fraction,
                "overlay_reasons": list(overlay_reasons),
                "opposing_signal": opposing_candidate is not None,
                "opposing_signal_direction": (
                    opposing_candidate.direction.value if opposing_candidate is not None else None
                ),
                "opposing_signal_candidate_id": (
                    opposing_candidate.candidate_id if opposing_candidate is not None else None
                ),
                "opposing_signal_strategy_version": (
                    opposing_candidate.strategy_version if opposing_candidate is not None else None
                ),
                "opposing_signal_votes": (
                    opposing_candidate.feature_snapshot.get("votes")
                    if opposing_candidate is not None
                    else None
                ),
                "opposing_signal_long_votes": (
                    opposing_candidate.feature_snapshot.get("long_votes")
                    if opposing_candidate is not None
                    else None
                ),
                "opposing_signal_short_votes": (
                    opposing_candidate.feature_snapshot.get("short_votes")
                    if opposing_candidate is not None
                    else None
                ),
            },
            created_at=now,
            expires_at=now + timedelta(seconds=self.settings.candidate_ttl_seconds),
        )

    def _management_candidate_id(
        self,
        *,
        symbol: str,
        direction: TradeDirection,
        hourly_close: datetime,
        now: datetime,
    ) -> str:
        cycle_bucket = int(now.timestamp()) // self.settings.decision_cycle_seconds
        identity = "|".join(
            (
                self.strategy.spec.version,
                "position-management",
                symbol,
                direction.value,
                hourly_close.isoformat(),
                str(cycle_bucket),
            )
        )
        return f"cand_{sha256(identity.encode('utf-8')).hexdigest()[:40]}"

    @staticmethod
    def _strategy_evidence(candidate: TradeCandidate) -> dict[str, JsonValue]:
        return {
            "evidence_id": f"strategy:{candidate.candidate_id}",
            "source": "champion_strategy",
            "first_observed_at": candidate.created_at.isoformat(),
            "payload": candidate.feature_snapshot,
        }

    @staticmethod
    def _derivatives_evidence(
        snapshot: DerivativesRiskSnapshot, reasons: Sequence[str]
    ) -> dict[str, JsonValue]:
        observed = int(snapshot.observed_at.timestamp() * 1_000)
        return {
            "evidence_id": f"binance:{snapshot.symbol}:derivatives:{observed}",
            "source": "binance_futures",
            "first_observed_at": snapshot.observed_at.isoformat(),
            "mark_price": snapshot.mark_price,
            "index_price": snapshot.index_price,
            "funding_rate": snapshot.funding_rate,
            "open_interest": snapshot.open_interest,
            "open_interest_change_24h_fraction": (
                snapshot.open_interest_change_24h_fraction
            ),
            "adl_quantile": snapshot.adl_quantile,
            "spread_bps": snapshot.spread_bps,
            "depth_multiple": snapshot.depth_multiple,
            "risk_overlay_reasons": list(reasons),
        }


def aligned_cycle_delay(
    *, now: datetime | None = None, interval_seconds: int = 900
) -> float:
    """Seconds to the next UTC interval boundary, useful for an external worker."""

    reference = (now or datetime.now(UTC)).astimezone(UTC)
    epoch = reference.timestamp()
    return interval_seconds - epoch % interval_seconds


def wait_until_next_cycle(interval_seconds: int = 900) -> None:
    time.sleep(aligned_cycle_delay(interval_seconds=interval_seconds))
