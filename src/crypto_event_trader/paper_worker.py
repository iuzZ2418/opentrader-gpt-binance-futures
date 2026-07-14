from __future__ import annotations

import asyncio
import signal
import socket
from datetime import UTC, date, datetime
from typing import Any

from .config import Settings
from .distributed_control import RedisTradingControl
from .futures_portfolio import FuturesAccountSnapshot
from .intelligence_worker import EvidenceInbox, build_evidence_inbox_queue
from .learning_runtime import LearningPromotionScheduler, build_champion_strategy
from .openai_research import OpenAIStrategyResearcher
from .paper_runtime import build_paper_approval_runtime, validate_paper_runtime_settings
from .strategy_registry import StrategyRegistry
from .trading_cycle import FuturesTradingCycle, aligned_cycle_delay
from .universe import DynamicUniverseManager, LiquidityObservationStore
from .worker import (
    _consume_intelligence,
    _external_evidence_for_symbols,
    _persist_and_enforce_risk,
    _position_symbols,
    _renew_leader_lease,
    _sync_cycle_champion,
    _wait_for_cycle_trigger,
    _watch_control,
)

PAPER_PROTECTIVE_POLL_SECONDS = 1.0


class _PaperProtectiveCoverageError(RuntimeError):
    """A stop filled but its closed-episode funding seal still needs retrying."""


def _paper_protective_poll_once(runtime: Any) -> int:
    """Mark and enforce audited paper stops without waiting for a strategy cycle."""

    reference = datetime.now(UTC)
    quotes = runtime.mark_positions()
    results = runtime.enforce_protective_stops(quotes, now=reference)
    if results:
        # Closing creates a new closed episode. Seal public funding coverage before the
        # mutex is released so performance can never observe a half-accounted close.
        try:
            runtime.sync_funding(now=datetime.now(UTC))
        except Exception as error:
            raise _PaperProtectiveCoverageError(
                "protective close funding coverage is not sealed"
            ) from error
        runtime.persist_account_snapshot(observed_at=datetime.now(UTC))
    return len(results)


async def _poll_paper_protective_stops(
    runtime: Any,
    stop: asyncio.Event,
    *,
    poll_seconds: float = PAPER_PROTECTIVE_POLL_SECONDS,
) -> None:
    if poll_seconds <= 0:
        raise ValueError("paper protective poll interval must be positive")
    loop = asyncio.get_running_loop()
    coverage_recovery_required = False
    while not stop.is_set():
        started = loop.time()
        try:
            if coverage_recovery_required:
                await asyncio.to_thread(
                    runtime.run_serialized,
                    runtime.sync_funding,
                    now=datetime.now(UTC),
                )
                await asyncio.to_thread(
                    runtime.run_serialized,
                    runtime.persist_account_snapshot,
                    observed_at=datetime.now(UTC),
                )
                coverage_recovery_required = False
            await asyncio.to_thread(
                runtime.run_serialized,
                _paper_protective_poll_once,
                runtime,
            )
        except _PaperProtectiveCoverageError:
            coverage_recovery_required = True
            runtime.control.engage_kill_switch("paper_protective_poll_unavailable")
        except Exception:
            # A stale/unavailable public mark or unresolved paper close blocks all new risk,
            # but the loop remains alive and keeps attempting deterministic protection.
            runtime.control.engage_kill_switch("paper_protective_poll_unavailable")
        elapsed = loop.time() - started
        delay = max(0.05, poll_seconds - elapsed)
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except TimeoutError:
            pass


async def run_paper_worker() -> None:
    """Run the automatic, public-data-only internal paper trading process."""

    settings = Settings.from_env()
    # Validate before touching Redis, OpenAI, Binance, or the audit database.
    validate_paper_runtime_settings(settings)
    control = RedisTradingControl(settings, lock_on_start=False)
    leader = control.client.lock(
        f"{control.key}:worker-leader", timeout=120, blocking_timeout=0
    )
    acquired = await asyncio.to_thread(leader.acquire, blocking=False)
    if not acquired:
        control.close()
        raise RuntimeError("another paper worker owns the distributed leader lease")

    runtime = None
    researcher: OpenAIStrategyResearcher | None = None
    evidence_queue: Any | None = None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    lease_task = asyncio.create_task(_renew_leader_lease(leader, stop, control))
    try:
        runtime = build_paper_approval_runtime(settings, control=control)
        await asyncio.to_thread(runtime.startup_check)
        registry = StrategyRegistry(settings.strategy_registry_file())
        researcher = OpenAIStrategyResearcher(
            api_key=settings.openai_api_key,
            model=settings.openai_research_model,
            project=settings.openai_project,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_request_timeout_seconds,
            allow_web_search=bool(settings.web_search_allowed_domains),
            allowed_search_domains=settings.web_search_allowed_domains,
            x_content_to_openai_allowed=settings.x_content_to_openai_allowed,
        )
        try:
            exact_model = getattr(researcher, "model", None) == settings.openai_research_model
            research_ready = exact_model and await asyncio.to_thread(
                researcher.check_model_access
            )
            if not research_ready:
                raise RuntimeError("exact OpenAI research model is not accessible")
        except Exception:
            control.engage_kill_switch("paper_research_model_unavailable")
            researcher.close()
            researcher = None

        learning = LearningPromotionScheduler(
            audit=runtime.audit,
            registry=registry,
            market_data=runtime.market_data,
            researcher=researcher,
            performance_venue=runtime.gateway.venue,
            state_path=settings.strategy_registry_file().with_suffix(
                ".paper-learning-state.json"
            ),
        )
        cycle = FuturesTradingCycle(
            settings=settings,
            market_data=runtime.market_data,
            strategy=build_champion_strategy(registry),
            approvals=runtime.approvals,
        )
        store = LiquidityObservationStore(runtime.audit)
        store.initialize()
        universe = DynamicUniverseManager(
            client=runtime.client,
            store=store,
            # No fallback is configured: 30 complete point-in-time dates are mandatory.
            fallback_symbols=(),
            coverage_days=30,
        )
        evidence_inbox = EvidenceInbox()
        await asyncio.to_thread(evidence_inbox.hydrate_from_audit, runtime.audit)
        evidence_queue = build_evidence_inbox_queue(
            settings,
            consumer=f"paper-trader-{socket.gethostname()}",
        )
        intelligence_wakeup = asyncio.Event()
        intelligence_task = asyncio.create_task(
            _consume_intelligence(
                evidence_queue,
                evidence_inbox,
                intelligence_wakeup,
                stop,
            )
        )
        control_task = asyncio.create_task(_watch_control(runtime, stop))
        protective_task = asyncio.create_task(
            _poll_paper_protective_stops(runtime, stop)
        )
        last_collection_day: date | None = None
        selected_symbols: tuple[str, ...] = ()

        def maintain_and_refresh_universe() -> tuple[tuple[str, ...], FuturesAccountSnapshot]:
            nonlocal last_collection_day
            now = datetime.now(UTC)
            # A UTC-day baseline must use current marks. If public marking fails, execution
            # leaves the prior durable baseline untouched instead of resetting it with stale PnL.
            quotes = runtime.mark_positions()
            funding_complete = True
            try:
                runtime.sync_funding(now=now)
            except Exception:
                # Exact accounting is a hard entry gate, but deterministic marks/stops stay live.
                funding_complete = False
                runtime.control.engage_kill_switch("paper_funding_accounting_unresolved")
            if funding_complete:
                runtime.portfolio.roll_day(now.date())
            stopped = runtime.enforce_protective_stops(quotes, now=now)
            if stopped:
                runtime.sync_funding(now=datetime.now(UTC))
            account = _persist_and_enforce_risk(runtime)
            expected_notional = max(
                5.0,
                account.equity
                * settings.capital_allocation_fraction
                * min(0.10, settings.max_asset_exposure),
            )
            if last_collection_day != now.date():
                universe.collect_daily(
                    as_of=now,
                    expected_order_notional=expected_notional,
                )
                last_collection_day = now.date()
            selection = universe.select_weekly(as_of=now, allow_fallback=False)
            return selection.symbols, account

        try:
            selected_symbols, _ = await asyncio.to_thread(
                runtime.run_serialized,
                maintain_and_refresh_universe,
            )
        except Exception:
            # No fabricated fallback universe is permitted. The process remains alive for
            # control, marking, deterministic exits, and a later public-data recovery.
            selected_symbols = ()

        last_cycle_started = 0.0
        while not stop.is_set():
            trigger = await _wait_for_cycle_trigger(
                stop,
                intelligence_wakeup,
                timeout=aligned_cycle_delay(
                    interval_seconds=settings.decision_cycle_seconds
                ),
            )
            if trigger == "STOP":
                continue
            if trigger == "INTELLIGENCE":
                cooldown = max(0.0, 60.0 - (loop.time() - last_cycle_started))
                if cooldown:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=cooldown)
                    except TimeoutError:
                        pass
                    if stop.is_set():
                        continue
            last_cycle_started = loop.time()
            try:
                selected_symbols, account = await asyncio.to_thread(
                    runtime.run_serialized,
                    maintain_and_refresh_universe,
                )
            except Exception:
                runtime.control.engage_kill_switch("paper_market_maintenance_unavailable")
                account = await asyncio.to_thread(
                    runtime.run_serialized,
                    runtime.portfolio.snapshot,
                )
                selection = universe.select_weekly(
                    as_of=datetime.now(UTC), allow_fallback=False
                )
                selected_symbols = selection.symbols
            learning_result = await asyncio.to_thread(
                learning.tick,
                now=datetime.now(UTC),
            )
            if learning_result.performance_status == "FAIL_CLOSED":
                runtime.control.engage_kill_switch(
                    "authoritative_performance_accounting_failed"
                )
            _sync_cycle_champion(cycle, registry)
            cycle_symbols = tuple(
                dict.fromkeys((*selected_symbols, *_position_symbols(account)))
            )
            evidence_by_symbol = _external_evidence_for_symbols(
                evidence_inbox,
                cycle_symbols,
                now=datetime.now(UTC),
            )

            def run_cycle_and_finalize(
                symbols: tuple[str, ...],
                evidence: dict[str, Any],
            ) -> None:
                cycle.run_once(
                    symbols,
                    external_evidence=evidence,
                )
                try:
                    runtime.sync_funding(now=datetime.now(UTC))
                except Exception:
                    runtime.control.engage_kill_switch(
                        "paper_funding_accounting_unresolved"
                    )
                runtime.persist_account_snapshot(observed_at=datetime.now(UTC))

            await asyncio.to_thread(
                runtime.run_serialized,
                run_cycle_and_finalize,
                cycle_symbols,
                evidence_by_symbol,
            )
    finally:
        stop.set()
        for task in (
            locals().get("control_task"),
            locals().get("intelligence_task"),
            locals().get("protective_task"),
            lease_task,
        ):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(
                task
                for task in (
                    locals().get("control_task"),
                    locals().get("intelligence_task"),
                    locals().get("protective_task"),
                    lease_task,
                )
                if task is not None
            ),
            return_exceptions=True,
        )
        if runtime is not None:
            runtime.close()
        if researcher is not None:
            researcher.close()
        if evidence_queue is not None:
            try:
                evidence_queue.client.close()
            except Exception:
                pass
        try:
            await asyncio.to_thread(leader.release)
        except Exception:
            pass
        control.close()


def main() -> None:
    asyncio.run(run_paper_worker())


if __name__ == "__main__":
    main()
