from __future__ import annotations

import hmac
import json
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from .config import Settings
from .control import TradingControlSnapshot


class RedisTradingControl:
    """Shared safety latch for separately deployed API and worker processes."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        key: str | None = None,
        lock_on_start: bool = True,
    ) -> None:
        if client is None:
            try:
                import redis
            except ImportError as error:  # pragma: no cover - optional production dependency
                raise RuntimeError("Redis control requires the trader extra") from error
            client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.settings = settings
        self.client = client
        self.key = key or f"trader:control:{settings.trading_stage}:{settings.execution_venue}"
        self.lock_key = f"{self.key}:lock"
        with self._lock():
            state = self._read_unlocked()
            if state is None:
                state = self._default_state()
            if lock_on_start and settings.trading_stage in {"canary", "scaled", "live"}:
                state["runtime_live_unlocked"] = False
                state["reason"] = "process_startup_locked"
                state["changed_at"] = datetime.now(UTC).isoformat()
            self._write_unlocked(state)

    def snapshot(self) -> TradingControlSnapshot:
        state = self._read_unlocked() or self._default_state()
        freeze_until = self._datetime(state.get("freeze_until"))
        changed_at = self._datetime(state.get("changed_at")) or datetime.now(UTC)
        is_live = self.settings.trading_stage in {"canary", "scaled", "live"}
        enabled = not bool(state["kill_switch_active"]) and (
            not is_live
            or (
                self.settings.production_trading_unlocked
                and bool(state["runtime_live_unlocked"])
            )
        )
        return TradingControlSnapshot(
            stage=self.settings.trading_stage,
            new_positions_enabled=enabled,
            runtime_live_unlocked=bool(state["runtime_live_unlocked"]),
            kill_switch_active=bool(state["kill_switch_active"]),
            reason=str(state["reason"]),
            changed_at=changed_at,
            freeze_until=freeze_until,
            permanent_risk_lock=bool(state.get("permanent_risk_lock", False)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.snapshot())

    def unlock_live(self, token: str) -> TradingControlSnapshot:
        if not self.settings.production_trading_unlocked:
            raise PermissionError("production trading is not enabled by configuration")
        self._require_token(token)
        with self._lock():
            state = self._state()
            if state["kill_switch_active"]:
                raise RuntimeError("kill switch must be reset before unlocking live trading")
            state["runtime_live_unlocked"] = True
            state["reason"] = "authenticated_manual_unlock"
            state["changed_at"] = datetime.now(UTC).isoformat()
            self._write_unlocked(state)
        return self.snapshot()

    def engage_kill_switch(self, reason: str) -> TradingControlSnapshot:
        with self._lock():
            state = self._state()
            state["kill_switch_active"] = True
            state["runtime_live_unlocked"] = False
            state["reason"] = reason.strip() or "manual_kill_switch"
            state["changed_at"] = datetime.now(UTC).isoformat()
            self._write_unlocked(state)
        return self.snapshot()

    def engage_risk_lock(
        self, reason: str, *, at: datetime | None = None
    ) -> TradingControlSnapshot:
        reference = (at or datetime.now(UTC)).astimezone(UTC)
        with self._lock():
            state = self._state()
            state["kill_switch_active"] = True
            state["runtime_live_unlocked"] = False
            state["reason"] = reason
            if reason == "daily_loss_limit":
                next_day = (reference + timedelta(days=1)).date()
                state["freeze_until"] = datetime.combine(
                    next_day, datetime.min.time(), UTC
                ).isoformat()
            elif reason == "total_drawdown_limit":
                state["permanent_risk_lock"] = True
            state["changed_at"] = reference.isoformat()
            self._write_unlocked(state)
        return self.snapshot()

    def reset_kill_switch(self, token: str) -> TradingControlSnapshot:
        self._require_token(token)
        with self._lock():
            state = self._state()
            now = datetime.now(UTC)
            freeze_until = self._datetime(state.get("freeze_until"))
            if freeze_until is not None and now < freeze_until:
                raise RuntimeError(
                    f"daily loss freeze remains active until {freeze_until.isoformat()}"
                )
            state.update(
                {
                    "kill_switch_active": False,
                    "runtime_live_unlocked": False,
                    "freeze_until": None,
                    "permanent_risk_lock": False,
                    "reason": "kill_switch_reset_live_remains_locked",
                    "changed_at": now.isoformat(),
                }
            )
            self._write_unlocked(state)
        return self.snapshot()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def _require_token(self, token: str) -> None:
        expected = self.settings.control_api_token
        if not expected or not hmac.compare_digest(token, expected):
            raise PermissionError("invalid control token")

    def _state(self) -> dict[str, Any]:
        return self._read_unlocked() or self._default_state()

    def _read_unlocked(self) -> dict[str, Any] | None:
        raw = self.client.get(self.key)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not raw:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("invalid distributed trading control state")
        return value

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        self.client.set(self.key, json.dumps(state, sort_keys=True, separators=(",", ":")))

    def _lock(self) -> Any:
        lock = getattr(self.client, "lock", None)
        if callable(lock):
            return lock(self.lock_key, timeout=10, blocking_timeout=5)
        return nullcontext()

    def _default_state(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "runtime_live_unlocked": False,
            "kill_switch_active": False,
            "reason": (
                "startup_locked"
                if self.settings.trading_stage in {"canary", "scaled", "live"}
                else "ready"
            ),
            "changed_at": now.isoformat(),
            "freeze_until": None,
            "permanent_risk_lock": False,
        }

    @staticmethod
    def _datetime(value: Any) -> datetime | None:
        if not value:
            return None
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
