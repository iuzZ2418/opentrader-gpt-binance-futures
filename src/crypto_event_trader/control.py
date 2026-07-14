from __future__ import annotations

import hmac
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from .config import Settings


@dataclass(frozen=True, slots=True)
class TradingControlSnapshot:
    stage: str
    new_positions_enabled: bool
    runtime_live_unlocked: bool
    kill_switch_active: bool
    reason: str
    changed_at: datetime
    freeze_until: datetime | None = None
    permanent_risk_lock: bool = False


class TradingControl:
    """Process-local safety latch layered on top of environment configuration.

    A production process always starts locked. Environment configuration can make
    live trading *eligible*, but an authenticated runtime action is still required.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._runtime_live_unlocked = False
        self._kill_switch_active = False
        self._reason = (
            "startup_locked"
            if settings.trading_stage in {"canary", "scaled", "live"}
            else "ready"
        )
        self._changed_at = datetime.now(UTC)
        self._freeze_until: datetime | None = None
        self._permanent_risk_lock = False

    def snapshot(self) -> TradingControlSnapshot:
        with self._lock:
            is_live = self.settings.trading_stage in {"canary", "scaled", "live"}
            enabled = not self._kill_switch_active and (
                not is_live
                or (self.settings.production_trading_unlocked and self._runtime_live_unlocked)
            )
            return TradingControlSnapshot(
                stage=self.settings.trading_stage,
                new_positions_enabled=enabled,
                runtime_live_unlocked=self._runtime_live_unlocked,
                kill_switch_active=self._kill_switch_active,
                reason=self._reason,
                changed_at=self._changed_at,
                freeze_until=self._freeze_until,
                permanent_risk_lock=self._permanent_risk_lock,
            )

    def as_dict(self) -> dict:
        return asdict(self.snapshot())

    def unlock_live(self, token: str) -> TradingControlSnapshot:
        with self._lock:
            if not self.settings.production_trading_unlocked:
                raise PermissionError("production trading is not enabled by configuration")
            expected = self.settings.control_api_token
            if not expected:
                raise PermissionError("CONTROL_API_TOKEN is not configured")
            if not hmac.compare_digest(token, expected):
                raise PermissionError("invalid control token")
            if self._kill_switch_active:
                raise RuntimeError("kill switch must be reset before unlocking live trading")
            self._runtime_live_unlocked = True
            self._reason = "authenticated_manual_unlock"
            self._changed_at = datetime.now(UTC)
            return self.snapshot()

    def engage_kill_switch(self, reason: str) -> TradingControlSnapshot:
        with self._lock:
            self._kill_switch_active = True
            self._runtime_live_unlocked = False
            self._reason = reason.strip() or "manual_kill_switch"
            self._changed_at = datetime.now(UTC)
            return self.snapshot()

    def engage_risk_lock(
        self, reason: str, *, at: datetime | None = None
    ) -> TradingControlSnapshot:
        """Apply the plan's daily freeze or total-drawdown permanent live lock."""

        reference = (at or datetime.now(UTC)).astimezone(UTC)
        with self._lock:
            self._kill_switch_active = True
            self._runtime_live_unlocked = False
            self._reason = reason
            if reason == "daily_loss_limit":
                next_day = (reference + timedelta(days=1)).date()
                self._freeze_until = datetime.combine(next_day, datetime.min.time(), UTC)
            elif reason == "total_drawdown_limit":
                self._permanent_risk_lock = True
            self._changed_at = reference
            return self.snapshot()

    def reset_kill_switch(self, token: str) -> TradingControlSnapshot:
        with self._lock:
            expected = self.settings.control_api_token
            if not expected or not hmac.compare_digest(token, expected):
                raise PermissionError("invalid control token")
            now = datetime.now(UTC)
            if self._freeze_until is not None and now < self._freeze_until:
                raise RuntimeError(
                    f"daily loss freeze remains active until {self._freeze_until.isoformat()}"
                )
            self._kill_switch_active = False
            self._runtime_live_unlocked = False
            self._freeze_until = None
            self._permanent_risk_lock = False
            self._reason = "kill_switch_reset_live_remains_locked"
            self._changed_at = now
            return self.snapshot()
