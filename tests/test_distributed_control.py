from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import pytest

from crypto_event_trader.config import Settings
from crypto_event_trader.distributed_control import RedisTradingControl


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def lock(self, *_: object, **__: object):  # type: ignore[no-untyped-def]
        return nullcontext()


def _live_settings() -> Settings:
    return replace(
        Settings.from_env(),
        trading_stage="live",
        execution_venue="binance_futures_live",
        live_trading_enabled=True,
        allow_binance_production=True,
        control_api_token="secret",
    )


def test_unlock_is_visible_to_other_process_instances() -> None:
    redis = FakeRedis()
    first = RedisTradingControl(_live_settings(), client=redis, lock_on_start=True)
    second = RedisTradingControl(_live_settings(), client=redis, lock_on_start=False)

    assert first.unlock_live("secret").new_positions_enabled is True
    assert second.snapshot().new_positions_enabled is True
    second.engage_kill_switch("operator")
    assert first.snapshot().new_positions_enabled is False


def test_any_live_process_restart_relocks_new_exposure() -> None:
    redis = FakeRedis()
    first = RedisTradingControl(_live_settings(), client=redis)
    first.unlock_live("secret")
    restarted = RedisTradingControl(_live_settings(), client=redis, lock_on_start=True)
    assert restarted.snapshot().runtime_live_unlocked is False


def test_wrong_token_cannot_reset_or_unlock() -> None:
    control = RedisTradingControl(_live_settings(), client=FakeRedis())
    with pytest.raises(PermissionError):
        control.unlock_live("wrong")
    control.engage_kill_switch("operator")
    with pytest.raises(PermissionError):
        control.reset_kill_switch("wrong")
