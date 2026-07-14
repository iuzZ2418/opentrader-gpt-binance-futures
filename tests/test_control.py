from dataclasses import replace

import pytest

from crypto_event_trader.config import Settings
from crypto_event_trader.control import TradingControl


def test_live_stage_requires_environment_and_runtime_unlock() -> None:
    settings = replace(
        Settings.from_env(),
        trading_stage="live",
        live_trading_enabled=True,
        allow_binance_production=True,
        control_api_token="secret",
    )
    control = TradingControl(settings)

    assert control.snapshot().new_positions_enabled is False
    assert control.unlock_live("secret").new_positions_enabled is True
    assert control.engage_kill_switch("test").new_positions_enabled is False


def test_live_capital_stages_require_exact_allocation_fraction() -> None:
    base = Settings.from_env()
    with pytest.raises(ValueError, match="CANARY requires"):
        replace(
            base,
            trading_stage="canary",
            execution_venue="binance_futures_live",
            capital_allocation_fraction=1.0,
        )

    canary = replace(
        base,
        trading_stage="canary",
        execution_venue="binance_futures_live",
        capital_allocation_fraction=0.10,
        live_trading_enabled=True,
        allow_binance_production=True,
        control_api_token="secret",
    )
    assert canary.production_trading_unlocked is True
    assert TradingControl(canary).snapshot().new_positions_enabled is False


def test_live_unlock_rejects_bad_token_and_kill_reset_stays_locked() -> None:
    settings = replace(
        Settings.from_env(),
        trading_stage="live",
        live_trading_enabled=True,
        allow_binance_production=True,
        control_api_token="secret",
    )
    control = TradingControl(settings)

    with pytest.raises(PermissionError):
        control.unlock_live("wrong")
    control.engage_kill_switch("risk_limit")
    assert control.reset_kill_switch("secret").runtime_live_unlocked is False


def test_paper_stage_is_enabled_but_killable() -> None:
    control = TradingControl(replace(Settings.from_env(), trading_stage="paper"))
    assert control.snapshot().new_positions_enabled is True
    assert control.engage_kill_switch("operator").new_positions_enabled is False
