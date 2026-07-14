from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    app_env: str
    database_url: str
    audit_database_url: str
    redis_url: str
    trading_stage: str
    live_trading_enabled: bool
    allow_binance_production: bool
    control_api_token: str | None
    initial_cash: float
    asset_universe: tuple[str, ...]
    futures_universe: tuple[str, ...]
    min_signal_score: float
    risk_per_trade: float
    max_asset_exposure: float
    max_open_positions: int
    daily_drawdown_limit: float
    taker_fee_bps: float
    base_slippage_bps: float
    execution_venue: str
    coingecko_api_key: str | None
    binance_api_key: str | None
    binance_api_secret: str | None
    binance_futures_demo_url: str
    binance_futures_live_url: str
    binance_futures_demo_ws_url: str
    binance_futures_live_ws_url: str
    binance_recv_window_ms: int
    openai_api_key: str | None
    openai_project: str | None
    openai_base_url: str
    openai_extraction_model: str
    openai_decision_model: str
    openai_research_model: str
    openai_request_timeout_seconds: float
    x_content_to_openai_allowed: bool
    x_bearer_token: str | None
    x_allowed_account_ids: tuple[str, ...]
    github_token: str | None
    github_webhook_secret: str | None
    github_allowed_repositories: tuple[str, ...]
    web_search_allowed_domains: tuple[str, ...]
    strategy_registry_path: str
    max_gross_exposure: float
    capital_episode_loss_limit: float
    market_data_max_age_seconds: int
    max_spread_bps: float
    max_net_exposure: float
    max_correlation_cluster_exposure: float
    max_leverage: int
    initial_position_risk: float
    add_position_risk: float
    total_drawdown_limit: float
    decision_open_confidence: float
    decision_add_confidence: float
    candidate_ttl_seconds: int
    decision_cycle_seconds: int
    entry_order_wait_seconds: float
    entry_price_protection_bps: float
    capital_allocation_fraction: float
    intelligence_stream_name: str
    intelligence_poll_seconds: float
    github_poll_limit: int
    x_stream_reconnect_seconds: float
    intelligence_evidence_ttl_seconds: int

    def __post_init__(self) -> None:
        if self.trading_stage not in {
            "backtest",
            "paper",
            "demo",
            "canary",
            "scaled",
            "live",
        }:
            raise ValueError(f"Unsupported TRADING_STAGE: {self.trading_stage}")
        if not 0 < self.initial_position_risk <= self.risk_per_trade:
            raise ValueError("INITIAL_POSITION_RISK must be positive and <= RISK_PER_TRADE")
        if self.initial_position_risk > 0.0075:
            raise ValueError("INITIAL_POSITION_RISK cannot exceed the 0.75% hard budget")
        if not 0 <= self.add_position_risk <= 0.0025:
            raise ValueError("ADD_POSITION_RISK must be between 0 and the 0.25% hard budget")
        if self.initial_position_risk + self.add_position_risk > self.risk_per_trade:
            raise ValueError("initial and add risk budgets exceed RISK_PER_TRADE")
        if not 1 <= self.max_leverage <= 3 or not 0 < self.max_gross_exposure <= 3:
            raise ValueError("leverage and exposure limits must be positive")
        hard_caps = {
            "RISK_PER_TRADE": (self.risk_per_trade, 0.01),
            "DAILY_DRAWDOWN_LIMIT": (self.daily_drawdown_limit, 0.03),
            "TOTAL_DRAWDOWN_LIMIT": (self.total_drawdown_limit, 0.20),
            "MAX_NET_EXPOSURE": (self.max_net_exposure, 1.5),
            "MAX_ASSET_EXPOSURE": (self.max_asset_exposure, 0.5),
            "MAX_CORRELATION_CLUSTER_EXPOSURE": (
                self.max_correlation_cluster_exposure,
                1.0,
            ),
        }
        for name, (value, maximum) in hard_caps.items():
            if not 0 < value <= maximum:
                raise ValueError(f"{name} must be positive and cannot exceed {maximum}")
        if not 1 <= self.max_open_positions <= 10:
            raise ValueError("MAX_OPEN_POSITIONS must be between 1 and the 10-symbol hard cap")
        if not 0 < self.max_spread_bps <= 10:
            raise ValueError("MAX_SPREAD_BPS must be positive and cannot exceed 10")
        if not 0.70 <= self.decision_open_confidence <= 1:
            raise ValueError("DECISION_OPEN_CONFIDENCE cannot be below the 0.70 hard floor")
        if not 0.80 <= self.decision_add_confidence <= 1:
            raise ValueError("DECISION_ADD_CONFIDENCE cannot be below the 0.80 hard floor")
        if not 0 <= self.entry_order_wait_seconds <= 5:
            raise ValueError("ENTRY_ORDER_WAIT_SECONDS cannot exceed the five-second hard cap")
        if not 0 < self.entry_price_protection_bps <= 20:
            raise ValueError("ENTRY_PRICE_PROTECTION_BPS cannot exceed the 20 bps hard cap")
        if self.taker_fee_bps < 0 or self.base_slippage_bps < 0:
            raise ValueError("paper fee and slippage assumptions cannot be negative")
        if not 1 <= self.candidate_ttl_seconds <= 120:
            raise ValueError("CANDIDATE_TTL_SECONDS must be between 1 and 120")
        if self.decision_cycle_seconds <= 0:
            raise ValueError("DECISION_CYCLE_SECONDS must be positive")
        if self.trading_stage in {"demo", "canary", "scaled", "live"}:
            if self.decision_cycle_seconds != 900:
                raise ValueError(
                    "Demo/live trading stages require DECISION_CYCLE_SECONDS=900"
                )
            if not 1 <= self.market_data_max_age_seconds <= 10:
                raise ValueError(
                    "Demo/live trading stages require MARKET_DATA_MAX_AGE_SECONDS "
                    "between 1 and 10"
                )
        if self.capital_allocation_fraction not in {0.10, 0.25, 1.0}:
            raise ValueError(
                "CAPITAL_ALLOCATION_FRACTION must be one of 0.10, 0.25, or 1.0"
            )
        expected_stage_fraction = {"canary": 0.10, "scaled": 0.25, "live": 1.0}.get(
            self.trading_stage
        )
        if (
            self.execution_venue == "binance_futures_live"
            and expected_stage_fraction is not None
            and self.capital_allocation_fraction != expected_stage_fraction
        ):
            raise ValueError(
                f"{self.trading_stage.upper()} requires "
                f"CAPITAL_ALLOCATION_FRACTION={expected_stage_fraction}"
            )
        if not self.intelligence_stream_name.strip():
            raise ValueError("INTELLIGENCE_STREAM_NAME must be non-empty")
        if self.intelligence_poll_seconds <= 0 or self.x_stream_reconnect_seconds <= 0:
            raise ValueError("intelligence polling intervals must be positive")
        if not 1 <= self.github_poll_limit <= 100:
            raise ValueError("GITHUB_POLL_LIMIT must be between 1 and 100")
        if self.intelligence_evidence_ttl_seconds <= 0:
            raise ValueError("INTELLIGENCE_EVIDENCE_TTL_SECONDS must be positive")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            database_url=os.getenv("DATABASE_URL", "sqlite:///data/trader.db"),
            audit_database_url=os.getenv(
                "AUDIT_DATABASE_URL", "sqlite:///data/trader_audit.db"
            ),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            trading_stage=os.getenv("TRADING_STAGE", "paper").strip().lower(),
            live_trading_enabled=_bool("LIVE_TRADING_ENABLED", False),
            allow_binance_production=_bool("ALLOW_BINANCE_PRODUCTION", False),
            control_api_token=os.getenv("CONTROL_API_TOKEN") or None,
            initial_cash=_float("INITIAL_CASH", 100_000),
            asset_universe=tuple(
                item.strip().upper()
                for item in os.getenv("ASSET_UNIVERSE", "BTC,ETH,SOL").split(",")
                if item.strip()
            ),
            futures_universe=tuple(
                item.strip().upper()
                for item in os.getenv(
                    "FUTURES_UNIVERSE", "BTCUSDT,ETHUSDT,SOLUSDT"
                ).split(",")
                if item.strip()
            ),
            min_signal_score=_float("MIN_SIGNAL_SCORE", 0.82),
            risk_per_trade=_float("RISK_PER_TRADE", 0.01),
            max_asset_exposure=_float("MAX_ASSET_EXPOSURE", 0.50),
            max_open_positions=_int("MAX_OPEN_POSITIONS", 10),
            daily_drawdown_limit=_float("DAILY_DRAWDOWN_LIMIT", 0.03),
            taker_fee_bps=_float("TAKER_FEE_BPS", 8),
            base_slippage_bps=_float("BASE_SLIPPAGE_BPS", 3),
            execution_venue=os.getenv("EXECUTION_VENUE", "internal"),
            coingecko_api_key=os.getenv("COINGECKO_API_KEY") or None,
            binance_api_key=os.getenv("BINANCE_API_KEY") or None,
            binance_api_secret=os.getenv("BINANCE_API_SECRET") or None,
            binance_futures_demo_url=os.getenv(
                "BINANCE_FUTURES_DEMO_URL", "https://demo-fapi.binance.com"
            ).rstrip("/"),
            binance_futures_live_url=os.getenv(
                "BINANCE_FUTURES_LIVE_URL", "https://fapi.binance.com"
            ).rstrip("/"),
            binance_futures_demo_ws_url=os.getenv(
                "BINANCE_FUTURES_DEMO_WS_URL", "wss://demo-fstream.binance.com"
            ).rstrip("/"),
            binance_futures_live_ws_url=os.getenv(
                "BINANCE_FUTURES_LIVE_WS_URL", "wss://fstream.binance.com"
            ).rstrip("/"),
            binance_recv_window_ms=_int("BINANCE_RECV_WINDOW_MS", 5_000),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_project=os.getenv("OPENAI_PROJECT") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_extraction_model=os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-5.6-luna"),
            openai_decision_model=os.getenv("OPENAI_DECISION_MODEL", "gpt-5.6-terra"),
            openai_research_model=os.getenv("OPENAI_RESEARCH_MODEL", "gpt-5.6-sol"),
            openai_request_timeout_seconds=_float("OPENAI_REQUEST_TIMEOUT_SECONDS", 20),
            x_content_to_openai_allowed=_bool("X_CONTENT_TO_OPENAI_ALLOWED", False),
            x_bearer_token=os.getenv("X_BEARER_TOKEN") or None,
            x_allowed_account_ids=tuple(
                item.strip()
                for item in os.getenv("X_ALLOWED_ACCOUNT_IDS", "").split(",")
                if item.strip()
            ),
            github_token=os.getenv("GITHUB_TOKEN") or None,
            github_webhook_secret=os.getenv("GITHUB_WEBHOOK_SECRET") or None,
            github_allowed_repositories=tuple(
                item.strip()
                for item in os.getenv("GITHUB_ALLOWED_REPOSITORIES", "").split(",")
                if item.strip()
            ),
            web_search_allowed_domains=tuple(
                item.strip().lower()
                for item in os.getenv("WEB_SEARCH_ALLOWED_DOMAINS", "").split(",")
                if item.strip()
            ),
            strategy_registry_path=os.getenv(
                "STRATEGY_REGISTRY_PATH", "data/strategy_registry.json"
            ),
            max_gross_exposure=_float("MAX_GROSS_EXPOSURE", 3.0),
            capital_episode_loss_limit=_float("CAPITAL_EPISODE_LOSS_LIMIT", 0.05),
            market_data_max_age_seconds=_int("MARKET_DATA_MAX_AGE_SECONDS", 10),
            max_spread_bps=_float("MAX_SPREAD_BPS", 10),
            max_net_exposure=_float("MAX_NET_EXPOSURE", 1.5),
            max_correlation_cluster_exposure=_float(
                "MAX_CORRELATION_CLUSTER_EXPOSURE", 1.0
            ),
            max_leverage=_int("MAX_LEVERAGE", 3),
            initial_position_risk=_float("INITIAL_POSITION_RISK", 0.0075),
            add_position_risk=_float("ADD_POSITION_RISK", 0.0025),
            total_drawdown_limit=_float("TOTAL_DRAWDOWN_LIMIT", 0.20),
            decision_open_confidence=_float("DECISION_OPEN_CONFIDENCE", 0.70),
            decision_add_confidence=_float("DECISION_ADD_CONFIDENCE", 0.80),
            candidate_ttl_seconds=_int("CANDIDATE_TTL_SECONDS", 120),
            decision_cycle_seconds=_int("DECISION_CYCLE_SECONDS", 900),
            entry_order_wait_seconds=_float("ENTRY_ORDER_WAIT_SECONDS", 5),
            entry_price_protection_bps=_float("ENTRY_PRICE_PROTECTION_BPS", 20),
            capital_allocation_fraction=_float(
                "CAPITAL_ALLOCATION_FRACTION", 1.0
            ),
            intelligence_stream_name=os.getenv(
                "INTELLIGENCE_STREAM_NAME", "trader:external-evidence"
            ).strip(),
            intelligence_poll_seconds=_float("INTELLIGENCE_POLL_SECONDS", 300),
            github_poll_limit=_int("GITHUB_POLL_LIMIT", 30),
            x_stream_reconnect_seconds=_float("X_STREAM_RECONNECT_SECONDS", 5),
            intelligence_evidence_ttl_seconds=_int(
                "INTELLIGENCE_EVIDENCE_TTL_SECONDS", 3_600
            ),
        )

    @property
    def binance_credentials_ready(self) -> bool:
        return bool(self.binance_api_key and self.binance_api_secret)

    @property
    def openai_credentials_ready(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def production_trading_unlocked(self) -> bool:
        return (
            self.trading_stage in {"canary", "scaled", "live"}
            and self.live_trading_enabled
            and self.allow_binance_production
        )

    @property
    def binance_base_url(self) -> str:
        if self.production_trading_unlocked:
            return self.binance_futures_live_url
        return self.binance_futures_demo_url

    @property
    def binance_ws_base_url(self) -> str:
        if self.production_trading_unlocked:
            return self.binance_futures_live_ws_url
        return self.binance_futures_demo_ws_url

    def sqlite_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("The MVP runtime currently supports sqlite:/// URLs")
        raw = self.database_url.removeprefix(prefix)
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def strategy_registry_file(self) -> Path:
        path = Path(self.strategy_registry_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
