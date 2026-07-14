from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .approval import ApprovalTradingService
from .audit import AuditRepository
from .config import Settings
from .control import TradingControl
from .openai_decision import OpenAIResponsesDecisionProvider
from .security import SecurityBoundaryError
from .strategy_registry import StrategyRegistry


@dataclass(slots=True)
class ApiRuntime:
    """Resources used by the control API, with explicit fail-closed state."""

    settings: Settings
    control: TradingControl
    audit: AuditRepository
    approval_service: ApprovalTradingService | None
    strategy_registry: StrategyRegistry | None = None
    decision_provider: Any | None = None
    errors: list[str] = field(default_factory=list)
    owns_audit: bool = False
    owns_decision_provider: bool = False
    model_access_verified: bool = False
    worker_managed_execution: bool = False

    @property
    def gateway_venue(self) -> str | None:
        service = self.approval_service
        gateway = getattr(service, "gateway", None) if service is not None else None
        venue = getattr(gateway, "venue", None)
        return str(venue) if venue else None

    def readiness(self) -> dict[str, Any]:
        service = self.approval_service
        account_source = getattr(service, "account_source", None) if service else None
        account_ready = getattr(account_source, "ready_for_new_orders", True)
        if callable(account_ready):
            account_ready = account_ready()
        model_ready = self.decision_provider is not None and self.model_access_verified
        control_snapshot = self.control.snapshot()
        reasons = list(self.errors)
        if not self.worker_managed_execution:
            if service is None:
                reasons.append("approval_runtime_unavailable")
            if not model_ready:
                reasons.append("decision_model_unavailable")
            if not account_ready:
                reasons.append("account_or_private_stream_not_ready")
        if (
            self.settings.execution_venue == "binance_futures_live"
            and not self.settings.production_trading_unlocked
        ):
            reasons.append("live_static_gates_closed")
        if not control_snapshot.new_positions_enabled:
            # The detailed operator/incident reason is available only from the authenticated
            # control endpoint.  Readiness is public and must not echo arbitrary operator text.
            reasons.append("control_locked")
        reasons = list(dict.fromkeys(reasons))
        return {
            "fail_closed": bool(reasons),
            "ready_for_new_positions": not reasons and not self.worker_managed_execution,
            "ready_for_position_management": (
                service is not None and not self.errors and not self.worker_managed_execution
            ),
            "worker_managed_execution": self.worker_managed_execution,
            "approval_runtime_available": service is not None,
            "decision_model_configured": model_ready,
            "decision_model_access_verified": self.model_access_verified,
            "account_and_private_stream_ready": bool(account_ready),
            "configured_venue": self.settings.execution_venue,
            "gateway_venue": self.gateway_venue,
            "reasons": reasons,
        }

    def close(self) -> None:
        if self.owns_decision_provider and self.decision_provider is not None:
            close = getattr(self.decision_provider, "close", None)
            if callable(close):
                close()
        if self.owns_audit:
            self.audit.close()


def build_api_runtime(
    settings: Settings,
    *,
    approval_service: ApprovalTradingService | None = None,
    audit: AuditRepository | None = None,
    control: TradingControl | None = None,
    worker_managed_execution: bool = False,
) -> ApiRuntime:
    """Build the internal-paper runtime and reject implicit venue substitutions.

    Binance Demo and Live services carry exchange state and therefore must be assembled by the
    dedicated worker and injected here. The API never turns an external stage into a paper stage
    behind the operator's back.
    """

    service_control = getattr(approval_service, "control", None)
    service_audit = getattr(approval_service, "audit", None)
    runtime_control = control or service_control or TradingControl(settings)
    runtime_audit = audit or service_audit or AuditRepository(settings.audit_database_url)
    runtime = ApiRuntime(
        settings=settings,
        control=runtime_control,
        audit=runtime_audit,
        approval_service=None,
        owns_audit=audit is None and service_audit is None,
        worker_managed_execution=worker_managed_execution,
    )
    if control is not None and service_control is not None and control is not service_control:
        runtime.errors.append("injected_runtime_control_mismatch")
    if audit is not None and service_audit is not None and audit is not service_audit:
        runtime.errors.append("injected_runtime_audit_mismatch")
    try:
        runtime_audit.initialize()
    except Exception as error:  # database failure must leave the API observable but locked
        runtime.errors.append(f"audit_unavailable:{type(error).__name__}")
        return runtime

    try:
        runtime.strategy_registry = StrategyRegistry(settings.strategy_registry_file())
    except Exception as error:
        runtime.errors.append(f"strategy_registry_unavailable:{type(error).__name__}")

    if worker_managed_execution:
        if approval_service is not None:
            runtime.errors.append("worker_managed_api_refuses_local_approval_service")
        return runtime

    if approval_service is not None:
        if runtime.errors:
            return runtime
        mismatch = _runtime_mismatch(settings, approval_service)
        if mismatch:
            runtime.errors.append(mismatch)
            return runtime
        runtime.approval_service = approval_service
        runtime.decision_provider = getattr(approval_service, "decision_provider", None)
        runtime.model_access_verified = _verify_model_access(
            runtime.decision_provider, runtime.errors
        )
        if not runtime.model_access_verified:
            runtime.approval_service = None
        return runtime

    if settings.execution_venue != "internal" or settings.trading_stage not in {
        "backtest",
        "paper",
    }:
        runtime.errors.append(
            "external_runtime_requires_explicit_matching_approval_service"
        )
        return runtime

    try:
        provider = OpenAIResponsesDecisionProvider(
            api_key=settings.openai_api_key,
            project=settings.openai_project,
            model=settings.openai_decision_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_request_timeout_seconds,
            allow_web_search=False,
            x_content_to_openai_allowed=settings.x_content_to_openai_allowed,
        )
    except SecurityBoundaryError as error:
        runtime.errors.append(f"openai_url_security_boundary:{error}")
        return runtime
    runtime.decision_provider = provider
    runtime.owns_decision_provider = True
    runtime.model_access_verified = _verify_model_access(provider, runtime.errors)
    if not runtime.model_access_verified and provider.api_key:
        return runtime
    try:
        runtime.approval_service = ApprovalTradingService.paper(
            settings=settings,
            decision_provider=provider,
            audit=runtime_audit,
            control=runtime_control,
        )
    except Exception as error:
        runtime.errors.append(f"approval_runtime_unavailable:{type(error).__name__}")
    return runtime


def _verify_model_access(provider: Any | None, errors: list[str]) -> bool:
    if provider is None:
        return False
    if not isinstance(provider, OpenAIResponsesDecisionProvider):
        # An injected deterministic/test provider has no remote model capability to probe.
        return True
    if not provider.api_key:
        return False
    try:
        verified = provider.check_model_access()
    except Exception as error:
        errors.append(f"decision_model_access_check_failed:{type(error).__name__}")
        return False
    if not verified:
        errors.append("decision_model_access_check_failed:not_accessible")
        return False
    return True


def _runtime_mismatch(
    settings: Settings, approval_service: ApprovalTradingService
) -> str | None:
    service_settings = getattr(approval_service, "settings", None)
    if service_settings is not None and (
        service_settings.execution_venue != settings.execution_venue
        or service_settings.trading_stage != settings.trading_stage
    ):
        return "injected_runtime_settings_mismatch"
    gateway = getattr(approval_service, "gateway", None)
    actual = str(getattr(gateway, "venue", ""))
    expected = settings.execution_venue
    if expected == "internal":
        matches = actual == "internal-paper" and settings.trading_stage in {
            "backtest",
            "paper",
        }
    elif expected == "binance_futures_demo":
        matches = actual == expected and settings.trading_stage == "demo"
    elif expected == "binance_futures_live":
        matches = actual == expected and settings.trading_stage in {
            "canary",
            "scaled",
            "live",
        }
    else:
        matches = False
    return None if matches else f"gateway_venue_mismatch:{expected}:{actual or 'missing'}"
