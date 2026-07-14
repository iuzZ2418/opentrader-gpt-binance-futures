from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any
from urllib.parse import urlsplit

from pydantic import JsonValue

SAFE_SOURCE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
SOURCE_LABEL_KEYS = frozenset({"source", "source_type", "platform", "provider"})
OPENAI_API_HOSTS = frozenset({"api.openai.com"})
DEEPSEEK_API_HOSTS = frozenset({"api.deepseek.com"})
BINANCE_DEMO_REST_HOSTS = frozenset(
    {"demo-fapi.binance.com", "testnet.binancefuture.com"}
)
BINANCE_LIVE_REST_HOSTS = frozenset(
    {"fapi.binance.com", "fapi1.binance.com", "fapi2.binance.com", "fapi3.binance.com"}
)
BINANCE_DEMO_WS_HOSTS = frozenset({"demo-fstream.binance.com"})
BINANCE_LIVE_WS_HOSTS = frozenset({"fstream.binance.com"})


class SecurityBoundaryError(ValueError):
    """Untrusted data or an outbound destination exceeded a hard boundary."""


def is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    segments = set(normalized.split("_"))
    fragments = (
        "api_key",
        "apikey",
        "api_secret",
        "private_key",
        "signing_key",
        "webhook_secret",
        "access_token",
        "bearer_token",
        "authorization",
        "password",
        "passwd",
        "credential",
        "cookie",
        "session_secret",
        "binance_secret",
    )
    return (
        normalized in {"secret", "token", "session", "cookie"}
        or bool(
            segments
            & {
                "secret",
                "token",
                "password",
                "passwd",
                "credential",
                "credentials",
                "cookie",
                "session",
            }
        )
        or any(fragment in normalized for fragment in fragments)
    )


def safe_source_label(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    stripped = value.strip()
    if not SAFE_SOURCE_LABEL.fullmatch(stripped):
        return "unknown"
    return stripped.lower()


def looks_like_x_document(value: Mapping[str, Any]) -> bool:
    for key in SOURCE_LABEL_KEYS:
        label = safe_source_label(value.get(key))
        if label in {"x", "twitter", "x_official_api", "x_post"} or label.startswith(
            ("x:", "x_", "x-", "twitter:", "twitter_", "twitter-")
        ):
            return True
    for key in ("url", "source_url"):
        raw_url = value.get(key)
        if not isinstance(raw_url, str):
            continue
        try:
            host = (urlsplit(raw_url).hostname or "").lower()
        except ValueError:
            continue
        if host in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}:
            return True
    return False


def sanitize_untrusted_json(
    value: Any,
    *,
    max_depth: int = 8,
    max_mapping_items: int = 64,
    max_sequence_items: int = 64,
    max_string_chars: int = 4_000,
    max_nodes: int = 2_000,
    max_bytes: int = 64_000,
) -> JsonValue:
    """Recursively redact secrets and bound data before it crosses an LLM boundary.

    Shape limits truncate individual containers and strings. A node or encoded-byte overflow
    rejects the packet instead of silently dropping evidence and changing its meaning.
    """

    if min(
        max_depth,
        max_mapping_items,
        max_sequence_items,
        max_string_chars,
        max_nodes,
        max_bytes,
    ) <= 0:
        raise ValueError("sanitizer limits must be positive")
    nodes = 0

    def visit(item: Any, depth: int) -> JsonValue:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise SecurityBoundaryError("untrusted_json_node_limit_exceeded")
        if depth > max_depth:
            return "[DEPTH_LIMIT]"
        if isinstance(item, Mapping):
            result: dict[str, JsonValue] = {}
            for raw_key, child in islice(item.items(), max_mapping_items):
                key = str(raw_key)[:128]
                if not key or key in result:
                    continue
                if is_sensitive_key(key):
                    result[key] = "[REDACTED]"
                elif key.strip().lower() in SOURCE_LABEL_KEYS:
                    result[key] = safe_source_label(child)
                else:
                    result[key] = visit(child, depth + 1)
            return result
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [
                visit(child, depth + 1)
                for child in islice(iter(item), max_sequence_items)
            ]
        if item is None or isinstance(item, bool | int):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise SecurityBoundaryError("untrusted_json_non_finite_number")
            return item
        if isinstance(item, str):
            return item[:max_string_chars]
        return str(item)[: min(max_string_chars, 2_000)]

    sanitized = visit(value, 0)
    encoded = json.dumps(
        sanitized,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise SecurityBoundaryError("untrusted_json_byte_limit_exceeded")
    return sanitized


def validate_service_base_url(
    url: str,
    *,
    service: str,
    scheme: str,
    allowed_hosts: frozenset[str],
    allowed_paths: frozenset[str],
) -> str:
    """Require a canonical TLS endpoint with no redirectable URL components."""

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise SecurityBoundaryError(f"{service}_url_invalid") from error
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme.lower() != scheme
        or not host
        or host not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in allowed_paths
    ):
        raise SecurityBoundaryError(f"{service}_url_not_allowlisted")
    return url.rstrip("/")


def validate_openai_base_url(url: str) -> str:
    if (urlsplit(url).hostname or "").lower() in DEEPSEEK_API_HOSTS:
        return validate_service_base_url(
            url,
            service="deepseek",
            scheme="https",
            allowed_hosts=DEEPSEEK_API_HOSTS,
            allowed_paths=frozenset({"", "/"}),
        )
    return validate_service_base_url(
        url,
        service="openai",
        scheme="https",
        allowed_hosts=OPENAI_API_HOSTS,
        allowed_paths=frozenset({"/v1", "/v1/"}),
    )


def validate_binance_runtime_urls(
    *,
    rest_url: str,
    ws_url: str,
    environment: str,
) -> tuple[str, str]:
    if environment == "demo":
        rest_hosts = BINANCE_DEMO_REST_HOSTS
        ws_hosts = BINANCE_DEMO_WS_HOSTS
    elif environment == "production":
        rest_hosts = BINANCE_LIVE_REST_HOSTS
        ws_hosts = BINANCE_LIVE_WS_HOSTS
    else:
        raise SecurityBoundaryError("binance_environment_invalid")
    return (
        validate_service_base_url(
            rest_url,
            service="binance_rest",
            scheme="https",
            allowed_hosts=rest_hosts,
            allowed_paths=frozenset({"", "/"}),
        ),
        validate_service_base_url(
            ws_url,
            service="binance_ws",
            scheme="wss",
            allowed_hosts=ws_hosts,
            allowed_paths=frozenset({"", "/"}),
        ),
    )
