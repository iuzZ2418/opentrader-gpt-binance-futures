from __future__ import annotations

import hashlib
import hmac
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from crypto_event_trader.binance import (
    BinanceApiError,
    BinanceFuturesDemoClient,
    BinanceSafetyError,
    FuturesRestSnapshot,
    reconcile_rest_snapshot,
)
from crypto_event_trader.config import Settings
from crypto_event_trader.database import Repository
from crypto_event_trader.service import TradingService


def _transport(secret: str = "test-secret") -> httpx.MockTransport:
    prices = {"BTCUSDT": "65000", "ETHUSDT": "3500", "SOLUSDT": "150"}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1_000_000})
        if path == "/fapi/v1/ping":
            return httpx.Response(200, json={})
        if path == "/fapi/v1/exchangeInfo":
            return httpx.Response(
                200,
                json={
                    "symbols": [
                        {
                            "symbol": symbol,
                            "filters": [
                                {
                                    "filterType": "MARKET_LOT_SIZE",
                                    "minQty": "0.001",
                                    "stepSize": "0.001",
                                }
                            ],
                        }
                        for symbol in prices
                    ]
                },
            )
        if path == "/fapi/v1/ticker/bookTicker":
            symbol = request.url.params["symbol"]
            price = float(prices[symbol])
            return httpx.Response(
                200,
                json={
                    "symbol": symbol,
                    "bidPrice": str(price * 0.9999),
                    "askPrice": str(price * 1.0001),
                },
            )
        if path == "/fapi/v1/ticker/24hr":
            symbol = request.url.params["symbol"]
            return httpx.Response(
                200,
                json={
                    "symbol": symbol,
                    "lastPrice": prices[symbol],
                    "quoteVolume": "1000000000",
                },
            )
        if path == "/fapi/v1/order" and request.method == "POST":
            assert request.headers["X-MBX-APIKEY"] == "test-key"
            raw_query = request.url.query.decode("ascii")
            payload, signature = raw_query.rsplit("&signature=", 1)
            expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
            assert signature == expected
            params = request.url.params
            symbol = params["symbol"]
            return httpx.Response(
                200,
                json={
                    "orderId": 1000 + int(params["newClientOrderId"].split("-")[-1]),
                    "clientOrderId": params["newClientOrderId"],
                    "symbol": symbol,
                    "status": "FILLED",
                    "avgPrice": prices[symbol],
                    "executedQty": params["quantity"],
                    "cumQuote": str(float(prices[symbol]) * float(params["quantity"])),
                },
            )
        return httpx.Response(404, json={"code": -1, "msg": f"Unhandled {path}"})

    return httpx.MockTransport(handler)


def test_binance_quotes_and_quantity_normalization() -> None:
    client = BinanceFuturesDemoClient(
        "test-key",
        "test-secret",
        transport=_transport(),
        time_provider=lambda: 1000,
    )
    quotes = client.fetch_quotes({"SOL": "SOLUSDT"})
    assert quotes["SOL"].bid < quotes["SOL"].ask
    assert quotes["SOL"].volume_24h == 1_000_000_000
    assert client.normalize_quantity("SOLUSDT", 1.23456) == "1.234"


def test_binance_demo_order_is_signed_and_mirrored(tmp_path: Path) -> None:
    settings = replace(
        Settings.from_env(),
        database_url=f"sqlite:///{tmp_path / 'binance.db'}",
        execution_venue="binance_futures_demo",
        binance_api_key="test-key",
        binance_api_secret="test-secret",
    )
    client = BinanceFuturesDemoClient(
        settings.binance_api_key,
        settings.binance_api_secret,
        transport=_transport(),
        time_provider=lambda: 1000,
    )
    repository = Repository(settings.sqlite_path())
    service = TradingService(settings, repository, client)
    sample = Path(__file__).parents[1] / "data" / "sample_documents.json"

    result = service.run_sample_cycle(sample)
    orders = repository.list_orders()

    assert result.orders_filled == 2
    assert {order["venue"] for order in orders} == {"binance-futures-demo"}
    assert all(order["external_order_id"] for order in orders)
    assert all(order["raw_response"]["status"] == "FILLED" for order in orders)


def _rules_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fapi/v1/exchangeInfo":
            return httpx.Response(
                200,
                json={
                    "symbols": [
                        {
                            "symbol": "BTCUSDT",
                            "status": "TRADING",
                            "contractType": "PERPETUAL",
                            "filters": [
                                {
                                    "filterType": "PRICE_FILTER",
                                    "minPrice": "1",
                                    "maxPrice": "1000000",
                                    "tickSize": "0.10",
                                },
                                {
                                    "filterType": "LOT_SIZE",
                                    "minQty": "0.001",
                                    "maxQty": "100",
                                    "stepSize": "0.001",
                                },
                                {
                                    "filterType": "MARKET_LOT_SIZE",
                                    "minQty": "0.01",
                                    "maxQty": "10",
                                    "stepSize": "0.01",
                                },
                                {"filterType": "MIN_NOTIONAL", "notional": "5"},
                                {
                                    "filterType": "NOTIONAL",
                                    "minNotional": "10",
                                    "maxNotional": "1000",
                                    "applyMinToMarket": "false",
                                    "applyMaxToMarket": "true",
                                },
                            ],
                        }
                    ]
                },
            )
        return httpx.Response(404, json={"code": -1, "msg": "not found"})

    return httpx.MockTransport(handler)


def test_dynamic_symbol_rules_cover_price_lot_market_and_notional_filters() -> None:
    client = BinanceFuturesDemoClient(None, None, transport=_rules_transport())
    rules = client.symbol_rules("BTCUSDT")

    assert rules.min_notional == Decimal("10")
    assert rules.max_notional == Decimal("1000")
    assert rules.min_notional_applies_to_market is False
    assert client.normalize_price("BTCUSDT", Decimal("100.19")) == "100.10"
    assert client.normalize_quantity("BTCUSDT", Decimal("0.1299")) == "0.12"
    assert client.normalize_quantity("BTCUSDT", Decimal("0.1299"), market=False) == "0.129"
    assert rules.validate_notional("0.01", "1", market=True) == Decimal("0.01")
    with pytest.raises(BinanceApiError, match="below Binance minimum"):
        rules.validate_notional("0.01", "100", market=False)
    with pytest.raises(BinanceApiError, match="above Binance maximum"):
        rules.validate_notional("2", "600", market=True)


def test_rate_limit_headers_and_retry_after_are_exposed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={
                "X-MBX-USED-WEIGHT-1M": "2400",
                "X-MBX-ORDER-COUNT-10S": "50",
                "Retry-After": "7",
            },
            json={"code": -1003, "msg": "Too many requests"},
        )

    client = BinanceFuturesDemoClient(None, None, transport=httpx.MockTransport(handler))
    with pytest.raises(BinanceApiError) as caught:
        client.ping()

    assert caught.value.status_code == 429
    assert caught.value.retry_after_seconds == 7
    assert client.rate_limits.used_weight["1m"] == 2400
    assert client.rate_limits.order_count["10s"] == 50


def test_rate_limit_response_enforces_local_embargo_without_rehitting_binance() -> None:
    calls = 0
    clock = [1_000.0]

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"Retry-After": "7"},
            json={"code": -1003, "msg": "Too many requests"},
        )

    client = BinanceFuturesDemoClient(
        None,
        None,
        transport=httpx.MockTransport(handler),
        time_provider=lambda: clock[0],
    )
    with pytest.raises(BinanceApiError) as first:
        client.ping()
    with pytest.raises(BinanceApiError, match="embargo") as blocked:
        client.ping()

    assert first.value.status_code == 429
    assert blocked.value.retry_after_seconds == pytest.approx(7)
    assert calls == 1

    clock[0] += 7
    with pytest.raises(BinanceApiError):
        client.ping()
    assert calls == 2


def test_ip_ban_without_retry_header_uses_conservative_local_embargo() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(418, json={"code": -1003, "msg": "IP banned"})

    client = BinanceFuturesDemoClient(
        None,
        None,
        transport=httpx.MockTransport(handler),
        time_provider=lambda: 1_000,
    )
    with pytest.raises(BinanceApiError) as first:
        client.ping()
    with pytest.raises(BinanceApiError, match="embargo") as blocked:
        client.ping()

    assert first.value.status_code == 418
    assert blocked.value.retry_after_seconds == pytest.approx(300)
    assert calls == 1


def test_unknown_503_order_is_queried_by_unique_client_id_without_resubmission() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/fapi/v1/exchangeInfo":
            return _transport().handle_request(request)
        if request.url.path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1_000_000})
        if request.url.path == "/fapi/v1/order" and request.method == "POST":
            return httpx.Response(
                503,
                json={"code": -1000, "msg": "Unknown error, check execution status"},
            )
        if request.url.path == "/fapi/v1/order" and request.method == "GET":
            assert request.url.params["origClientOrderId"] == "cet-recover-1"
            return httpx.Response(
                200,
                json={
                    "symbol": "BTCUSDT",
                    "clientOrderId": "cet-recover-1",
                    "orderId": 7,
                    "status": "NEW",
                },
            )
        return httpx.Response(404, json={"code": -1, "msg": "not found"})

    client = BinanceFuturesDemoClient(
        "test-key",
        "test-secret",
        transport=httpx.MockTransport(handler),
        time_provider=lambda: 1000,
    )
    order = client.place_market_order(
        symbol="BTCUSDT", side="BUY", quantity=Decimal("0.01"), client_order_id="cet-recover-1"
    )
    repeated = client.place_market_order(
        symbol="BTCUSDT", side="BUY", quantity=Decimal("0.01"), client_order_id="cet-recover-1"
    )

    assert order["orderId"] == repeated["orderId"] == 7
    assert calls.count(("POST", "/fapi/v1/order")) == 1
    assert calls.count(("GET", "/fapi/v1/order")) == 2


def test_unknown_503_is_preserved_when_order_cannot_be_reconciled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fapi/v1/exchangeInfo":
            return _transport().handle_request(request)
        if request.url.path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1_000_000})
        if request.url.path == "/fapi/v1/order" and request.method == "POST":
            return httpx.Response(503, json={"code": -1000, "msg": "Unknown error"})
        return httpx.Response(400, json={"code": -2013, "msg": "Order does not exist"})

    client = BinanceFuturesDemoClient(
        "test-key",
        "test-secret",
        transport=httpx.MockTransport(handler),
        time_provider=lambda: 1000,
    )
    with pytest.raises(BinanceApiError) as caught:
        client.place_market_order(
            symbol="BTCUSDT", side="BUY", quantity=0.01, client_order_id="cet-unknown-1"
        )
    assert caught.value.execution_unknown is True
    assert caught.value.status_code == 503


def test_read_timeout_is_execution_unknown_and_forces_order_query() -> None:
    queried = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal queried
        if request.url.path == "/fapi/v1/exchangeInfo":
            return _transport().handle_request(request)
        if request.url.path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1_000_000})
        if request.url.path == "/fapi/v1/order" and request.method == "POST":
            raise httpx.ReadTimeout("response timed out", request=request)
        if request.url.path == "/fapi/v1/order" and request.method == "GET":
            queried = True
            return httpx.Response(400, json={"code": -2013, "msg": "Order does not exist"})
        return httpx.Response(404)

    client = BinanceFuturesDemoClient(
        "test-key",
        "test-secret",
        transport=httpx.MockTransport(handler),
        time_provider=lambda: 1000,
    )
    with pytest.raises(BinanceApiError) as caught:
        client.place_market_order(
            symbol="BTCUSDT", side="BUY", quantity=0.01, client_order_id="cet-timeout-1"
        )

    assert queried is True
    assert caught.value.execution_unknown is True


def test_account_configuration_order_and_algo_endpoints() -> None:
    requests: list[tuple[str, str, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, dict(request.url.params)))
        path = request.url.path
        if path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1_000_000})
        if path == "/fapi/v1/exchangeInfo":
            return _rules_transport().handle_request(request)
        if path == "/fapi/v1/leverage":
            return httpx.Response(200, json={"symbol": "BTCUSDT", "leverage": 3})
        if path == "/fapi/v1/marginType":
            return httpx.Response(200, json={"code": 200, "msg": "success"})
        if path == "/fapi/v1/positionSide/dual":
            if request.method == "GET":
                return httpx.Response(200, json={"dualSidePosition": False})
            return httpx.Response(200, json={"code": 200, "msg": "success"})
        if path == "/fapi/v1/order":
            return httpx.Response(200, json={"orderId": 1, "status": "NEW"})
        if path == "/fapi/v1/algoOrder":
            return httpx.Response(200, json={"algoId": 2, "algoStatus": "NEW"})
        return httpx.Response(404, json={"code": -1, "msg": "not found"})

    client = BinanceFuturesDemoClient(
        "test-key",
        "test-secret",
        transport=httpx.MockTransport(handler),
        time_provider=lambda: 1000,
    )
    client.set_leverage(symbol="BTCUSDT", leverage=3)
    client.set_margin_type(symbol="BTCUSDT")
    assert client.get_position_mode() is False
    client.set_position_mode(dual_side_position=False)
    client.place_limit_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity="0.2",
        price="100.19",
        client_order_id="cet-limit-1",
    )
    client.place_reduce_only_market_order(
        symbol="BTCUSDT", side="SELL", quantity="0.2", client_order_id="cet-close-1"
    )
    client.place_stop_market_algo_order(
        symbol="BTCUSDT",
        side="SELL",
        trigger_price="90.19",
        quantity="0.2",
        client_algo_id="cet-stop-1",
    )

    by_path = {path: params for _method, path, params in requests}
    assert by_path["/fapi/v1/leverage"]["leverage"] == "3"
    assert by_path["/fapi/v1/marginType"]["marginType"] == "ISOLATED"
    assert by_path["/fapi/v1/positionSide/dual"]["dualSidePosition"] == "false"
    order_requests = [entry for entry in requests if entry[1] == "/fapi/v1/order"]
    assert order_requests[0][2]["price"] == "100.10"
    assert order_requests[1][2]["reduceOnly"] == "true"
    assert by_path["/fapi/v1/algoOrder"]["type"] == "STOP_MARKET"
    assert by_path["/fapi/v1/algoOrder"]["triggerPrice"] == "90.10"


def test_rest_snapshot_and_reconciliation_types() -> None:
    snapshot = FuturesRestSnapshot(
        observed_at_ms=100,
        account={"totalWalletBalance": "100"},
        balances=(),
        positions=(
            BinanceFuturesDemoClient._parse_position(
                {"symbol": "BTCUSDT", "positionAmt": "0.1", "positionSide": "BOTH"}
            ),
        ),
        open_orders=(
            BinanceFuturesDemoClient._parse_open_order(
                {"symbol": "BTCUSDT", "orderId": 1, "clientOrderId": "cet-open-1"}
            ),
        ),
    )
    consistent = reconcile_rest_snapshot(
        snapshot,
        expected_open_client_ids={"cet-open-1"},
        expected_positions={"BTCUSDT": Decimal("0.1")},
    )
    mismatch = reconcile_rest_snapshot(
        snapshot,
        expected_open_client_ids=set(),
        expected_positions={"BTCUSDT": Decimal("0")},
    )

    assert consistent.consistent is True
    assert {issue.category for issue in mismatch.issues} == {
        "unexpected_open_order",
        "position_quantity_mismatch",
    }


def test_rest_snapshot_reads_account_positions_orders_and_trades() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1_000_000})
        if request.url.path == "/fapi/v3/account":
            return httpx.Response(200, json={"totalWalletBalance": "100", "assets": []})
        if request.url.path == "/fapi/v3/balance":
            return httpx.Response(
                200,
                json=[
                    {
                        "asset": "USDT",
                        "balance": "100",
                        "availableBalance": "90",
                        "crossWalletBalance": "0",
                        "crossUnPnl": "0",
                        "updateTime": 10,
                    }
                ],
            )
        if request.url.path == "/fapi/v3/positionRisk":
            return httpx.Response(
                200,
                json=[
                    {
                        "symbol": "BTCUSDT",
                        "positionSide": "BOTH",
                        "positionAmt": "0.1",
                        "entryPrice": "65000",
                        "markPrice": "65100",
                        "unRealizedProfit": "10",
                        "leverage": "3",
                        "marginType": "isolated",
                    }
                ],
            )
        if request.url.path == "/fapi/v1/openOrders":
            return httpx.Response(
                200,
                json=[
                    {
                        "symbol": "BTCUSDT",
                        "orderId": 1,
                        "clientOrderId": "cet-open-1",
                        "origQty": "0.1",
                        "executedQty": "0",
                    }
                ],
            )
        if request.url.path == "/fapi/v1/userTrades":
            assert request.url.params["symbol"] == "BTCUSDT"
            return httpx.Response(
                200,
                json=[
                    {
                        "symbol": "BTCUSDT",
                        "id": 2,
                        "orderId": 1,
                        "side": "BUY",
                        "price": "65000",
                        "qty": "0.1",
                        "quoteQty": "6500",
                        "realizedPnl": "0",
                        "commission": "2.6",
                        "commissionAsset": "USDT",
                        "time": 11,
                    }
                ],
            )
        return httpx.Response(404, json={"code": -1, "msg": "not found"})

    client = BinanceFuturesDemoClient(
        "test-key",
        "test-secret",
        transport=httpx.MockTransport(handler),
        time_provider=lambda: 1000,
    )
    snapshot = client.rest_snapshot(symbol="BTCUSDT", trade_symbols=["BTCUSDT"])

    assert snapshot.account["totalWalletBalance"] == "100"
    assert snapshot.balances[0].available_balance == Decimal("90")
    assert snapshot.positions[0].quantity == Decimal("0.1")
    assert snapshot.open_orders[0].client_order_id == "cet-open-1"
    assert snapshot.user_trades[0].trade_id == 2
    assert {
        "/fapi/v3/account",
        "/fapi/v3/balance",
        "/fapi/v3/positionRisk",
        "/fapi/v1/openOrders",
        "/fapi/v1/userTrades",
    }.issubset(paths)


def test_production_host_requires_explicit_environment_and_unlock() -> None:
    with pytest.raises(BinanceSafetyError, match="environment='production'"):
        BinanceFuturesDemoClient(None, None, base_url="https://fapi.binance.com")

    client = BinanceFuturesDemoClient(
        "key",
        "secret",
        base_url="https://fapi.binance.com",
        environment="production",
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
    )
    with pytest.raises(BinanceSafetyError, match="locked"):
        client.set_leverage(symbol="BTCUSDT", leverage=1)


def test_production_mutation_authorization_is_scoped_even_on_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1_000_000})
        if request.url.path == "/fapi/v1/leverage":
            return httpx.Response(200, json={"leverage": 1})
        return httpx.Response(500, json={"msg": "failure"})

    client = BinanceFuturesDemoClient(
        "key",
        "secret",
        base_url="https://fapi.binance.com",
        environment="production",
        transport=httpx.MockTransport(handler),
    )
    assert client.allow_production_trading is False
    with client.mutation_authorization():
        client.set_leverage(symbol="BTCUSDT", leverage=1)
        assert client.allow_production_trading is True
    assert client.allow_production_trading is False

    with pytest.raises(RuntimeError):
        with client.mutation_authorization():
            raise RuntimeError("interrupt mutation")
    assert client.allow_production_trading is False


def test_algo_query_cancel_and_open_order_endpoints() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/fapi/v1/time":
            return httpx.Response(200, json={"serverTime": 1_000_000})
        if request.url.path == "/fapi/v1/openAlgoOrders":
            return httpx.Response(200, json=[])
        if request.url.path == "/fapi/v1/algoOrder":
            status = "CANCELED" if request.method == "DELETE" else "NEW"
            return httpx.Response(200, json={"algoId": 7, "algoStatus": status})
        return httpx.Response(404, json={"code": -1, "msg": "not found"})

    client = BinanceFuturesDemoClient(
        "key", "secret", transport=httpx.MockTransport(handler)
    )
    assert client.query_algo_order(client_algo_id="audit-stop")["algoStatus"] == "NEW"
    assert (
        client.cancel_algo_order(client_algo_id="audit-stop")["algoStatus"]
        == "CANCELED"
    )
    assert client.open_algo_orders(symbol="BTCUSDT") == []
    assert requests == [
        ("GET", "/fapi/v1/time"),
        ("GET", "/fapi/v1/algoOrder"),
        ("DELETE", "/fapi/v1/algoOrder"),
        ("GET", "/fapi/v1/openAlgoOrders"),
    ]


def test_binance_rest_rejects_plaintext_or_credential_bearing_base_url() -> None:
    with pytest.raises(BinanceSafetyError, match="binance_rest_url_not_allowlisted"):
        BinanceFuturesDemoClient(
            "key", "secret", base_url="http://demo-fapi.binance.com"
        )
    with pytest.raises(BinanceSafetyError, match="binance_rest_url_not_allowlisted"):
        BinanceFuturesDemoClient(
            "key", "secret", base_url="https://user@demo-fapi.binance.com"
        )
