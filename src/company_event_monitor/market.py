from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import httpx

from .domain import Company

EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
PRICE_SOURCE = "腾讯证券公开行情"
EASTMONEY_PRICE_SOURCE = "东方财富公开行情"
TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class PriceBar:
    trade_date: date
    open: float
    close: float
    high: float
    low: float
    volume: float
    amount: float
    amplitude: float = 0.0
    pct_change: float = 0.0
    turnover: float = 0.0
    source: str = PRICE_SOURCE

    def to_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["trade_date"] = self.trade_date.isoformat()
        return result


class EastMoneyMarketData:
    """Public A-share daily bars behind a replaceable market-data boundary."""

    def __init__(self, timeout: float = 20) -> None:
        self.timeout = timeout

    def fetch_company(self, company: Company, limit: int = 750) -> list[PriceBar]:
        return self.fetch_sec_id(_security_id(company), limit)

    def fetch_benchmark(self, company: Company, limit: int = 750) -> tuple[str, list[PriceBar]]:
        name, sec_id = _benchmark(company)
        return name, self.fetch_sec_id(sec_id, limit)

    def fetch_sec_id(self, sec_id: str, limit: int = 750) -> list[PriceBar]:
        params = {
            "secid": sec_id,
            "klt": "101",
            "fqt": "1",
            "lmt": str(limit),
            "end": "20500101",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 CompanyEventMonitor/0.6",
            "Referer": "https://quote.eastmoney.com/",
        }
        last_error = ""
        payload: dict[str, Any] = {}
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                    response = client.get(EASTMONEY_KLINE_URL, params=params, headers=headers)
                    response.raise_for_status()
                    payload = response.json()
                break
            except (httpx.HTTPError, ValueError) as error:
                last_error = str(error)
                if attempt < 2:
                    time.sleep(0.8 * (attempt + 1))
        if not payload:
            raise RuntimeError(f"行情源连续3次访问失败：{last_error}")
        data = payload.get("data") or {}
        rows = data.get("klines") or []
        if not rows:
            raise RuntimeError("行情源未返回有效日线数据")
        bars = [_parse_eastmoney_kline(row) for row in rows]
        return sorted(bars, key=lambda item: item.trade_date)


class TencentMarketData:
    """Tencent daily bars; used by default because it is stable in packaged Python."""

    def __init__(self, timeout: float = 20) -> None:
        self.timeout = timeout
        self.quotes: dict[str, dict[str, Any]] = {}

    def fetch_company(self, company: Company, limit: int = 750) -> list[PriceBar]:
        return self.fetch_code(_tencent_code(company), limit)

    def fetch_benchmark(self, company: Company, limit: int = 750) -> tuple[str, list[PriceBar]]:
        name, code = _tencent_benchmark(company)
        return name, self.fetch_code(code, limit)

    def quote_for_company(self, company: Company) -> dict[str, Any]:
        return self.quotes.get(_tencent_code(company), {})

    def fetch_code(self, code: str, limit: int = 750) -> list[PriceBar]:
        params = {"param": f"{code},day,,,{limit},qfq"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://gu.qq.com/",
        }
        last_error = ""
        payload: dict[str, Any] = {}
        for attempt in range(3):
            try:
                with httpx.Client(
                    timeout=self.timeout,
                    follow_redirects=True,
                    trust_env=False,
                ) as client:
                    response = client.get(TENCENT_KLINE_URL, params=params, headers=headers)
                    response.raise_for_status()
                    payload = response.json()
                break
            except (httpx.HTTPError, ValueError) as error:
                last_error = str(error)
                if attempt < 2:
                    time.sleep(0.8 * (attempt + 1))
        if not payload:
            raise RuntimeError(f"腾讯行情连续3次访问失败：{last_error}")
        data = (payload.get("data") or {}).get(code) or {}
        quote_values = (data.get("qt") or {}).get(code) or []
        self.quotes[code] = _parse_tencent_quote(quote_values)
        rows = data.get("qfqday") or data.get("day") or []
        if not rows:
            raise RuntimeError("腾讯行情未返回有效日线数据")
        bars: list[PriceBar] = []
        previous_close = 0.0
        for row in rows:
            bar = _parse_tencent_kline(row, previous_close)
            bars.append(bar)
            previous_close = bar.close
        return sorted(bars, key=lambda item: item.trade_date)


def update_market_analysis(
    repository: Any,
    company: Company,
    events: list[dict[str, Any]],
    *,
    provider: Any | None = None,
) -> dict[str, Any]:
    provider = provider or TencentMarketData()
    company_bars = provider.fetch_company(company)
    quote = provider.quote_for_company(company) if hasattr(provider, "quote_for_company") else {}
    benchmark_name, benchmark_bars = provider.fetch_benchmark(company)
    repository.upsert_price_bars(company.company_id, company_bars)
    analysis = analyze_market(
        company,
        company_bars,
        benchmark_name,
        benchmark_bars,
        events,
        quote=quote,
    )
    repository.save_market_analysis(company.company_id, analysis)
    return analysis


def analyze_market(
    company: Company,
    bars: list[PriceBar],
    benchmark_name: str,
    benchmark_bars: list[PriceBar],
    events: list[dict[str, Any]],
    *,
    quote: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(bars) < 30:
        raise ValueError("至少需要30个交易日才能生成价格分析")
    bars = sorted(bars, key=lambda item: item.trade_date)
    benchmark_bars = sorted(benchmark_bars, key=lambda item: item.trade_date)
    benchmark_by_date = {item.trade_date: item for item in benchmark_bars}
    aligned = [
        (item, benchmark_by_date[item.trade_date])
        for item in bars
        if item.trade_date in benchmark_by_date
    ]
    closes = [item.close for item in bars]
    returns = _daily_returns(closes)
    current = bars[-1]
    r5 = _period_return(closes, 5)
    r20 = _period_return(closes, 20)
    r60 = _period_return(closes, 60)
    benchmark_20 = _aligned_return(aligned, 20)
    excess_20 = r20 - benchmark_20
    volatility_20 = _annualized_volatility(returns[-20:])
    volatility_60 = _annualized_volatility(returns[-60:])
    drawdown_60 = _max_drawdown(closes[-61:])
    volume_ratio = _volume_ratio(bars)
    market_score, market_reasons = _market_signal(closes, aligned, bars)
    event_score, event_reasons = _event_signal(events)
    combined_score = max(-1.0, min(1.0, 0.78 * market_score + 0.22 * event_score))
    forecast = _forecast(closes, combined_score, event_score, market_score, aligned)
    event_links = _event_price_links(events, bars, benchmark_by_date)
    source_hash = hashlib.sha256(
        json.dumps(
            {
                "company": company.company_id,
                "as_of": current.trade_date.isoformat(),
                "close": current.close,
                "events": [item.get("id") for item in events[:30]],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "company_id": company.company_id,
        "ticker": company.ticker,
        "source": current.source,
        "as_of": current.trade_date.isoformat(),
        "latest_price": current.close,
        "latest_pct_change": current.pct_change / 100,
        "returns": {"5d": r5, "20d": r20, "60d": r60},
        "benchmark": {
            "name": benchmark_name,
            "20d_return": benchmark_20,
            "excess_20d": excess_20,
            "aligned_days": len(aligned),
        },
        "risk": {
            "annualized_volatility_20d": volatility_20,
            "annualized_volatility_60d": volatility_60,
            "max_drawdown_60d": drawdown_60,
            "volume_ratio_5d_vs_20d": volume_ratio,
        },
        "valuation": quote or {},
        "signals": {
            "market_score": round(market_score, 4),
            "event_score": round(event_score, 4),
            "combined_score": round(combined_score, 4),
            "reasons": [*market_reasons, *event_reasons][:8],
        },
        "forecast_20d": forecast,
        "event_price_links": event_links,
        "recent_prices": [item.to_record() for item in bars[-90:]],
        "data_hash": source_hash,
        "generated_at": datetime.now().astimezone().isoformat(),
        "disclaimer": (
            "情景预测基于历史价格、相对指数表现和已披露事件，不是目标价、买卖建议或收益承诺。"
        ),
    }


def _parse_eastmoney_kline(value: str) -> PriceBar:
    fields = value.split(",")
    if len(fields) < 11:
        raise ValueError("行情日线字段不完整")
    return PriceBar(
        trade_date=date.fromisoformat(fields[0]),
        open=float(fields[1]),
        close=float(fields[2]),
        high=float(fields[3]),
        low=float(fields[4]),
        volume=float(fields[5]),
        amount=float(fields[6]),
        amplitude=float(fields[7]),
        pct_change=float(fields[8]),
        turnover=float(fields[10]),
        source=EASTMONEY_PRICE_SOURCE,
    )


def _parse_tencent_kline(fields: list[str], previous_close: float) -> PriceBar:
    if len(fields) < 6:
        raise ValueError("腾讯行情日线字段不完整")
    open_price = float(fields[1])
    close = float(fields[2])
    high = float(fields[3])
    low = float(fields[4])
    volume = float(fields[5])
    pct_change = (close / previous_close - 1) * 100 if previous_close else 0.0
    amplitude = (high - low) / previous_close * 100 if previous_close else 0.0
    return PriceBar(
        trade_date=date.fromisoformat(fields[0]),
        open=open_price,
        close=close,
        high=high,
        low=low,
        volume=volume,
        amount=close * volume * 100,
        amplitude=amplitude,
        pct_change=pct_change,
        turnover=0.0,
        source=PRICE_SOURCE,
    )


def _parse_tencent_quote(values: list[Any]) -> dict[str, Any]:
    if len(values) < 50:
        return {}

    def number(index: int) -> float | None:
        try:
            text = str(values[index]).strip()
            return float(text) if text else None
        except (IndexError, TypeError, ValueError):
            return None

    return {
        "quote_time": str(values[30]) if len(values) > 30 else "",
        "turnover_rate": number(38),
        "pe_dynamic": number(39),
        "circulating_market_cap_yi": number(44),
        "total_market_cap_yi": number(45),
        "pb": number(46),
        "volume_ratio": number(49),
        "average_price": number(51),
        "pe_ttm": number(52),
        "currency": str(values[82]) if len(values) > 82 else "CNY",
        "note": "估值字段为行情源即时快照，仅用于研究背景，不直接进入方向预测。",
    }


def _security_id(company: Company) -> str:
    market = company.market.lower()
    if "沪" in company.market or market in {"sh", "sse", "shanghai"}:
        prefix = "1"
    elif "北" in company.market or market in {"bj", "bse", "beijing"}:
        prefix = "0"
    else:
        prefix = "1" if company.ticker.startswith(("5", "6")) else "0"
    return f"{prefix}.{company.ticker}"


def _tencent_code(company: Company) -> str:
    market = company.market.lower()
    if "北" in company.market or market in {"bj", "bse", "beijing"}:
        prefix = "bj"
    elif "沪" in company.market or market in {"sh", "sse", "shanghai"}:
        prefix = "sh"
    else:
        prefix = "sh" if company.ticker.startswith(("5", "6")) else "sz"
    return f"{prefix}{company.ticker}"


def _tencent_benchmark(company: Company) -> tuple[str, str]:
    market = company.market.lower()
    if "北" in company.market or market in {"bj", "bse", "beijing"}:
        return "北证50", "bj899050"
    if (
        "沪" in company.market
        or market in {"sh", "sse", "shanghai"}
        or company.ticker.startswith(("5", "6"))
    ):
        return "上证指数", "sh000001"
    return "深证成指", "sz399001"


def _benchmark(company: Company) -> tuple[str, str]:
    market = company.market.lower()
    if "北" in company.market or market in {"bj", "bse", "beijing"}:
        return "北证50", "0.899050"
    if (
        "沪" in company.market
        or market in {"sh", "sse", "shanghai"}
        or company.ticker.startswith(("5", "6"))
    ):
        return "上证指数", "1.000001"
    return "深证成指", "0.399001"


def _daily_returns(closes: list[float]) -> list[float]:
    return [
        closes[index] / closes[index - 1] - 1
        for index in range(1, len(closes))
        if closes[index - 1]
    ]


def _period_return(closes: list[float], days: int) -> float:
    if len(closes) <= days or not closes[-days - 1]:
        return 0.0
    return closes[-1] / closes[-days - 1] - 1


def _aligned_return(aligned: list[tuple[PriceBar, PriceBar]], days: int) -> float:
    if len(aligned) <= days:
        return 0.0
    start = aligned[-days - 1][1].close
    end = aligned[-1][1].close
    return end / start - 1 if start else 0.0


def _annualized_volatility(returns: list[float]) -> float:
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS) if len(returns) >= 2 else 0.0


def _max_drawdown(closes: list[float]) -> float:
    peak = closes[0]
    worst = 0.0
    for value in closes:
        peak = max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1)
    return worst


def _volume_ratio(bars: list[PriceBar]) -> float:
    if len(bars) < 25:
        return 1.0
    recent = statistics.mean(item.volume for item in bars[-5:])
    baseline = statistics.mean(item.volume for item in bars[-25:-5])
    return recent / baseline if baseline else 1.0


def _market_signal(
    closes: list[float],
    aligned: list[tuple[PriceBar, PriceBar]],
    bars: list[PriceBar],
) -> tuple[float, list[str]]:
    r20 = _period_return(closes, 20)
    r60 = _period_return(closes, 60)
    excess = r20 - _aligned_return(aligned, 20)
    price_vs_ma20 = closes[-1] / statistics.mean(closes[-20:]) - 1
    volume = _volume_ratio(bars)
    score = (
        0.30 * math.tanh(r20 / 0.10)
        + 0.25 * math.tanh(r60 / 0.20)
        + 0.25 * math.tanh(excess / 0.10)
        + 0.20 * math.tanh(price_vs_ma20 / 0.06)
    )
    if volume > 1.4:
        score *= 1.08
    reasons = [
        f"近20日收益{r20:+.1%}",
        f"近60日收益{r60:+.1%}",
        f"相对基准近20日超额{excess:+.1%}",
        f"最新价相对20日均线{price_vs_ma20:+.1%}",
        f"近5日成交量为此前20日的{volume:.2f}倍",
    ]
    return max(-1.0, min(1.0, score)), reasons


def _event_signal(events: list[dict[str, Any]]) -> tuple[float, list[str]]:
    if not events:
        return 0.0, ["当前窗口没有可用于价格联动的标准化事件"]
    weighted = 0.0
    total = 0.0
    escalation = 0
    for event in events[:30]:
        weight = max(0.2, float(event.get("value_score") or 0.5))
        direction = int(event.get("direction") or 0)
        if event.get("change_type") in {"escalation", "conflict"} and direction < 0:
            weight *= 1.25
            escalation += 1
        weighted += direction * weight
        total += weight
    score = weighted / total if total else 0.0
    positive = sum(int(item.get("direction") or 0) > 0 for item in events[:30])
    negative = sum(int(item.get("direction") or 0) < 0 for item in events[:30])
    return max(-1.0, min(1.0, score)), [
        f"高关注事件中正向{positive}项、负向{negative}项",
        f"负向升级或冲突{escalation}项",
    ]


def _forecast(
    closes: list[float],
    combined_score: float,
    event_score: float,
    market_score: float,
    aligned: list[tuple[PriceBar, PriceBar]],
) -> dict[str, Any]:
    forward = [
        closes[index + 20] / closes[index] - 1
        for index in range(max(0, len(closes) - 520), len(closes) - 20)
    ]
    if not forward:
        forward = [0.0]
    backtest = _walk_forward_backtest(closes, aligned)
    sample_count = int(backtest.get("sample_count") or 0)
    hit_rate = backtest.get("direction_hit_rate")
    if sample_count >= 25 and hit_rate is not None:
        reliability = min(1.0, abs(float(hit_rate) - 0.5) * 2)
        orientation = 1.0 if float(hit_rate) >= 0.5 else -1.0
        calibrated_market = market_score * orientation * reliability
        calibration_note = (
            "当前公司历史上趋势信号呈延续特征"
            if orientation > 0
            else "当前公司历史上趋势信号呈均值回归特征"
        )
    else:
        reliability = 0.25
        calibrated_market = market_score * 0.25
        calibration_note = "历史回测样本较少，市场信号已主动收缩"
    forecast_score = max(-1.0, min(1.0, 0.78 * calibrated_market + 0.22 * event_score))
    event_shift = max(-0.025, min(0.025, event_score * 0.025))
    low = _percentile(forward, 0.10) + event_shift
    median = _percentile(forward, 0.50) + event_shift
    high = _percentile(forward, 0.90) + event_shift
    neutral = max(0.20, min(0.38, 0.32 - 0.10 * abs(forecast_score)))
    up_share = max(0.12, min(0.88, 0.5 + 0.38 * forecast_score))
    p_up = (1 - neutral) * up_share
    p_down = 1 - neutral - p_up
    current = closes[-1]
    if forecast_score > 0.18:
        regime = "偏强情景"
    elif forecast_score < -0.18:
        regime = "偏弱情景"
    else:
        regime = "震荡情景"
    agreement = 1 - min(1.0, abs(market_score - event_score) / 2)
    confidence = min(
        0.85,
        0.25 + 0.20 * min(1, len(forward) / 300) + 0.25 * reliability + 0.10 * agreement,
    )
    return {
        "horizon_trading_days": 20,
        "regime": regime,
        "probabilities": {
            "up": round(p_up, 4),
            "neutral": round(neutral, 4),
            "down": round(p_down, 4),
        },
        "return_range": {
            "downside_p10": round(low, 4),
            "median_p50": round(median, 4),
            "upside_p90": round(high, 4),
        },
        "price_range": {
            "downside_p10": round(current * (1 + low), 2),
            "median_p50": round(current * (1 + median), 2),
            "upside_p90": round(current * (1 + high), 2),
        },
        "confidence": round(confidence, 3),
        "raw_signal_score": round(combined_score, 4),
        "calibrated_signal_score": round(forecast_score, 4),
        "calibration_note": calibration_note,
        "backtest": backtest,
        "method": "历史20日收益分布 + 多周期动量 + 相对指数强弱 + 标准化事件方向",
    }


def _walk_forward_backtest(
    closes: list[float], aligned: list[tuple[PriceBar, PriceBar]]
) -> dict[str, Any]:
    benchmark = {pair[0].trade_date: pair[1].close for pair in aligned}
    dates = [pair[0].trade_date for pair in aligned]
    close_by_date = {pair[0].trade_date: pair[0].close for pair in aligned}
    aligned_closes = [close_by_date[item] for item in dates]
    predictions = 0
    correct = 0
    for index in range(max(80, len(dates) - 300), len(dates) - 20, 5):
        history = aligned_closes[: index + 1]
        r20 = _period_return(history, 20)
        r60 = _period_return(history, 60)
        bench_dates = dates[: index + 1]
        bench_closes = [benchmark[item] for item in bench_dates]
        excess = r20 - _period_return(bench_closes, 20)
        signal = (
            0.4 * math.tanh(r20 / 0.1) + 0.3 * math.tanh(r60 / 0.2) + 0.3 * math.tanh(excess / 0.1)
        )
        actual = aligned_closes[index + 20] / aligned_closes[index] - 1
        if abs(signal) < 0.08:
            continue
        predictions += 1
        correct += int((signal > 0) == (actual > 0))
    return {
        "sample_count": predictions,
        "direction_hit_rate": round(correct / predictions, 4) if predictions else None,
        "note": "仅用于检验方向信号历史稳定性，未计交易成本，也不代表未来表现。",
    }


def _event_price_links(
    events: list[dict[str, Any]],
    bars: list[PriceBar],
    benchmark_by_date: dict[date, PriceBar],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in events[:30]:
        try:
            event_date = datetime.fromisoformat(str(event.get("published_at"))).date()
        except (TypeError, ValueError):
            continue
        index = next((i for i, bar in enumerate(bars) if bar.trade_date >= event_date), None)
        if index is None or index + 5 >= len(bars):
            continue
        bar = bars[index]
        forward_5 = bars[index + 5].close / bar.close - 1
        result: dict[str, Any] = {
            "event_id": event.get("id"),
            "date": event_date.isoformat(),
            "event": event.get("standardized_text", ""),
            "direction": event.get("direction", 0),
            "forward_5d": round(forward_5, 4),
            "forward_20d": None,
            "excess_20d": None,
        }
        if index + 20 < len(bars):
            end = bars[index + 20]
            forward_20 = end.close / bar.close - 1
            benchmark_start = benchmark_by_date.get(bar.trade_date)
            benchmark_end = benchmark_by_date.get(end.trade_date)
            benchmark_return = (
                benchmark_end.close / benchmark_start.close - 1
                if benchmark_start and benchmark_end and benchmark_start.close
                else 0.0
            )
            result["forward_20d"] = round(forward_20, 4)
            result["excess_20d"] = round(forward_20 - benchmark_return, 4)
        results.append(result)
    return results[:12]


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
