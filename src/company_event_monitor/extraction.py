from __future__ import annotations

import re
from dataclasses import dataclass

from .domain import Company, Document, EventStatus, EventType, EvidenceSegment, FundamentalEvent


@dataclass(frozen=True, slots=True)
class EventRule:
    event_type: EventType
    terms: tuple[str, ...]
    direction: int
    label: str
    impacts: tuple[str, ...]
    base_confidence: float = 0.78


RULES = (
    EventRule(
        EventType.REVENUE_DECLINE,
        ("收入下降", "营收下降", "收入下滑", "营收下滑"),
        -1,
        "营业收入下降",
        ("revenue",),
    ),
    EventRule(
        EventType.REVENUE_GROWTH,
        ("收入增长", "营收增长", "收入上升"),
        1,
        "营业收入增长",
        ("revenue",),
    ),
    EventRule(
        EventType.ORDER_DECLINE,
        ("订单下降", "订单减少", "订单不足", "需求下降"),
        -1,
        "订单或需求下降",
        ("revenue", "orders"),
    ),
    EventRule(
        EventType.ORDER_GROWTH,
        ("订单增长", "订单增加", "订单充足", "需求增长"),
        1,
        "订单或需求增长",
        ("revenue", "orders"),
    ),
    EventRule(
        EventType.MARGIN_PRESSURE,
        ("毛利率下降", "毛利率承压", "盈利空间收窄"),
        -1,
        "毛利率面临下降压力",
        ("gross_margin", "profit"),
    ),
    EventRule(
        EventType.COST_INCREASE,
        ("成本上升", "成本增加", "原材料价格上涨"),
        -1,
        "经营成本上升",
        ("cost", "gross_margin"),
    ),
    EventRule(
        EventType.RECEIVABLE_PRESSURE,
        ("回款周期延长", "回款放缓", "回款可能放缓", "应收账款增加", "逾期增加"),
        -1,
        "客户回款或应收账款压力上升",
        ("accounts_receivable", "cash_flow"),
    ),
    EventRule(
        EventType.INVENTORY_PRESSURE,
        ("存货增加", "库存增加", "库存积压", "存货周转下降"),
        -1,
        "存货或库存压力上升",
        ("inventory", "cash_flow"),
    ),
    EventRule(
        EventType.CASH_FLOW_PRESSURE,
        ("现金流转负", "现金流下降", "现金流承压", "资金紧张"),
        -1,
        "经营现金流压力上升",
        ("operating_cash_flow",),
    ),
    EventRule(
        EventType.CAPACITY_EXPANSION,
        ("扩产", "新增产能", "产能建设", "投产"),
        1,
        "公司推进产能扩张",
        ("capacity", "capex"),
    ),
    EventRule(
        EventType.PROJECT_DELAY,
        ("项目延期", "延期投产", "进度不及预期", "建设放缓"),
        -1,
        "项目建设或投产进度延迟",
        ("revenue", "capex"),
    ),
    EventRule(
        EventType.MAJOR_CONTRACT,
        ("重大合同", "签订合同", "中标", "获得订单"),
        1,
        "公司获得重要合同或订单",
        ("revenue", "orders"),
    ),
    EventRule(
        EventType.LITIGATION,
        ("重大诉讼", "提起诉讼", "仲裁事项"),
        -1,
        "公司涉及诉讼或仲裁",
        ("contingent_liability",),
    ),
    EventRule(
        EventType.REGULATORY_PENALTY,
        ("行政处罚", "监管处罚", "立案调查"),
        -1,
        "公司受到监管调查或处罚",
        ("compliance",),
    ),
    EventRule(
        EventType.FINANCING,
        ("定向增发", "发行可转债", "融资计划"),
        0,
        "公司计划或实施融资",
        ("financing", "dilution"),
    ),
    EventRule(
        EventType.R_AND_D_PROGRESS,
        ("研发取得进展", "获得认证", "产品获批", "研发成功"),
        1,
        "研发或产品认证取得进展",
        ("product", "revenue"),
    ),
    EventRule(
        EventType.CUSTOMER_CONCENTRATION,
        ("客户集中度上升", "大客户依赖"),
        -1,
        "客户集中度或大客户依赖上升",
        ("customer", "revenue"),
    ),
    EventRule(
        EventType.CUSTOMER_LOSS,
        ("客户流失", "终止合作", "丢失客户"),
        -1,
        "公司出现客户流失",
        ("customer", "revenue"),
    ),
    EventRule(
        EventType.SUPPLIER_DISRUPTION,
        ("供应商中断", "供应链中断", "供应受限"),
        -1,
        "供应链或供应商供货受到影响",
        ("supply_chain", "production"),
    ),
    EventRule(
        EventType.RAW_MATERIAL_SHORTAGE,
        ("原材料短缺", "原料紧缺", "芯片短缺"),
        -1,
        "生产所需原材料供应短缺",
        ("supply_chain", "production"),
    ),
    EventRule(
        EventType.CAPACITY_UTILIZATION_UP,
        ("产能利用率提升", "产能利用率上升", "满产"),
        1,
        "产能利用率提升",
        ("capacity", "gross_margin"),
    ),
    EventRule(
        EventType.CAPACITY_UTILIZATION_DOWN,
        ("产能利用率下降", "开工率下降", "产能闲置"),
        -1,
        "产能利用率下降",
        ("capacity", "gross_margin"),
    ),
    EventRule(
        EventType.PRODUCTION_SUSPENSION,
        ("停产", "暂停生产", "临时停工"),
        -1,
        "部分生产活动暂停",
        ("production", "revenue"),
    ),
    EventRule(
        EventType.NEW_FACTORY,
        ("新建工厂", "建设生产基地", "新生产基地"),
        1,
        "公司建设新的生产基地",
        ("capacity", "capex"),
    ),
    EventRule(
        EventType.NEW_PRODUCT,
        ("新产品量产", "推出新产品", "新产品发布"),
        1,
        "公司推出或量产新产品",
        ("product", "revenue"),
    ),
    EventRule(
        EventType.PRODUCT_RECALL,
        ("产品召回", "实施召回", "召回车辆"),
        -1,
        "公司相关产品被召回",
        ("quality", "cost"),
    ),
    EventRule(
        EventType.QUALITY_ISSUE,
        ("质量问题", "质量缺陷", "产品缺陷"),
        -1,
        "产品出现质量或缺陷问题",
        ("quality", "cost"),
    ),
    EventRule(
        EventType.OVERSEAS_EXPANSION,
        ("海外扩张", "海外建厂", "拓展海外市场"),
        1,
        "公司扩展海外业务",
        ("overseas", "capex", "revenue"),
    ),
    EventRule(
        EventType.OVERSEAS_DEMAND_PRESSURE,
        ("海外需求下降", "海外市场承压", "出口下降"),
        -1,
        "海外市场需求或出口承压",
        ("overseas", "revenue"),
    ),
    EventRule(
        EventType.TARIFF_PRESSURE,
        ("关税上升", "加征关税", "关税影响"),
        -1,
        "关税变化对公司经营形成压力",
        ("overseas", "cost"),
    ),
    EventRule(
        EventType.FX_IMPACT,
        ("汇率波动", "汇兑损失", "汇率影响"),
        0,
        "汇率变化影响公司经营结果",
        ("overseas", "profit"),
    ),
    EventRule(
        EventType.CAPEX_INCREASE,
        ("资本开支增加", "资本性支出增加"),
        0,
        "公司增加资本性支出",
        ("capex", "cash_flow"),
    ),
    EventRule(
        EventType.CAPEX_REDUCTION,
        ("削减资本开支", "资本开支下降", "减少资本性支出"),
        0,
        "公司减少资本性支出",
        ("capex", "cash_flow"),
    ),
    EventRule(
        EventType.DEBT_PRESSURE,
        ("偿债压力", "债务压力", "流动性压力"),
        -1,
        "公司债务或流动性压力上升",
        ("debt", "cash_flow"),
    ),
    EventRule(
        EventType.GOODWILL_IMPAIRMENT,
        ("商誉减值", "计提商誉减值"),
        -1,
        "公司确认或面临商誉减值",
        ("impairment", "profit"),
    ),
    EventRule(
        EventType.SUBSIDY,
        ("获得政府补助", "收到政府补助", "政府补贴"),
        1,
        "公司获得政府补助",
        ("profit", "cash_flow"),
    ),
    EventRule(
        EventType.MANAGEMENT_CHANGE,
        ("董事长辞职", "总经理辞职", "高管变动"),
        0,
        "公司核心管理人员发生变动",
        ("governance",),
    ),
    EventRule(
        EventType.EQUITY_INCENTIVE,
        ("股权激励计划", "限制性股票激励"),
        1,
        "公司推出股权激励计划",
        ("governance", "dilution"),
    ),
    EventRule(
        EventType.SHARE_REPURCHASE,
        ("回购股份", "股份回购"),
        1,
        "公司实施或计划股份回购",
        ("capital_market",),
    ),
    EventRule(
        EventType.SHAREHOLDER_REDUCTION,
        ("股东减持", "减持计划"),
        -1,
        "公司股东实施或计划减持",
        ("capital_market",),
    ),
    EventRule(
        EventType.RELATED_PARTY_TRANSACTION,
        ("关联交易", "关联方交易"),
        0,
        "公司发生关联交易",
        ("governance",),
    ),
)

HEDGE_TERMS = ("可能", "或将", "预计", "有望", "存在一定", "面临")
RESOLVED_TERMS = ("已解决", "恢复正常", "风险消除", "得到缓解")
NEGATION_TERMS = ("不存在", "不涉及", "未发生", "无需", "不构成", "并非")
LEGAL_REFERENCE_TERMS = ("管理办法", "法律法规", "相关规定", "实施细则", "业务规则")
ACTION_TERMS = ("计划", "拟", "实施", "发生", "完成", "正在", "将", "已")
LEGAL_REFERENCE_SENSITIVE = {
    EventType.SHAREHOLDER_REDUCTION,
    EventType.SHARE_REPURCHASE,
    EventType.PRODUCT_RECALL,
    EventType.RELATED_PARTY_TRANSACTION,
}
NUMBER_PATTERN = re.compile(r"(?:\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:亿元|万元|元|个|台|天|日))")


class BaselineChineseExtractor:
    """Deterministic baseline. An LLM adapter can implement the same extract contract."""

    def __init__(self, companies: list[Company]) -> None:
        self.companies = companies

    def extract(self, document: Document) -> list[FundamentalEvent]:
        companies = [company for company in self.companies if self._mentions(document, company)]
        if not companies:
            return []
        segments = (
            list(document.segments)
            if document.segments
            else [
                EvidenceSegment(part.strip())
                for part in re.split(r"[\n。；]+", document.text)
                if part.strip()
            ]
        )
        results: list[FundamentalEvent] = []
        seen: set[tuple[str, EventType, str]] = set()
        for segment in segments:
            paragraph = segment.text
            for rule in RULES:
                matched = next((term for term in rule.terms if term in paragraph), None)
                if not matched:
                    continue
                if self._excluded(paragraph, rule.event_type):
                    continue
                for company in companies:
                    key = (company.company_id, rule.event_type, paragraph)
                    if key in seen:
                        continue
                    seen.add(key)
                    status, certainty = self._status(paragraph)
                    numbers = tuple(NUMBER_PATTERN.findall(paragraph))
                    standard = self._standardize(company.name, rule.label, status, numbers)
                    results.append(
                        FundamentalEvent(
                            company_id=company.company_id,
                            ticker=company.ticker,
                            company_name=company.name,
                            event_type=rule.event_type,
                            status=status,
                            direction=rule.direction,
                            standardized_text=standard,
                            evidence_text=paragraph,
                            evidence_page=segment.page,
                            evidence_section=segment.section,
                            source_id=document.source_id,
                            source_name=document.source_name,
                            source_tier=document.source_tier,
                            published_at=document.published_at,
                            certainty=certainty,
                            impact_dimensions=rule.impacts,
                            numeric_evidence=numbers,
                            confidence=min(0.98, rule.base_confidence + (0.08 if numbers else 0)),
                        )
                    )
        return results

    @staticmethod
    def _mentions(document: Document, company: Company) -> bool:
        haystack = f"{document.title}\n{document.text}"
        aliases = (company.name, company.ticker, *company.aliases)
        return any(alias and alias in haystack for alias in aliases)

    @staticmethod
    def _excluded(text: str, event_type: EventType) -> bool:
        if any(term in text for term in NEGATION_TERMS):
            return True
        if event_type == EventType.ORDER_DECLINE and "海外需求下降" in text:
            return True
        return (
            event_type in LEGAL_REFERENCE_SENSITIVE
            and any(term in text for term in LEGAL_REFERENCE_TERMS)
            and not any(term in text for term in ACTION_TERMS)
        )

    @staticmethod
    def _status(text: str) -> tuple[EventStatus, float]:
        if any(term in text for term in RESOLVED_TERMS):
            return EventStatus.RESOLVED, 0.95
        if "计划" in text or "拟" in text or "预计" in text:
            return EventStatus.EXPECTED, 0.68
        if any(term in text for term in HEDGE_TERMS):
            return EventStatus.POSSIBLE, 0.55
        return EventStatus.OCCURRED, 0.9

    @staticmethod
    def _standardize(name: str, label: str, status: EventStatus, numbers: tuple[str, ...]) -> str:
        status_text = {
            EventStatus.OCCURRED: "已发生",
            EventStatus.EXPECTED: "预计发生",
            EventStatus.POSSIBLE: "可能发生",
            EventStatus.RESOLVED: "已缓解",
        }[status]
        numeric = f"，原文数据：{'、'.join(numbers)}" if numbers else ""
        return f"{name}{label}（{status_text}）{numeric}。"
