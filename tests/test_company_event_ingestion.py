from datetime import date

import httpx

from company_event_monitor.domain import Company, SourceTier
from company_event_monitor.ingestion.cninfo import lookup_szse_company, query_announcements
from company_event_monitor.ingestion.rss import FeedConfig, FeedState, fetch_feed, parse_feed
from company_event_monitor.ingestion.web import enrich_document

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>公告</title><item>
<guid>notice-1</guid><title>示例公司重大合同公告</title>
<link>https://example.invalid/1</link>
<description><![CDATA[示例公司签订合同，金额2.5亿元。]]></description>
<pubDate>Mon, 13 Jul 2026 10:00:00 +0800</pubDate>
</item></channel></rss>"""


def test_parse_rss_to_document() -> None:
    documents = parse_feed(RSS, FeedConfig("https://example.invalid/rss", "公司公告", SourceTier.A))
    assert len(documents) == 1
    assert documents[0].source_id == "notice-1"
    assert documents[0].text == "示例公司签订合同，金额2.5亿元。"
    assert documents[0].published_at.utcoffset().total_seconds() == 8 * 3600


def test_conditional_fetch_handles_not_modified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["if-none-match"] == '"v1"'
        return httpx.Response(304)

    config = FeedConfig("https://example.invalid/rss", "公司公告", SourceTier.A)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        documents, state, changed = fetch_feed(config, FeedState(etag='"v1"'), client)
    assert documents == []
    assert state.etag == '"v1"'
    assert changed is False


def test_enriches_feed_document_from_html_source() -> None:
    document = parse_feed(RSS, FeedConfig("https://example.invalid/rss", "公司公告", SourceTier.A))[
        0
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<h2>一、经营情况</h2><p>示例公司订单下降。</p>",
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        enriched = enrich_document(document, client=client)
    assert enriched.text == "一、经营情况\n示例公司订单下降。"
    assert enriched.segments[0].section == "一、经营情况"


def test_cninfo_company_query_maps_official_pdf() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        body = request.content.decode()
        assert "stock=301487%2C9900051422" in body
        return httpx.Response(
            200,
            json={
                "announcements": [
                    {
                        "announcementId": "1225422508",
                        "announcementTitle": "示例公司重大合同公告",
                        "announcementTime": 1783946048000,
                        "adjunctUrl": "finalpage/2026-07-13/1225422508.PDF",
                    }
                ],
                "hasMore": False,
            },
        )

    company = Company(
        "c1",
        "示例公司",
        "301487",
        market="szse",
        source_org_id="9900051422",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        documents = query_announcements(
            company,
            date(2026, 7, 1),
            date(2026, 7, 13),
            client=client,
        )
    assert len(documents) == 1
    assert documents[0].source_tier == SourceTier.A
    assert documents[0].url == ("https://static.cninfo.com.cn/finalpage/2026-07-13/1225422508.PDF")


def test_lookup_szse_company_uses_official_catalog() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"stockList": [{"code": "000001", "orgId": "gssz0000001", "zwjc": "平安银行"}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        company = lookup_szse_company("000001", client=client)
    assert company is not None
    assert company.market == "szse"
    assert company.source_org_id == "gssz0000001"
