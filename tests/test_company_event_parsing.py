from company_event_monitor.parsing import parse_html, segment_text


def test_html_parser_removes_noise_and_keeps_sections() -> None:
    text, segments = parse_html(
        """
        <html><head><style>.x{display:none}</style><script>ignore()</script></head>
        <body><h2>一、经营情况</h2><p>示例公司订单下降。</p>
        <p>应收账款增加45%。</p></body></html>
        """
    )
    assert "ignore" not in text
    assert "订单下降" in text
    assert [item.section for item in segments] == ["一、经营情况", "一、经营情况"]


def test_segments_retain_page_and_section() -> None:
    segments = segment_text("二、风险因素\n示例公司回款周期延长。", page=7)
    assert len(segments) == 1
    assert segments[0].page == 7
    assert segments[0].section == "二、风险因素"
