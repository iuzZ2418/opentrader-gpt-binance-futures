from __future__ import annotations

import io
import re
from html.parser import HTMLParser

import pdfplumber

from .domain import EvidenceSegment

SPACE_PATTERN = re.compile(r"[ \t\u3000]+")
EMPTY_LINES_PATTERN = re.compile(r"\n{3,}")
PAGE_NUMBER_PATTERN = re.compile(r"^\s*(?:第\s*)?\d+\s*(?:页)?\s*$")
SECTION_PATTERN = re.compile(
    r"^(?:第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+[、.]|\d+(?:\.\d+)*[、.\s])"
)


class TextHTMLParser(HTMLParser):
    BLOCKS = {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "br"}
    IGNORED = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.IGNORED:
            self.ignore_depth += 1
        elif tag in self.BLOCKS and self.parts and self.parts[-1] != "\n":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.IGNORED:
            self.ignore_depth = max(0, self.ignore_depth - 1)
        elif tag in self.BLOCKS and self.parts and self.parts[-1] != "\n":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignore_depth:
            self.parts.append(data)

    def text(self) -> str:
        return normalize_text("".join(self.parts))


def normalize_text(text: str) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned = SPACE_PATTERN.sub(" ", line).strip()
        if cleaned and not PAGE_NUMBER_PATTERN.match(cleaned):
            lines.append(cleaned)
    return EMPTY_LINES_PATTERN.sub("\n\n", "\n".join(lines)).strip()


def parse_html(html_text: str) -> tuple[str, tuple[EvidenceSegment, ...]]:
    parser = TextHTMLParser()
    parser.feed(html_text)
    text = parser.text()
    return text, segment_text(text)


def parse_pdf(content: bytes) -> tuple[str, tuple[EvidenceSegment, ...]]:
    page_texts: list[str] = []
    segments: list[EvidenceSegment] = []
    current_section = ""
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = normalize_text(page.extract_text(x_tolerance=2, y_tolerance=3) or "")
            if not text:
                continue
            page_texts.append(text)
            page_segments = segment_text(text, page=page_number, initial_section=current_section)
            if page_segments:
                current_section = page_segments[-1].section or current_section
                segments.extend(page_segments)
    return "\n\n".join(page_texts), tuple(segments)


def segment_text(
    text: str,
    *,
    page: int | None = None,
    initial_section: str = "",
) -> tuple[EvidenceSegment, ...]:
    section = initial_section
    segments: list[EvidenceSegment] = []
    for block in re.split(r"\n+|(?<=[。！？；])", text):
        cleaned = block.strip()
        if not cleaned:
            continue
        if len(cleaned) <= 80 and SECTION_PATTERN.match(cleaned):
            section = cleaned
            continue
        segments.append(EvidenceSegment(cleaned, page, section))
    return tuple(segments)
