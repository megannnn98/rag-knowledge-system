"""Confluence Storage Format -> plain text.

Ported from the reference implementation in integra-rag
(confluence_rag/loader/parser.py). That version exists because the naive
approach (BeautifulSoup.get_text() over the storage body) pulled macro and
blueprint METADATA into the user-visible text — `<ac:parameter>` values like
layout names, panel types and macro ids ended up indexed as if an author had
written them, polluting both embeddings and BM25. The structural walk below
is what fixed it, so it is kept intact rather than re-derived:

  - `ac:parameter` and every `ri:*` element (ri:page, ri:user,
    ri:attachment, ...) are dropped entirely — they are macro plumbing, not
    content.
  - `ac:rich-text-body` is descended into as a block (it holds real authored
    text inside a macro), and `ac:plain-text-body` is treated as code.
  - tables become "cell | cell | cell" lines and lists become "- item"
    lines, so structure survives into the chunker as something a language
    model can read, instead of collapsing into one run-on paragraph.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

TECHNICAL_TAGS = {"script", "style", "nav", "ac:parameter"}
BLOCK_TAGS = {
    "ac:layout",
    "ac:layout-cell",
    "ac:layout-section",
    "ac:rich-text-body",
    "article",
    "blockquote",
    "div",
    "section",
}
TEXT_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p"}
CODE_TAGS = {"ac:plain-text-body", "pre", "code"}


class ConfluenceHtmlParser:
    def parse(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style", "nav", "ac:parameter"]):
            element.decompose()

        for element in soup.select("[role='navigation'], .navigation, #navigation"):
            element.decompose()

        lines: list[str] = []
        root = soup.body or soup
        for child in root.children:
            self._append_text(child, lines)

        return self._normalize_lines(lines)

    def _append_text(self, node: object, lines: list[str]) -> None:
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                lines.append(text)
            return
        if not isinstance(node, Tag):
            return

        self._append_tag_text(node, lines)

    def _append_tag_text(self, node: Tag, lines: list[str]) -> None:
        name = self._tag_name(node)
        if self._is_technical_tag(name):
            return

        if name in TEXT_BLOCK_TAGS:
            self._append_block_text(node, lines)
        elif name in {"ul", "ol"}:
            self._append_list(node, lines)
        elif name == "table":
            self._append_table(node, lines)
        elif name in CODE_TAGS:
            self._append_code(node, lines)
        elif name in {"br", "hr"}:
            lines.append("")
        else:
            for child in node.children:
                self._append_text(child, lines)
            if name in BLOCK_TAGS:
                lines.append("")

    def _append_block_text(self, tag: Tag, lines: list[str]) -> None:
        text = self._inline_text(tag)
        if text:
            lines.append(text)
            lines.append("")

    def _append_list(self, tag: Tag, lines: list[str]) -> None:
        for item in tag.find_all("li", recursive=False):
            text = self._inline_text(item)
            if text:
                lines.append(f"- {text}")
        lines.append("")

    def _append_table(self, tag: Tag, lines: list[str]) -> None:
        for row in tag.find_all("tr"):
            cells = [self._inline_text(cell) for cell in row.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if cells:
                lines.append(" | ".join(cells))
        lines.append("")

    def _append_code(self, tag: Tag, lines: list[str]) -> None:
        text = tag.get_text("\n", strip=True)
        if text:
            lines.extend(line.rstrip() for line in text.splitlines())
            lines.append("")

    def _inline_text(self, tag: Tag) -> str:
        parts = list(self._iter_text(tag))
        text = " ".join(part for part in parts if part.strip())
        return re.sub(r"[ \t\r\f\v]+", " ", text).strip()

    def _iter_text(self, node: object) -> list[str]:
        if isinstance(node, NavigableString):
            return [str(node)]
        if not isinstance(node, Tag):
            return []

        name = self._tag_name(node)
        if self._is_technical_tag(name):
            return []
        if name == "br":
            return ["\n"]

        parts: list[str] = []
        for child in node.children:
            parts.extend(self._iter_text(child))
        return parts

    def _tag_name(self, tag: Tag) -> str:
        return str(tag.name).lower()

    def _is_technical_tag(self, name: str) -> bool:
        return name in TECHNICAL_TAGS or name.startswith("ri:")

    def _normalize_lines(self, lines: list[str]) -> str:
        normalized: list[str] = []
        previous_blank = True
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if not previous_blank:
                    normalized.append("")
                previous_blank = True
                continue
            normalized.append(stripped)
            previous_blank = False

        while normalized and normalized[-1] == "":
            normalized.pop()

        return "\n".join(normalized)
