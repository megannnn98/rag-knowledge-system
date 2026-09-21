"""Tests for ingestion/confluence/adapter.py — page -> ParsedDocument, and
the title actually reaching retrieval.

The title case is a regression from the reference project, where a page's
title lived only in metadata: an exact-title query could not find its own
page, because nothing the retriever scored ever contained the title. Here the
title is carried as ParsedDocument.filename, which chunk_context_text()
prefixes onto every chunk before embedding AND before reranking — so this
asserts on that prefixed text, which is the string retrieval actually sees.
"""
from datetime import datetime, timezone

from ingestion.chunker import SmartChunker, chunk_context_text
from ingestion.confluence.adapter import confluence_identity, confluence_page_to_parsed_document
from ingestion.confluence.client import ConfluencePage
from vector_db.sparse_encoder import tokenize

TITLE = "Глоссарий основных сущностей ISMT"
# Body deliberately never repeats the title as a phrase — the point is that
# retrieval must not depend on the body happening to contain it.
BODY = (
    "Резервуар — учётная единица системы. Паспорт содержит объём в литрах и тип жидкости. "
    "Датчик уровня жидкости измеряет уровень ультразвуковым методом и передаёт результат "
    "концентратору раз в 15 минут. Концентратор собирает данные по радиоканалу Zigbee."
)


def _page(title: str = TITLE, text: str = BODY, version: int = 4) -> ConfluencePage:
    return ConfluencePage(
        id="12345",
        title=title,
        space_key="AISMT",
        url="https://confluence.example.com/display/AISMT/12345",
        version=version,
        html_content="<p>irrelevant, already parsed</p>",
        text_content=text,
        created_at=datetime(2026, 9, 20, tzinfo=timezone.utc),
        updated_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )


def test_identity_is_the_page_id_not_a_content_hash():
    """Content-hash identity would make an edited page a second document and
    two identical pages one document — both wrong."""
    assert confluence_identity("12345") == "confluence:12345"
    assert confluence_identity("12345") == confluence_identity("12345")
    assert len(confluence_identity("123456789012")) <= 32  # file_hashes.hash is VARCHAR(32)


def test_page_becomes_a_single_page_parsed_document():
    parsed = confluence_page_to_parsed_document(_page())

    assert parsed.filename == TITLE
    assert parsed.total_pages == 1
    assert len(parsed.pages) == 1
    assert parsed.pages[0]["page_num"] == 1
    assert parsed.pages[0]["text"] == BODY
    assert parsed.pages[0]["has_ocr"] is False
    assert parsed.pages[0]["char_count"] == len(BODY)
    assert parsed.file_size_kb > 0


def test_metadata_carries_everything_the_ui_and_sync_need():
    parsed = confluence_page_to_parsed_document(_page())

    assert parsed.metadata == {
        "source": "confluence",
        "title": TITLE,
        "page_id": "12345",
        "space_key": "AISMT",
        "confluence_url": "https://confluence.example.com/display/AISMT/12345",
        "version": 4,
        "updated_at": "2026-09-21T00:00:00+00:00",
    }


def test_parsed_document_flows_through_the_standard_chunker():
    parsed = confluence_page_to_parsed_document(_page())
    chunks = SmartChunker(chunk_size=512, chunk_overlap=50).chunk_document(parsed.pages, "doc-1")

    assert chunks
    assert all(c.page_num == 1 for c in chunks)
    assert all(c.document_id == "doc-1" for c in chunks)


def test_title_is_prefixed_onto_every_indexed_chunk():
    """chunk_context_text() is what gets embedded, indexed and reranked — if
    the title isn't in there, it isn't in retrieval."""
    parsed = confluence_page_to_parsed_document(_page())
    chunks = SmartChunker(chunk_size=512, chunk_overlap=50).chunk_document(parsed.pages, "doc-1")
    for chunk in chunks:
        chunk.filename = parsed.filename

    for chunk in chunks:
        assert chunk_context_text(chunk).startswith(f"{TITLE}: ")
        # The stored/displayed text stays clean — only the indexed form is prefixed.
        assert not chunk.text.startswith(TITLE)


def test_exact_title_query_matches_the_indexed_text_lexically():
    """Model-free half of the exact-title guarantee: every term of the title
    query is present in the chunk's indexed form, so BM25/sparse retrieval
    can find the page even though the body never states the title."""
    parsed = confluence_page_to_parsed_document(_page())
    chunks = SmartChunker(chunk_size=512, chunk_overlap=50).chunk_document(parsed.pages, "doc-1")
    for chunk in chunks:
        chunk.filename = parsed.filename

    query_terms = set(tokenize(TITLE))
    assert query_terms, "the title must tokenize to something — see sparse_encoder"

    indexed_terms = set(tokenize(chunk_context_text(chunks[0])))
    assert query_terms <= indexed_terms

    # Without the title prefix (the reference project's bug), the body alone
    # would NOT carry those terms — this is what makes the assertion above
    # meaningful rather than accidentally true.
    body_terms = set(tokenize(chunks[0].text))
    assert not query_terms <= body_terms
