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
from ingestion.confluence.adapter import (
    confluence_comment_identity,
    confluence_page_identity,
    confluence_page_to_parsed_document,
    page_context_prefix,
    split_project_title,
)
from ingestion.confluence.client import ConfluencePage
from vector_db.sparse_encoder import tokenize

PROJECT = "A174. Гидроснаб"
PAGE_PATH = "Проекты компании / A174. Гидроснаб / A174. Прошивка / Уровнемер"
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
    assert confluence_page_identity("12345") == "confluence:page:12345"
    assert confluence_page_identity("12345") == confluence_page_identity("12345")
    assert len(confluence_page_identity("123456789012")) <= 32  # file_hashes.hash is VARCHAR(32)


def test_page_and_comment_identities_never_collide():
    """Both live in the same file_hashes table; a page id and a comment id
    are independent sequences and can be equal."""
    assert confluence_page_identity("777") != confluence_comment_identity("777")
    assert len(confluence_comment_identity("123456789012")) <= 32


def test_project_code_and_name_are_split_when_present():
    assert split_project_title("A174. Гидроснаб") == ("A174", "Гидроснаб")
    assert split_project_title("A054.Maintenance 2026") == ("A054", "Maintenance 2026")
    assert split_project_title("A091") == ("A091", None)
    # Cyrillic А — a real title in the tree uses it.
    assert split_project_title("А187.Киберпротект") == ("А187", "Киберпротект")
    # Not a project title: no code, so nothing is invented.
    assert split_project_title("Требования") == (None, None)


def test_page_becomes_a_single_page_parsed_document():
    parsed = confluence_page_to_parsed_document(_page(), project=PROJECT, page_path=PAGE_PATH)

    assert parsed.filename == TITLE
    assert parsed.total_pages == 1
    assert len(parsed.pages) == 1
    assert parsed.pages[0]["page_num"] == 1
    assert parsed.pages[0]["text"] == BODY
    assert parsed.pages[0]["has_ocr"] is False
    assert parsed.pages[0]["char_count"] == len(BODY)
    assert parsed.file_size_kb > 0


def test_metadata_carries_everything_the_ui_and_sync_need():
    parsed = confluence_page_to_parsed_document(_page(), project=PROJECT, page_path=PAGE_PATH)

    assert parsed.metadata["source"] == "confluence"
    assert parsed.metadata["content_type"] == "page"
    assert parsed.metadata["project"] == PROJECT
    assert parsed.metadata["project_code"] == "A174"
    assert parsed.metadata["project_name"] == "Гидроснаб"
    assert parsed.metadata["page_id"] == "12345"
    assert parsed.metadata["page_title"] == TITLE
    assert parsed.metadata["page_path"] == PAGE_PATH
    assert parsed.metadata["page_url"] == "https://confluence.example.com/display/AISMT/12345"
    assert parsed.metadata["version"] == 4
    assert parsed.metadata["updated_at"] == "2026-09-21T00:00:00+00:00"


def test_page_context_prefix_carries_project_page_and_path():
    """This is the string that gets embedded and reranked — the project code
    has to be in it, or "компоненты A174" can never match a chunk of a page
    whose own words never mention the project."""
    prefix = page_context_prefix(PROJECT, TITLE, PAGE_PATH)
    assert "A174" in prefix
    assert TITLE in prefix
    assert PAGE_PATH in prefix


def test_parsed_document_flows_through_the_standard_chunker():
    parsed = confluence_page_to_parsed_document(_page(), project=PROJECT, page_path=PAGE_PATH)
    chunks = SmartChunker(chunk_size=512, chunk_overlap=50).chunk_document(parsed.pages, "doc-1")

    assert chunks
    assert all(c.page_num == 1 for c in chunks)
    assert all(c.document_id == "doc-1" for c in chunks)


def test_title_and_project_are_prefixed_onto_every_indexed_chunk():
    """chunk_context_text() is what gets embedded, indexed and reranked — if
    the title and project aren't in there, they aren't in retrieval."""
    parsed = confluence_page_to_parsed_document(_page(), project=PROJECT, page_path=PAGE_PATH)
    chunks = SmartChunker(chunk_size=512, chunk_overlap=50).chunk_document(parsed.pages, "doc-1")
    for chunk in chunks:
        chunk.filename = parsed.filename
        chunk.context_prefix = page_context_prefix(PROJECT, TITLE, PAGE_PATH)

    for chunk in chunks:
        indexed = chunk_context_text(chunk)
        assert indexed.startswith("Project: A174. Гидроснаб")
        assert TITLE in indexed
        assert chunk.text in indexed
        # The stored/displayed text stays clean — only the indexed form is prefixed.
        assert not chunk.text.startswith("Project:")


def test_exact_title_query_matches_the_indexed_text_lexically():
    """Model-free half of the exact-title guarantee: every term of the title
    query is present in the chunk's indexed form, so BM25/sparse retrieval
    can find the page even though the body never states the title."""
    parsed = confluence_page_to_parsed_document(_page(), project=PROJECT, page_path=PAGE_PATH)
    chunks = SmartChunker(chunk_size=512, chunk_overlap=50).chunk_document(parsed.pages, "doc-1")
    for chunk in chunks:
        chunk.filename = parsed.filename

    for chunk in chunks:
        chunk.context_prefix = page_context_prefix(PROJECT, TITLE, PAGE_PATH)

    query_terms = set(tokenize(TITLE))
    assert query_terms, "the title must tokenize to something — see sparse_encoder"

    indexed_terms = set(tokenize(chunk_context_text(chunks[0])))
    assert query_terms <= indexed_terms

    # Without the title prefix (the reference project's bug), the body alone
    # would NOT carry those terms — this is what makes the assertion above
    # meaningful rather than accidentally true.
    body_terms = set(tokenize(chunks[0].text))
    assert not query_terms <= body_terms


# ── comments ──────────────────────────────────────────────────────────────

def _comment(text: str = "Должен отображать уровень радиосигнала и остаток в литрах.", version: int = 2):
    from ingestion.confluence.client import ConfluenceComment
    return ConfluenceComment(
        id="99001",
        page_id="12345",
        version=version,
        url="https://confluence.example.com/display/AISMT/12345?focusedCommentId=99001#comment-99001",
        html_content="<p>irrelevant, already parsed</p>",
        text_content=text,
        author="Anton V. Kochekov",
        created_at=datetime(2026, 2, 27, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )


def test_comment_becomes_its_own_document_titled_after_its_page():
    from ingestion.confluence.adapter import confluence_comment_to_parsed_document

    parsed = confluence_comment_to_parsed_document(
        _comment(), _page(), project=PROJECT, page_path=PAGE_PATH
    )

    # Titled after the page: a comment has no title of its own, and a
    # citation has to read as "said on this page".
    assert parsed.filename == TITLE
    assert parsed.pages[0]["text"].startswith("Должен отображать")
    assert parsed.metadata["content_type"] == "comment"
    assert parsed.metadata["comment_id"] == "99001"
    assert parsed.metadata["page_id"] == "12345"
    assert parsed.metadata["project"] == PROJECT
    assert parsed.metadata["page_path"] == PAGE_PATH
    assert parsed.metadata["comment_author"] == "Anton V. Kochekov"
    assert parsed.metadata["version"] == 2


def test_comment_links_to_its_own_permalink_when_confluence_gave_one():
    from ingestion.confluence.adapter import confluence_comment_to_parsed_document

    parsed = confluence_comment_to_parsed_document(
        _comment(), _page(), project=PROJECT, page_path=PAGE_PATH
    )
    assert "focusedCommentId=99001" in parsed.metadata["page_url"]


def test_comment_without_permalink_falls_back_to_its_page():
    from ingestion.confluence.adapter import confluence_comment_to_parsed_document
    from ingestion.confluence.client import ConfluenceComment

    bare = ConfluenceComment(id="1", page_id="12345", version=1, url="", html_content="",
                             text_content="text", author=None, created_at=None, updated_at=None)
    parsed = confluence_comment_to_parsed_document(bare, _page(), project=PROJECT, page_path=PAGE_PATH)
    assert parsed.metadata["page_url"] == _page().url


def test_comment_context_prefix_marks_the_type_and_omits_the_author():
    """The author is metadata, not a retrieval signal: putting a person's
    name into every chunk of their comments would let the name compete with
    the actual content for matches."""
    from ingestion.confluence.adapter import comment_context_prefix

    prefix = comment_context_prefix(PROJECT, TITLE, PAGE_PATH)
    assert "Type: comment" in prefix
    assert "A174" in prefix
    assert "Kochekov" not in prefix


def test_project_code_query_matches_a_chunk_that_never_says_it():
    """The retrieval half of requirement 21: a page called "BOM" under A174
    contains no project code in its own text, so only the context prefix can
    make "компоненты A174" findable."""
    from ingestion.confluence.adapter import page_context_prefix

    body = "Корпус, разъёмы, ультразвуковой датчик, аккумулятор и антенна."
    parsed = confluence_page_to_parsed_document(
        ConfluencePage(id="7", title="BOM", space_key="PROJ",
                       url="https://confluence.example.com/x", version=1,
                       html_content="", text_content=body, created_at=None, updated_at=None),
        project=PROJECT, page_path=PAGE_PATH,
    )
    chunks = SmartChunker(chunk_size=512, chunk_overlap=50, min_chunk_size=10).chunk_document(parsed.pages, "doc-1")
    for chunk in chunks:
        chunk.filename = parsed.filename
        chunk.context_prefix = page_context_prefix(PROJECT, "BOM", PAGE_PATH)

    indexed_terms = set(tokenize(chunk_context_text(chunks[0])))
    assert "a174" in indexed_terms
    assert "a174" not in set(tokenize(chunks[0].text))
