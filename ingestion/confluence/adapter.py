"""Confluence page -> ParsedDocument.

The whole point of this module is that it is the ONLY Confluence-aware thing
downstream of the REST client: everything after it (SmartChunker,
chunk_context_text, EmbeddingService, VectorStore, Postgres persistence,
retrieval, reranking, generation) sees a ParsedDocument exactly like the one
PDFParser/TxtParser produce, so Confluence gets no pipeline of its own.

The page title is carried as ParsedDocument.filename, not just as metadata.
That is deliberate and load-bearing: chunk_context_text() prefixes the
filename to every chunk before embedding AND before reranking, so the title
participates in retrieval. The reference project had a bug where the title
lived only in metadata and an exact-title query could not find its own page
(see tests/test_confluence_adapter.py::test_exact_title_query_ranks_its_own_page).
"""
from __future__ import annotations

from ingestion.confluence.client import ConfluencePage
from ingestion.document import ParsedDocument


def confluence_identity(page_id: str) -> str:
    """Stable local identity for a Confluence page.

    A page's content hash is NOT usable as identity the way an uploaded
    file's is: editing a page must update the SAME document, and two pages
    can legitimately hold identical text. The page id is the only thing
    Confluence itself guarantees stable across edits, moves and renames.

    Stored in the `file_hashes` table (VARCHAR(32)), which is what maps an
    identity to a doc_id for the whole app — so the prefix keeps Confluence
    identities from ever colliding with a real 32-hex-char file md5."""
    return f"confluence:{page_id}"


def confluence_page_to_parsed_document(page: ConfluencePage) -> ParsedDocument:
    """One page becomes a one-page ParsedDocument — Confluence has no
    pagination of its own, same as TxtParser's treatment of a text file."""
    text = page.text_content
    return ParsedDocument(
        filename=page.title,
        total_pages=1,
        pages=[{
            "page_num": 1,
            "text": text,
            "has_ocr": False,
            "char_count": len(text),
        }],
        metadata={
            "source": "confluence",
            "title": page.title,
            "page_id": page.id,
            "space_key": page.space_key,
            "confluence_url": page.url,
            "version": page.version,
            "updated_at": page.updated_at.isoformat() if page.updated_at else None,
        },
        file_size_kb=len(text.encode("utf-8")) / 1024,
    )
