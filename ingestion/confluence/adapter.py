"""Confluence page/comment -> ParsedDocument.

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
(see tests/test_confluence_adapter.py::test_exact_title_query_matches...).

A comment becomes its own document rather than being appended to its page's
text. Two reasons: a page's requirements often live entirely in a comment
(so it has to be retrievable on its own merits, not diluted into a long
page), and a comment has its own lifecycle — it can be added or deleted
without the page changing at all, which only a separate document can track.
"""
from __future__ import annotations

import re

from ingestion.confluence.client import ConfluenceComment, ConfluencePage
from ingestion.document import ParsedDocument

CONTENT_TYPE_PAGE = "page"
CONTENT_TYPE_COMMENT = "comment"

# Project roots are titled "<CODE>. <NAME>" ("A174. Гидроснаб"), but not
# always: some are a bare code ("A091", "A138"), some omit the space after
# the dot ("A054.Maintenance 2026"), and one uses a Cyrillic А ("А187.
# Киберпротект"). The code is matched on either alphabet, and `project` — the
# full title — is what's authoritative; code/name are the optional extras
# requirement 3 asks for only when they can be extracted reliably.
_PROJECT_TITLE_RE = re.compile(r"^\s*([AА]\d{2,4})\s*[.．]?\s*(.*)$")


def confluence_page_identity(page_id: str) -> str:
    """Stable local identity for a Confluence page.

    Neither the title nor the path can serve here — pages get renamed and
    moved, and either would make that look like "old document deleted, new
    document appeared". A content hash is equally wrong: an edit must update
    the SAME document, and two pages with identical text must not collapse
    into one. The page id is the only thing Confluence guarantees stable
    across edits, renames and moves.

    Stored in the `file_hashes` table (VARCHAR(32)) — the prefix keeps these
    from ever colliding with a real 32-hex-char file md5, and the `:page:`
    segment keeps them distinct from comment identities."""
    return f"confluence:page:{page_id}"


def confluence_comment_identity(comment_id: str) -> str:
    """Stable local identity for a comment — see confluence_page_identity."""
    return f"confluence:comment:{comment_id}"


def split_project_title(title: str) -> tuple[str | None, str | None]:
    """("A174. Гидроснаб") -> ("A174", "Гидроснаб"); ("A091") -> ("A091", None).

    Returns (None, None) for a title that doesn't start with a project code,
    rather than guessing — `project` (the full title) is the required field,
    these two are best-effort extras."""
    match = _PROJECT_TITLE_RE.match(title)
    if not match:
        return None, None
    code, name = match.group(1), match.group(2).strip()
    return code, (name or None)


def page_context_prefix(project: str, page_title: str, page_path: str) -> str:
    """What gets prepended to a chunk for embedding/reranking ONLY (see
    ingestion/chunker.py::chunk_context_text) — never to the stored or
    displayed text.

    Without it, a chunk deep in a project's subtree ("BOM", "Схема") carries
    no trace of which project it belongs to, so "компоненты A174" cannot
    match it: the code appears nowhere in the chunk's own words. The path
    does the same job for sibling pages that share a generic title —
    "Требования" exists under many projects."""
    return f"Project: {project}\nPage: {page_title}\nPath: {page_path}"


def comment_context_prefix(project: str, page_title: str, page_path: str) -> str:
    """Same idea for a comment, plus an explicit type marker so the model can
    tell a discussion remark from the page's own statement of fact.

    The author is deliberately NOT here: it belongs in metadata, and putting
    a person's name in every chunk of their comments would make the name a
    retrieval signal competing with the actual content."""
    return f"Project: {project}\nPage: {page_title}\nPath: {page_path}\nType: comment"


def confluence_page_to_parsed_document(
    page: ConfluencePage, *, project: str, page_path: str,
) -> ParsedDocument:
    """One page becomes a one-page ParsedDocument — Confluence has no
    pagination of its own, same as TxtParser's treatment of a text file."""
    text = page.text_content
    project_code, project_name = split_project_title(project)
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
            "content_type": CONTENT_TYPE_PAGE,
            "project": project,
            "project_code": project_code,
            "project_name": project_name,
            "page_id": page.id,
            "page_title": page.title,
            "page_path": page_path,
            "page_url": page.url,
            "space_key": page.space_key,
            "confluence_url": page.url,  # what api/query.py links citations to
            "version": page.version,
            "updated_at": page.updated_at.isoformat() if page.updated_at else None,
        },
        file_size_kb=len(text.encode("utf-8")) / 1024,
    )


def confluence_comment_to_parsed_document(
    comment: ConfluenceComment, page: ConfluencePage, *, project: str, page_path: str,
) -> ParsedDocument:
    """A comment's document is titled after its page, not after the comment
    (comments have no title of their own), so a citation reads as "the page
    this was said on" — which is what a reader needs to place it."""
    text = comment.text_content
    project_code, project_name = split_project_title(project)
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
            "content_type": CONTENT_TYPE_COMMENT,
            "project": project,
            "project_code": project_code,
            "project_name": project_name,
            "page_id": page.id,
            "page_title": page.title,
            "page_path": page_path,
            # The comment's own permalink when Confluence gave one
            # (focusedCommentId lands the reader on the comment itself);
            # otherwise the page, which is still the right place to look.
            "page_url": comment.url or page.url,
            "confluence_url": comment.url or page.url,
            "space_key": page.space_key,
            "comment_id": comment.id,
            "comment_author": comment.author,
            "comment_created_at": comment.created_at.isoformat() if comment.created_at else None,
            "comment_updated_at": comment.updated_at.isoformat() if comment.updated_at else None,
            "version": comment.version,
            "updated_at": comment.updated_at.isoformat() if comment.updated_at else None,
        },
        file_size_kb=len(text.encode("utf-8")) / 1024,
    )
