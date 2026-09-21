"""The one ingestion path: ParsedDocument -> chunks -> vectors -> Postgres.

Everything here used to live inline inside api/upload.py's upload_document()
body. It moved out when Confluence became a second source, so that both
sources run the SAME chunk/embed/upsert/persist code instead of Confluence
getting a parallel copy that could drift (different chunker settings, a
forgotten payload field, a missing db_save_ingestion call).

Format-specific work — parsing bytes off disk, or fetching a page over REST
— stays with each source. This module starts where they converge: a
ParsedDocument.

`import api.main as m` is done lazily INSIDE the functions, same as every
other api/* module: api.main imports these modules to wire its routers, and
tests monkeypatch attributes on the api.main module object itself.
"""
import asyncio
import contextlib
import logging
import os
import time

from ingestion.chunker import chunk_context_text
from rag.executors import run_on_gpu

logger = logging.getLogger(__name__)

# Bounds concurrent CPU-bound parse/OCR + GPU-bound embed work. Embedding is
# already serialized onto one GPU worker by run_on_gpu (rag/executors.py),
# but parsing/OCR runs via plain asyncio.to_thread, so nothing else limits
# how many of those can run in parallel. Shared by /upload, /upload-batch and
# Confluence sync — a sync running while someone uploads must not be able to
# double the in-flight ingestion work.
MAX_CONCURRENT_INGESTIONS = int(os.getenv("MAX_CONCURRENT_INGESTIONS", "2"))
_ingestion_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INGESTIONS)
_active_ingestions = 0  # only ever mutated between an `await` boundary and the next, see ingestion_slot() — safe without a separate lock under asyncio's single-threaded cooperative scheduling


@contextlib.asynccontextmanager
async def ingestion_slot(doc_id: str):
    """Wraps _ingestion_semaphore with observability: logs how many
    ingestion jobs are actually concurrently in this block right now, not
    just that a semaphore object exists — the thing worth being able to
    verify from logs under real concurrent load, not just trust from
    reading the semaphore size.

    NOT re-entrant: a caller already holding a slot must not call something
    that takes another, or two concurrent ingestions can deadlock against
    each other. Acquire it once, around the whole parse-to-persist span."""
    global _active_ingestions
    async with _ingestion_semaphore:
        _active_ingestions += 1
        logger.info(f"Ingestion slot acquired for {doc_id} ({_active_ingestions}/{MAX_CONCURRENT_INGESTIONS} in use)")
        try:
            yield
        finally:
            _active_ingestions -= 1
            logger.info(f"Ingestion slot released for {doc_id} ({_active_ingestions}/{MAX_CONCURRENT_INGESTIONS} in use)")


async def ingest_parsed_document(
    parsed,
    *,
    doc_id: str,
    filename: str,
    folder: str,
    doc_format: str,
    identity_key: str,
) -> dict:
    """Chunk, embed, upsert to Qdrant, then commit to Postgres — in that
    order, because the Postgres commit is what makes the document visible to
    the rest of the app, and it must never be the thing that's true while
    the vectors behind it are missing.

    `identity_key` is what goes into the file_hashes table: an uploaded
    file's md5, or `confluence:<page_id>` for a synced page. Whatever it is,
    it must be stable for the same logical document across re-ingestions —
    that's what makes a duplicate detectable at all.

    Raises ValueError when the document yields no chunks, and whatever
    db_save_ingestion raises (notably DuplicateFileHashError) untouched: the
    caller owns rollback, since only it knows what else it created (a file
    on disk, a half-finished sync) that this function never saw.

    The in-memory registries are updated only AFTER the commit returns, for
    the reason spelled out in db_save_ingestion()'s docstring: an entry with
    nothing behind it turns every later retry of the same document into a
    false 409."""
    import api.main as m

    chunks = m.chunker.chunk_document(parsed.pages, doc_id)
    if not chunks:
        raise ValueError("Could not extract text from document")

    for chunk in chunks:
        chunk.filename = filename
        chunk.pages = parsed.total_pages
        chunk.folder = folder or ""

    texts = [chunk_context_text(c) for c in chunks]
    t_embed = time.time()
    vectors = await run_on_gpu(m.embedder.embed_batch, texts)
    embed_ms = int((time.time() - t_embed) * 1000)
    await asyncio.to_thread(m.vector_store.upsert_chunks, chunks, vectors)

    doc_meta = {
        "doc_id": doc_id,
        "filename": filename,
        "pages": parsed.total_pages,
        "chunks": len(chunks),
        "size_kb": parsed.file_size_kb,
        "metadata": parsed.metadata,
        "folder": folder or "",
        "format": doc_format,
    }
    await asyncio.to_thread(m.db_save_ingestion, doc_meta, parsed.pages, identity_key)

    m.file_hashes[identity_key] = doc_id
    if folder:
        m.folders_registry.add(folder)
    m.documents_registry[doc_id] = doc_meta

    logger.info(
        f"Ingested {filename!r} | format={doc_format} pages={parsed.total_pages} "
        f"chunks={len(chunks)} embed_ms={embed_ms}"
    )
    return doc_meta
