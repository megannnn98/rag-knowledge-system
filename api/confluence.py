"""Confluence space sync — the second source feeding api/ingest.py.

Everything Confluence-specific ends here: below this module a synced page is
an ordinary ParsedDocument, indistinguishable to the chunker, the embedder,
Qdrant, the retriever, the reranker and the generator from an uploaded PDF.
There is deliberately no Confluence-specific retrieval path.

Identity is `confluence:<page_id>` (see ingestion/confluence/adapter.py),
stored in the same `file_hashes` table an upload's md5 goes into — so the
existing "do I already have this document" machinery works unchanged, while
a page edit still maps to the SAME local document instead of creating a
second one the way a content hash would.

Change detection uses the page's Confluence `version` number, recorded in
documents.metadata at ingestion. list_pages() expands version but not body,
so an unchanged space costs one listing request and no page downloads.

Update strategy is delete-then-reingest, not upsert-in-place. A page that
shrinks from 5 chunks to 3 would otherwise leave 2 stale chunks in Qdrant
forever: the deterministic point IDs only overwrite the chunk indices the
NEW version happens to reach (see vector_db/qdrant_client.py::
point_id_for_chunk), and nothing else would ever revisit the tail. Deleting
the whole document first makes stale vectors structurally impossible rather
than something a future edit has to remember to clean up.
"""
import asyncio
import logging
import os
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import require_not_backing_up
from api.ingest import ingest_parsed_document, ingestion_slot
from ingestion.confluence.adapter import confluence_identity, confluence_page_to_parsed_document
from ingestion.confluence.client import ConfluenceClient, ConfluenceClientError, ConfluenceConfigError

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_SPACE = os.getenv("CONFLUENCE_SPACE", "")
CONFLUENCE_FORMAT = "confluence"


class ConfluenceSyncRequest(BaseModel):
    space_key: str = Field(default="", max_length=255)
    # Folder the synced pages land in, so they can be scoped/filtered in the
    # UI like any other folder. Defaults to the space key.
    folder: str = Field(default="", max_length=255)
    limit: int | None = Field(default=None, ge=1, le=10000)


def _local_confluence_documents(space_key: str) -> dict[str, dict]:
    """page_id -> document metadata, for pages of this space we already hold.

    Reads the in-memory registry (loaded from Postgres at startup and kept
    current by every ingest/delete), not Qdrant: Postgres is the source of
    truth for document metadata everywhere else in this app, and staying
    consistent with that is what lets a sync reason about "what do I have"
    without a vector-store scan."""
    import api.main as m

    local: dict[str, dict] = {}
    for doc in m.documents_registry.values():
        if doc.get("status", "active") != "active":
            continue
        meta = doc.get("metadata") or {}
        if meta.get("source") != "confluence":
            continue
        if space_key and meta.get("space_key") != space_key:
            continue
        page_id = str(meta.get("page_id", ""))
        if page_id:
            local[page_id] = doc
    return local


async def _delete_document(doc_id: str) -> None:
    """Full cross-store delete, reusing the exact path the DELETE
    /documents/{id} endpoint uses — durable 'deleting' tombstone first, then
    Qdrant points, Postgres rows and the file. Going through the same
    reconciliation helper means a sync interrupted mid-delete is picked up
    by the startup sweep like any other half-finished delete, instead of
    leaving vectors only this module knew about."""
    import api.main as m

    await asyncio.to_thread(m.db_mark_document_deleting, doc_id)
    doc = m.documents_registry.get(doc_id)
    if doc is not None:
        doc["status"] = "deleting"
    await m._reconcile_document_deletion(doc_id)


async def _ingest_page(client: ConfluenceClient, page_id: str, folder: str) -> int:
    """Fetches one page's body and runs it through the shared pipeline.
    Returns the number of chunks created."""
    page = await client.get_page(page_id)
    parsed = confluence_page_to_parsed_document(page)
    if not parsed.pages[0]["text"].strip():
        logger.info(f"Confluence page {page_id} ({page.title!r}) has no extractable text — skipped")
        return 0

    doc_id = str(uuid.uuid4())
    async with ingestion_slot(doc_id):
        doc_meta = await ingest_parsed_document(
            parsed,
            doc_id=doc_id,
            filename=page.title,
            folder=folder,
            doc_format=CONFLUENCE_FORMAT,
            identity_key=confluence_identity(page_id),
        )
    return doc_meta["chunks"]


async def sync_space(space_key: str, folder: str = "", limit: int | None = None) -> dict:
    """Idempotent one-way sync Confluence -> local index.

    Running it twice in a row with nothing changed upstream performs zero
    writes: every page lands in `unchanged`. That property is what makes it
    safe to schedule.
    """
    import api.main as m

    started = time.time()
    async with ConfluenceClient() as client:
        remote_pages = await client.list_pages(space_key, limit=limit)
        logger.info(f"Confluence sync {space_key}: {len(remote_pages)} page(s) listed")

        local = _local_confluence_documents(space_key)
        stats = {"space": space_key, "remote_pages": len(remote_pages),
                 "added": 0, "updated": 0, "unchanged": 0, "deleted": 0,
                 "skipped_empty": 0, "failed": 0, "chunks_indexed": 0}
        seen: set[str] = set()

        for page in remote_pages:
            seen.add(page.id)
            existing = local.get(page.id)
            try:
                if existing is None:
                    chunks = await _ingest_page(client, page.id, folder)
                    if chunks:
                        stats["added"] += 1
                        stats["chunks_indexed"] += chunks
                    else:
                        stats["skipped_empty"] += 1
                    continue

                local_version = (existing.get("metadata") or {}).get("version")
                if local_version == page.version:
                    stats["unchanged"] += 1
                    continue

                # Changed: the old document (and every vector behind it) goes
                # first, so the new version can never be merged with leftovers
                # of the old one.
                await _delete_document(existing["doc_id"])
                chunks = await _ingest_page(client, page.id, folder)
                if chunks:
                    stats["updated"] += 1
                    stats["chunks_indexed"] += chunks
                else:
                    stats["skipped_empty"] += 1
            except ValueError:
                # ingest_parsed_document() raises this when the page yields
                # no chunks — in practice a page whose whole body is a
                # diagram/macro, or one shorter than SmartChunker's
                # min_chunk_size. Nothing went wrong; there is simply
                # nothing indexable, same as the explicit empty-text case
                # above, so it must not be reported as a failure.
                logger.info(f"Confluence page {page.id} ({page.title!r}) produced no chunks — skipped")
                stats["skipped_empty"] += 1
            except ConfluenceClientError as e:
                # One unreachable page must not abort the whole space — the
                # rest of the sync is still useful, and the next run retries
                # this page from wherever it left off.
                logger.error(f"Confluence sync: page {page.id} failed: {e}")
                stats["failed"] += 1
            except Exception as e:
                logger.error(f"Confluence sync: page {page.id} failed to ingest: {e}")
                stats["failed"] += 1

        # Pages we hold that the space no longer lists. Only done for a full
        # sync: with `limit` set, the listing is a prefix of the space, not
        # the whole of it, so absence proves nothing.
        if limit is None:
            for page_id, doc in local.items():
                if page_id in seen:
                    continue
                logger.info(f"Confluence page {page_id} is gone upstream — deleting local document {doc['doc_id']}")
                await _delete_document(doc["doc_id"])
                stats["deleted"] += 1

    stats["duration_s"] = round(time.time() - started, 1)
    logger.info(f"Confluence sync {space_key} done: {stats}")
    return stats


@router.post("/confluence/sync", dependencies=[Depends(require_not_backing_up)])
async def confluence_sync(request: ConfluenceSyncRequest):
    space_key = request.space_key or DEFAULT_SPACE
    if not space_key:
        raise HTTPException(422, "space_key is required (or set CONFLUENCE_SPACE)")
    folder = request.folder or space_key
    try:
        return await sync_space(space_key, folder=folder, limit=request.limit)
    except ConfluenceConfigError as e:
        # The message names the missing variable, never its value.
        raise HTTPException(503, f"Confluence is not configured: {e}")
    except ConfluenceClientError as e:
        raise HTTPException(502, f"Confluence sync failed: {e}")
