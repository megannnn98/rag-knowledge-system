"""Confluence sync — the second source feeding api/ingest.py.

Everything Confluence-specific ends here: below this module a synced page or
comment is an ordinary ParsedDocument, indistinguishable to the chunker, the
embedder, Qdrant, the retriever, the reranker and the generator from an
uploaded PDF. There is deliberately no Confluence-specific retrieval path.

The primary scope is a TREE: one root page id, everything under it, at any
depth. `/descendant/page` answers HTTP 500 on the target instance, so the
walk recurses through `/child/page` one level at a time. Space-wide sync
(sync_space) is kept as a backward-compatible secondary entry point.

Two things the tree gives every document that a flat space sync could not:

  - `project` — the first-level child of the root a page descends from. It
    comes from position in the tree, never from the page's text, so a page
    called "BOM" four levels down is still attributable to its project.
  - `page_path` — the titles from the root down to the page, which
    disambiguates the many identically-named "Требования"/"Схема" pages.

Both are also fed into the embedded/reranked text (see
ingestion/confluence/adapter.py::page_context_prefix), because a chunk of
"BOM" contains no trace of its project in its own words and could otherwise
never match "компоненты A174".

Identity is `confluence:page:<page_id>` / `confluence:comment:<comment_id>`,
stored in the same `file_hashes` table an upload's md5 goes into — so the
existing "do I already have this document" machinery works unchanged, while
an edited, renamed or MOVED page still maps to the same local document.

Update strategy is delete-then-reingest, not upsert-in-place. A page that
shrinks from 5 chunks to 3 would otherwise leave 2 stale chunks in Qdrant
forever: the deterministic point IDs only overwrite the chunk indices the
NEW version happens to reach (see vector_db/qdrant_client.py::
point_id_for_chunk), and nothing else would ever revisit the tail.
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
from ingestion.confluence.adapter import (
    CONTENT_TYPE_COMMENT,
    CONTENT_TYPE_PAGE,
    comment_context_prefix,
    confluence_comment_identity,
    confluence_comment_to_parsed_document,
    confluence_page_identity,
    confluence_page_to_parsed_document,
    page_context_prefix,
)
from ingestion.confluence.client import ConfluenceClient, ConfluenceClientError, ConfluenceConfigError

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_SPACE = os.getenv("CONFLUENCE_SPACE", "")
DEFAULT_ROOT_PAGE_ID = os.getenv("CONFLUENCE_ROOT_PAGE_ID", "")
CONFLUENCE_FORMAT = "confluence"

# A tree this deep is a structural anomaly (or a cycle the visited-set below
# somehow didn't catch), not real documentation. Bounded so a malformed tree
# cannot turn into unbounded recursion.
MAX_TREE_DEPTH = 25


class ConfluenceSyncRequest(BaseModel):
    """`root_page_id` (or the configured CONFLUENCE_ROOT_PAGE_ID) selects the
    tree sync; `space_key` selects the older flat space sync."""
    root_page_id: str = Field(default="", max_length=64)
    space_key: str = Field(default="", max_length=255)
    folder: str = Field(default="", max_length=255)
    limit: int | None = Field(default=None, ge=1, le=10000)


class TreeNode:
    """One page as the walk found it: the page itself plus where it sits."""

    def __init__(self, page, project: str, path: list[str], depth: int):
        self.page = page
        self.project = project
        self.path = path
        self.depth = depth

    @property
    def page_path(self) -> str:
        return " / ".join(self.path)


def _local_confluence_documents() -> dict[str, dict]:
    """identity -> document metadata, for everything Confluence we hold.

    Reads the in-memory registry (loaded from Postgres at startup and kept
    current by every ingest/delete), not Qdrant: Postgres is the source of
    truth for document metadata everywhere else in this app, and staying
    consistent with that is what lets a sync reason about "what do I have"
    without a vector-store scan."""
    import api.main as m

    local: dict[str, dict] = {}
    for identity, doc_id in m.file_hashes.items():
        if not identity.startswith("confluence:"):
            continue
        doc = m.documents_registry.get(doc_id)
        if doc is None or doc.get("status", "active") != "active":
            continue
        local[identity] = doc
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


async def walk_tree(client: ConfluenceClient, root_page_id: str, stats: dict) -> list[TreeNode]:
    """Every page under `root_page_id`, breadth-first, with its project and
    path. The root itself is NOT included — it is the container, not content.

    `stats["traversal_complete"]` is the safety interlock for deletions: it
    goes False the moment any subtree could not be listed, because a page
    missing from a partial walk is indistinguishable from a page that was
    deleted upstream, and acting on that confusion would delete real
    documents over a transient API error."""
    root_page = await client.get_page(root_page_id)
    stats["root_title"] = root_page.title
    stats["root_space"] = root_page.space_key

    visited: set[str] = {root_page_id}
    nodes: list[TreeNode] = []
    # (page_id, project, path_titles, depth) — the project of a first-level
    # child is itself; everything deeper inherits its ancestor's.
    queue: list[tuple[str, str, list[str], int]] = [(root_page_id, "", [root_page.title], 0)]

    while queue:
        page_id, project, path, depth = queue.pop(0)
        if depth >= MAX_TREE_DEPTH:
            logger.warning(f"Confluence tree: depth limit {MAX_TREE_DEPTH} reached at page {page_id} — not descending further")
            stats["depth_limited"] += 1
            continue
        try:
            children = await client.list_child_pages(page_id)
        except ConfluenceClientError as e:
            # One unreachable subtree must not abort the walk, but it does
            # mean we no longer know the full set of pages — see
            # traversal_complete above.
            logger.error(f"Confluence tree: listing children of {page_id} failed: {e}")
            stats["failed_listings"] += 1
            stats["traversal_complete"] = False
            continue

        for child in children:
            if child.id in visited:
                # A page can legitimately be reached twice only in a
                # malformed tree; indexing it twice would give one document
                # two identities and two sets of vectors.
                logger.warning(f"Confluence tree: page {child.id} ({child.title!r}) seen twice — skipping the repeat")
                stats["duplicates_skipped"] += 1
                continue
            visited.add(child.id)
            child_project = child.title if depth == 0 else project
            child_path = path + [child.title]
            nodes.append(TreeNode(child, child_project, child_path, depth + 1))
            queue.append((child.id, child_project, child_path, depth + 1))

    stats["pages_walked"] = len(nodes)
    stats["projects"] = sorted({n.project for n in nodes if n.depth == 1})
    stats["max_depth"] = max((n.depth for n in nodes), default=0)
    return nodes


def _has_text(text: str) -> bool:
    return bool(text and text.strip())


async def _ingest_page(node: TreeNode, page, folder: str) -> int:
    """Indexes one page's body. Returns chunks created, or 0 when the text
    was too short for the chunker to produce any.

    ValueError is swallowed HERE rather than at the sync loop: it means "this
    particular content is not indexable", which must not abort the rest of
    the node — a page whose body is too short still has comments worth
    indexing, and a short comment must not stop the comments after it."""
    parsed = confluence_page_to_parsed_document(page, project=node.project, page_path=node.page_path)
    doc_id = str(uuid.uuid4())
    try:
        async with ingestion_slot(doc_id):
            doc_meta = await ingest_parsed_document(
                parsed,
                doc_id=doc_id,
                filename=page.title,
                folder=folder,
                doc_format=CONFLUENCE_FORMAT,
                identity_key=confluence_page_identity(page.id),
                context_prefix=page_context_prefix(node.project, page.title, node.page_path),
                payload_metadata=parsed.metadata,
            )
    except ValueError:
        logger.info(f"Confluence page {page.id} ({page.title!r}) produced no chunks — skipped")
        return 0
    return doc_meta["chunks"]


async def _ingest_comment(node: TreeNode, page, comment, folder: str) -> int:
    """Indexes one comment. Returns chunks created, or 0 when it was too
    short to chunk — see _ingest_page for why that is handled here."""
    parsed = confluence_comment_to_parsed_document(
        comment, page, project=node.project, page_path=node.page_path
    )
    doc_id = str(uuid.uuid4())
    try:
        async with ingestion_slot(doc_id):
            doc_meta = await ingest_parsed_document(
                parsed,
                doc_id=doc_id,
                filename=page.title,
                folder=folder,
                doc_format=CONFLUENCE_FORMAT,
                identity_key=confluence_comment_identity(comment.id),
                context_prefix=comment_context_prefix(node.project, page.title, node.page_path),
                payload_metadata=parsed.metadata,
            )
    except ValueError:
        logger.debug(f"Confluence comment {comment.id} on page {page.id} produced no chunks — skipped")
        return 0
    return doc_meta["chunks"]


def _page_needs_reingest(existing: dict, page, node: TreeNode) -> bool:
    """True when the stored copy no longer matches Confluence.

    Version alone is not enough: MOVING a page does not necessarily bump its
    version, yet its project and path — which are part of what gets embedded
    — change. Comparing those too is what makes requirement 11's "moved
    page" case actually re-index rather than silently keep stale context."""
    meta = existing.get("metadata") or {}
    return (
        meta.get("version") != page.version
        or meta.get("project") != node.project
        or meta.get("page_path") != node.page_path
        or meta.get("page_title") != page.title
    )


def _comment_needs_reingest(existing: dict, comment, node: TreeNode, page) -> bool:
    meta = existing.get("metadata") or {}
    return (
        meta.get("version") != comment.version
        or meta.get("project") != node.project
        or meta.get("page_path") != node.page_path
        or meta.get("page_title") != page.title
    )


async def _sync_node(client: ConfluenceClient, node: TreeNode, local: dict, folder: str,
                     stats: dict, seen: set[str]) -> None:
    """Reconciles one page and its comments against what we already hold."""
    page = await client.get_page(node.page.id)
    page_identity = confluence_page_identity(page.id)
    page_has_text = _has_text(page.text_content)

    if page_has_text:
        seen.add(page_identity)
        existing = local.get(page_identity)
        if existing is None:
            chunks = await _ingest_page(node, page, folder)
            if chunks:
                stats["pages_indexed"] += 1
                stats["chunks_indexed"] += chunks
            else:
                stats["pages_skipped_empty"] += 1
                seen.discard(page_identity)
        elif _page_needs_reingest(existing, page, node):
            await _delete_document(existing["doc_id"])
            chunks = await _ingest_page(node, page, folder)
            if chunks:
                stats["pages_updated"] += 1
                stats["chunks_indexed"] += chunks
            else:
                stats["pages_skipped_empty"] += 1
                seen.discard(page_identity)
        else:
            stats["pages_unchanged"] += 1
    else:
        # A structural page (a project root that is just a container, a page
        # whose whole body is a diagram macro). It still participated in the
        # walk — its children have their project and path from it — but
        # there is nothing to embed, so no vector is created for it.
        stats["pages_structural"] += 1

    comments = await client.list_comments(page.id)
    stats["comments_found"] += len(comments)
    remote_comment_identities = set()
    for comment in comments:
        if not _has_text(comment.text_content):
            stats["comments_skipped_empty"] += 1
            continue
        identity = confluence_comment_identity(comment.id)
        remote_comment_identities.add(identity)
        seen.add(identity)
        existing = local.get(identity)
        if existing is None:
            chunks = await _ingest_comment(node, page, comment, folder)
            if chunks:
                stats["comments_indexed"] += 1
                stats["chunks_indexed"] += chunks
            else:
                stats["comments_skipped_empty"] += 1
                seen.discard(identity)
        elif _comment_needs_reingest(existing, comment, node, page):
            await _delete_document(existing["doc_id"])
            chunks = await _ingest_comment(node, page, comment, folder)
            if chunks:
                stats["comments_updated"] += 1
                stats["chunks_indexed"] += chunks
            else:
                stats["comments_skipped_empty"] += 1
                seen.discard(identity)
        else:
            stats["comments_unchanged"] += 1

    # Comments deleted upstream. Safe to act on without the global
    # traversal_complete interlock: this page WAS successfully listed just
    # now, so its comment set is known exactly — unlike a page's absence
    # from a partial walk, which proves nothing.
    #
    # Collected first, then deleted: `local` is what the end-of-sync cleanup
    # iterates too, and mutating it mid-iteration would both raise and leave
    # the cleanup trying to delete a document this loop already removed.
    stale = [
        identity for identity, doc in local.items()
        if identity.startswith("confluence:comment:")
        and (doc.get("metadata") or {}).get("page_id") == page.id
        and identity not in remote_comment_identities
    ]
    for identity in stale:
        doc = local.pop(identity)
        logger.info(f"Confluence comment {identity} is gone upstream — deleting local document {doc['doc_id']}")
        await _delete_document(doc["doc_id"])
        stats["comments_deleted"] += 1


def _new_stats(scope: str) -> dict:
    return {
        "scope": scope,
        "traversal_complete": True,
        "pages_walked": 0, "pages_indexed": 0, "pages_updated": 0, "pages_unchanged": 0,
        "pages_structural": 0, "pages_skipped_empty": 0, "pages_deleted": 0,
        "comments_found": 0, "comments_indexed": 0, "comments_updated": 0,
        "comments_unchanged": 0, "comments_skipped_empty": 0, "comments_deleted": 0,
        "chunks_indexed": 0, "failed_pages": 0, "failed_listings": 0,
        "duplicates_skipped": 0, "depth_limited": 0,
    }


async def sync_tree(root_page_id: str, folder: str = "", limit: int | None = None) -> dict:
    """Idempotent one-way sync of a Confluence page tree -> local index.

    Running it twice in a row with nothing changed upstream performs zero
    writes. That property is what makes it safe to schedule.
    """
    started = time.time()
    stats = _new_stats(f"tree:{root_page_id}")

    async with ConfluenceClient() as client:
        nodes = await walk_tree(client, root_page_id, stats)
        if limit is not None:
            nodes = nodes[:limit]
            stats["limited_to"] = limit
        logger.info(
            f"Confluence tree {root_page_id}: {len(nodes)} page(s) to reconcile, "
            f"{len(stats.get('projects', []))} project(s)"
        )

        local = _local_confluence_documents()
        seen: set[str] = set()

        for node in nodes:
            try:
                await _sync_node(client, node, local, folder, stats, seen)
            except ConfluenceClientError as e:
                logger.error(f"Confluence sync: page {node.page.id} failed: {e}")
                stats["failed_pages"] += 1
                stats["traversal_complete"] = False
            except ValueError:
                # ingest_parsed_document() raises this when a document yields
                # no chunks — text shorter than SmartChunker's minimum.
                # Nothing went wrong; there is nothing indexable.
                logger.info(f"Confluence page {node.page.id} ({node.page.title!r}) produced no chunks — skipped")
                stats["pages_skipped_empty"] += 1
            except Exception as e:
                logger.error(f"Confluence sync: page {node.page.id} failed to ingest: {e}")
                stats["failed_pages"] += 1
                stats["traversal_complete"] = False

        # Deleting local documents that the tree no longer contains is the
        # one destructive step, so it is gated on having actually seen the
        # whole tree. A partial walk (a failed listing, a failed page, or a
        # caller-imposed limit) makes "absent from the walk" mean nothing,
        # and acting on it would delete real documents over a transient API
        # error — the failure mode requirement 12 exists to prevent.
        can_delete = stats["traversal_complete"] and limit is None
        if can_delete:
            import api.main as m

            for identity, doc in list(local.items()):
                if identity in seen:
                    continue
                if doc["doc_id"] not in m.documents_registry:
                    # Already removed earlier in this same run (a comment
                    # whose page was re-synced). Deleting it again would hit
                    # a row that no longer exists.
                    continue
                logger.info(f"{identity} is gone upstream — deleting local document {doc['doc_id']}")
                await _delete_document(doc["doc_id"])
                if identity.startswith("confluence:comment:"):
                    stats["comments_deleted"] += 1
                else:
                    stats["pages_deleted"] += 1
        else:
            stale = [i for i in local if i not in seen]
            stats["deletions_skipped"] = len(stale)
            if stale:
                logger.warning(
                    f"Confluence sync: traversal incomplete (failed_listings="
                    f"{stats['failed_listings']}, failed_pages={stats['failed_pages']}, "
                    f"limit={limit}) — NOT deleting {len(stale)} document(s) absent from this run"
                )

    stats["duration_s"] = round(time.time() - started, 1)
    logger.info(f"Confluence tree sync {root_page_id} done: {stats}")
    return stats


async def sync_space(space_key: str, folder: str = "", limit: int | None = None) -> dict:
    """Flat, single-space sync — the original entry point, kept working.

    Indexes page bodies only (no tree, so no project/path context, and no
    comments): a space listing gives no parent/child structure to derive
    them from. Prefer sync_tree() for anything project-shaped."""
    started = time.time()
    stats = _new_stats(f"space:{space_key}")

    async with ConfluenceClient() as client:
        remote_pages = await client.list_pages(space_key, limit=limit)
        logger.info(f"Confluence space {space_key}: {len(remote_pages)} page(s) listed")
        stats["pages_walked"] = len(remote_pages)

        local = _local_confluence_documents()
        seen: set[str] = set()

        for listed in remote_pages:
            identity = confluence_page_identity(listed.id)
            existing = local.get(identity)
            try:
                if existing is not None and (existing.get("metadata") or {}).get("version") == listed.version:
                    stats["pages_unchanged"] += 1
                    seen.add(identity)
                    continue

                page = await client.get_page(listed.id)
                if not _has_text(page.text_content):
                    stats["pages_structural"] += 1
                    continue

                node = TreeNode(page, space_key, [space_key, page.title], 1)
                if existing is not None:
                    await _delete_document(existing["doc_id"])
                chunks = await _ingest_page(node, page, folder)
                if chunks:
                    seen.add(identity)
                    if existing is None:
                        stats["pages_indexed"] += 1
                    else:
                        stats["pages_updated"] += 1
                    stats["chunks_indexed"] += chunks
                else:
                    stats["pages_skipped_empty"] += 1
            except ConfluenceClientError as e:
                logger.error(f"Confluence sync: page {listed.id} failed: {e}")
                stats["failed_pages"] += 1
                stats["traversal_complete"] = False
            except ValueError:
                logger.info(f"Confluence page {listed.id} ({listed.title!r}) produced no chunks — skipped")
                stats["pages_skipped_empty"] += 1
            except Exception as e:
                logger.error(f"Confluence sync: page {listed.id} failed to ingest: {e}")
                stats["failed_pages"] += 1
                stats["traversal_complete"] = False

        if stats["traversal_complete"] and limit is None:
            for identity, doc in local.items():
                if identity in seen or identity.startswith("confluence:comment:"):
                    continue
                await _delete_document(doc["doc_id"])
                stats["pages_deleted"] += 1
        else:
            stats["deletions_skipped"] = len([i for i in local if i not in seen])

    stats["duration_s"] = round(time.time() - started, 1)
    logger.info(f"Confluence space sync {space_key} done: {stats}")
    return stats


@router.post("/confluence/sync", dependencies=[Depends(require_not_backing_up)])
async def confluence_sync(request: ConfluenceSyncRequest):
    root_page_id = request.root_page_id or DEFAULT_ROOT_PAGE_ID
    space_key = request.space_key or (DEFAULT_SPACE if not root_page_id else "")
    if not root_page_id and not space_key:
        raise HTTPException(422, "root_page_id is required (or set CONFLUENCE_ROOT_PAGE_ID)")
    try:
        if root_page_id:
            folder = request.folder or "Confluence"
            return await sync_tree(root_page_id, folder=folder, limit=request.limit)
        return await sync_space(space_key, folder=request.folder or space_key, limit=request.limit)
    except ConfluenceConfigError as e:
        # The message names the missing variable, never its value.
        raise HTTPException(503, f"Confluence is not configured: {e}")
    except ConfluenceClientError as e:
        raise HTTPException(502, f"Confluence sync failed: {e}")
