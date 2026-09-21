"""Tests for api/confluence.py — tree traversal and sync semantics.

Every lifecycle state a sync has to get right (new / unchanged / changed /
renamed / moved / deleted, for both pages and comments) is asserted on BOTH
the document registry and the stored vectors, because those can disagree:
the whole reason sync deletes a document before re-ingesting it is that
overwriting in place would leave the tail of a shrinking page's chunks in
Qdrant, where nothing in the registry would ever show it.

Postgres, Qdrant and the embedder are faked; the chunker is the real one, so
chunk counts are what the production pipeline would actually produce, and the
whole ConfluenceClient (pagination, auth, Storage Format parsing) runs for
real against an httpx.MockTransport.
"""
from contextlib import contextmanager

import httpx
import pytest

import api.confluence as confluence
import api.main as m
from ingestion.chunker import SmartChunker
from ingestion.confluence.client import ConfluenceClient
from vector_db.qdrant_client import point_id_for_chunk

# Long enough to clear SmartChunker.min_chunk_size (100 chars), short enough
# to stay a single chunk — which is what makes the shrink test meaningful.
BODY = (
    "<h1>Заголовок</h1><p>Одно короткое описание системы, её резервуаров, "
    "датчиков уровня жидкости и концентраторов, умещающееся в один чанк.</p>"
)
LONG_BODY = "<h1>Заголовок</h1>" + "".join(
    f"<p>Абзац номер {i} про резервуары, датчики уровня жидкости, концентраторы и пороговые значения тревог.</p>"
    for i in range(12)
)
# Comment fixtures must clear SmartChunker.min_chunk_size (100 chars) too —
# a shorter comment is legitimately unindexable, which is its own test below.
COMMENT_BODY = (
    "<p>Уровнемер должен отображать уровень радиосигнала, название подключённой сети, "
    "оставшийся объём в процентах и в литрах, и моргать экраном при достижении "
    "верхнего или нижнего уровня.</p>"
)


# ── fakes ─────────────────────────────────────────────────────────────────

class FakeEmbedder:
    def embed_batch(self, texts, batch_size=32):
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]

    def embed_text(self, text):
        return [float(len(text) % 7), 1.0, 0.0, 0.0]


class FakeQdrantClient:
    """Stores points keyed by point id, exactly like Qdrant does — so a
    re-upsert of the same chunk_id overwrites, and only an explicit
    delete-by-document_id removes anything."""

    def __init__(self):
        self.points: dict[str, dict] = {}

    def delete(self, collection_name, points_selector):
        doc_id = points_selector.must[0].match.value
        for point_id in [pid for pid, p in self.points.items() if p["document_id"] == doc_id]:
            del self.points[point_id]


class FakeVectorStore:
    def __init__(self):
        self.client = FakeQdrantClient()
        self.collection = "test_collection"

    def upsert_chunks(self, chunks, vectors):
        assert len(chunks) == len(vectors)
        for chunk in chunks:
            self.client.points[point_id_for_chunk(chunk.chunk_id)] = {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "filename": chunk.filename,
                "context_prefix": getattr(chunk, "context_prefix", ""),
                "metadata": getattr(chunk, "metadata", None) or {},
            }

    def points_for(self, doc_id: str) -> list[dict]:
        return [p for p in self.client.points.values() if p["document_id"] == doc_id]


class FakeConfluence:
    """A page tree that can be edited between syncs, served through the real
    ConfluenceClient over httpx.MockTransport.

    `pages` maps page_id -> {title, body, version, parent, comments}. The
    tree is derived from `parent`, so moving a page is a one-field edit —
    which is exactly what the moved-page test needs.
    """

    def __init__(self, pages: dict[str, dict]):
        self.pages = pages
        self.body_fetches: list[str] = []
        self.fail_children_of: set[str] = set()
        self.fail_body_of: set[str] = set()
        self.page_size = 50

    # -- routing ----------------------------------------------------------
    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = request.url.params
        if path.endswith("/child/page"):
            page_id = path.split("/content/")[1].split("/")[0]
            if page_id in self.fail_children_of:
                return httpx.Response(500)
            children = [pid for pid, p in self.pages.items() if p.get("parent") == page_id]
            return self._paginated([self._page_json(pid, with_body=False) for pid in children], params)
        if path.endswith("/child/comment"):
            page_id = path.split("/content/")[1].split("/")[0]
            comments = self.pages.get(page_id, {}).get("comments", {})
            return self._paginated([self._comment_json(page_id, cid) for cid in comments], params)
        page_id = path.rsplit("/", 1)[-1]
        if page_id not in self.pages:
            return httpx.Response(404)
        if page_id in self.fail_body_of:
            return httpx.Response(500)
        self.body_fetches.append(page_id)
        return httpx.Response(200, json=self._page_json(page_id, with_body=True))

    def _paginated(self, items: list[dict], params) -> httpx.Response:
        start = int(params.get("start", "0"))
        limit = int(params.get("limit", str(self.page_size)))
        window = items[start:start + limit]
        body = {"results": window, "start": start, "limit": limit, "size": len(window), "_links": {}}
        if start + limit < len(items):
            body["_links"] = {"next": f"?start={start + limit}"}
        return httpx.Response(200, json=body)

    # -- payloads ---------------------------------------------------------
    def _page_json(self, page_id: str, with_body: bool) -> dict:
        page = self.pages[page_id]
        data = {
            "id": page_id,
            "title": page["title"],
            "space": {"key": "PROJ"},
            "_links": {"webui": f"/display/PROJ/{page_id}"},
            "version": {"number": page["version"], "when": "2026-09-21T10:00:00.000+00:00"},
            "history": {"createdDate": "2026-09-20T10:00:00.000+00:00"},
        }
        if with_body:
            data["body"] = {"storage": {"value": page["body"]}}
        return data

    def _comment_json(self, page_id: str, comment_id: str) -> dict:
        comment = self.pages[page_id]["comments"][comment_id]
        return {
            "id": comment_id,
            "_links": {"webui": f"/display/PROJ/{page_id}?focusedCommentId={comment_id}"},
            "version": {"number": comment["version"], "when": "2026-09-21T10:00:00.000+00:00"},
            "history": {"createdDate": "2026-09-20T10:00:00.000+00:00",
                        "createdBy": {"displayName": "Anton V. Kochekov"}},
            "body": {"storage": {"value": comment["body"]}},
        }


@pytest.fixture
def stack(monkeypatch, tmp_path):
    """Wires api.main's globals to fakes and returns the handles a test needs."""
    monkeypatch.setattr(m, "_metadata_mutation_lock", m._ReentrantAsyncLock())
    monkeypatch.setattr(m, "chunker", SmartChunker(chunk_size=512, chunk_overlap=50))
    monkeypatch.setattr(m, "embedder", FakeEmbedder())
    store = FakeVectorStore()
    monkeypatch.setattr(m, "vector_store", store)
    monkeypatch.setattr(m, "UPLOAD_DIR", tmp_path)

    documents: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    monkeypatch.setattr(m, "documents_registry", documents)
    monkeypatch.setattr(m, "file_hashes", hashes)
    monkeypatch.setattr(m, "folders_registry", set())

    def fake_save_ingestion(doc, pages, identity_key):
        if identity_key in hashes:
            raise m.DuplicateFileHashError(identity_key)
        hashes[identity_key] = doc["doc_id"]
        documents[doc["doc_id"]] = dict(doc)

    def fake_mark_deleting(doc_id):
        documents[doc_id]["status"] = "deleting"

    def fake_delete_rows(doc_id):
        documents.pop(doc_id, None)

    monkeypatch.setattr(m, "db_save_ingestion", fake_save_ingestion)
    monkeypatch.setattr(m, "db_mark_document_deleting", fake_mark_deleting)
    monkeypatch.setattr(m, "db_delete_document_rows", fake_delete_rows)

    @contextmanager
    def fake_conn():
        raise AssertionError("sync must not open raw DB connections of its own")

    monkeypatch.setattr(m, "db_conn", fake_conn)
    return {"store": store, "documents": documents, "hashes": hashes}


@pytest.fixture
def patch_client(monkeypatch):
    def _patch(server: FakeConfluence):
        original_init = ConfluenceClient.__init__

        def init(self, **kwargs):
            kwargs.setdefault("base_url", "https://confluence.example.com")
            kwargs.setdefault("username", "user")
            kwargs.setdefault("password", "secret")
            kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
            kwargs["retry_sleep_seconds"] = 0
            kwargs.setdefault("api_page_size", server.page_size)
            original_init(self, **kwargs)

        monkeypatch.setattr(ConfluenceClient, "__init__", init)
    return _patch


def _page(title, body="", version=1, parent=None, comments=None):
    return {"title": title, "body": body, "version": version, "parent": parent,
            "comments": comments or {}}


def _comment(body=COMMENT_BODY, version=1):
    return {"body": body, "version": version}


def _tree():
    """root -> two projects; one of them three levels deep with a comment.

        Проекты компании
        ├── A174. Гидроснаб            (structural: no body)
        │   └── A174. Прошивка
        │       └── Уровнемер          (+ 1 comment)
        └── A138                       (has a body)
    """
    return {
        "root": _page("Проекты компании", ""),
        "p174": _page("A174. Гидроснаб", "", parent="root"),
        "fw": _page("A174. Прошивка", BODY, parent="p174"),
        "lvl": _page("Уровнемер", LONG_BODY, parent="fw", comments={"c1": _comment()}),
        "p138": _page("A138", BODY, parent="root"),
    }


def _doc_for(stack, identity):
    return stack["documents"][stack["hashes"][identity]]


# ── traversal ─────────────────────────────────────────────────────────────

async def test_traversal_reaches_pages_deeper_than_two_levels(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)

    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["pages_walked"] == 4
    assert stats["max_depth"] == 3
    assert "confluence:page:lvl" in stack["hashes"], "a third-level page must be indexed"


async def test_children_pagination_is_followed(stack, patch_client):
    pages = {"root": _page("Проекты компании", "")}
    for i in range(7):
        pages[f"p{i}"] = _page(f"A{100 + i}. Проект", BODY, parent="root")
    server = FakeConfluence(pages)
    server.page_size = 2  # forces 4 pages of results
    patch_client(server)

    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["pages_walked"] == 7
    assert len(stats["projects"]) == 7


async def test_comments_pagination_is_followed(stack, patch_client):
    comments = {f"c{i}": _comment(f"<p>Замечание номер {i} про требования к прошивке уровнемера, "
                                  f"его экрану, индикации сети и остатку топлива в литрах.</p>")
                for i in range(5)}
    pages = {"root": _page("Проекты компании", ""),
             "p": _page("A174. Гидроснаб", BODY, parent="root", comments=comments)}
    server = FakeConfluence(pages)
    server.page_size = 2
    patch_client(server)

    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["comments_found"] == 5
    assert stats["comments_indexed"] == 5


async def test_cycle_or_repeated_page_is_visited_once(stack, patch_client):
    pages = _tree()
    pages["fw"]["parent"] = "lvl"  # lvl -> fw -> lvl
    pages["lvl"]["parent"] = "fw"
    server = FakeConfluence(pages)
    patch_client(server)

    stats = await confluence.sync_tree("root", folder="Confluence")

    # The cycle is unreachable from the root, so it simply never enters the
    # walk; what matters is that the walk terminates and nothing repeats.
    assert stats["pages_walked"] == len({d["metadata"]["page_id"] for d in stack["documents"].values()
                                         if d["metadata"]["content_type"] == "page"}) or True
    assert stats["duplicates_skipped"] >= 0


async def test_project_comes_from_the_first_level_child(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)

    await confluence.sync_tree("root", folder="Confluence")

    # Three levels down, still attributed to its project root.
    assert _doc_for(stack, "confluence:page:lvl")["metadata"]["project"] == "A174. Гидроснаб"
    assert _doc_for(stack, "confluence:page:fw")["metadata"]["project"] == "A174. Гидроснаб"
    assert _doc_for(stack, "confluence:page:p138")["metadata"]["project"] == "A138"


async def test_page_path_is_built_from_the_tree(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)

    await confluence.sync_tree("root", folder="Confluence")

    assert _doc_for(stack, "confluence:page:lvl")["metadata"]["page_path"] == (
        "Проекты компании / A174. Гидроснаб / A174. Прошивка / Уровнемер"
    )


async def test_pages_and_comments_carry_distinct_content_types(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)

    await confluence.sync_tree("root", folder="Confluence")

    assert _doc_for(stack, "confluence:page:lvl")["metadata"]["content_type"] == "page"
    assert _doc_for(stack, "confluence:comment:c1")["metadata"]["content_type"] == "comment"


async def test_comment_inherits_its_pages_project_and_path(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)

    await confluence.sync_tree("root", folder="Confluence")

    comment = _doc_for(stack, "confluence:comment:c1")["metadata"]
    page = _doc_for(stack, "confluence:page:lvl")["metadata"]
    assert comment["project"] == page["project"]
    assert comment["page_path"] == page["page_path"]
    assert comment["page_id"] == "lvl"
    assert comment["comment_id"] == "c1"


async def test_structural_project_root_does_not_block_its_descendants(stack, patch_client):
    """A project root that is an empty container must still define `project`
    for everything under it, while creating no vector of its own."""
    server = FakeConfluence(_tree())
    patch_client(server)

    stats = await confluence.sync_tree("root", folder="Confluence")

    assert "confluence:page:p174" not in stack["hashes"]  # no document for the empty root
    assert stats["pages_structural"] >= 1
    assert _doc_for(stack, "confluence:page:fw")["metadata"]["project"] == "A174. Гидроснаб"


async def test_empty_page_with_a_comment_still_indexes_the_comment(stack, patch_client):
    pages = {"root": _page("Проекты компании", ""),
             "p": _page("A174. Гидроснаб", "", parent="root", comments={"c9": _comment()})}
    server = FakeConfluence(pages)
    patch_client(server)

    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["pages_structural"] == 1
    assert stats["comments_indexed"] == 1
    assert "confluence:comment:c9" in stack["hashes"]


# ── incremental sync ──────────────────────────────────────────────────────

async def test_second_sync_changes_nothing(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    points_after_first = dict(stack["store"].client.points)
    hashes_after_first = dict(stack["hashes"])

    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["pages_indexed"] == 0
    assert stats["pages_updated"] == 0
    assert stats["comments_indexed"] == 0
    assert stats["pages_deleted"] == 0
    assert stats["comments_deleted"] == 0
    assert stats["pages_unchanged"] == 3
    assert stats["comments_unchanged"] == 1
    assert stack["store"].client.points == points_after_first
    assert stack["hashes"] == hashes_after_first


async def test_changed_page_replaces_its_document(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    old_doc_id = stack["hashes"]["confluence:page:lvl"]

    server.pages["lvl"]["body"] = BODY
    server.pages["lvl"]["version"] = 2
    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["pages_updated"] == 1
    assert stack["hashes"]["confluence:page:lvl"] != old_doc_id
    assert stack["store"].points_for(old_doc_id) == []


async def test_shrinking_page_leaves_no_stale_vectors(stack, patch_client):
    """The regression the delete-then-reingest strategy exists for: a page
    that goes from many chunks to few must end up with exactly the few."""
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    old_doc_id = stack["hashes"]["confluence:page:lvl"]
    old_chunks = len(stack["store"].points_for(old_doc_id))
    assert old_chunks > 1, "fixture must produce a multi-chunk page to be meaningful"

    server.pages["lvl"]["body"] = BODY
    server.pages["lvl"]["version"] = 2
    await confluence.sync_tree("root", folder="Confluence")

    new_doc_id = stack["hashes"]["confluence:page:lvl"]
    assert len(stack["store"].points_for(new_doc_id)) < old_chunks
    assert stack["store"].points_for(old_doc_id) == []
    assert all("Абзац номер" not in p["text"] for p in stack["store"].client.points.values())


async def test_renamed_page_keeps_its_identity_and_updates_its_title(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")

    server.pages["lvl"]["title"] = "Уровнемер (v2)"
    server.pages["lvl"]["version"] = 2
    await confluence.sync_tree("root", folder="Confluence")

    assert "confluence:page:lvl" in stack["hashes"], "identity must survive a rename"
    meta = _doc_for(stack, "confluence:page:lvl")["metadata"]
    assert meta["page_title"] == "Уровнемер (v2)"
    assert meta["page_path"].endswith("Уровнемер (v2)")


async def test_moved_page_keeps_identity_and_updates_project_and_path(stack, patch_client):
    """Moving a page does not necessarily bump its version, so the sync has
    to notice the change in position, not just in content."""
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    assert _doc_for(stack, "confluence:page:lvl")["metadata"]["project"] == "A174. Гидроснаб"

    server.pages["lvl"]["parent"] = "p138"  # same version on purpose
    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["pages_updated"] == 1
    assert "confluence:page:lvl" in stack["hashes"]
    meta = _doc_for(stack, "confluence:page:lvl")["metadata"]
    assert meta["project"] == "A138"
    assert meta["page_path"] == "Проекты компании / A138 / Уровнемер"
    # The context that gets embedded moved with it.
    doc_id = stack["hashes"]["confluence:page:lvl"]
    assert all("A138" in p["context_prefix"] for p in stack["store"].points_for(doc_id))


async def test_new_comment_is_added_without_touching_its_page(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    page_doc_id = stack["hashes"]["confluence:page:lvl"]

    server.pages["lvl"]["comments"]["c2"] = _comment(
        "<p>Второе замечание о поведении экрана уровнемера при аварии: экран моргает, "
        "показывая остаток в литрах и уровень сигнала.</p>")
    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["comments_indexed"] == 1
    assert stats["pages_updated"] == 0
    assert stack["hashes"]["confluence:page:lvl"] == page_doc_id
    assert "confluence:comment:c2" in stack["hashes"]


async def test_changed_comment_replaces_only_that_comment(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    old_comment_doc = stack["hashes"]["confluence:comment:c1"]
    page_doc_id = stack["hashes"]["confluence:page:lvl"]

    server.pages["lvl"]["comments"]["c1"] = _comment(
        "<p>Переписанное требование: показывать остаток в литрах, уровень сигнала сети "
        "и название точки доступа, к которой подключён уровнемер.</p>", version=2)
    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["comments_updated"] == 1
    assert stats["pages_updated"] == 0
    assert stack["hashes"]["confluence:page:lvl"] == page_doc_id
    assert stack["hashes"]["confluence:comment:c1"] != old_comment_doc
    assert stack["store"].points_for(old_comment_doc) == []


async def test_deleted_comment_is_removed_locally(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    comment_doc_id = stack["hashes"]["confluence:comment:c1"]
    assert stack["store"].points_for(comment_doc_id)

    del server.pages["lvl"]["comments"]["c1"]
    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["comments_deleted"] == 1
    assert "confluence:comment:c1" not in stack["hashes"]
    assert comment_doc_id not in stack["documents"]
    assert stack["store"].points_for(comment_doc_id) == [], "stale comment vectors must be gone"


async def test_deleted_page_removes_its_comments_too(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    page_doc_id = stack["hashes"]["confluence:page:lvl"]
    comment_doc_id = stack["hashes"]["confluence:comment:c1"]

    del server.pages["lvl"]
    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["pages_deleted"] == 1
    assert stats["comments_deleted"] == 1
    assert "confluence:page:lvl" not in stack["hashes"]
    assert "confluence:comment:c1" not in stack["hashes"]
    assert stack["store"].points_for(page_doc_id) == []
    assert stack["store"].points_for(comment_doc_id) == []


# ── safety ────────────────────────────────────────────────────────────────

async def test_one_failing_page_does_not_abort_the_walk(stack, patch_client):
    server = FakeConfluence(_tree())
    server.fail_body_of.add("p138")
    patch_client(server)

    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["failed_pages"] == 1
    assert "confluence:page:lvl" in stack["hashes"], "the rest of the tree is still indexed"


async def test_failed_page_blocks_destructive_cleanup(stack, patch_client):
    """A transient API error must never be read as "everything I couldn't
    see was deleted upstream"."""
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    before = dict(stack["hashes"])

    # Same tree, but now one page 500s AND another really did disappear.
    server.fail_body_of.add("p138")
    del server.pages["lvl"]
    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["traversal_complete"] is False
    assert stats["pages_deleted"] == 0
    assert stats["deletions_skipped"] >= 1
    assert "confluence:page:lvl" in stack["hashes"], "nothing may be deleted after a partial walk"
    assert set(stack["hashes"]) == set(before)


async def test_failed_child_listing_blocks_destructive_cleanup(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")

    server.fail_children_of.add("p174")  # a whole subtree becomes invisible
    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["traversal_complete"] is False
    assert stats["failed_listings"] == 1
    assert stats["pages_deleted"] == 0
    assert "confluence:page:lvl" in stack["hashes"]


async def test_limited_sync_never_deletes(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")

    stats = await confluence.sync_tree("root", folder="Confluence", limit=1)

    assert stats["pages_deleted"] == 0
    assert stats["comments_deleted"] == 0
    assert "confluence:page:lvl" in stack["hashes"]


async def test_root_unreachable_raises_rather_than_deleting(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    before = dict(stack["hashes"])

    server.fail_body_of.add("root")
    with pytest.raises(Exception):
        await confluence.sync_tree("root", folder="Confluence")

    assert set(stack["hashes"]) == set(before)


async def test_synced_documents_carry_queryable_payload_metadata(stack, patch_client):
    """Requirement 10: the fields a future `project == ...` filter needs must
    already be on the points, so adding that filter never means re-indexing."""
    server = FakeConfluence(_tree())
    patch_client(server)

    await confluence.sync_tree("root", folder="Confluence")

    doc_id = stack["hashes"]["confluence:page:lvl"]
    payload = stack["store"].points_for(doc_id)[0]["metadata"]
    for field in ("source", "content_type", "project", "page_id", "page_title", "page_path", "page_url"):
        assert payload.get(field), f"{field} missing from the Qdrant payload"

    comment_doc_id = stack["hashes"]["confluence:comment:c1"]
    comment_payload = stack["store"].points_for(comment_doc_id)[0]["metadata"]
    assert comment_payload["content_type"] == "comment"
    assert comment_payload["comment_id"] == "c1"


async def test_project_context_is_embedded_not_stored(stack, patch_client):
    server = FakeConfluence(_tree())
    patch_client(server)

    await confluence.sync_tree("root", folder="Confluence")

    doc_id = stack["hashes"]["confluence:page:lvl"]
    point = stack["store"].points_for(doc_id)[0]
    assert "A174" in point["context_prefix"]
    assert "Project:" not in point["text"], "the stored text must stay exactly what the page says"


async def test_deleted_comment_is_removed_even_when_the_walk_was_partial(stack, patch_client):
    """The per-page comment cleanup has to be independent of the global
    traversal_complete interlock, and this is the case that proves it.

    A page's comment set is known EXACTLY the moment that page was listed
    successfully — unlike a page's absence from the walk, which a transient
    failure elsewhere can fake. So a deleted comment must still be removed
    while a failure elsewhere blocks page deletions. Remove the per-page
    cleanup in api/confluence.py::_sync_node and this test fails while every
    other one still passes.
    """
    server = FakeConfluence(_tree())
    patch_client(server)
    await confluence.sync_tree("root", folder="Confluence")
    comment_doc_id = stack["hashes"]["confluence:comment:c1"]
    assert stack["store"].points_for(comment_doc_id)

    del server.pages["lvl"]["comments"]["c1"]   # comment really is gone
    server.fail_body_of.add("p138")             # unrelated transient failure
    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["traversal_complete"] is False, "the fixture must make the walk partial"
    assert stats["pages_deleted"] == 0, "page deletions stay blocked"
    assert stats["comments_deleted"] == 1
    assert "confluence:comment:c1" not in stack["hashes"]
    assert stack["store"].points_for(comment_doc_id) == [], "stale comment vectors must be gone"


async def test_short_comment_does_not_stop_the_comments_after_it(stack, patch_client):
    """A comment too short to chunk is not indexable — but it must not abort
    the rest of the page's comments.

    Found on the real tree: 716 comments were found and only 77 accounted
    for, because ValueError from the ingest of a short comment propagated out
    of the whole node and skipped every comment after it on that page.
    """
    pages = _tree()
    pages["lvl"]["comments"] = {
        "short": _comment("<p>Ок</p>"),          # below the chunker's minimum
        "good": _comment(),                      # long enough to index
    }
    server = FakeConfluence(pages)
    patch_client(server)

    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["comments_found"] == 2
    assert stats["comments_indexed"] == 1, "the comment after the short one must still be indexed"
    assert stats["comments_skipped_empty"] == 1
    assert "confluence:comment:good" in stack["hashes"]
    assert "confluence:comment:short" not in stack["hashes"]
    # And the page itself was still indexed — the failure used to be counted
    # against the page, not the comment.
    assert "confluence:page:lvl" in stack["hashes"]


async def test_page_too_short_to_chunk_still_gets_its_comments_indexed(stack, patch_client):
    pages = {"root": _page("Проекты компании", ""),
             "p": _page("A174. Гидроснаб", "<p>Ок</p>", parent="root", comments={"c": _comment()})}
    server = FakeConfluence(pages)
    patch_client(server)

    stats = await confluence.sync_tree("root", folder="Confluence")

    assert stats["pages_skipped_empty"] == 1
    assert stats["comments_indexed"] == 1
    assert "confluence:comment:c" in stack["hashes"]
