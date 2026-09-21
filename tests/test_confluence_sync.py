"""Tests for api/confluence.py — space sync semantics.

The four states a sync has to get right (new / unchanged / changed /
deleted upstream) are each asserted on BOTH the document registry and the
stored vectors, because those can disagree: the whole reason sync deletes a
changed document before re-ingesting it is that overwriting in place would
leave the tail of a shrinking page's chunks behind in Qdrant, where nothing
in the registry would ever show it.

Postgres, Qdrant and the embedder are faked; the chunker is the real one, so
chunk counts are what the production pipeline would actually produce.
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
SHORT_BODY = (
    "<h1>Заголовок</h1><p>Одно короткое описание системы ISMT, её резервуаров, "
    "датчиков уровня жидкости и концентраторов, умещающееся в один чанк.</p>"
)
LONG_BODY = "<h1>Заголовок</h1>" + "".join(
    f"<p>Абзац номер {i} про резервуары, датчики уровня жидкости, концентраторы и пороговые значения тревог в системе ISMT.</p>"
    for i in range(12)
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
            }

    def points_for(self, doc_id: str) -> list[dict]:
        return [p for p in self.client.points.values() if p["document_id"] == doc_id]


class FakeConfluenceServer:
    """A space whose pages can be edited/removed between syncs, served
    through the real ConfluenceClient over httpx.MockTransport — so the
    client's own pagination/auth/parsing runs in these tests too."""

    def __init__(self, pages: dict[str, dict]):
        self.pages = pages
        self.body_fetches: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/rest/api/content"):
            return httpx.Response(200, json={
                "results": [self._page_json(pid, with_body=False) for pid in self.pages],
                "_links": {},
            })
        page_id = path.rsplit("/", 1)[-1]
        if page_id not in self.pages:
            return httpx.Response(404)
        self.body_fetches.append(page_id)
        return httpx.Response(200, json=self._page_json(page_id, with_body=True))

    def _page_json(self, page_id: str, with_body: bool) -> dict:
        page = self.pages[page_id]
        data = {
            "id": page_id,
            "title": page["title"],
            "space": {"key": "AISMT"},
            "_links": {"webui": f"/display/AISMT/{page_id}"},
            "version": {"number": page["version"], "when": "2026-09-21T10:00:00.000+00:00"},
            "history": {"createdDate": "2026-09-20T10:00:00.000+00:00"},
        }
        if with_body:
            data["body"] = {"storage": {"value": page["body"]}}
        return data


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
    def _patch(server: FakeConfluenceServer):
        original_init = ConfluenceClient.__init__

        def init(self, **kwargs):
            kwargs.setdefault("base_url", "https://confluence.example.com")
            kwargs.setdefault("username", "user")
            kwargs.setdefault("password", "secret")
            kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
            kwargs["retry_sleep_seconds"] = 0
            original_init(self, **kwargs)

        monkeypatch.setattr(ConfluenceClient, "__init__", init)
    return _patch


def _page(title: str, body: str, version: int = 1) -> dict:
    return {"title": title, "body": body, "version": version}


# ── sync semantics ────────────────────────────────────────────────────────

async def test_initial_sync_indexes_every_page(stack, patch_client):
    server = FakeConfluenceServer({
        "1": _page("Глоссарий основных сущностей ISMT", LONG_BODY),
        "2": _page("External Invoice", SHORT_BODY),
    })
    patch_client(server)

    stats = await confluence.sync_space("AISMT", folder="AISMT")

    assert stats["added"] == 2
    assert stats["unchanged"] == 0
    assert stats["deleted"] == 0
    assert len(stack["documents"]) == 2
    assert set(stack["hashes"]) == {"confluence:1", "confluence:2"}
    titles = {d["filename"] for d in stack["documents"].values()}
    assert titles == {"Глоссарий основных сущностей ISMT", "External Invoice"}


async def test_sync_is_idempotent_and_skips_unchanged_pages(stack, patch_client):
    server = FakeConfluenceServer({"1": _page("Глоссарий основных сущностей ISMT", LONG_BODY)})
    patch_client(server)

    await confluence.sync_space("AISMT", folder="AISMT")
    doc_id_after_first = stack["hashes"]["confluence:1"]
    points_after_first = dict(stack["store"].client.points)
    fetches_after_first = len(server.body_fetches)

    stats = await confluence.sync_space("AISMT", folder="AISMT")

    assert stats == {**stats, "added": 0, "updated": 0, "deleted": 0, "unchanged": 1}
    assert stack["hashes"]["confluence:1"] == doc_id_after_first
    assert stack["store"].client.points == points_after_first
    # An unchanged page must not even be downloaded — only listed.
    assert len(server.body_fetches) == fetches_after_first


async def test_changed_page_replaces_its_document(stack, patch_client):
    server = FakeConfluenceServer({"1": _page("Глоссарий основных сущностей ISMT", LONG_BODY)})
    patch_client(server)
    await confluence.sync_space("AISMT", folder="AISMT")
    old_doc_id = stack["hashes"]["confluence:1"]

    server.pages["1"] = _page("Глоссарий основных сущностей ISMT (v2)", SHORT_BODY, version=2)
    stats = await confluence.sync_space("AISMT", folder="AISMT")

    assert stats["updated"] == 1
    assert stats["added"] == 0
    assert len(stack["documents"]) == 1
    new_doc_id = stack["hashes"]["confluence:1"]
    assert new_doc_id != old_doc_id
    assert stack["documents"][new_doc_id]["filename"] == "Глоссарий основных сущностей ISMT (v2)"
    assert stack["documents"][new_doc_id]["metadata"]["version"] == 2


async def test_shrinking_page_leaves_no_stale_vectors(stack, patch_client):
    """The regression this whole delete-then-reingest strategy exists for: a
    page that goes from many chunks to few must end up with exactly the few,
    not the few plus the old tail."""
    server = FakeConfluenceServer({"1": _page("Длинная страница", LONG_BODY)})
    patch_client(server)
    await confluence.sync_space("AISMT", folder="AISMT")
    old_doc_id = stack["hashes"]["confluence:1"]
    old_chunk_count = len(stack["store"].points_for(old_doc_id))
    assert old_chunk_count > 1, "fixture must produce a multi-chunk page to be meaningful"

    server.pages["1"] = _page("Длинная страница", SHORT_BODY, version=2)
    await confluence.sync_space("AISMT", folder="AISMT")

    new_doc_id = stack["hashes"]["confluence:1"]
    new_points = stack["store"].points_for(new_doc_id)
    assert len(new_points) < old_chunk_count
    # Nothing at all is left from the previous version, under any document id.
    assert stack["store"].points_for(old_doc_id) == []
    assert len(stack["store"].client.points) == len(new_points)
    assert all("Абзац номер" not in p["text"] for p in stack["store"].client.points.values())


async def test_page_deleted_upstream_is_deleted_locally(stack, patch_client):
    server = FakeConfluenceServer({
        "1": _page("Остаётся", SHORT_BODY),
        "2": _page("Удаляется", SHORT_BODY),
    })
    patch_client(server)
    await confluence.sync_space("AISMT", folder="AISMT")
    removed_doc_id = stack["hashes"]["confluence:2"]

    del server.pages["2"]
    stats = await confluence.sync_space("AISMT", folder="AISMT")

    assert stats["deleted"] == 1
    assert "confluence:2" not in stack["hashes"]
    assert removed_doc_id not in stack["documents"]
    assert stack["store"].points_for(removed_doc_id) == []
    # The surviving page is untouched.
    assert stack["hashes"]["confluence:1"] in stack["documents"]


async def test_limited_sync_never_deletes(stack, patch_client):
    """With a limit the listing is a prefix of the space, so a page's absence
    from it proves nothing — deleting on that basis would wipe real pages."""
    server = FakeConfluenceServer({
        "1": _page("Первая", SHORT_BODY),
        "2": _page("Вторая", SHORT_BODY),
    })
    patch_client(server)
    await confluence.sync_space("AISMT", folder="AISMT")

    stats = await confluence.sync_space("AISMT", folder="AISMT", limit=1)

    assert stats["deleted"] == 0
    assert len(stack["documents"]) == 2


async def test_empty_page_is_skipped_not_indexed(stack, patch_client):
    server = FakeConfluenceServer({
        "1": _page("Пустая", "<ac:structured-macro ac:name='toc'><ac:parameter ac:name='style'>none</ac:parameter></ac:structured-macro>"),
    })
    patch_client(server)

    stats = await confluence.sync_space("AISMT", folder="AISMT")

    assert stats["skipped_empty"] == 1
    assert stats["added"] == 0
    assert stack["documents"] == {}


async def test_one_failing_page_does_not_abort_the_space(stack, patch_client):
    server = FakeConfluenceServer({
        "1": _page("Хорошая", SHORT_BODY),
        "2": _page("Битая", SHORT_BODY),
    })

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/2"):
            return httpx.Response(500)
        return server.handler(request)

    broken = FakeConfluenceServer(server.pages)
    broken.handler = handler  # type: ignore[method-assign]
    patch_client(broken)

    stats = await confluence.sync_space("AISMT", folder="AISMT")

    assert stats["failed"] == 1
    assert stats["added"] == 1
    assert "confluence:1" in stack["hashes"]


async def test_synced_document_carries_confluence_metadata(stack, patch_client):
    server = FakeConfluenceServer({"7": _page("Привилегии", SHORT_BODY, version=3)})
    patch_client(server)

    await confluence.sync_space("AISMT", folder="AISMT")

    doc = stack["documents"][stack["hashes"]["confluence:7"]]
    assert doc["format"] == "confluence"
    assert doc["folder"] == "AISMT"
    meta = doc["metadata"]
    assert meta["source"] == "confluence"
    assert meta["page_id"] == "7"
    assert meta["space_key"] == "AISMT"
    assert meta["confluence_url"] == "https://confluence.example.com/display/AISMT/7"
    assert meta["version"] == 3


async def test_page_too_short_to_chunk_is_skipped_not_failed(stack, patch_client):
    """A page with real but sub-min_chunk_size text (a stub, or a body that
    is entirely a diagram macro) produces no chunks. That is an empty page,
    not a sync failure — reporting it as failed would make every run of a
    space containing one look broken."""
    server = FakeConfluenceServer({"1": _page("Заглушка", "<p>Слишком коротко.</p>")})
    patch_client(server)

    stats = await confluence.sync_space("AISMT", folder="AISMT")

    assert stats["failed"] == 0
    assert stats["skipped_empty"] == 1
    assert stack["documents"] == {}
