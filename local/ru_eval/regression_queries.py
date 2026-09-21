"""RU/EN end-to-end regression trace.

For each query, prints every retrieval stage separately — dense-only
candidates, sparse-only candidates, the server-side RRF fusion the app
actually uses, the reranker's ordering, and the context that would reach the
generator — so a regression can be attributed to one stage instead of being
visible only as a worse final answer.

Runs against the same Qdrant collection the API serves, but loads its own
embedder/reranker (CPU by default, so it can run while the API holds the GPU):

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= ./venv/bin/python local/ru_eval/regression_queries.py
"""
import os
import textwrap

from embeddings.embedding_service import EmbeddingService
from ingestion.chunker import chunk_context_text
from rag.prompt_builder import PromptBuilder
from rag.reranker import CrossEncoderReranker
from vector_db.qdrant_client import VectorStore
from vector_db.sparse_encoder import build_sparse_vector

QUERIES = [
    ("A", "Какие основные сущности есть в ISMT?"),
    ("B", "Что такое External Invoice?"),
    ("C", "A016ISMT-901"),
    ("D", "Какой сегодня курс биткоина на бирже?"),
]
POOL = 10
TOP_K = 5


def _label(hit: dict) -> str:
    return f"{hit.get('filename', '?')[:28]:28s} c{hit.get('chunk_index', 0):<2d}"


def main() -> None:
    store = VectorStore(
        url=os.getenv("QDRANT_URL", "http://localhost:6335"),
        collection=os.getenv("QDRANT_COLLECTION", "knowledge_base"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=30.0,
    )
    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    reranker = CrossEncoderReranker(model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.1"))
    prompt_builder = PromptBuilder()

    for tag, query in QUERIES:
        print("\n" + "=" * 78)
        print(f"[{tag}] {query}")
        print("=" * 78)

        query_vector = embedder.embed_text(query)
        sparse = build_sparse_vector(query)
        print(f"sparse terms in query: {len(sparse.indices)}")

        dense = store.search(query_vector, top_k=POOL)
        print(f"\n-- dense candidates ({len(dense)}) --")
        for hit in dense[:5]:
            print(f"   {hit['score']:.4f}  {_label(hit)}")

        sparse_hits = store.client.query_points(
            collection_name=store.collection, query=sparse, using="bm25",
            limit=POOL, with_payload=True,
        ).points if sparse.indices else []
        print(f"\n-- sparse (BM25) candidates ({len(sparse_hits)}) --")
        for point in sparse_hits[:5]:
            payload = point.payload or {}
            print(f"   {point.score:.4f}  {payload.get('filename', '?')[:28]:28s} c{payload.get('chunk_index', 0):<2d}")
        if not sparse.indices:
            print("   (none — query produced an empty sparse vector)")

        fused = store.hybrid_search(query_vector, query, top_k=POOL)
        print(f"\n-- RRF fusion ({len(fused)}) --")
        for hit in fused[:5]:
            print(f"   {hit['score']:.4f}  {_label(hit)}")

        reranked = reranker.rerank(query, [dict(h) for h in fused], top_k=TOP_K)
        print(f"\n-- reranker top {TOP_K} --")
        for hit in reranked:
            print(f"   {hit['rerank_score']:.4f}  {_label(hit)}")

        best = max((h.get("rerank_score", 0) for h in reranked), default=0)
        print(f"\n-- final context (threshold {threshold}, best {best:.4f}) --")
        if best < threshold:
            print("   REFUSED — nothing above the relevance threshold")
            continue
        messages = prompt_builder.build(query=query, chunks=reranked)
        context = messages[-1]["content"].split("<question>")[0].strip()
        print(textwrap.indent(context[:900], "   "))


if __name__ == "__main__":
    main()
