"""Stage-by-stage trace of the retrieval pipeline for one or more queries.

Prints what each stage actually produced — expansion, dense candidates,
sparse candidates, the server-side RRF fusion, the reranker, the threshold
decision, the context handed to the generator — so a regression can be
attributed to a stage instead of being visible only as a worse answer.

Loads its own embedder/reranker (CPU by default) so it can run while the API
holds the GPU; the final answer and sources come from the live API, so they
are the real ones a user would get.

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
      ./venv/bin/python local/ru_eval/trace_queries.py "Какие основные сущности есть в ISMT?"
"""
import os
import sys
import textwrap

import httpx

from embeddings.embedding_service import EmbeddingService
from rag.prompt_builder import PromptBuilder
from rag.reranker import CrossEncoderReranker
from vector_db.qdrant_client import VectorStore
from vector_db.sparse_encoder import build_sparse_vector, tokenize

POOL = 10
TOP_K = 5
API = os.getenv("RAG_API_URL", "http://localhost:8000")


def _label(hit: dict) -> str:
    return f"{(hit.get('filename') or '?')[:44]:44s} c{hit.get('chunk_index', 0):<3d}"


def main(queries: list[str]) -> None:
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

    for query in queries:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)

        with httpx.Client(timeout=900) as client:
            answer = client.post(f"{API}/query", json={"question": query, "top_k": TOP_K, "debug": True}).json()
        debug = answer.get("debug") or {}

        print("\n--- 1. query expansion ---")
        for variant in debug.get("expanded_queries") or [query]:
            print(f"   {variant}")

        query_vector = embedder.embed_text(query)
        sparse = build_sparse_vector(query)
        print(f"\n--- 2. sparse terms ({len(sparse.indices)}) ---")
        print(f"   {tokenize(query)}")

        dense = store.search(query_vector, top_k=POOL)
        print(f"\n--- 3. dense candidates (top {min(5, len(dense))} of {len(dense)}) ---")
        for hit in dense[:5]:
            print(f"   {hit['score']:.4f}  {_label(hit)}")

        sparse_hits = store.client.query_points(
            collection_name=store.collection, query=sparse, using="bm25",
            limit=POOL, with_payload=True,
        ).points if sparse.indices else []
        print(f"\n--- 4. sparse BM25 candidates (top {min(5, len(sparse_hits))} of {len(sparse_hits)}) ---")
        for point in sparse_hits[:5]:
            payload = point.payload or {}
            print(f"   {point.score:.4f}  {(payload.get('filename') or '?')[:44]:44s} c{payload.get('chunk_index', 0):<3d}")
        if not sparse.indices:
            print("   (none — query produced an empty sparse vector)")

        fused = store.hybrid_search(query_vector, query, top_k=POOL)
        print(f"\n--- 5. RRF fusion (top {min(5, len(fused))} of {len(fused)}) ---")
        for hit in fused[:5]:
            print(f"   {hit['score']:.4f}  {_label(hit)}")

        reranked = reranker.rerank(query, [dict(h) for h in fused], top_k=TOP_K)
        print(f"\n--- 6. bge-reranker-v2-m3 (top {TOP_K}) ---")
        for hit in reranked:
            print(f"   {hit['rerank_score']:.4f}  {_label(hit)}")

        best = max((h.get("rerank_score", 0) for h in reranked), default=0)
        print(f"\n--- 7. threshold ({threshold}) ---")
        print(f"   best={best:.4f} -> {'ANSWER' if best >= threshold else 'REFUSE'}")

        if best >= threshold:
            messages = prompt_builder.build(query=query, chunks=reranked)
            context = messages[-1]["content"].split("<question>")[0].strip()
            print("\n--- 8. final context ---")
            print(textwrap.indent(context[:1400], "   "))

        print("\n--- 9. final answer (live API) ---")
        print(textwrap.indent(answer.get("answer", ""), "   "))
        print("\n--- 10. sources ---")
        for source in answer.get("sources", []):
            print(f"   {source['relevance_score']:.3f}  {source.get('title', '?')}")
            print(f"          {source.get('url', '(local document)')}")
            print(f"          {source.get('excerpt', '')[:110]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Какие основные сущности есть в ISMT?"])
