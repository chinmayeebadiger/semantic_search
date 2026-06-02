"""FastAPI routes for semantic search and cache management."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.schemas import (
    CacheClearResponse,
    CacheStatsResponse,
    QueryRequest,
    QueryResponse,
    SearchResultResponse,
)
from cache.semantic_cache import SemanticCache, build_query_clusterer
from core.embedder import TextEmbedder
from core.qdrant_store import QdrantDocumentStore
from core.search import SearchResult, SemanticSearcher


router = APIRouter()
DEFAULT_TOP_K = 5
DEFAULT_CACHE_THRESHOLD = 0.70


@lru_cache(maxsize=1)
def get_semantic_cache() -> SemanticCache:
    """Build and reuse the API's in-memory semantic cache.

    This is the service dependency graph:
    QdrantDocumentStore -> SemanticSearcher -> SemanticCache.
    The query clusterer is trained from embeddings already stored in Qdrant, so
    cluster-aware cache checks can compare only entries from the same dominant
    semantic region.
    """

    store = QdrantDocumentStore(mode="localhost")
    embedder = TextEmbedder()
    searcher = SemanticSearcher(store=store, embedder=embedder)
    query_clusterer = build_query_clusterer(store=store)

    return SemanticCache(
        embedder=embedder,
        searcher=searcher,
        similarity_threshold=DEFAULT_CACHE_THRESHOLD,
        query_clusterer=query_clusterer,
        cluster_aware=True,
    )


def _to_result_response(result: SearchResult) -> SearchResultResponse:
    """Convert internal search dataclass into a Pydantic response model."""

    return SearchResultResponse(
        document_text=result.document_text,
        similarity_score=result.similarity_score,
        category=result.category,
    )


@router.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    cache: SemanticCache = Depends(get_semantic_cache),
) -> QueryResponse:
    """Embed a query, check semantic cache, and fall back to Qdrant retrieval.

    Request flow:
    1. The cache embeds the incoming query.
    2. The query is assigned to a dominant GMM cluster.
    3. Cached query embeddings in that cluster are compared with cosine
       similarity.
    4. On a hit, the cached retrieval result is returned.
    5. On a miss, Qdrant vector retrieval runs and the result is cached.
    """

    try:
        lookup = cache.search(query=request.query, top_k=DEFAULT_TOP_K)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Semantic search service is not ready. "
                "Confirm Qdrant is running and semantic_documents has vectors."
            ),
        ) from exc

    return QueryResponse(
        query=request.query,
        cache_hit=lookup.cache_hit,
        matched_query=lookup.matched_query,
        similarity_score=lookup.similarity_score,
        dominant_cluster=lookup.dominant_cluster,
        results=[
            _to_result_response(result)
            for result in lookup.retrieval_result
        ],
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats(
    cache: SemanticCache = Depends(get_semantic_cache),
) -> CacheStatsResponse:
    """Return in-memory cache size and hit/miss metrics."""

    return CacheStatsResponse(
        total_entries=len(cache.entries),
        hit_count=cache.metrics.hit_count,
        miss_count=cache.metrics.miss_count,
        hit_rate=cache.metrics.hit_rate,
    )


@router.delete("/cache", response_model=CacheClearResponse)
def clear_cache(
    cache: SemanticCache = Depends(get_semantic_cache),
) -> CacheClearResponse:
    """Clear cached query results and reset cache statistics."""

    cache.clear()
    return CacheClearResponse(
        cleared=True,
        total_entries=len(cache.entries),
        hit_count=cache.metrics.hit_count,
        miss_count=cache.metrics.miss_count,
        hit_rate=cache.metrics.hit_rate,
    )
