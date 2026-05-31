"""Smoke test for Part 3 semantic cache.

Run Qdrant first and populate vectors:
    docker compose up -d
    python test_semantic_search.py

Run:
    python test_semantic_cache.py
"""

from __future__ import annotations

from embedder import TextEmbedder
from qdrant_store import QdrantDocumentStore
from search import SemanticSearcher
from semantic_cache import SemanticCache, build_query_clusterer


def print_cache_result(query: str, cache_hit: bool, matched_query: str | None, score: float) -> None:
    """Print one cache lookup summary."""

    print(
        f"query={query!r} "
        f"cache_hit={cache_hit} "
        f"matched_query={matched_query!r} "
        f"similarity={score:.4f}"
    )


def main() -> None:
    """Exercise misses, hits, metrics, and threshold experiments."""

    store = QdrantDocumentStore(mode="localhost")
    embedder = TextEmbedder()
    searcher = SemanticSearcher(store=store, embedder=embedder)
    query_clusterer = build_query_clusterer(store=store)

    cache = SemanticCache(
        embedder=embedder,
        searcher=searcher,
        similarity_threshold=0.70,
        query_clusterer=query_clusterer,
        cluster_aware=True,
    )

    queries = [
        "NASA space shuttle orbit mission",
        "space shuttle mission and nasa orbit",
        "baseball pitching statistics",
        "best baseball pitcher stats",
        "windows driver configuration issue",
        "middle east politics and diplomacy",
    ]

    for query in queries:
        result = cache.search(query=query, top_k=5)
        print_cache_result(
            query=query,
            cache_hit=result.cache_hit,
            matched_query=result.matched_query,
            score=result.similarity_score,
        )

    print(
        "\nCache metrics: "
        f"hits={cache.metrics.hit_count} "
        f"misses={cache.metrics.miss_count} "
        f"hit_rate={cache.metrics.hit_rate:.3f}"
    )

    cache.evaluate_thresholds(
        thresholds=[0.7, 0.8, 0.9],
        labeled_query_pairs=[
            (
                "NASA space shuttle orbit mission",
                "space shuttle mission and nasa orbit",
                True,
            ),
            (
                "baseball pitching statistics",
                "best baseball pitcher stats",
                True,
            ),
            (
                "windows driver configuration issue",
                "fix display driver in windows",
                True,
            ),
            (
                "NASA space shuttle orbit mission",
                "used motorcycle paint repair",
                False,
            ),
            (
                "baseball pitching statistics",
                "middle east diplomacy conflict",
                False,
            ),
        ],
    )


if __name__ == "__main__":
    main()
