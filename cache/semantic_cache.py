"""In-memory semantic cache for query-result reuse.

Semantic caching reuses a previous retrieval result when a new query is close
enough in embedding space. This is different from an exact string cache: the
queries "NASA shuttle launch" and "space shuttle mission" may be phrased
differently but still ask for similar results.

The similarity threshold controls the main tradeoff:
- lower thresholds increase hits but risk false positives
- higher thresholds reduce false positives but create more misses

Cluster-aware matching narrows comparisons to cached queries in the same
dominant GMM cluster. That keeps lookup cost lower as the cache grows and also
reduces accidental matches across unrelated semantic regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from cache.cache_metrics import (
    CacheMetrics,
    ThresholdEvaluation,
    print_threshold_evaluations,
)
from clustering.gmm_cluster import load_document_embeddings, train_gmm, vectors_to_matrix
from core.embedder import TextEmbedder
from core.qdrant_store import QdrantDocumentStore
from core.search import SearchResult, SemanticSearcher


@dataclass(frozen=True)
class CacheEntry:
    """One cached query and its reusable retrieval result."""

    original_query: str
    query_embedding: list[float]
    retrieval_result: list[SearchResult]
    dominant_cluster: int | None
    timestamp: datetime


@dataclass(frozen=True)
class CacheLookupResult:
    """Result returned by a semantic cache lookup."""

    cache_hit: bool
    matched_query: str | None
    similarity_score: float
    retrieval_result: list[SearchResult]
    dominant_cluster: int | None


class QueryClusterer:
    """Assign incoming query embeddings to the GMM's dominant cluster."""

    def __init__(self, pca: PCA, model: GaussianMixture) -> None:
        self.pca = pca
        self.model = model

    def predict_cluster(self, query_embedding: list[float]) -> int:
        """Return the dominant cluster for one query embedding."""

        vector = np.asarray([query_embedding], dtype=np.float32)
        features = self.pca.transform(vector)
        probabilities = self.model.predict_proba(features)[0]
        return int(probabilities.argmax())


def build_query_clusterer(
    store: QdrantDocumentStore,
    n_clusters: int = 20,
    pca_components: int = 25,
    random_state: int = 42,
) -> QueryClusterer:
    """Train a query clusterer from document embeddings already in Qdrant."""

    documents = load_document_embeddings(store)
    embeddings = vectors_to_matrix(documents)
    max_components = min(embeddings.shape[0] - 1, embeddings.shape[1], pca_components)
    pca = PCA(n_components=max_components, random_state=random_state)
    cluster_features = pca.fit_transform(embeddings)
    model = train_gmm(
        embeddings=cluster_features,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    return QueryClusterer(pca=pca, model=model)


class SemanticCache:
    """In-memory semantic cache with optional cluster-aware lookup."""

    def __init__(
        self,
        embedder: TextEmbedder,
        searcher: SemanticSearcher,
        similarity_threshold: float = 0.85,
        query_clusterer: QueryClusterer | None = None,
        cluster_aware: bool = True,
    ) -> None:
        self.embedder = embedder
        self.searcher = searcher
        self.similarity_threshold = similarity_threshold
        self.query_clusterer = query_clusterer
        self.cluster_aware = cluster_aware
        self.entries: list[CacheEntry] = []
        self.metrics = CacheMetrics()

    def search(self, query: str, top_k: int = 5) -> CacheLookupResult:
        """Return cached results when the query is semantically similar enough."""

        query_embedding = self.embedder.embed_query(query)
        dominant_cluster = self._dominant_cluster(query_embedding)
        matched_entry, similarity_score = self._best_match(
            query_embedding=query_embedding,
            dominant_cluster=dominant_cluster,
            threshold=self.similarity_threshold,
        )

        if matched_entry is not None:
            self.metrics.record_hit()
            return CacheLookupResult(
                cache_hit=True,
                matched_query=matched_entry.original_query,
                similarity_score=similarity_score,
                retrieval_result=matched_entry.retrieval_result,
                dominant_cluster=dominant_cluster,
            )

        self.metrics.record_miss()
        retrieval_result = self.searcher.search(query=query, top_k=top_k)
        self.add_entry(
            query=query,
            query_embedding=query_embedding,
            retrieval_result=retrieval_result,
            dominant_cluster=dominant_cluster,
        )
        return CacheLookupResult(
            cache_hit=False,
            matched_query=None,
            similarity_score=similarity_score,
            retrieval_result=retrieval_result,
            dominant_cluster=dominant_cluster,
        )

    def add_entry(
        self,
        query: str,
        query_embedding: list[float],
        retrieval_result: list[SearchResult],
        dominant_cluster: int | None,
    ) -> None:
        """Store query, embedding, result, cluster, and timestamp in memory."""

        self.entries.append(
            CacheEntry(
                original_query=query,
                query_embedding=query_embedding,
                retrieval_result=retrieval_result,
                dominant_cluster=dominant_cluster,
                timestamp=datetime.now(timezone.utc),
            )
        )

    def clear(self) -> None:
        """Remove all cached entries and reset cache metrics."""

        self.entries.clear()
        self.metrics.reset()

    def evaluate_thresholds(
        self,
        thresholds: Iterable[float],
        labeled_query_pairs: list[tuple[str, str, bool]],
    ) -> list[ThresholdEvaluation]:
        """Evaluate threshold choices with labeled query pairs.

        Each pair is `(cached_query, incoming_query, should_hit)`. A false
        positive means the threshold reused a result when it should not have.
        A false negative means it missed a useful reusable result.
        """

        evaluations: list[ThresholdEvaluation] = []
        for threshold in thresholds:
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            for cached_query, incoming_query, should_hit in labeled_query_pairs:
                cached_embedding = self.embedder.embed_query(cached_query)
                incoming_embedding = self.embedder.embed_query(incoming_query)
                similarity = cosine_similarity(cached_embedding, incoming_embedding)
                predicted_hit = similarity >= threshold

                if predicted_hit and should_hit:
                    true_positives += 1
                elif predicted_hit and not should_hit:
                    false_positives += 1
                elif not predicted_hit and should_hit:
                    false_negatives += 1
                else:
                    true_negatives += 1

            evaluations.append(
                ThresholdEvaluation(
                    threshold=float(threshold),
                    true_positives=true_positives,
                    false_positives=false_positives,
                    true_negatives=true_negatives,
                    false_negatives=false_negatives,
                )
            )

        print_threshold_evaluations(evaluations)
        return evaluations

    def _dominant_cluster(self, query_embedding: list[float]) -> int | None:
        """Predict the query cluster when cluster-aware cache search is enabled."""

        if self.query_clusterer is None:
            return None
        return self.query_clusterer.predict_cluster(query_embedding)

    def _candidate_entries(
        self,
        dominant_cluster: int | None,
    ) -> list[CacheEntry]:
        """Return cache entries to compare against.

        If cluster-aware mode is active, only entries with the same dominant
        cluster are compared. If no clusterer is configured, the cache falls
        back to scanning all entries.
        """

        if not self.cluster_aware or dominant_cluster is None:
            return self.entries

        return [
            entry
            for entry in self.entries
            if entry.dominant_cluster == dominant_cluster
        ]

    def _best_match(
        self,
        query_embedding: list[float],
        dominant_cluster: int | None,
        threshold: float,
    ) -> tuple[CacheEntry | None, float]:
        """Find the highest-similarity cache entry above threshold."""

        best_entry: CacheEntry | None = None
        best_similarity = 0.0
        for entry in self._candidate_entries(dominant_cluster):
            similarity = cosine_similarity(query_embedding, entry.query_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_entry is None or best_similarity < threshold:
            return None, best_similarity

        return best_entry, best_similarity


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two query embeddings."""

    left_vector = np.asarray(left, dtype=np.float32)
    right_vector = np.asarray(right, dtype=np.float32)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)
