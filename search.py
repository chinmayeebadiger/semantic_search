"""Semantic search interface over the in-memory Qdrant document store."""

from __future__ import annotations

from dataclasses import dataclass

from embedder import TextEmbedder
from qdrant_store import QdrantDocumentStore


@dataclass(frozen=True)
class SearchResult:
    """A single semantic search result."""

    document_text: str
    similarity_score: float
    category: str


class SemanticSearcher:
    """Embed text queries and retrieve nearest documents from Qdrant."""

    def __init__(self, store: QdrantDocumentStore, embedder: TextEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return document text, cosine similarity score, and category."""

        query_vector = self.embedder.embed_query(query)
        matches = self.store.search_vectors(query_vector=query_vector, top_k=top_k)

        results: list[SearchResult] = []
        for match in matches:
            payload = match.payload or {}
            results.append(
                SearchResult(
                    document_text=str(payload.get("text", "")),
                    similarity_score=float(match.score),
                    category=str(payload.get("category", "")),
                )
            )

        return results
