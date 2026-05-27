"""Qdrant collection setup, indexing, and vector search."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from embedder import EMBEDDING_DIMENSION
from preprocess import Document


COLLECTION_NAME = "semantic_documents"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


class QdrantDocumentStore:
    """A lightweight Qdrant wrapper for localhost or in-memory storage."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        vector_size: int = EMBEDDING_DIMENSION,
        mode: str = "localhost",
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.mode = mode

        if mode == "memory":
            self.client = QdrantClient(":memory:")
        elif mode == "localhost":
            self.client = QdrantClient(host=host, port=port)
        else:
            raise ValueError("mode must be either 'localhost' or 'memory'")

    def recreate_collection(self) -> None:
        """Create a fresh cosine-similarity collection."""

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        batch_size: int = 128,
    ) -> None:
        """Store document id, original text, embedding, and category label."""

        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")

        for start in range(0, len(documents), batch_size):
            batch_documents = documents[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]

            points = [
                PointStruct(
                    id=document.id,
                    vector=embedding,
                    payload={
                        "document_id": document.id,
                        "text": document.original_text,
                        "category": document.category,
                    },
                )
                for document, embedding in zip(batch_documents, batch_embeddings)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> Any:
        """Search Qdrant and return scored points with payloads."""

        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return response.points
