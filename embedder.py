"""Sentence-transformers embedding utilities."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class TextEmbedder:
    """Generate normalized text embeddings with all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of documents or queries."""

        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,  #instead of pythorch tensors
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, query: str) -> list[float]:
        """Embed ONE search query as a Qdrant-ready vector."""

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.astype(float).tolist()  # numpy array to python list
