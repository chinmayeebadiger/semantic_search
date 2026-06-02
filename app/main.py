"""FastAPI entry point for the semantic search service."""

from __future__ import annotations

from fastapi import FastAPI

from app.routes import router


app = FastAPI(
    title="Semantic Search Service",
    description=(
        "Semantic search API with Qdrant retrieval, GMM cluster-aware "
        "semantic cache, and in-memory cache metrics."
    ),
    version="0.4.0",
)

app.include_router(router)
