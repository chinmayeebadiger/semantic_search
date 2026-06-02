"""Pydantic request and response models for the semantic search API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., min_length=1, description="Natural language query")


class SearchResultResponse(BaseModel):
    """One retrieved document returned by the API."""

    document_text: str
    similarity_score: float
    category: str


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float
    dominant_cluster: int | None
    results: list[SearchResultResponse]


class CacheStatsResponse(BaseModel):
    """Response body for GET /cache/stats."""

    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class CacheClearResponse(BaseModel):
    """Response body for DELETE /cache."""

    cleared: bool
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
