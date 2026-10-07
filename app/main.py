"""FastAPI entry point for the semantic search service."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    """Serve the presentation UI."""

    return FileResponse("app/static/index.html")
