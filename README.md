# Semantic Search Pipeline

Part 1 semantic search pipeline using Python, Qdrant, sentence-transformers, and scikit-learn.

## Project Structure

```text
app/          FastAPI app, routes, and Pydantic schemas
cache/        In-memory semantic cache and cache metrics
clustering/   GMM fuzzy clustering, cluster analysis, and visualization
core/         Preprocessing, embeddings, Qdrant storage, and search
scripts/      Smoke-test scripts for each project phase
artifacts/    Generated plots and visual outputs
```

## Run with Localhost Qdrant

Start Qdrant locally first:

```bash
docker compose up -d
```

Then install dependencies and run the smoke test:

```bash
pip install -r requirements.txt
python scripts/test_semantic_search.py
```

Qdrant will be available at `http://localhost:6333`.

The test script loads the local 20 Newsgroups archive, embeds documents with
`sentence-transformers/all-MiniLM-L6-v2`, stores vectors in the
`semantic_documents` collection, and runs one semantic search query.

## Run Fuzzy Clustering

After the vector index has been populated, run:

```bash
python scripts/test_fuzzy_clustering.py
```

This loads all document embeddings from Qdrant, trains a Gaussian Mixture Model,
stores `dominant_cluster` and `cluster_probabilities` back into Qdrant payloads,
prints representative, ambiguous, and high-confidence documents, runs cluster
count experiments with BIC, AIC, and silhouette score, and saves:

- `artifacts/cluster_visualization.png`
- `artifacts/cluster_model_selection.png`

## Run Semantic Cache

After Qdrant has vectors loaded, run:

```bash
python scripts/test_semantic_cache.py
```

The cache is in memory only. It stores the original query, query embedding,
retrieval result, dominant cluster, and timestamp. Incoming queries are embedded
and compared with cached query embeddings using cosine similarity. When
cluster-aware mode is enabled, only cached entries in the same dominant GMM
cluster are compared.

## Run FastAPI Service

Make sure Qdrant is running and the `semantic_documents` collection has vectors:

```bash
docker compose up -d
python scripts/test_semantic_search.py
```

Start the API:

```bash
python -m uvicorn app.main:app --reload
```

The service is available at `http://127.0.0.1:8000`.

### Query Documents

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "space shuttle mission and nasa orbit"}'
```

Response shape:

```json
{
  "query": "space shuttle mission and nasa orbit",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.0,
  "dominant_cluster": 3,
  "results": [
    {
      "document_text": "...",
      "similarity_score": 0.42,
      "category": "sci.space"
    }
  ]
}
```

### Cache Stats

```bash
curl "http://127.0.0.1:8000/cache/stats"
```

Response shape:

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### Clear Cache

```bash
curl -X DELETE "http://127.0.0.1:8000/cache"
```

## In-Memory Option

For a no-server run, instantiate the store with:

```python
QdrantDocumentStore(mode="memory")
```
