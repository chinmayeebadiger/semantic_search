# Semantic Search Pipeline

Part 1 semantic search pipeline using Python, Qdrant, sentence-transformers, and scikit-learn.

## Run with Localhost Qdrant

Start Qdrant locally first:

```bash
docker compose up -d
```

Then install dependencies and run the smoke test:

```bash
pip install -r requirements.txt
python test_semantic_search.py
```

Qdrant will be available at `http://localhost:6333`.

The test script loads the local 20 Newsgroups archive, embeds documents with
`sentence-transformers/all-MiniLM-L6-v2`, stores vectors in the
`semantic_documents` collection, and runs one semantic search query.

## Run Fuzzy Clustering

After the vector index has been populated, run:

```bash
python test_fuzzy_clustering.py
```

This loads all document embeddings from Qdrant, trains a Gaussian Mixture Model,
stores `dominant_cluster` and `cluster_probabilities` back into Qdrant payloads,
prints representative, ambiguous, and high-confidence documents, runs cluster
count experiments with BIC, AIC, and silhouette score, and saves:

- `cluster_visualization.png`
- `cluster_model_selection.png`

## Run Semantic Cache

After Qdrant has vectors loaded, run:

```bash
python test_semantic_cache.py
```

The cache is in memory only. It stores the original query, query embedding,
retrieval result, dominant cluster, and timestamp. Incoming queries are embedded
and compared with cached query embeddings using cosine similarity. When
cluster-aware mode is enabled, only cached entries in the same dominant GMM
cluster are compared.

## In-Memory Option

For a no-server run, instantiate the store with:

```python
QdrantDocumentStore(mode="memory")
```
