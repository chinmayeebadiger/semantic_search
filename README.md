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

## In-Memory Option

For a no-server run, instantiate the store with:

```python
QdrantDocumentStore(mode="memory")
```
