"""Smoke test for the Part 1 semantic search pipeline."""

#end to end pipeline test  

from __future__ import annotations

from embedder import TextEmbedder
from preprocess import DEFAULT_LOCAL_ARCHIVE, load_20newsgroups_documents
from qdrant_store import QdrantDocumentStore
from search import SemanticSearcher


def main() -> None:
    """Load data, insert vectors, and perform one semantic search query."""

    documents = load_20newsgroups_documents(
        max_documents=500,
        archive_path=DEFAULT_LOCAL_ARCHIVE,
    )
    print(f"Loaded {len(documents)} preprocessed documents.")

    embedder = TextEmbedder()       #initializes embedding model    
    embeddings = embedder.embed_texts([document.text for document in documents])    #generate embeddings

    store = QdrantDocumentStore(mode="localhost")
    store.recreate_collection()
    store.upsert_documents(documents, embeddings.astype(float).tolist())    #embedding models return numpy. but quadrant returns python lists
    print("Inserted vectors into the semantic_documents collection.")

    searcher = SemanticSearcher(store=store, embedder=embedder)
    results = searcher.search(
        query="space shuttle mission and nasa orbit",
        top_k=5,
    )

    print("\nSearch results:")
    for index, result in enumerate(results, start=1):
        preview = result.document_text[:240].replace("\n", " ")
        print(f"\n{index}. score={result.similarity_score:.4f}")
        print(f"category={result.category}")
        print(f"text={preview}...")


if __name__ == "__main__":
    main()
